import gym
import numpy as np
import scipy.signal
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# ———————————————— MPI并行相关 ———————————————— # 可通过导包mpi_tf和mpi_tools，此处直接贴代码

import os, subprocess, sys
from mpi4py import MPI


class MpiTool:
    def mpi_fork(self, n, bind_to_core=False):
        if n <= 1:
            return
        if os.getenv("IN_MPI") is None:
            env = os.environ.copy()
            env.update(
                MKL_NUM_THREADS="1",
                OMP_NUM_THREADS="1",
                IN_MPI="1"
            )
            args = ["mpirun", "-np", str(n)]
            if bind_to_core:
                args += ["-bind-to", "core"]
            args += [sys.executable] + sys.argv
            subprocess.check_call(args, env=env)
            sys.exit()

    def msg(self, m, string=''):
        print(('Message from %d: %s \t ' % (MPI.COMM_WORLD.Get_rank(), string)) + str(m))

    def proc_id(self):
        return MPI.COMM_WORLD.Get_rank()

    def allreduce(self, *args, **kwargs):
        return MPI.COMM_WORLD.Allreduce(*args, **kwargs)

    def num_procs(self):
        return MPI.COMM_WORLD.Get_size()

    def broadcast(self, x, root=0):
        MPI.COMM_WORLD.Bcast(x, root=root)

    def mpi_op(self, x, op):
        x, scalar = ([x], True) if np.isscalar(x) else (x, False)
        x = np.asarray(x, dtype=np.float32)
        buff = np.zeros_like(x, dtype=np.float32)
        self.allreduce(x, buff, op=op)
        return buff[0] if scalar else buff

    def mpi_sum(self, x):
        return self.mpi_op(x, MPI.SUM)

    def mpi_avg(self, x):
        return self.mpi_sum(x) / self.num_procs()

    def mpi_statistics_scalar(self, x, with_min_and_max=False):
        x = np.array(x, dtype=np.float32)
        global_sum, global_n = self.mpi_sum([np.sum(x), len(x)])
        mean = global_sum / global_n
        global_sum_sq = self.mpi_sum(np.sum((x - mean) ** 2))
        std = np.sqrt(global_sum_sq / global_n)  # compute global std
        if with_min_and_max:
            global_min = self.mpi_op(np.min(x) if len(x) > 0 else np.inf, op=MPI.MIN)
            global_max = self.mpi_op(np.max(x) if len(x) > 0 else -np.inf, op=MPI.MAX)
            return mean, std, global_min, global_max
        return mean, std

    def flat_concat(self, xs):
        return tf.concat([tf.reshape(x, (-1,)) for x in xs], axis=0)

    def assign_params_from_flat(self, x, params):
        flat_size = lambda p: int(np.prod(p.shape.as_list()))  # the 'int' is important for scalars
        splits = tf.split(x, [flat_size(p) for p in params])
        new_params = [tf.reshape(p_new, p.shape) for p, p_new in zip(params, splits)]
        return tf.group([tf.assign(p, p_new) for p, p_new in zip(params, new_params)])

    def sync_params(self, params):
        get_params = self.flat_concat(params)
        def _broadcast(x):
            self.broadcast(x)
            return x
        synced_params = tf.py_func(_broadcast, [get_params], tf.float32)
        return self.assign_params_from_flat(synced_params, params)

    def sync_all_params(self):
        return self.sync_params(tf.global_variables())


mpi = MpiTool()


class MpiAdamOptimizer(tf.train.AdamOptimizer):
    def __init__(self, **kwargs):
        self.comm = MPI.COMM_WORLD
        tf.train.AdamOptimizer.__init__(self, **kwargs)

    def compute_gradients(self, loss, var_list, **kwargs):
        grads_and_vars = super().compute_gradients(loss, var_list, **kwargs)
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        flat_grad = mpi.flat_concat([g for g, v in grads_and_vars])
        shapes = [v.shape.as_list() for g, v in grads_and_vars]
        sizes = [int(np.prod(s)) for s in shapes]
        num_tasks = self.comm.Get_size()
        buf = np.zeros(flat_grad.shape, np.float32)
        def _collect_grads(flat_grad):
            self.comm.Allreduce(flat_grad, buf, op=MPI.SUM)
            np.divide(buf, float(num_tasks), out=buf)
            return buf
        avg_flat_grad = tf.py_func(_collect_grads, [flat_grad], tf.float32)
        avg_flat_grad.set_shape(flat_grad.shape)
        avg_grads = tf.split(avg_flat_grad, sizes, axis=0)
        avg_grads_and_vars = [(tf.reshape(g, v.shape), v)
                    for g, (_, v) in zip(avg_grads, grads_and_vars)]
        return avg_grads_and_vars

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        opt = super().apply_gradients(grads_and_vars, global_step, name)
        with tf.control_dependencies([opt]):
            sync = mpi.sync_params([v for g,v in grads_and_vars])
        return tf.group([opt, sync])


mpi.mpi_fork(4)  # 4并行


# ———————————————— 算法定义 ———————————————— #

class PPO:
    def __init__(self, sess, env):
        self.sess = sess

        # ———————— 传参封装及占位符定义 ———————— #

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]  # 连续空间

        self.clip_ratio = 0.2
        self.target_kl = 0.01  # 控制停止训练的程度
        self.train_policy_times = 80
        self.train_value_times = 80
        self.policy_lr = 3e-4
        self.value_lr = 1e-3

        self.states_ph = tf.placeholder(tf.float32, (None, self.state_dim))
        self.actions_ph = tf.placeholder(tf.float32, (None, self.action_dim))
        self.returns_ph = tf.placeholder(tf.float32, (None,))
        self.advantages_ph = tf.placeholder(tf.float32, (None,))
        self.old_log_probs_ph = tf.placeholder(tf.float32, (None,))

        # ———————— 神经网络定义 ———————— #

        # 此处十分奇怪的是policy定义需要在value前，否则性能会极大下降，具体原因不明
        with tf.variable_scope('policy'):
            layer = tf.layers.dense(self.states_ph, 64, tf.tanh)
            layer = tf.layers.dense(layer, 64, tf.tanh)
            mu = tf.layers.dense(layer, self.action_dim)  # 连续相关

        with tf.variable_scope('value'):
            layer = tf.layers.dense(self.states_ph, 64, tf.tanh)
            layer = tf.layers.dense(layer, 64, tf.tanh)
            self.state_values = tf.squeeze(tf.layers.dense(layer, 1), axis=1)  # [None]

        # ———————— 连续空间的动作选择与对数概率输出定义 ———————— #

        def gaussian_likelihood(x, mu, log_std):
            pre_sum = -0.5 * (((x - mu) / (tf.exp(log_std) + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))
            return tf.reduce_sum(pre_sum, axis=1)

        log_std = tf.get_variable(name='log_std', initializer=-0.5 * np.ones(self.action_dim, dtype=np.float32))
        self.action = mu + tf.random_normal(tf.shape(mu)) * tf.exp(log_std)  # pi
        action_log_probs = gaussian_likelihood(self.actions_ph, mu, log_std)  # logp
        self.old_log_probs = gaussian_likelihood(self.action, mu, log_std)  # logp_pi

        # ———————— 训练更新定义 ———————— #

        ratio = tf.exp(action_log_probs - self.old_log_probs_ph)

        surr_loss = ratio * self.advantages_ph
        clip_loss = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * self.advantages_ph
        policy_loss = -tf.reduce_mean(tf.minimum(surr_loss, clip_loss))
        value_loss = tf.losses.mean_squared_error(self.returns_ph, self.state_values)

        self.train_actor = MpiAdamOptimizer(learning_rate=self.policy_lr).minimize(policy_loss)
        self.train_critic = MpiAdamOptimizer(learning_rate=self.value_lr).minimize(value_loss)

        self.approx_kl = tf.reduce_mean(self.old_log_probs_ph - action_log_probs)

    def choose_action(self, state):
        action, state_value, old_log_prob = self.sess.run(
            [self.action, self.state_values, self.old_log_probs],
            feed_dict={self.states_ph: state})
        return action, state_value, old_log_prob

    def calculate_state_value(self, state):  # 计算单个state的value
        state_value = self.sess.run(self.state_values, feed_dict={self.states_ph: state})
        return state_value

    def update(self, buffer):
        states, actions, returns, advantages, old_log_probs = buffer.generate()

        feed = {self.states_ph: states,
                self.actions_ph: actions,
                self.returns_ph: returns,
                self.advantages_ph: advantages,
                self.old_log_probs_ph: old_log_probs}

        for _ in range(self.train_policy_times):
            _, kl = self.sess.run([self.train_actor, self.approx_kl], feed_dict=feed)
            kl = mpi.mpi_avg(kl)
            if kl > 1.5 * self.target_kl:
                break

        for _ in range(self.train_value_times):
            self.sess.run(self.train_critic, feed_dict=feed)


class Buffer:
    def __init__(self, buffer_size, state_dim, action_dim):
        self.buffer_size = buffer_size

        self.states_bf = np.zeros((self.buffer_size, state_dim), np.float32)
        self.actions_bf = np.zeros((self.buffer_size, action_dim), np.float32)
        self.rewards_bf = np.zeros(self.buffer_size, np.float32)
        self.old_state_values_bf = np.zeros(self.buffer_size, np.float32)
        self.old_log_probs_bf = np.zeros(self.buffer_size, np.float32)
        self.advantages_bf = np.zeros(self.buffer_size, np.float32)
        self.returns_bf = np.zeros(self.buffer_size, np.float32)

        self.step = 0
        self.game_start_step = 0

    def store(self, state, action, reward, old_state_value, old_log_prob):
        self.states_bf[self.step % self.buffer_size] = state.reshape(1, -1)  # 等同于[state]
        self.actions_bf[self.step % self.buffer_size] = action
        self.rewards_bf[self.step % self.buffer_size] = reward
        self.old_state_values_bf[self.step % self.buffer_size] = old_state_value
        self.old_log_probs_bf[self.step % self.buffer_size] = old_log_prob

        self.step += 1

    def calculate_value(self, last_state_value):
        game_slice = slice(self.game_start_step, self.step)

        # # ———————————————— 标准return（reward-to-go）版本 ———————————————— #
        #
        # gamma = 0.99
        #
        # # 以下函数作用是：自底向上求动态规划反传值, 1:nv, 2:0.9*nv+r1, 3:0.9*(0.9*nv+r1)+r2
        # append_rewards = np.append(self.rewards_bf[game_slice], last_state_value)
        # discount_rewards = scipy.signal.lfilter([1], [1, - gamma], append_rewards[::-1], axis=0)[::-1]
        #
        # self.returns_bf[game_slice] = discount_rewards[:-1]  # td_error=r+0.9*nv, 最后一项不包含nv
        # self.advantages_bf[game_slice] = self.returns_bf[game_slice] - self.old_state_values_bf[game_slice]

        # ———————————————— GAE版本 ———————————————— #

        gamma = 0.99
        gae_lambda = 0.97

        game_rewards = np.append(self.rewards_bf[game_slice], last_state_value)
        game_values = np.append(self.old_state_values_bf[game_slice], last_state_value)

        deltas = game_rewards[:-1] + gamma * game_values[1:] - game_values[:-1]
        discount_deltas = scipy.signal.lfilter([1], [1, float(- gamma * gae_lambda)], deltas[::-1], axis=0)
        self.advantages_bf[game_slice] = discount_deltas[::-1]

        discount_rewards = scipy.signal.lfilter([1], [1, float(-gamma)], game_rewards[::-1], axis=0)
        self.returns_bf[game_slice] = discount_rewards[::-1][:-1]

        self.game_start_step = self.step  # 计算时定然是游戏结束或buffer满时了

    def generate(self):  # 每次取后都从0开始存
        self.step = 0
        self.game_start_step = 0

        advantage_mean, advantage_std = mpi.mpi_statistics_scalar(self.advantages_bf)
        self.advantages_bf = (self.advantages_bf - advantage_mean) / advantage_std
        return self.states_bf, self.actions_bf, self.returns_bf, self.advantages_bf, self.old_log_probs_bf


# ———————————————— 超参数及初始化 ———————————————— #

env_name = 'HalfCheetah-v2'

max_epoch = 1000
game_max_step = 1000  # 针对HalfCheetah环境（一般用max_step / num_process）
buffer_size = game_max_step  # 意味着游戏长度不会超过buffer

env = gym.make(env_name).unwrapped
env.seed(10 * mpi.proc_id())
np.random.seed(10 * mpi.proc_id())
tf.set_random_seed(10 * mpi.proc_id())

sess = tf.Session()
agent = PPO(sess, env)
sess.run(tf.global_variables_initializer())
sess.run(mpi.sync_all_params())

buffer = Buffer(buffer_size, env.observation_space.shape[0], env.action_space.shape[0])

# ———————————————— 主逻辑 ———————————————— #

for epoch in range(max_epoch):

    sum_reward = 0
    state = env.reset()
    game_start_step = buffer.step

    step = 0  # 用于判断游戏是否达到最大步数
    while True:
        step += 1

        action, old_state_value, old_log_prob = agent.choose_action(state.reshape(1, -1))
        next_state, reward, done, _ = env.step(action[0])
        sum_reward += reward

        buffer.store(state, action, reward, old_state_value, old_log_prob)

        if done or step >= game_max_step or buffer.step == buffer_size:  # 游戏结束或buffer存满时计算值
            last_state_value = 0 if done else agent.calculate_state_value(next_state.reshape(1, -1))
            buffer.calculate_value(last_state_value)

        if buffer.step == buffer_size:  # 仅buffer存满时更新
            agent.update(buffer)

        if done or step >= game_max_step:  # 仅游戏结束时重置游戏
            if mpi.proc_id() == 0:
                print('Epoch: %d, Reward: %.2f' % (epoch, sum_reward))
            break

        state = next_state
