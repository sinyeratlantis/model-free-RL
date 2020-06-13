import gym
import numpy as np
import scipy.signal
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


# ———————————————— 算法说明 ———————————————— #

# 本版相比SpinningUp做了如下修改：原代码主逻辑中首个buffer reward固定为0，本版前移一位，
# 并置last_value reward为0，由于SpinningUp SAC中采用如此实现，所以采取该定义方式。


# ———————————————— 算法定义 ———————————————— #

class PPO:
    def __init__(self, sess, env):
        self.sess = sess

        # ———————— 传参封装及占位符定义 ———————— #

        self.state_dim = env.observation_space.shape[0]
        self.action_num = env.action_space.n

        self.clip_ratio = 0.2
        self.target_kl = 0.01  # 控制停止训练的程度
        self.train_times = 80
        self.learning_rate = 1e-3

        self.states_ph = tf.placeholder(tf.float32, (None, self.state_dim))
        self.actions_ph = tf.placeholder(tf.int32, (None,))
        self.returns_ph = tf.placeholder(tf.float32, (None,))
        self.advantages_ph = tf.placeholder(tf.float32, (None,))
        self.old_log_probs_ph = tf.placeholder(tf.float32, (None,))

        # ———————— 神经网络定义 ———————— #

        with tf.variable_scope('policy'):
            layer = tf.layers.dense(self.states_ph, 64, tf.tanh)
            layer = tf.layers.dense(layer, 64, tf.tanh)
            action_values = tf.layers.dense(layer, self.action_num)

        with tf.variable_scope('value'):
            layer = tf.layers.dense(self.states_ph, 64, tf.tanh)
            layer = tf.layers.dense(layer, 64, tf.tanh)
            self.state_values = tf.squeeze(tf.layers.dense(layer, 1), axis=1)  # [None]

        # ———————— 动作选择与对数概率输出定义 ———————— #

        self.action = tf.squeeze(tf.random.categorical(action_values, 1), axis=1)

        actions_log_probs = tf.nn.log_softmax(action_values)  # 也用于policy loss

        one_hot_output_action = tf.one_hot(self.action, self.action_num)
        self.old_log_prob = tf.reduce_sum(one_hot_output_action * actions_log_probs, axis=1)

        # ———————— 训练更新定义 ———————— #

        one_hot_actions = tf.one_hot(self.actions_ph, self.action_num)
        action_log_probs = tf.reduce_sum(actions_log_probs * one_hot_actions, axis=1)
        ratio = tf.exp(action_log_probs - self.old_log_probs_ph)

        surr_loss = ratio * self.advantages_ph
        clip_loss = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * self.advantages_ph
        policy_loss = -tf.reduce_mean(tf.minimum(surr_loss, clip_loss))

        value_loss = tf.losses.mean_squared_error(self.returns_ph, self.state_values)
        loss = policy_loss + 2. * value_loss

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        self.approx_kl = tf.reduce_mean(self.old_log_probs_ph - action_log_probs)

    def choose_action(self, state):
        action, state_value, old_log_prob = sess.run(
            [self.action, self.state_values, self.old_log_prob],
            feed_dict={self.states_ph: state})
        return action, state_value, old_log_prob

    def calculate_state_value(self, state):  # 计算单个state的value
        state_value = sess.run(self.state_values, feed_dict={self.states_ph: state})
        return state_value

    def update(self, buffer):
        states, actions, returns, advantages, old_log_probs = buffer.generate()

        feed = {self.states_ph: states,
                self.actions_ph: actions,
                self.returns_ph: returns,
                self.advantages_ph: advantages,
                self.old_log_probs_ph: old_log_probs}

        for _ in range(self.train_times):
            _, kl = sess.run([self.train_op, self.approx_kl], feed_dict=feed)
            if kl > 1.5 * self.target_kl:
                break


class Buffer:  # 离散空间的buffer
    def __init__(self, buffer_size, state_dim):
        self.buffer_size = buffer_size
        self.state_dim = state_dim

        self.states_bf = np.zeros((self.buffer_size, self.state_dim), np.float32)
        self.actions_bf = np.zeros(self.buffer_size, np.float32)
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
        gamma = 0.99
        game_slice = slice(self.game_start_step, self.step)

        # ———————————————— 标准return（reward-to-go）版本 ———————————————— #
        # 以下函数作用是：自底向上求动态规划反传值, 1:nv, 2:0.9*nv+r1, 3:0.9*(0.9*nv+r1)+r2

        append_rewards = np.append(self.rewards_bf[game_slice], last_state_value)
        discount_rewards = scipy.signal.lfilter([1], [1, - gamma], append_rewards[::-1], axis=0)[::-1]

        self.returns_bf[game_slice] = discount_rewards[:-1]  # td_error=r+0.9*nv, 最后一项不包含nv
        self.advantages_bf[game_slice] = self.returns_bf[game_slice] - self.old_state_values_bf[game_slice]

        # # ———————————————— GAE版本 ———————————————— #
        #
        # gae_lambda = 0.97
        #
        # game_rewards = np.append(self.rewards_bf[game_slice], last_state_value)
        # game_values = np.append(self.old_state_values_bf[game_slice], last_state_value)
        #
        # deltas = game_rewards[:-1] + gamma * game_values[1:] - game_values[:-1]
        #
        # discount_deltas = scipy.signal.lfilter([1], [1, float(- gamma * gae_lambda)], deltas[::-1], axis=0)
        # self.advantages_bf[game_slice] = discount_deltas[::-1]
        #
        # discount_rewards = scipy.signal.lfilter([1], [1, float(-gamma)], game_rewards[::-1], axis=0)
        # self.returns_bf[game_slice] = discount_rewards[::-1][:-1]

        self.game_start_step = self.step  # 计算时定然是游戏结束或buffer满时了

    def generate(self):  # buffer清零和返回数据（on-policy）
        self.step = 0
        self.game_start_step = 0

        advantage_mean = np.mean(self.advantages_bf)
        advantage_std = np.std(self.advantages_bf)
        self.advantages_bf = (self.advantages_bf - advantage_mean) / advantage_std
        return self.states_bf, self.actions_bf, self.returns_bf, self.advantages_bf, self.old_log_probs_bf


# ———————————————— 超参数及初始化 ———————————————— #

env_name = 'CartPole-v0'

max_epoch = 1000
game_max_step = 400
buffer_size = 4000

env = gym.make(env_name).unwrapped
env.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

sess = tf.Session()
agent = PPO(sess, env)
sess.run(tf.global_variables_initializer())

buffer = Buffer(buffer_size, env.observation_space.shape[0])

# ———————————————— 主逻辑 ———————————————— #

for epoch in range(max_epoch):

    state = env.reset()
    step = 0  # 用于判断游戏是否达到最大步数
    while True:
        step += 1

        action, old_state_value, old_log_prob = agent.choose_action(state.reshape(1, -1))
        next_state, reward, done, _ = env.step(action[0])

        buffer.store(state, action, reward, old_state_value, old_log_prob)

        if done or step >= game_max_step or buffer.step == buffer_size:  # 游戏结束或buffer存满时计算值
            if done:
                last_state_value = 0
            else:
                last_state_value = agent.calculate_state_value(next_state.reshape(1, -1))
            buffer.calculate_value(last_state_value)

        if buffer.step == buffer_size:  # 仅buffer存满时更新
            agent.update(buffer)
            print('———————————————— update ————————————————')

        if done or step >= game_max_step:  # 仅游戏结束时重置游戏
            print('Epoch: %d, Reward: %d' % (epoch, step))
            break

        state = next_state
