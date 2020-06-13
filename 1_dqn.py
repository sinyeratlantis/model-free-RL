import gym
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


# ———————————————— 算法定义 ———————————————— # TODO 算法还未完善逻辑整理和性能最优化

class FullDQN:
    def __init__(self, sess, env):
        self.sess = sess

        # ———————— 传参封装及占位符定义 ———————— #

        self.state_dim = env.observation_space.shape[0]
        self.action_num = env.action_space.n
        self.lr = 0.005
        self.gamma = 0.9

        self.states_ph = tf.placeholder(tf.float32, (None, self.state_dim))  # 顺序：s-a-r-else
        self.actions_ph = tf.placeholder(tf.int32, (None, ))
        self.rewards_ph = tf.placeholder(tf.float32, (None, 1))

        # next_q_value由target_net给出，由外传入
        self.next_states_ph = tf.placeholder(tf.float32, (None, self.state_dim))
        self.next_q_values_ph = tf.placeholder(tf.float32, (None, self.action_num))

        # ———————— 神经网络定义 ———————— #

        with tf.variable_scope('eval_net'):  # 输出某状态下选择各个动作的价值，action's'表示多动作，value's'表示batch
            layer_eval = tf.layers.dense(self.states_ph, 20, tf.nn.relu)
            self.q_value_eval = tf.layers.dense(layer_eval, self.action_num, None)

        with tf.variable_scope('target_net'):
            # 由target_net负责计算next_q_value，但最终状态价值为0不由target_net给出，所以输出后修正再传入
            layer_target = tf.layers.dense(self.next_states_ph, 20, tf.nn.relu)
            self.next_q_value = tf.layers.dense(layer_target, self.action_num, None)

        # ———————— 训练更新定义 ———————— # q_target = r + gamma * max_a(q(s', a'))  返回list

        q_target = tf.stop_gradient(self.rewards_ph[0] + self.gamma * tf.reduce_max(self.next_q_values_ph[0], axis=0))

        loss = tf.reduce_mean(tf.squared_difference(q_target, self.q_value_eval[0][self.actions_ph[0]]))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

        target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        self.replace_op = [tf.assign(t, e) for t, e in zip(target_params, eval_params)]

    def choose_action(self, state):
        action_value = self.sess.run(self.q_value_eval, feed_dict={self.states_ph: [state]})
        if np.random.uniform() < 0.8:
            action_id = np.argmax(action_value)
        else:
            action_id = np.random.randint(0, 2)
        return action_id

    def update(self, buffer):
        states, actions, rewards, next_states, dones = buffer.random_sample(batch_size)

        # 最终状态价值不传入训练states中，奖励已经彻底描述了最终失败的不好，所以估计终态价值会引入偏差
        next_q_values = []
        for i, next_state in enumerate(next_states):
            if dones[i]:
                next_q_value = [[0 for _ in range(self.action_num)]]
            else:
                next_q_value = self.sess.run(self.next_q_value, feed_dict={self.next_states_ph: [next_state]})
            next_q_values.append(next_q_value[0])

        feed = {self.states_ph: states,
                self.actions_ph: actions,
                self.next_q_values_ph: next_q_values,
                self.rewards_ph: rewards}
        self.sess.run(self.train_op, feed_dict=feed)

    def replace(self):
        self.sess.run(self.replace_op)


class Buffer:
    def __init__(self, buffer_size, state_dim):
        self.step = 0
        self.buffer_size = buffer_size

        self.states_bf = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions_bf = np.zeros((buffer_size,), dtype=np.float32)
        self.rewards_bf = np.zeros((buffer_size, 1), dtype=np.float32)  # 与tf输出的q_value维度保持一致
        self.next_states_bf = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.dones_bf = np.zeros((buffer_size,), dtype=np.bool)

    def store(self, state, action, reward, next_state, done):
        self.states_bf[self.step % self.buffer_size] = state
        self.actions_bf[self.step % self.buffer_size] = action
        self.rewards_bf[self.step % self.buffer_size] = [reward]
        self.next_states_bf[self.step % self.buffer_size] = next_state
        self.dones_bf[self.step % self.buffer_size] = done

        self.step += 1

    def random_sample(self, batch_size):  # 大于batch_size就开始训练
        max_index = min(self.step, self.buffer_size)
        indices = np.random.choice(max_index, size=batch_size)
        return (self.states_bf[indices], self.actions_bf[indices], self.rewards_bf[indices],
                self.next_states_bf[indices], self.dones_bf[indices])


# ———————————————— 超参数及初始化 ———————————————— #

env_name = 'CartPole-v0'

buffer_size = 10000
batch_size = 64
max_epoch = 1000
max_step = 400  # 单回合游戏最长步数设定

update_interval = 1  # 网络更新的间隔，1即为单步更新
replace_interval = 30  # 每更新多少次替换一次参数

env = gym.make(env_name).unwrapped
env.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

sess = tf.Session()
agent = FullDQN(sess, env)
sess.run(tf.global_variables_initializer())

buffer = Buffer(buffer_size, env.observation_space.shape[0])

# ———————————————— 主逻辑 ———————————————— #

update_counter = 0  # 更新间隔计数指标
for epoch in range(max_epoch):
    state = env.reset()

    step = 0  # 用于游戏超时判断
    while True:
        step += 1

        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        if done:
            reward = -10

        buffer.store(state, action, reward, next_state, done)

        if buffer.step > batch_size and buffer.step % update_interval == 0:
            agent.update(buffer)
            update_counter += 1

            if update_counter % replace_interval == 0:  # 影响方差和性能
                agent.replace()

        if done or step >= max_step:
            print('Epoch: %d, Reward: %d' % (epoch, step))
            break

        state = next_state
