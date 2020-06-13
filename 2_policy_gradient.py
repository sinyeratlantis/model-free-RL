import gym
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


# ———————————————— 算法定义 ———————————————— # 实现为无baseline版，未性能最优化

class PG:
    def __init__(self, sess, env):
        self.sess = sess

        # ———————— 传参封装及占位符定义 ———————— #

        self.state_dim = env.observation_space.shape[0]
        self.action_num = env.action_space.n
        self.lr = 0.005
        self.gamma = 0.99

        self.states_ph = tf.placeholder(tf.float32, (None, self.state_dim))
        self.actions_ph = tf.placeholder(tf.int32, (None, ))
        self.returns_ph = tf.placeholder(tf.float32, (None, ))  # 后向累计奖励

        # ———————— 神经网络定义 ———————— #

        layer = tf.layers.dense(self.states_ph, 50, tf.nn.tanh)
        self.actions_probs = tf.layers.dense(layer, self.action_num, tf.nn.softmax)

        # ———————— 训练更新定义 ———————— #

        action_log_probs = tf.log(self.actions_probs) * tf.one_hot(self.actions_ph, self.action_num)
        action_log_probs = tf.reduce_sum(action_log_probs, axis=1)  # 所选动作的log概率

        loss = -tf.reduce_mean(action_log_probs * self.returns_ph)  # loss = log(p) * r
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, state):
        actions_probs = self.sess.run(self.actions_probs, feed_dict={self.states_ph: [state]})
        action_id = np.random.choice(range(actions_probs.shape[1]), p=actions_probs[0])
        return action_id

    def update(self, epoch_buffer):
        epoch_states, epoch_actions, epoch_rewards = [], [], []
        for [state, action, reward] in epoch_buffer:
            epoch_states.append(state)
            epoch_actions.append(action)
            epoch_rewards.append(reward)

        epoch_returns = np.zeros_like(epoch_rewards, dtype='float64')
        running_add = 0
        for i in reversed(range(len(epoch_rewards))):  # i即取值不含终态的所有状态
            running_add = epoch_rewards[i] + self.gamma * running_add
            epoch_returns[i] = running_add

        feed = {self.states_ph: epoch_states,  # 无baseline版
                self.actions_ph: epoch_actions,
                self.returns_ph: epoch_returns}
        self.sess.run(self.train_op, feed)


# ———————————————— 超参数及初始化 ———————————————— #

env_name = 'CartPole-v0'

max_epoch = 1000
max_step = 400

env = gym.make(env_name).unwrapped
env.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

sess = tf.Session()
agent = PG(sess, env)
sess.run(tf.global_variables_initializer())

# ———————————————— 主逻辑 ———————————————— #

for epoch in range(max_epoch):
    state = env.reset()

    step = 0
    epoch_buffer = []
    while True:
        step += 1

        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        if done:
            reward = -1

        epoch_buffer.append([state, action, reward])

        if done or step >= max_step:
            agent.update(epoch_buffer)
            print('Epoch: %d, Reward: %d' % (epoch, step))
            break

        state = next_state
