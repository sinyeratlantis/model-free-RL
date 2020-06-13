import gym
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


# ———————————————— 算法定义 ———————————————— # 单步更新版，未性能最优化

class AC:
    def __init__(self, sess, env):
        self.sess = sess

        # ———————— 传参封装及占位符定义 ———————— #

        self.state_dim = env.observation_space.shape[0]
        self.action_num = env.action_space.n
        self.lr = 0.001
        self.gamma = 0.99

        self.states_ph = tf.placeholder(tf.float32, (None, self.state_dim))
        self.actions_ph = tf.placeholder(tf.int32, (None,))
        self.target_values_ph = tf.placeholder(tf.float32, (None, 1))  # 代替reward的expected_return

        # ———————— 神经网络定义 ———————— #

        # Critic net
        critic_layer = tf.layers.dense(self.states_ph, 50, tf.nn.relu)
        self.state_values = tf.layers.dense(critic_layer, 1, None)
        advantages = self.target_values_ph - self.state_values

        # Actor net
        actor_layer = tf.layers.dense(self.states_ph, 50, tf.nn.relu)
        self.actions_probs = tf.layers.dense(actor_layer, self.action_num, tf.nn.softmax)

        # ———————— 训练更新定义 ———————— #

        action_log_probs = tf.log(self.actions_probs) * tf.one_hot(self.actions_ph, self.action_num)
        action_log_probs = tf.reduce_sum(action_log_probs, axis=1)

        policy_loss = -tf.reduce_mean(action_log_probs * tf.stop_gradient(advantages))
        value_loss = 0.25 * tf.square(advantages)  # 加权

        loss = policy_loss + value_loss  # 合并训练
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, state):
        actions_probs = self.sess.run(self.actions_probs, feed_dict={self.states_ph: [state]})
        action_id = np.random.choice(range(actions_probs.shape[1]), p=actions_probs[0])
        return action_id

    def update(self, state, action, reward, next_state, done):
        if done:
            next_state_value = np.array([[0]])
        else:
            next_state_value = self.sess.run(self.state_values, {self.states_ph: [next_state]})
        target_values = [[reward]] + self.gamma * next_state_value

        feed = {self.states_ph: [state],
                self.actions_ph: [action],
                self.target_values_ph: target_values}
        self.sess.run(self.train_op, feed)


# ———————————————— 超参数及初始化 ———————————————— #

env_name = 'CartPole-v0'

max_epoch = 2000
max_step = 400

env = gym.make(env_name).unwrapped
env.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

sess = tf.Session()
agent = AC(sess, env)
sess.run(tf.global_variables_initializer())

# ———————————————— 主逻辑 ———————————————— #

for epoch in range(max_epoch):
    state = env.reset()

    step = 0
    while True:
        step += 1

        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        if done:
            reward = -1

        agent.update(state, action, reward, next_state, done)

        if done or step >= max_step:
            print('Epoch: %d, Reward: %d' % (epoch, step))
            break

        state = next_state
