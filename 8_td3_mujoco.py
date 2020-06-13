import gym
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


# ———————————————— 算法定义 ———————————————— #

class TD3:
    def __init__(self, sess, env):
        self.sess = sess

        # ———————— 传参封装及占位符定义 ———————— #

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_limit = env.action_space.high[0]

        self.gamma = 0.99
        self.polyak = 0.995
        self.target_noise = 0.2
        self.noise_clip = 0.5
        self.policy_delay = 2
        self.actor_lr = 1e-3
        self.critic_lr = 1e-3

        self.states_ph = tf.placeholder(tf.float32, (None, self.state_dim))
        self.actions_ph = tf.placeholder(tf.float32, (None, self.action_dim))
        self.rewards_ph = tf.placeholder(tf.float32, (None,))
        self.next_states_ph = tf.placeholder(tf.float32, (None, self.state_dim))
        self.dones_ph = tf.placeholder(tf.float32, (None,))

        # ———————— 神经网络定义 ———————— #

        with tf.variable_scope('main'):
            self.policy, q_value1, q_value2, policy_value = self.actor_critic_net(self.states_ph, self.actions_ph)

        with tf.variable_scope('target'):
            target_policy, _, _, _ = self.actor_critic_net(self.next_states_ph, self.actions_ph)

        with tf.variable_scope('target', reuse=True):  # 平滑Q值
            epsilon = tf.random_normal(tf.shape(target_policy), stddev=self.target_noise)
            epsilon = tf.clip_by_value(epsilon, -self.noise_clip, self.noise_clip)

            target_action = tf.clip_by_value(target_policy + epsilon, -self.action_limit, self.action_limit)
            _, target_q_value1, target_q_value2, _ = self.actor_critic_net(self.next_states_ph, target_action)

        # ———————— 训练更新定义 ———————— #

        actor_loss = -tf.reduce_mean(policy_value)

        min_target_value = tf.minimum(target_q_value1, target_q_value2)
        returns = tf.stop_gradient(self.rewards_ph + self.gamma * (1 - self.dones_ph) * min_target_value)
        critic_loss = tf.reduce_mean((q_value1 - returns) ** 2) + tf.reduce_mean((q_value2 - returns) ** 2)

        actor_vars = [var for var in tf.global_variables() if 'main/actor' in var.name]
        critic_vars = [var for var in tf.global_variables() if 'main/critic' in var.name]
        self.actor_optimizer = tf.train.AdamOptimizer(self.actor_lr).minimize(actor_loss, var_list=actor_vars)
        self.critic_optimizer = tf.train.AdamOptimizer(self.critic_lr).minimize(critic_loss, var_list=critic_vars)

        # ———————— 参数替换定义 ———————— #

        main_vars = [var for var in tf.global_variables() if 'main' in var.name]
        target_vars = [var for var in tf.global_variables() if 'target' in var.name]

        self.target_update = tf.group([tf.assign(target_var, self.polyak * target_var + (1 - self.polyak) * main_var)
                                       for main_var, target_var in zip(main_vars, target_vars)])
        self.target_init = tf.group([tf.assign(target_var, main_var)
                                     for main_var, target_var in zip(main_vars, target_vars)])

    def actor_critic_net(self, state, action):
        with tf.variable_scope('actor'):
            layer = tf.layers.dense(state, 64, tf.nn.relu)
            layer = tf.layers.dense(layer, 64, tf.nn.relu)
            policy = self.action_limit * tf.layers.dense(layer, self.action_dim, tf.tanh)
        with tf.variable_scope('critic1'):
            layer = tf.layers.dense(tf.concat([state, action], axis=-1), 64, tf.nn.relu)
            layer = tf.layers.dense(layer, 64, tf.nn.relu)
            q_value1 = tf.squeeze(tf.layers.dense(layer, 1, None), axis=1)
        with tf.variable_scope('critic1', reuse=True):
            layer = tf.layers.dense(tf.concat([state, policy], axis=-1), 64, tf.nn.relu)
            layer = tf.layers.dense(layer, 64, tf.nn.relu)
            policy_value = tf.squeeze(tf.layers.dense(layer, 1, None), axis=1)
        with tf.variable_scope('critic2'):
            layer = tf.layers.dense(tf.concat([state, action], axis=-1), 64, tf.nn.relu)
            layer = tf.layers.dense(layer, 64, tf.nn.relu)
            q_value2 = tf.squeeze(tf.layers.dense(layer, 1, None), axis=1)
        return policy, q_value1, q_value2, policy_value

    def initial_target(self):
        self.sess.run(self.target_init)

    def choose_action(self, state, noise=0.1):
        action = self.sess.run(self.policy, feed_dict={self.states_ph: state.reshape(1, -1)})[0]
        action += noise * np.random.randn(self.action_dim)
        return np.clip(action, -self.action_limit, self.action_limit)

    def update(self, buffer, train_step, batch_size):
        states, actions, rewards, next_states, dones = buffer.random_sample(batch_size)
        feed = {self.states_ph: states,
                self.actions_ph: actions,
                self.rewards_ph: rewards,
                self.next_states_ph: next_states,
                self.dones_ph: dones}
        self.sess.run(self.critic_optimizer, feed)
        if train_step % self.policy_delay == 0:
            self.sess.run([self.actor_optimizer, self.target_update], feed)


class Buffer:
    def __init__(self, buffer_size, state_dim, action_dim):
        self.buffer_size = buffer_size

        self.states_bf = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions_bf = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards_bf = np.zeros((buffer_size, ), dtype=np.float32)  # 与tf输出的q_value维度保持一致
        self.next_states_bf = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.dones_bf = np.zeros((buffer_size,), dtype=np.bool)

        self.step = 0

    def store(self, state, action, reward, next_state, done):
        self.states_bf[self.step % self.buffer_size] = state
        self.actions_bf[self.step % self.buffer_size] = action
        self.rewards_bf[self.step % self.buffer_size] = reward
        self.next_states_bf[self.step % self.buffer_size] = next_state
        self.dones_bf[self.step % self.buffer_size] = done

        self.step += 1

    def random_sample(self, batch_size):  # 大于batch_size就开始训练
        max_index = min(self.step, self.buffer_size)
        indices = np.random.choice(max_index, size=batch_size)
        return (self.states_bf[indices], self.actions_bf[indices], self.rewards_bf[indices],
                self.next_states_bf[indices], self.dones_bf[indices])


# ———————————————— 超参数及初始化 ———————————————— #

env_name = 'HalfCheetah-v2'

buffer_size = int(1e6)
batch_size = 100
max_step = 1000
max_epoch = 500

env = gym.make(env_name).unwrapped
env.seed(1)
np.random.seed(1)
tf.random.set_random_seed(1)

sess = tf.Session()
agent = TD3(sess, env)
sess.run(tf.global_variables_initializer())
agent.initial_target()

buffer = Buffer(buffer_size, env.observation_space.shape[0], env.action_space.shape[0])

# ———————————————— 主逻辑 ———————————————— #

for epoch in range(max_epoch):

    sum_reward = 0
    state = env.reset()

    step = 0
    while True:
        step += 1

        if epoch < 10:
            action = env.action_space.sample()
        else:
            action = agent.choose_action(state)

        next_state, reward, done, _ = env.step(action)
        done = False if step >= max_step else done
        sum_reward += reward

        buffer.store(state, action, reward, next_state, done)

        if done or step >= max_step:
            if epoch >= 10:  # SpinningUp的实现中将更新间隔设置为了超参数，效果其实也不怎么明显
                for train_step in range(step):
                    agent.update(buffer, train_step, batch_size)

            print('Epoch: %d, Reward: %.2f' % (epoch, sum_reward))
            break

        state = next_state
