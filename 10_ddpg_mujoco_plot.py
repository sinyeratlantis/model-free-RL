import gym
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


# ———————————————— 算法定义 ———————————————— #

class DDPG:
    def __init__(self, sess, env, batch_size):
        self.sess = sess

        # ———————— 传参封装及占位符定义 ———————— #

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_limit = env.action_space.high

        self.tau = 0.001
        self.gamma = 0.99
        self.actor_lr = 1e-4
        self.critic_lr = 1e-3

        self.actor_noise = np.zeros(self.action_dim)
        self.kernel_init = tf.initializers.random_uniform(-0.003, 0.003)

        self.states_ph = tf.placeholder(tf.float32, (None, self.state_dim))
        self.actions_ph = tf.placeholder(tf.float32, (None, self.action_dim))
        self.returns_ph = tf.placeholder(tf.float32, (None, 1))
        self.critic_grad_ph = tf.placeholder(tf.float32, (None, self.action_dim))

        # ———————— 神经网络定义 ———————— #

        self.actions = self.build_actor_net('actor')
        self.target_actions = self.build_actor_net('target_actor')

        self.q_value = self.build_critic_net('critic')
        self.target_value = self.build_critic_net('target_critic')

        # ———————— 参数替换定义 ———————— #

        actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
        target_actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_actor')

        critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
        critic_target_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_critic')

        self.update_actor_params = self.assign_params(actor_params, target_actor_params)
        self.update_critic_params = self.assign_params(critic_params, critic_target_params)

        # ———————— 训练更新定义 ———————— #

        self.critic_loss = tf.losses.mean_squared_error(self.returns_ph, self.q_value)

        # 这里的逻辑实际上就是：d(q_value)/d(a_param) = (d(a)/d(a_param))*(d(q)/d(a))
        self.critic_grad = tf.gradients(self.q_value, self.actions_ph)
        grad = tf.gradients(self.actions, actor_params, - self.critic_grad_ph)

        # 之所以使用梯度而不是直接最大化q_value，可能就是因为batch_size归一化
        grad = [g / batch_size for g in grad]

        self.train_actor = tf.train.AdamOptimizer(self.actor_lr).apply_gradients(zip(grad, actor_params))
        self.train_critic = tf.train.AdamOptimizer(self.critic_lr).minimize(self.critic_loss)

    def build_actor_net(self, scope):
        with tf.variable_scope(scope):
            layer = tf.layers.dense(self.states_ph, 400)
            layer = tf.nn.relu(tf.layers.batch_normalization(layer))
            layer = tf.layers.dense(layer, 300)
            layer = tf.nn.relu(tf.layers.batch_normalization(layer))
            return tf.layers.dense(layer, self.action_dim, tf.tanh, kernel_initializer=self.kernel_init)

    def build_critic_net(self, scope):
        with tf.variable_scope(scope):
            layer = tf.layers.dense(self.states_ph, 400)
            layer = tf.nn.relu(tf.layers.batch_normalization(layer))
            state_features = tf.layers.dense(layer, 300)
            action_features = tf.layers.dense(self.actions_ph, 300)
            layer = tf.nn.relu(state_features + action_features)
            return tf.layers.dense(layer, 1, kernel_initializer=self.kernel_init)

    def assign_params(self, params, target_params):
        def assign_param(param, target_param, tau):
            return target_param.assign(param * tau + target_param * (1 - tau))

        return [assign_param(params[i], target_params[i], self.tau) for i in range(len(params))]

    def choose_action(self, state):
        feed = {self.states_ph: np.reshape(state, (1, self.state_dim))}
        action = self.sess.run(self.actions, feed_dict=feed)
        self.actor_noise = 0.9985 * self.actor_noise + 0.02 * np.random.normal(size=self.action_dim)
        action += self.actor_noise
        return action

    def update(self, buffer, batch_size):
        states, actions, rewards, next_states, dones = buffer.random_sample(batch_size)

        actor_actions = self.sess.run(self.actions, feed_dict={self.states_ph: states})
        target_action = self.sess.run(self.target_actions, feed_dict={self.states_ph: next_states})

        # ———————— 更新actor ———————— #

        feed_grad = {self.states_ph: states, self.actions_ph: actor_actions}
        critic_grads = self.sess.run(self.critic_grad, feed_dict=feed_grad)

        feed_actor = {self.states_ph: states, self.critic_grad_ph: critic_grads[0]}
        self.sess.run(self.train_actor, feed_dict=feed_actor)

        # ———————— 更新critic ———————— #

        feed_value = {self.states_ph: next_states, self.actions_ph: target_action}
        next_q_value = self.sess.run(self.target_value, feed_dict=feed_value)

        returns = np.zeros((batch_size, 1))  # TD方法求解returns
        for i in range(batch_size):
            if dones[i]:
                returns[i] = rewards[i]
            else:
                returns[i] = rewards[i] + self.gamma * next_q_value[i]

        feed_critic = {self.states_ph: states, self.actions_ph: actions, self.returns_ph: returns}
        self.sess.run(self.train_critic, feed_dict=feed_critic)

    def replace_params(self):
        self.sess.run(self.update_actor_params)
        self.sess.run(self.update_critic_params)


class Buffer:
    def __init__(self, buffer_size, state_dim, action_dim):
        self.buffer_size = buffer_size

        self.states_bf = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions_bf = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards_bf = np.zeros((buffer_size, 1), dtype=np.float32)  # 与tf输出的q_value维度保持一致
        self.next_states_bf = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.dones_bf = np.zeros((buffer_size,), dtype=np.bool)

        self.step = 0

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


class Recorder:  # 存储x轴的值，并画图
    def __init__(self, sess, save_path, start_save_epoch=0, savefig_interval=100, save_interval=500):
        self.xs = []
        self.sess = sess
        self.saver = tf.train.Saver()
        self.save_name = save_path
        self.save_path = save_path
        self.start_save_epoch = start_save_epoch
        self.savefig_interval = savefig_interval
        self.save_interval = save_interval
        import os
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def store(self, x):
        self.xs.append(x)

    def record(self, epoch, title_name, end=False):
        if not end:
            if epoch > self.start_save_epoch and epoch % self.savefig_interval == 0:
                save_name = self.save_path + '/' + self.save_name + '_%d.pdf' % epoch
                self._savefig(save_name, title_name)
            if epoch > self.start_save_epoch and epoch % self.save_interval == 0:
                self.saver.save(self.sess, self.save_path + "/model_%d.ckpt" % epoch)
        else:
            savefig_name = self.save_path + '/' + self.save_name + '.pdf'
            save_np_name = self.save_path + '/' + self.save_name + '_plot.npy'
            self._savefig(savefig_name, title_name)
            np.save(save_np_name, self.xs)
            self.saver.save(self.sess, self.save_path + "/model_final.ckpt")

    def _savefig(self, save_name, title_name):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 5))
        plt.title(title_name)
        plt.plot(np.arange(len(self.xs)), self.xs)
        plt.savefig(save_name)
        plt.close()


# ———————————————— 超参数及初始化 ———————————————— #

env_name = 'HalfCheetah-v2'

buffer_size = int(1e6)
batch_size = 64
max_epoch = 500

env = gym.make(env_name)
env.seed(1)
np.random.seed(1)
tf.random.set_random_seed(1)

sess = tf.Session()
agent = DDPG(sess, env, batch_size)  # batch_size用于归一化grad
sess.run(tf.global_variables_initializer())
agent.replace_params()

buffer = Buffer(buffer_size, env.observation_space.shape[0], env.action_space.shape[0])
save_path = 'mujoco_ddpg_baseline'
recorder = Recorder(sess, save_path, start_save_epoch=0, savefig_interval=100, save_interval=500)

# ———————————————— 主逻辑 ———————————————— #

for epoch in range(max_epoch):

    sum_reward = 0
    state = env.reset()

    step = 0
    while True:
        step += 1

        if epoch < 10:  # 增强exploration
            action = env.action_space.sample()
        else:
            action = agent.choose_action(state)[0]  # 对tf输出的squeeze

        next_state, reward, done, _ = env.step(action)
        sum_reward += reward

        buffer.store(state, action, reward, next_state, done)

        if done:
            for _ in range(step):  # 前提：HalfCheetah恒定终止长度大于batch_size
                agent.update(buffer, batch_size)
                agent.replace_params()

            print('Epoch: %d, Reward: %.2f' % (epoch, sum_reward))
            recorder.store(sum_reward)
            break

        state = next_state

    recorder.record(epoch, 'Mujoco DDPG')

recorder.record(None, 'Mujoco DDPG', True)
