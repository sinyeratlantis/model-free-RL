import gym
import numpy as np
import tensorflow.compat.v1 as tf  # 针对tf1.15以上版本

tf.disable_v2_behavior()
config = tf.ConfigProto(allow_soft_placement=True)  # auto distribute device
config.gpu_options.allow_growth = True  # gpu memory dependent on require


# ———————————————— 环境封装定义 ———————————————— #

class Wrapper:
    def __init__(self, env, stack_image_num):
        self.env = env
        self.stack_image_num = stack_image_num

    def reset(self):  # 初始化堆叠帧的同时，初始化reward_memory
        self.finish = False
        self.reward_mean = self.recent_reward_mean()  # 初始化reward均值记录
        gray_image = self.rgb_to_gray(self.env.reset())
        self.state = [gray_image] * self.stack_image_num  # 用第一帧重复堆叠初始化状态
        return np.array(self.state)

    def step(self, action, render=False):
        total_reward = 0
        action_repeat = 10  # 每个动作重复数帧
        for i in range(action_repeat):
            rgb_image, reward, finish, _ = self.env.step(action)
            if finish:  # 完成游戏奖励
                reward += 100
            if np.mean(rgb_image[:, :, 1]) > 185.0:  # 惩罚绿色区域
                reward -= 0.05
            total_reward += reward
            done = True if self.reward_mean(reward) <= -0.1 else False
            if done or finish:
                break
            if render: self.env.render()
        self.state.pop(0)
        self.state.append(self.rgb_to_gray(rgb_image))
        return np.array(self.state), total_reward, done, finish

    @staticmethod
    def recent_reward_mean():  # 返回最近轮的reward均值
        count = 0
        length = 100
        history = np.zeros(length)
        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)
        return memory

    def rgb_to_gray(self, rgb):
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        gray = gray / 128. - 1.
        return gray


# ———————————————— 算法定义 ———————————————— # 特征维度为[96, 96, 3]，动作空间维度为3

class PPO:
    def __init__(self, sess, stack_image_num):
        self.sess = sess

        # ———————— 传参封装及占位符定义 ———————— #

        self.lr = 1e-3
        self.clip_ratio = 0.1
        self.stack_image_num = stack_image_num
        self.update_step = 0

        self.states_ph = tf.placeholder(tf.float32, shape=(None, self.stack_image_num, 96, 96))
        self.actions_ph = tf.placeholder(tf.float32, shape=(None, 3))
        self.returns_ph = tf.placeholder(tf.float32, shape=(None,))  # critic更新的衡量对象
        self.advantages_ph = tf.placeholder(tf.float32, shape=(None,))  # actor更新的衡量对象
        self.old_log_probs_ph = tf.placeholder(tf.float32, shape=(None,))  # actor更新的约束条件
        self.update_step_ph = tf.placeholder(tf.int32, shape=())

        # ———————— 图特征提取 ———————— # input filter kernel stride

        conv = tf.layers.conv2d(tf.transpose(self.states_ph, [0, 2, 3, 1]), 8, 4, 2, 'valid', activation='relu')
        conv = tf.layers.conv2d(conv, 16, 3, 2, 'valid', activation='relu')
        conv = tf.layers.conv2d(conv, 32, 3, 2, 'valid', activation='relu')
        conv = tf.layers.conv2d(conv, 64, 3, 2, 'valid', activation='relu')
        conv = tf.layers.conv2d(conv, 128, 3, 1, 'valid', activation='relu')
        feature = tf.layers.flatten(tf.layers.conv2d(conv, 256, 3, 1, 'valid', activation='relu'))

        # ———————— 神经网络定义 ———————— #
        
        with tf.variable_scope('policy'):
            fc = tf.layers.dense(feature, 100, tf.nn.relu)
            alpha_head = tf.layers.dense(fc, 3, tf.nn.softplus)
            beta_head = tf.layers.dense(fc, 3, tf.nn.softplus)
            distrib = tf.distributions.Beta(alpha_head + 1, beta_head + 1)

        with tf.variable_scope('value'):
            layer = tf.layers.dense(feature, 100, tf.nn.relu)
            self.state_values = tf.squeeze(tf.layers.dense(layer, 1), axis=1)  # [None]

        # ———————— 动作选择定义 ———————— #

        self.action = tf.squeeze(distrib.sample(), axis=0)  # squeeze([action_dim])
        self.old_log_prob = tf.reduce_sum(distrib.log_prob(self.action))  # 返回单动作各维度概率然后求和

        # ———————— 训练更新定义 ———————— #

        action_log_probs = tf.reduce_sum(distrib.log_prob(self.actions_ph), axis=1)  # 对action的多维prob求和
        ratio = tf.exp(action_log_probs - self.old_log_probs_ph)

        surr_loss = ratio * self.advantages_ph
        clip_loss = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * self.advantages_ph

        policy_loss = -tf.reduce_mean(tf.minimum(surr_loss, clip_loss))
        value_loss = tf.losses.huber_loss(self.state_values, self.returns_ph)
        loss = policy_loss + 2. * value_loss

        decay_lr = tf.train.exponential_decay(self.lr, self.update_step_ph, 1, 0.99, staircase=True)
        self.train_op = tf.train.AdamOptimizer(decay_lr).minimize(loss)

    def choose_action(self, state):
        action, old_log_prob = self.sess.run([self.action, self.old_log_prob], feed_dict={self.states_ph: [state]})
        return action, old_log_prob

    def calcalute_state_value(self, state):
        return self.sess.run(self.state_values, feed_dict={self.states_ph: state})

    def update(self, buffer, batch_size):
        self.update_step += 1
        for i in range(8):  # 训练次数暂固定 TODO
            for states, actions, returns, advantages, old_log_probs in buffer.random_iterator(batch_size):
                feed = {self.states_ph: states,
                        self.actions_ph: actions,
                        self.returns_ph: returns,
                        self.advantages_ph: advantages,
                        self.old_log_probs_ph: old_log_probs,
                        self.update_step_ph: self.update_step}
                self.sess.run(self.train_op, feed_dict=feed)


class Buffer:
    def __init__(self, buffer_size, stack_image_num):
        self.buffer_size = buffer_size
        self.stack_image_num = stack_image_num

        self.states_bf = np.empty((self.buffer_size, self.stack_image_num, 96, 96), np.float32)
        self.actions_bf = np.empty((self.buffer_size, 3), np.float32)
        self.rewards_bf = np.empty((self.buffer_size,), np.float32)
        self.next_states_bf = np.empty((self.buffer_size, self.stack_image_num, 96, 96), np.float32)
        self.old_log_probs_bf = np.ones((self.buffer_size,), np.float32)  # 每个action仅有一值

        self.step = 0

    def store(self, state, action, reward, next_state, policy_log_prob):
        self.states_bf[self.step % self.buffer_size] = state
        self.actions_bf[self.step % self.buffer_size] = action
        self.rewards_bf[self.step % self.buffer_size] = reward
        self.next_states_bf[self.step % self.buffer_size] = next_state
        self.old_log_probs_bf[self.step % self.buffer_size] = policy_log_prob

        self.step += 1

    def calculate_values(self, agent):  # TODO TD法求return
        state_values = agent.calcalute_state_value(self.states_bf)
        next_state_values = agent.calcalute_state_value(self.next_states_bf)

        self.returns_bf = self.rewards_bf + 0.99 * next_state_values
        self.advantages_bf = self.returns_bf - state_values

    def random_iterator(self, batch_size):  # 在线更新，不重复采样经验
        all_indices = np.arange(self.buffer_size)
        np.random.shuffle(all_indices)

        i = 0
        while i < len(all_indices):
            indices = all_indices[i:i + batch_size]
            yield (self.states_bf[indices], self.actions_bf[indices], self.returns_bf[indices],
                   self.advantages_bf[indices], self.old_log_probs_bf[indices])
            i += batch_size


# ———————————————— 超参数及初始化 ———————————————— #

stack_image_num = 4
batch_size = 128
buffer_size = 2000
max_epoch = 5000

env = gym.make('CarRacing-v0', verbose=0)
env.seed(0)
np.random.seed(0)
tf.set_random_seed(0)
env = Wrapper(env, stack_image_num)

sess = tf.Session(config=config)
agent = PPO(sess, stack_image_num)
sess.run(tf.global_variables_initializer())

buffer = Buffer(buffer_size, stack_image_num)

# ———————————————— 主逻辑 ———————————————— #

for epoch in range(max_epoch):

    game_reward = 0
    state = env.reset()
    while True:
        action, action_log_prob = agent.choose_action(state)

        # 需对action第一项输出扩展到-1到1
        next_state, reward, done, finish = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
        game_reward += reward

        buffer.store(state, action, reward, next_state, action_log_prob)
        if buffer.step % buffer_size == 0:
            buffer.calculate_values(agent)
            agent.update(buffer, batch_size)

        if done or finish:  # 长时间无reward，或游戏结束
            print('Epoch: %d, Reward: %.2f' % (epoch, game_reward))
            break

        state = next_state
