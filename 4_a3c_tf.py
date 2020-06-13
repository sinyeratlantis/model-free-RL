import gym
import threading
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


# ———————————————— 算法定义 ———————————————— #

# TODO 此处实现是使用了reuse_variables()方法做多线程同步，是否有效存疑；
# 可参考MorvanZhou的实现和A3C_Atari的实现，受限于时间此处不做扩展了。

class A3C:  #
    def __init__(self, sess, env, reuse):
        self.sess = sess

        # ———————— 传参封装及占位符定义 ———————— #

        self.state_dim = env.observation_space.shape[0]
        self.action_num = env.action_space.n
        self.lr = 0.001
        self.gamma = 0.99

        self.states_ph = tf.placeholder(tf.float32, [None, self.state_dim])
        self.actions_ph = tf.placeholder(tf.int32, [None, ])
        self.returns_ph = tf.placeholder(tf.float32, [None, 1])

        # ———————— 神经网络定义 ———————— #

        if reuse:
            tf.get_variable_scope().reuse_variables()

        with tf.variable_scope("actor"):
            actor_layer = tf.layers.dense(self.states_ph, 100, tf.nn.relu)
            logits = tf.layers.dense(actor_layer, self.action_num, None)
            self.actions_probs = tf.nn.softmax(logits)

        with tf.variable_scope("critic"):
            critic_layer = tf.layers.dense(self.states_ph, 100, tf.nn.relu)
            self.state_values = tf.layers.dense(critic_layer, 1, None)

        # ———————— 训练更新定义 ———————— #

        # value loss
        advantages = self.returns_ph - self.state_values
        value_loss = tf.reduce_mean(tf.square(advantages))

        # policy loss
        action_log_probs = tf.log(self.actions_probs) * tf.one_hot(self.actions_ph, self.action_num)
        action_log_probs = tf.reduce_sum(action_log_probs, axis=1)
        policy_loss = - tf.reduce_mean(action_log_probs * tf.stop_gradient(advantages))

        # entropy loss
        actions_log_probs = tf.log(self.actions_probs)
        entropy = - tf.reduce_mean(tf.reduce_sum(self.actions_probs * actions_log_probs, axis=1))

        policy_loss -= 0.01 * entropy
        loss = 0.5 * value_loss + policy_loss
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, state):
        actions_probs = self.sess.run(self.actions_probs, {self.states_ph: [state]})
        action = np.random.choice(range(actions_probs.shape[1]), p=actions_probs[0])
        return action
    
    def update(self, buffer):
        last_done = buffer.dones_bf[-1]
        last_state = buffer.next_states_bf[-1]

        if last_done:  # buffer最后状态的状态价值
            last_state_value = 0
        else:
            last_state_value = self.sess.run(self.state_values, {self.states_ph: [last_state]})[0][0]

        returns = np.zeros([len(buffer.rewards_bf), 1], dtype=np.float32)
        running_add = last_state_value
        for j in reversed(range(len(buffer.rewards_bf))):
            running_add = buffer.rewards_bf[j] + self.gamma * running_add
            returns[j][0] = running_add

        feed = {self.states_ph: buffer.states_bf,
                self.actions_ph: buffer.actions_bf,
                self.returns_ph: returns}
        self.sess.run(self.train_op, feed)


class Buffer:
    def __init__(self):
        self.states_bf = []
        self.actions_bf = []
        self.rewards_bf = []
        self.next_states_bf = []
        self.dones_bf = []

    def store(self, state, action, reward, next_state, done):
        self.states_bf.append(state)
        self.actions_bf.append(action)
        self.rewards_bf.append(reward)
        self.next_states_bf.append(next_state)
        self.dones_bf.append(done)

    def clear(self):
        self.states_bf = []
        self.actions_bf = []
        self.rewards_bf = []
        self.next_states_bf = []
        self.dones_bf = []

        
class Worker(threading.Thread):
    def __init__(self, sess, index):
        super(Worker, self).__init__()  # 继承threading类相关

        self.sess = sess
        self.index = index
        self.env = gym.make('CartPole-v0').unwrapped
        
        self.env.seed(index)
        np.random.seed(index)
        tf.set_random_seed(index)

        self.local_agent = A3C(self.sess, self.env, reuse=True)
        
        self.buffer = Buffer()

    def run(self):  # 内部主逻辑
        for epoch in range(max_epoch):
            state = self.env.reset()

            step = 0
            while True:
                step += 1

                action = self.local_agent.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                if done:
                    reward = -1

                self.buffer.store(state, action, reward, next_state, done)

                if step % update_freq == 0 or done or step >= max_step:
                    self.local_agent.update(self.buffer)
                    self.buffer.clear()
                    if done or step >= max_step:
                        if self.index == 0:
                            print('Epoch: %d, Reward: %d' % (epoch, step))
                        break

                state = next_state


# ———————————————— 超参数及初始化定义 ———————————————— #

env_name = 'CartPole-v0'

parallel = 8
update_freq = 20  # 每收集20条数据更新一次，on-policy
max_epoch = 1000
max_step = 400

sess = tf.Session()
global_agent = A3C(sess, gym.make(env_name), reuse=False)
workers = [Worker(sess, i) for i in range(parallel)]
sess.run(tf.global_variables_initializer())

# ———————————————— 主逻辑 ———————————————— #

for worker in workers:
    worker.daemon = True  # 守护线程，主线程停止时停止所有工作线程
    worker.start()

[w.join() for w in workers]

coord = tf.train.Coordinator()
coord.join(workers)
