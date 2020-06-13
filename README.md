## Model-free RL算法实现原理简介

本文档仅用于实验代码所涉及的算法原理的必要介绍，更细节和完整的推导可参见：

### 1. Deep Q Network

- 策略：根据状态价值选取动作。

- 模型：模型用于得到状态价值或状态动作价值：$Q(s,a)=f(s,a)$。

  一种实现是按照公式输入state和action，输出Q值；另一种是输入state。输出各action下的Q值，我们采用第二种实现。

#### 更新

根据公式：

$Q_{target} = reward + gamma * \max_a Q(s', a')$

$loss = MSE(Q_{target}, Q_{eval})$

$Q_{target}$ 是更新神经网络的参考标准，如公式所示，其会随着Q网络的更新而更新，而原理上对 $Q(s, a)$ 的更新不应该导致 $Q(s', a')$ 的变化。所以第一步是延迟更新 $Q_{target}$，通过创建一个同样的网络模型输出 $Q_{target}$，然后每隔一段时间与不断更新的网络Q同步参数，即可做到延迟更新。另外，连续收集的样本相关性严重，所以DQN采用了replay buffer。

具体实现上，$Q_{target}$ 通过target_net计算next_state对所有action的Q值，然后取max得到 $\max_a Q(s', a')$。考虑到buffer的最终状态价值不由网络输出，所以 $Q_{next}$ 更新时需根据具体done的真假值修正，实践中采用buffer更新前输出所有 $Q_{next}$，修正后再作为网络更新的传参提供 $Q_{target}$ 信息。

### 2. Policy Gradient

- 策略：将动作建模为所观测的状态的函数。

- 模型：$p(a) = f(s,a)$，与DQN同样，我们直接输入state，输出各动作的选择概率。

#### 更新

我们的目标可认为是最大化路径奖励的期望：$J(\theta)=\mathbb{E}_{\tau \sim p_{\theta}(\tau)}[r(\tau)]$

策略为动作选择的概率，我们稍微扩展到路径选择的概率，则有：$J(\theta)=\int p_{\theta}(\tau) r(\tau) \mathrm{d} \tau$

求梯度：$\nabla_{\theta} J(\theta)=\int \nabla_{\theta} p_{\theta}(\tau) r(\tau) \mathrm{d} \tau$

由公式：$\nabla_{\theta} p_{\theta}(\tau)=\nabla_{\theta} \log p_{\theta}(\tau)p_{\theta}(\tau)$

可得：$\nabla_{\theta} J(\theta)=\mathbb{E}_{\tau \sim p_{\theta}(\tau)}\left[\nabla_{\theta} \log p_{\theta}(\tau) r(\tau)\right]$

展开路径可得：$\nabla_{\theta} J(\theta)=\mathbb{E}_{\tau \sim p_{\theta}(\tau)}\left[\left(\sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta}\left(a_{t} | s_{t}\right)\right)\left(\sum_{t=1}^{T} r\left(s_{t}, a_{t}\right)\right)\right]$

最后一项是路径总奖励，我们考虑到因果性后转化后向累加奖励，并考虑削弱较遥远的奖励的作用加入gamma衰减，在实现中记为return。

我们更新的思想采用的是蒙特卡洛思想，即采样完整的一条路径进行更新。那么，结合公式，更新梯度的计算就是计算一条路径下 $\left(\sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta}\left(a_{t} | s_{t}\right)\right)\left(\sum_{t=1}^{T} r\left(s_{t}, a_{t}\right)\right)$ 的值。

可得，梯度上升的方向是 mean( log p(a_t) * return) 。p(a_t) 由模型给出：p(a_all) = f(s)，再找到实际选择的动作 a_t 将 p 值从 a_all 中筛选出来。具体见代码。

### 3. Actor Critic

AC算法是 Q-learning 系算法与 Policy Gradient 系算法的结合，目标是使PG能够实现单步更新。

我们学习并非完全根据reward的绝对值大小，而是根据我们相对于我们习惯的reward大小更好还是更坏来学习。基于这个思想PG系算法希望能对return减去一个学习的习惯量，称为baseline。AC算法即是用Q network提供了这个信息。在单步更新情况下，由于没有累计任何奖励，我们使用另一个更准确的词表示return，称为target_value。

critic的输出可以认为是基于以往所有经验对当前状态价值的估计，这也是其作为代替baseline存在的原因。对于单步更新的AC算法而言，我们通过TD采样r+v(s')来作为当前状态价值的目标值（因为reward使这个值得估计相对更加准确），这个价值与我们估计的价值的差距自然就是critic loss。我们会看到，为降低方差所使用的on-policy buffer更新方式会使用MC的思想计算return以提供比单步的TD更准确的估计。

然而不可忽略的是，我们的确用了从真实环境获得的参考数据来提升critic的估计，但这个参考数据不是想监督学习的标签那样固定用来提升critic的估计的，它是策略在环境中采样得到的，能获得怎样的数据与策略息息相关。于是，采样的数据可以反映两方面的信息：第一，它提供的状态价值能让critic知道估计是否准确；第二，它提供的状态价值能让actor知道自己的行为好还是坏。所以对critic而言，return - state_value是其估计误差，而对actor而言，return - state_value是当前行为相对自己习惯的状态的好坏评价，所以actor下的state_value即baseline，return - baseline才是advantage的定义。critic的mse误差只是恰好等于advantage的平方。

至于为什么这么可以用参见大文档。其余方面与PG均是一般。

### 4. A3C

单步更新存在稳定性问题，A3C通过并行使稳定性提升，并增强了算法的探索能力。考虑到并行的稳定性可能有限，实现中采用了on-policy buffer更新的方式。具体实现中，每收集20条transition或游戏结束时，更新算法。其余与AC系基本一致。

注意到一定步数更新这里的步数实际上是不一定的，可能是集满20条数据，也可能是收集1条游戏就结束了，所以这里存在不稳定的因素。

### 5. PPO

PG采用收集的路径数据更新，属于on-policy更新，包括A3C算法也是如此。为提高数据利用效率，一种方式是replay buffer的模式。另一种可能，就是看能否最大化单次收集数据的利用效率，而不是仅仅更新一次就丢弃了。基于这样的观点，TRPO探索了这种实际上是off-policy更新方式的可能性。其认为，只要更新后的策略与更新前的策略的差异满足一定条件的约束，那么可以近似认为两种策略是接近的，可以用同一次收集的数据集更新。

具体推导可见大文档，推导中可以看到使用老数据集更新与正常更新有两项区别：首先是改变了计算形式，与更新前的策略 log p(a) 相关；其次是做了一项近似，这项近似转变为一项约束。

现在我们直接给出结论：要求其 log p(a) 与 log p'(a) 的KL散度接近，且 policy loss 需要修改成随 p 的更新而改变其值以降低偏差。

SpinningUp的实现中采用了不同于A3C实现中所用的游戏结束时也更新的策略，而是使用了定长的buffer，这里就出现一项问题，MC思想下的后向累计奖励是一条路径的反传，但收集的buffer可能是多条路径的融合，所以可能出现很多种情况：

1. 从游戏开始到游戏还未结束（等同于A3C情况）；
2. 从游戏中途到游戏还未结束（等同于A3C情况）；
3. 从游戏开始到游戏结束，再开一局游戏，再到游戏未结束；
4. 从游戏中途到游戏结束，再开一局游戏，再到游戏未结束；
5. 从游戏开始到游戏结束，再开一局游戏，再到游戏结束，再开游戏，最后到当前游戏未结束；
6. 从游戏中途到游戏结束，再开一局游戏，再到游戏结束，再开游戏，最后到当前游戏未结束。

值得注意的是，buffer长度有可能小于游戏长度，在这种情况下，在游戏未结束时是不能重置游戏的，否则有些长尺度游戏永远看不到结束，所以A3C实现中对所有buffer长度和游戏长度是均适用的。按照我们在A3C中实现的思路，后向累计奖励与从游戏开始计算还是从中途计算无关，因为只涉及后向，所以情况可以约减为：

1. 游戏还未结束（等同于A3C情况）；
2. 游戏结束，再开一局游戏，再到游戏未结束；
3. 游戏结束，再开一局游戏，再到游戏结束，再开游戏，最后到当前游戏未结束。

第一种直接全buffer后向计算即可，第二种需要在buffer中确认上局游戏结束的标志，并以该标志为反传计算点，两部分反传后再合并构成定长的buffer。第三种则是需要确认buffer中所有的结束标志，并分别反传，得到完整buffer，第三种显然适用于所有定长buffer的情况。

虽然return是agent update相关，但太多操作与buffer相关了，所以在PPO实现中，我们将计算return的工作封装在了buffer中，当然，转移到agent update中也并非十分复杂。根据done和结束标志反传价值后再合并显得十分繁琐，PPO中我们采用了每局游戏结束后自动计算反传的returns再存储到buffer中，于是，结束位点便不再重要，重要的是何时开始，以免影响到先前已经计算好的returns。

policy loss 的公式仅有一条，看似不难，实际上我们需要分析 policy loss 具体需要哪些输入才能计算。注意到更新是基于老数据集的，不可忽略的是，advantage是一项客观指标，其值是由老数据集和老策略所决定的，不可用新策略来算。也就是说，在 advantages = returns - state_value 中，由数据集决定的returns自不用说（因为也没有新数据集给你算），state_value 也必须是老策略的值。

为什么要强调这点呢？因为一般PPO的实现是创建两个相同的网络，一个用于存储老策略，一个用于更新，两者每次更新完后同步。SpinningUp中采用了一种更加聪明的方式，其不创建多余的网络了，而是将更新时真正要用的老策略相关的数据给存储起来，然后直接提供给新策略使用。

涉及到老策略提供的信息的只有两个：一个是actor网络提供给ratio计算的old_log_probs，另一个就是critic网络提供给advantage计算的state_value。那么，我们就可以令agent在与环境交互时输出这两项的值（action_log_prob和state_value）存储为old_log_prob和old_state_value。

如果将值计算封装到buffer中，那么buffer需要提供给agent更新的信息有：

- states - actions：用于当前策略critic的state_values和actor的action_log_probs计算；
- returns：在外部执行更加方便的反传计算，注意此处critic loss并不是advantage的平方，而是预测的state_value与真实return的差异，在mse loss且on-policy更新的情况下才会相等；
- advantages：由于其 state_value 不是由当前策略给出，要用存储的 old_state_value 计算；
- old_log_prob：policy loss 计算和 KL 计算相关。

至此，PPO SpinningUp实现的具体逻辑便足够清晰了，具体细节请参见代码。

### 6. DPPO

本版代码为PPO代码的简单扩展，但从这里开始，之后的代码便开始应用于复杂环境并具有实际应用价值了。

本版对应用于CartPole的上版PPO改变很少，主要是依据SpinningUp的实现加入了并行机制，以及将离散动作空间的CartPole应用到连续空间的HalfCheetah上，还有就是加入了GAE机制。由于并行部分的代码我直接挪用了SpinningUp的实现，也没仔细看，所以代码量相对其他实现而言就略大了，但总代码量也不会超过400行。

注意到SpinningUp的循环逻辑是以存满buffer为中心，我们更改为完成一局游戏为中心，以便更好地观察reward的变化。

### 7. DDPG

从这里开始，我们进入到连续动作空间的强化学习算法的讨论与研究。DDPG基于AC架构，直接的思想是actor直接输出action的value：a = f(s)，由于action不是离散的，f(s) 不能输出各action的value，所以critic只能建模为 q_value = f(s, a)。

之后便是如何更新的问题。critic loss比较显然，拿return和q_value的mse算即可，两者都可以直接求得。问题在actor，AC架构用的是 log p(a) * adv，但当action是连续value时，无法得到概率值。

DDPG令PG算法中的策略不再表示成概率，而是确定值。原本 $\pi_\theta(s)$ 输出的是选择各个 action 的概率，现在输出一个值，这个值是唯一输出，$J(\theta)$ 便因 log p(a) 而无法计算了。注意到AC算法中我们用stop_gradient禁止了adv的梯度，因为critic在AC思想中是用于提供baseline的，adv是确定值用于actor更新方向的参考。但在DDPG思想中就不一样了，AC策略是policy-based的，但DDPG转变为了value-based，其策略的目的不再是最大化奖励期望，而是找到能最大化Q值的动作，actor的梯度由此来源于critic，具体细节还是看论文吧，这部分我个人理解也比较局限。直观上，q值对actor参数的梯度没办法求，所以转化为：

d(q value) / d(actor params) = d(q value) / d(action) * d(action) / d(a_param)

注意到 q_value = f(s, a)，a = f(s)，所以q值对action有梯度，action对actor有梯度。

最终在代码里表现即为：

```python
self.critic_grad = tf.gradients(self.q_value, self.actions_ph)
grad = tf.gradients(self.actions, actor_params, - self.critic_grad_ph)
```

在代码中使用placeholder的原因是 critic_grad 需要计算出来才能传参，不能直接传，所以外调一下，本质与直接传参无区别。

DDPG是value-based方法为主，原本的更新方式是off-policy单步更新，即具有replay buffer、target net，每与环境交互一步更新一次，SpinningUp中采用了TD3思想中延迟更新的方法，将更新延后到游戏结束时一并更新，并对action选择加入了噪声。对于off-policy更新而言，buffer自然有一项方法random_sample，buffer_size也无疑比较大，这是与先前代码不一样的地方。

### 8. TD3

TD3是DDPG算法的扩展，目标是解决DDPG中出现过的问题。我们已经说过，DDPG是value-based为主的算法，所以value-based的问题在DDPG中也会存在。

首先是Q值得过高估计问题，TD3采用类似于Double-Q Learning的方式建立两个critic网络，取估计较小的那一个。单步更新（TD）存在不稳定（方差大）的问题，target net的soft更新即是为了解决这个问题，然而，DDPG与AC算法不一致，其十分依赖critic的估计，导致critic的偏差对模型性能的影响极高，所以为了降低critic更新的不稳定性对actor的过度影响，TD3采用了actor和target相对critic延迟更新的思想，在代码中实现为critic更新两次actor和target更新一次。除了对action增加噪声提高exploration能力，为了使Q值得估计更加平滑，TD3在Q值得估计上也加入了噪声。

TD3的三项改进比较直观，但实现上细节还是比较多的，在此不详述了。

### 9. PPO实现像素级控制

由于这版代码是参考了Udacity的实现，所以很多原理性的问题其实不一定正确（没错，原版可能就是错的）。预期之后有空会实现一版SpinningUp版的像素级控制算法。

与标准PPO实现不同的地方主要有：

- 图特征提取；
- 策略建模为beta分布；
- critic loss通过huber_loss计算；
- 学习率衰减；
- 固定训练8次，无KL散度判别；
- 采用TD思想计算return（最大存疑问题）；
- 使用遍历buffer的随机batch更新（可能是用于避免像素数据全buffer更新导致显存溢出的问题）。

之所以认为采用TD思想计算return是存在问题的，是因为TD思想仅在单步更新时使用。对于能收集到路径连续数据的情况，MC能提供比TD更准确的估计，所以这里的实现可能存在不足。另外一种可能是，batch的抽取是随机的，可能打乱数据连续性，但return的计算应该与训练时的采样无关了，所以只能作为推测。

本版实现仅为参考，算法还待完善。

















