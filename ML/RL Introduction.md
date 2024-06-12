## 参考

[动手学强化学习](https://hrl.boyuai.com/)
drl

## MDP

Markov Decision Process 是一个五元组$<S, A, T, R, \gamma>$

- $S$ 是状态空间
- $A$ 是动作空间
- $T: S \times A \times S \to \mathbb{R}$ 是状态转移概率，$T(s, a, s')$ 表示在状态$s$下采取动作$a$转移到状态$s'$的概率
- $R: S \times A \times S \to \mathbb{R}$ 是奖励函数，$R(s, a, s')$ 表示在状态$s$下采取动作$a$转移到状态$s'$的奖励
- $\gamma$ 是折扣因子，用于平衡当前奖励和未来奖励的重要性

MDP 的目标是找到一个策略$\pi: S \to A$，使得在这个策略下，能够最大化期望回报（Return），注意，是一个随机过程，因此我们需要考虑期望回报

Bellman Equation:

$$
Q(s, a) = E[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots | S_t = s, A_t = a] = \sum_{s'} T(s, a, s') [R(s, a, s') + \gamma V(s')]
$$

$$
V(s) = E[U_t|S_t = s] = \sum_{a} \pi(a|s) \sum_{s'} T(s, a, s') [R(s, a, s') + \gamma V(s')]
$$

Bellman Optimality Equation:

$$
Q^*(s, a) = \sum_{s'} T(s, a, s') [R(s, a, s') + \gamma V^*(s')]
$$

$$
V^*(s) = \max_a \sum_{s'} T(s, a, s') [R(s, a, s') + \gamma V^*(s')]
$$

于是我们可以推导出最优方程的形式：

$$
Q^*(s, a) = R(s, a) + \gamma \sum_{s'} T(s, a, s') \max_{a'} Q^*(s', a')
$$

$$
V^*(s) = \max_a Q^*(s, a)
$$

而在回报和状态转移概率都是已知的情况下，我们可以有多种求解方法：

## Value Iteration

$$
V_{k+1}(s) = \max_a \sum_{s'} T(s, a, s') [R(s, a, s') + \gamma V_k(s')] = \max_a Q_k(s, a)
$$

依据如上的等式，在一次迭代的时候遍历所有的状态，找出每一个状态对应的最大估计 Q 值，然后更新 V 值，直到收敛。

```python
def value_iteration(env:GridWorld, gamma=0.9, theta=1e-6):
    """
    Value iteration algorithm for solving a given environment.

    Parameters:
    env: GridWorld
        An instance of the environment, which includes the state space, action space, rewards, and transition dynamics.

    gamma: float, optional, default=0.9
        The discount factor. It balances the importance of immediate rewards versus future rewards, ranging from 0 to 1. A higher gamma value makes the agent focus more on long-term rewards.

    theta: float, optional, default=1e-6
        The convergence threshold. It determines the stopping criterion for the value iteration algorithm. The algorithm stops when the maximum change in the value function is less than theta.

    Returns:
    policy: numpy.ndarray
        A policy matrix of shape (env.size, env.size), storing the optimal action for each state.
    """
    # initialize the value function and policy
    V = np.zeros((env.size, env.size))
    policy = np.zeros((env.size, env.size), dtype=int)
    ####
    # Implement the value iteration algorithm here

    iterations = 0

    while True:
        updated_V = V.copy()

        iterations += 1

        for now_state_x in range(env.size):
            for now_state_y in range(env.size):
                Q_values = []
                env.state = (now_state_x, now_state_y)

                for action in range(4):

                    # get s' and reward
                    next_state, reward = env.step(action=action)

                    next_state_x, next_state_y = next_state

                    # calc Q_value
                    Q_value = reward + gamma * V[next_state_x, next_state_y]

                    Q_values.append(Q_value)

                    # reset now_state
                    env.state = (now_state_x, now_state_y)

                # find max Q
                max_Q = max(Q_values)

                updated_V[now_state_x, now_state_y] = max_Q
                policy[now_state_x, now_state_y] = Q_values.index(max_Q)

        if np.amax(np.fabs(updated_V - V)) <= theta:
            print ('Value-iteration converged at iteration# %d.' %(iterations))
            break
        else:
            V = updated_V
    ####

    env.reset()
    return policy
```

## Policy Iteration

从一个初始化的策略出发，先进行**策略评估**，然后**改进策略**，评估改进的策略，再进一步改进策略，经过不断迭代更新，直达策略收敛，这种算法被称为“策略迭代”

- Policy Evaluation

根据 Bellman 期望方程得出迭代式

$$
V_{k+1}(s) = \sum_{a} \pi(a|s) \sum_{s'} T(s, a, s') [R(s, a, s') + \gamma V_k(s')]
$$

我们可以知道$V^k =V^\pi$是一个不动点

当迭代到收敛时，我们可以得到这个策略下的状态值函数

```python
def policy_evaluation(policy:np.ndarray, env:GridWorld, gamma=0.9, theta=1e-6):
    """
    Evaluate a policy given an environment.

    Parameters:
    policy: numpy.ndarray
        A matrix representing the policy. Each entry contains an action to take at that state.

    env: ComplexGridWorld
        An instance of the environment, which includes the state space, action space, rewards, and transition dynamics.

    gamma: float, optional, default=0.9
        The discount factor. It balances the importance of immediate rewards versus future rewards, ranging from 0 to 1.

    theta: float, optional, default=1e-6
        A threshold for the evaluation's convergence. When the change in value function is less than theta for all states, the evaluation stops.

    Returns:
    V: numpy.ndarray
        A value function representing the expected return for each state under the given policy.
    """
    V = np.zeros((env.size, env.size))
    ####
    # Implement the policy evaluation algorithm here

    while True:

            updated_V = V.copy()

            for now_state_x in range(env.size):
                for now_state_y in range(env.size):

                    env.state = (now_state_x, now_state_y)

                    action = policy[now_state_x, now_state_y]

                    next_state, reward = env.step(action=action)

                    updated_V[now_state_x, now_state_y] = reward + gamma * V[next_state[0], next_state[1]]

            if np.amax(np.fabs(updated_V - V)) <= theta:
                V = updated_V
                break
            else:
                V = updated_V

    ####

    return V
```

- Policy Improvement

假设我们在原来的状态价值函数的基础上，对于每一个状态，我们能够找到一个更优的动作$a$, 使得$Q^\pi (s, a) \geq V^\pi(s)$，那么能够获得更高的汇报

现在如果我们能够找到一个新的策略$\pi'$，使得$V^{\pi'}(s) \geq V^\pi(s)$，那么我们就可以得到一个更好的策略

因此我们可以贪心的选择每一个状态动作价值最大的那个动作，也就是

$$
\pi'(s) = \arg \max_a Q^\pi(s, a) = \arg \max_a \sum_{s'} T(s, a, s') [R(s, a, s') + \gamma V^\pi(s')]
$$

```python
def policy_iteration(env:GridWorld, gamma=0.9, theta=1e-6):
    """
    Perform policy iteration to find the optimal policy for a given environment.

    Parameters:
    env: ComplexGridWorld
        An instance of the environment, which includes the state space, action space, rewards, and transition dynamics.

    gamma: float, optional, default=0.9
        The discount factor. It balances the importance of immediate rewards versus future rewards, ranging from 0 to 1.

    theta: float, optional, default=1e-6
        A threshold for the evaluation's convergence. When the change in value function is less than theta for all states, the evaluation stops.

    Returns:
    policy: numpy.ndarray
        A matrix representing the optimal policy. Each entry contains the best action to take at that state.
    """
    policy = np.zeros((env.size, env.size), dtype=int)

    ####
    # Implement the policy iteration algorithm here

    while True:

        V = policy_evaluation(policy=policy, env=env)

        policy_stable = True

        for now_state_x in range(env.size):
            for now_state_y in range(env.size):
                Q_values = []
                env.state = (now_state_x, now_state_y)

                for action in range(4):

                    # get s' and reward
                    next_state, reward = env.step(action=action)

                    next_state_x, next_state_y = next_state

                    # calc Q_value
                    Q_value = reward + gamma * V[next_state_x, next_state_y]

                    Q_values.append(Q_value)

                    # reset now_state
                    env.state = (now_state_x, now_state_y)

                # update policy
                max_Q = max(Q_values)
                now_action = policy[now_state_x, now_state_y]
                new_action = Q_values.index(max_Q)

                if now_action != new_action:
                    policy_stable = False
                    policy[now_state_x, now_state_y] = Q_values.index(max_Q)

        if policy_stable:
            break

    ####
    env.reset()
    return policy
```

## State-Action-Reward-State-Action (SARSA)

一个表格由所有状态和动作组成，表格中的 Q-value 表示在某个状态下采取某个动作的价值，我们可以通过不断的更新这个表格来得到最优的策略

这个表格的值由策略决定，策略变化，表格的值也会变化

$$
Q^\pi(s_t, a_t) = \mathbb{E}[R_{t} + \gamma Q^\pi(s_{t+1}, a_{t+1}) | S_t = s_t, A_t = a_t]
$$

那么左右两边都是可以计算的，并且都是对 Q 值的估计，我们可以通过不断的迭代来更新这个表格

即使用观测到的$r_t$, $s_{t+1}$ 以及抽样的出的$a_{t+1}$，得到$r_t + \gamma q(s_{t+1}, a_{t+1})$

采用 TD 的思想，将$q(s_t, a_t) = (1-\alpha) q(s_t, a_t) + \alpha r_t + \gamma q(s_{t+1}, a_{t+1})$

SARSA 用到了五元组$(s_t, a_t, r_t, s_{t+1}, a_{t+1})$，因此我们可以通过不断的迭代来更新这个表格

在采样最佳策略的时候，使用$\epsilon$-greedy 策略，即以$\epsilon$的概率随机选择动作，以$1-\epsilon$的概率选择最优动作

$$
a = \begin{cases}
\text{random action} & \text{with probability } \epsilon \\
\arg \max_a Q(s, a) & \text{with probability } 1-\epsilon
\end{cases}
$$

```python
def sarsa(env:GridWorld, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    """
    SARSA algorithm for training an agent in a given environment.

    Parameters:
    env: ComplexGridWorld
        An instance of the environment, which includes the state space, action space, rewards, and transition dynamics.

    episodes: int, optional, default=1000
        The number of episodes for training. In each episode, the agent starts from the initial state and interacts with the environment until it reaches the goal or the episode terminates.

    alpha: float, optional, default=0.1
        The learning rate. It determines the step size for updating the Q-values, ranging from 0 to 1. A higher alpha value means faster learning but may lead to instability.

    gamma: float, optional, default=0.9
        The discount factor. It balances the importance of immediate rewards versus future rewards, ranging from 0 to 1. A higher gamma value makes the agent focus more on long-term rewards.

    epsilon: float, optional, default=0.1
        The exploration rate. It determines the probability of the agent taking a random action, used to balance exploration and exploitation. A higher epsilon value makes the agent explore more.

    Returns:
    q_table: numpy.ndarray
        A Q-value table of shape (env.size, env.size, 4), storing the Q-values for each state-action pair.
    """
    q_table = np.zeros((env.size, env.size, 4))

    ####
    # Implement the SARSA algorithm here

    import random

    for _ in range(episodes):

        x, y = env.start
        policy = extract_policy(q_table=q_table)
        action = policy[x, y]
        # 抽样
        if random.uniform(0, 1) <= epsilon:
                action = random.randint(0, 3)

        while True:
            policy = extract_policy(q_table=q_table)

            env.state = (x, y)
            new_state, reward = env.step(action=action)

            new_action = policy[new_state[0], new_state[1]]

            if random.uniform(0, 1) <= epsilon:
                new_action = random.randint(0, 3)

            q_table[x, y, action] += alpha * (reward + gamma * q_table[new_state[0], new_state[1], new_action] - q_table[x, y, action])

            x, y = new_state
            action = new_action

            if new_state == (9, 9):
                break

    ####

    env.reset()
    return q_table
```

## Q-Learning

Q-Learning 是一种无模型的学习方法，它不需要环境的转移概率，只需要环境的奖励即可

基于 TD 的思想，我们可以通过不断的迭代来更新 Q 值

$$
Q^*(s_t, a_t) = \sum_{s'} T(s, a, s') [R(s, a, s') + \gamma \max_{a'} Q^*(s', a')]
$$

```python
def q_learning(env:GridWorld, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    """
    Q-learning algorithm for training an agent in a given environment.

    Parameters:
    env: ComplexGridWorld
        An instance of the environment, which includes the state space, action space, rewards, and transition dynamics.

    episodes: int, optional, default=1000
        The number of episodes for training. In each episode, the agent starts from the initial state and interacts with the environment until it reaches the goal or the episode terminates.

    alpha: float, optional, default=0.1
        The learning rate. It determines the step size for updating the Q-values, ranging from 0 to 1. A higher alpha value means faster learning but may lead to instability.

    gamma: float, optional, default=0.9
        The discount factor. It balances the importance of immediate rewards versus future rewards, ranging from 0 to 1. A higher gamma value makes the agent focus more on long-term rewards.

    epsilon: float, optional, default=0.1
        The exploration rate. It determines the probability of the agent taking a random action, used to balance exploration and exploitation. A higher epsilon value makes the agent explore more.

    Returns:
    q_table: numpy.ndarray
        A Q-value table of shape (env.size, env.size, 4), storing the Q-values for each state-action pair.
    """
    q_table = np.zeros((env.size, env.size, 4))
    ####
    # Implement the Q-learning algorithm here

    for _ in range(episodes):
        x, y = env.start

        while True:
            policy = extract_policy(q_table=q_table)
            action = policy[x, y]

            if random.uniform(0,1) <= epsilon:
                action = random.randint(0,3)

            env.state = (x, y)
            new_state, reward = env.step(action=action)

            q_table[x, y, action] += alpha * (reward + gamma * np.amax(q_table[new_state[0], new_state[1]]) - q_table[x, y, action])

            x, y = new_state

            if new_state == env.goal:
                break
    ####

    env.reset()
    return q_table
```
