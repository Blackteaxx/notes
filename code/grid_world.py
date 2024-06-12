import numpy as np
import argparse
import random


class GridWorld:
    def __init__(self, size=10, obstacles=None):
        self.size = size
        self.start = (0, 0)
        self.goal = (9, 9)
        self.state = self.start
        
        # 设置障碍物的位置
        self.obstacles = obstacles if obstacles is not None else [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 0:  # up
            next_state = (max(0, x - 1), y)
        elif action == 1:  # down
            next_state = (min(self.size - 1, x + 1), y)
        elif action == 2:  # left
            next_state = (x, max(0, y - 1))
        elif action == 3:  # right
            next_state = (x, min(self.size - 1, y + 1))

        if next_state in self.obstacles:
            next_state = self.state

        if next_state == self.goal:
            reward = 10
        else:
            reward = -1

        self.state = next_state
        return next_state, reward


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
    
    iterations = 0
    
    while True:
            
        iterations += 1
        
        updated_V = V.copy()
        
        for now_state_x in range(env.size):
            for now_state_y in range(env.size):
                
                env.state = (now_state_x, now_state_y)
                
                action = policy[now_state_x, now_state_y]
                
                next_state, reward = env.step(action=action)
                
                updated_V[now_state_x, now_state_y] = reward + gamma * V[next_state[0], next_state[1]]
        
        if np.amax(np.fabs(updated_V - V)) <= theta:
            V = updated_V
            print ('Policy-evaluation converged at iteration# %d.' %(iterations))
            break
        else:
            V = updated_V
    
    ####
    
    return V


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

    iterations = 0
    
    while True:
        
        iterations += 1
        
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
            print ('Policy-iteration converged at iteration# %d.' %(iterations))
            break
    
    ####
    env.reset()
    return policy


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
    
    return_list = []
    
    for _ in range(episodes):
        x, y = env.start
        
        episode_return = 0
        
        while True:
            policy = extract_policy(q_table=q_table)
            action = policy[x, y]
            
            if random.uniform(0,1) <= epsilon:
                action = random.randint(0,3)
                
            env.state = (x, y)
            new_state, reward = env.step(action=action)
        
            episode_return += reward
            
            q_table[x, y, action] += alpha * (reward + gamma * 
                                              np.amax(q_table[new_state[0], new_state[1]]) - q_table[x, y, action])

            x, y = new_state
            
            if new_state == env.goal:
                break 
            
        return_list.append(episode_return)
    ####
    
    import matplotlib.pyplot as plt
    plt.scatter(range(episodes), return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.show()
    
    env.reset()
    return q_table


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
    
    return_list = []
    
    for _ in range(episodes):
        
        x, y = env.start
        policy = extract_policy(q_table=q_table)
        action = policy[x, y]
        # 抽样
        if random.uniform(0, 1) <= epsilon:
                action = random.randint(0, 3)
        
        episode_return = 0
        
        while True:
            policy = extract_policy(q_table=q_table)
              
            env.state = (x, y)
            new_state, reward = env.step(action=action)
            
            episode_return += reward
            
            new_action = policy[new_state[0], new_state[1]]
            
            if random.uniform(0, 1) <= epsilon:
                new_action = random.randint(0, 3)
                
            q_table[x, y, action] += alpha * (reward + gamma * q_table[new_state[0], new_state[1], new_action] - q_table[x, y, action])

            x, y = new_state
            action = new_action
            
            if new_state == (9, 9):
                break
        
        return_list.append(episode_return)
    ####

    # plot the return
    import matplotlib.pyplot as plt
    plt.scatter(range(episodes), return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.show()
    
    env.reset()
    return q_table


def display_policy(policy):
    directions = ['↑', '↓', '←', '→']
    for row in policy:
        print(' '.join([directions[action] for action in row]))


def extract_policy(q_table):
    """
    Extract the optimal policy from the Q-value table.

    Parameters:
    q_table: numpy.ndarray
        A Q-value table of shape (env.size, env.size, 4), storing the Q-values for each state-action pair.

    Returns:
    policy: numpy.ndarray
        A policy matrix of shape (q_table.shape[0], q_table.shape[1]), storing the optimal action for each state.
    """

    ####
    # Implement the function to extract the optimal policy from the Q-value table
    
    policy = np.argmax(q_table, axis=2)
    
    ####

    return policy

def display_path(env:GridWorld, policy):
    """
    Display the path from the start state to the goal state using the learned policy.

    Parameters:
    env: GridWorld
        The environment instance.
    policy: numpy.ndarray
        The policy matrix storing the optimal action for each state.
    """
    ####
    # Implement the function to display the path from the start state to the goal state using the learned policy
    
    print("\n========== SHOW PATH ==========")
    
    print(env.state, end=" ")
    
    while env.state != (9, 9):
    
        env.step(policy[env.state[0], env.state[1]])
        
        print("->", end=" ")
        print(env.state, end=" ")
    
    
    print("\n========== END  PATH ==========")
    ####


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Grid World Reinforcement Learning')
    parser.add_argument('--algorithm', type=str, choices=['value_iteration', 'policy_iteration', 'q_learning', 'sarsa'], default='q_learning', help='Algorithm to use for training')
    args = parser.parse_args()

    obstacles = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]
    env = GridWorld(obstacles=obstacles)
    if args.algorithm == 'value_iteration':
        policy = value_iteration(env)
    elif args.algorithm == 'policy_iteration':
        policy = policy_iteration(env)
    elif args.algorithm == 'q_learning':
        q_table = q_learning(env)
        policy = extract_policy(q_table)
    elif args.algorithm == 'sarsa':
        q_table = sarsa(env)
        policy = extract_policy(q_table)
    display_policy(policy)
    display_path(env, policy)
