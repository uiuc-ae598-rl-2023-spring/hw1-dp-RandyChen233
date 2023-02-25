import random
import numpy as np
from discrete_pendulum import Pendulum
import matplotlib.pyplot as plt

class SarsaTD0:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((env.num_states, env.num_actions))
        self.V = np.zeros(self.env.num_states)
        
        
    def state_to_xy(self, state):
        num_velocities = self.env.n_thetadot
        num_angles = self.env.n_theta
        max_velocity = self.env.max_thetadot
        angle = ((state // (num_velocities + 1)) * 2 * np.pi / num_angles) - np.pi
        velocity = ((state % (num_velocities + 1)) * 2 * max_velocity / num_velocities) - max_velocity
        x = np.sin(angle) + velocity
        y = np.cos(angle)
        
        return x, y



    def select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            # Choose a random action
            return random.randint(0, self.env.num_actions - 1)
        else:
            # Choose the action with highest Q-value
            return np.argmax(self.Q[state])
    
    def train(self, num_episodes):
        returns = []
        for _ in range(num_episodes):
            state = self.env.reset()
            action = self.select_action(state)
            total_return = 0
            done = False
            
            while not done:
                next_state, reward, done = self.env.step(action)
                
                next_action = self.select_action(next_state)
                td_target = reward + self.gamma * self.Q[next_state][next_action]
                td_error = td_target - self.Q[state][action]
                self.Q[state][action] += self.alpha * td_error
                total_return += reward
                state = next_state
                action = next_action
            returns.append(total_return)

        # Compute value function using TD(0)
        for state in range(self.env.num_states):
            for action in range(self.env.num_actions):
                self.V[state] += self.Q[state, action] * self.epsilon / self.env.num_actions + \
                                (1 - self.epsilon) * self.Q[state, np.argmax(self.Q[state, :])] \
                                * (1 - self.epsilon + self.epsilon / self.env.num_actions)
        return returns, self.V, self.Q
    def plot_learning_curves(self, alphas, epsilons, num_episodes):
        fig, axs = plt.subplots(nrows=len(alphas), ncols=len(epsilons), figsize=(15, 10))
        for i, alpha in enumerate(alphas):
            for j, epsilon in enumerate(epsilons):
                agent = SarsaTD0(self.env, alpha=alpha, gamma=0.95, epsilon=epsilon)
                returns, V, Q= agent.train(num_episodes)
                axs[i][j].plot(range(num_episodes), returns)
                axs[i][j].set_title(f'alpha={alpha}, epsilon={epsilon}')
        plt.tight_layout()
        # plt.show()
        plt.savefig('figures/pendulum/Learning_curves_SARSA.png')

    def plot_example_trajectories(self, num_trajectories, num_episodes):
        fig, axs = plt.subplots(nrows=num_trajectories, figsize=(6, 3*num_trajectories))
        agent = QLearning(self.env)
        _, _ , Q = agent.train(num_episodes)
        for i in range(num_trajectories):
            state = self.env.reset()
            done = False
            traj = [state]
            while not done:
                action = np.argmax(Q[state])
                next_state, _, done = self.env.step(action)
                traj.append(next_state)
                state = next_state
            traj = np.array(traj)
            xs, ys = zip(*(self.state_to_xy(s) for s in traj))
            axs[i].scatter(xs, ys, s=50, c='r')
            axs[i].set_xlim([-2.5, 2.5])
            axs[i].set_ylim([-2.5, 2.5])
            axs[i].grid(color='k', linestyle='-', linewidth=1)
            axs[i].set_title(f'Sarsa Trajectory {i+1}')
        plt.tight_layout()
        plt.savefig('figures/pendulum/example_trajectory_QLearning.png')


    def plot_policy(self, num_episodes):
        # Compute state-values and policy
        agent = SarsaTD0(self.env)
        returns, V, Q= agent.train(num_episodes)
        state_values = V
        policy = np.argmax(Q, axis=1)
        # Plot state-value function
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        axs[0].plot(state_values)
        axs[0].set_xlabel('State')
        axs[0].set_ylabel('State-value')
        axs[0].set_title('State-value function')
        
        # Plot policy
        axs[1].bar(range(self.env.num_states), policy)
        axs[1].set_xlabel('State')
        axs[1].set_ylabel('Action')
        axs[1].set_title('Policy')
        
        plt.savefig('figures/pendulum/policy_stateValue_SARSA.png')


class QLearning:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((env.num_states, env.num_actions))
        self.V = np.zeros(self.env.num_states)

    def state_to_xy(self, state):
        num_velocities = self.env.n_thetadot
        num_angles = self.env.n_theta
        max_velocity = self.env.max_thetadot
        angle = ((state // (num_velocities + 1)) * 2 * np.pi / num_angles) - np.pi
        velocity = ((state % (num_velocities + 1)) * 2 * max_velocity / num_velocities) - max_velocity
        x = np.sin(angle) + velocity
        y = np.cos(angle)
        
        return x, y

    def select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            # Choose a random action
            return random.randint(0, self.env.num_actions - 1)
        else:
            # Choose the action with highest Q-value
            return np.argmax(self.Q[state])
    
    def train(self, num_episodes):
        returns = []
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            total_returns = 0
            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.env.step(action)
                td_target = reward + self.gamma * np.max(self.Q[next_state])
                td_error = td_target - self.Q[state][action]
                self.Q[state][action] += self.alpha * td_error
                state = next_state
                total_returns += reward
            returns.append(total_returns)
        # Compute value function using TD(0)
        
        for state in range(self.env.num_states):
            self.V[state] = np.max(self.Q[state])
            
        return returns, self.V, self.Q
    
    def plot_learning_curves(self, alphas, epsilons, num_episodes):
        fig, axs = plt.subplots(nrows=len(alphas), ncols=len(epsilons), figsize=(15, 10))
        for i, alpha in enumerate(alphas):
            for j, epsilon in enumerate(epsilons):
                agent = QLearning(self.env, alpha=alpha, gamma=0.95, epsilon=epsilon)
                returns, V, Q= agent.train(num_episodes)
                axs[i][j].plot(range(num_episodes), returns)
                axs[i][j].set_title(f'alpha={alpha}, epsilon={epsilon}')
        plt.tight_layout()
        plt.savefig('figures/pendulum/Learning_curves_Qlearning.png')

    def plot_example_trajectories(self, num_trajectories, num_episodes):
        fig, axs = plt.subplots(nrows=num_trajectories, figsize=(6, 3*num_trajectories))
        agent = QLearning(self.env)
        _, _ , Q = agent.train(num_episodes)
        for i in range(num_trajectories):
            state = self.env.reset()
            done = False
            traj = [state]
            while not done:
                action = np.argmax(Q[state])
                next_state, _, done = self.env.step(action)
                traj.append(next_state)
                state = next_state
            traj = np.array(traj)
            xs, ys = zip(*([self.state_to_xy(s) for s in traj]))
            axs[i].scatter(xs, ys, s=50, c='r')
            axs[i].set_xlim([-2.5, 2.5])
            axs[i].set_ylim([-2.5, 2.5])
            axs[i].grid(color='k', linestyle='-', linewidth=1)
            axs[i].set_title(f'Q-learning Trajectory {i+1}')
        plt.tight_layout()
        plt.savefig('figures/pendulum/example_trajectory_QLearning.png')


    def plot_policy(self, num_episodes):
        # Compute state-values and policy
        agent = SarsaTD0(self.env)
        returns, V, Q= agent.train(num_episodes)
        state_values = V
        policy = np.argmax(Q, axis=1)
        # Plot state-value function
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        axs[0].plot(state_values)
        axs[0].set_xlabel('State')
        axs[0].set_ylabel('State-value')
        axs[0].set_title('State-value function')
        
        # Plot policy
        axs[1].bar(range(self.env.num_states), policy)
        axs[1].set_xlabel('State')
        axs[1].set_ylabel('Action')
        axs[1].set_title('Policy')
        plt.tight_layout()
        plt.savefig('figures/pendulum/policy_stateValue_QLearning.png')



"""

MAIN LOOP:


"""    
        
def main():
  
    env = Pendulum(n_theta=15, n_thetadot=21)
    
    
    """SARSA"""
    # by default: alpha=0.5, gamma=0.95, epsilon=0.1
    env.reset()
    alphas = [0.3, 0.5, 0.8]
    epsilons = [0.1, 0.3, 0.5]
    num_episodes = 50
    sarsa = SarsaTD0(env)
    #Learning curves with various alphas and epsilons
    sarsa.plot_learning_curves(alphas, epsilons, num_episodes)
    sarsa.plot_example_trajectories(5,num_episodes)
    sarsa.plot_policy(num_episodes)

    """Q Learning"""
    env.reset()
    alphas = [0.3, 0.5, 0.8]
    epsilons = [0.1, 0.3, 0.5]
    num_episodes = 50
    sarsa = QLearning(env)
    #Learning curves with various alphas and epsilons
    sarsa.plot_learning_curves(alphas, epsilons, num_episodes)
    sarsa.plot_example_trajectories(5,num_episodes)
    sarsa.plot_policy(num_episodes)


if __name__ == '__main__':
    main()