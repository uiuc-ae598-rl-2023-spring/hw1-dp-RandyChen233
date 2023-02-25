import random
import numpy as np
import matplotlib.pyplot as plt
import gridworld

"""Since this is a HW assignment, each method implemented has its own Class. Ideally we can use an abstract class"""
state_to_xy = {
    0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (3, 0), 4: (4, 0),
    5: (0, 1), 6: (1, 1), 7: (2, 1), 8: (3, 1), 9: (4, 1),
    10: (0, 2), 11: (1, 2), 12: (2, 2), 13: (3, 2), 14: (4, 2),
    15: (0, 3), 16: (1, 3), 17: (2, 3), 18: (3, 3), 19: (4, 3),
    20: (0, 4), 21: (1, 4), 22: (2, 4), 23: (3, 4), 24: (4, 4),
}


class PolicyIteration:
    def __init__(self, env, gamma):
        self.env = env
        self.num_states = self.env.num_states
        self.num_actions = self.env.num_actions
        self.max_num_steps = self.env.max_num_steps

        self.gamma = gamma #discount factor
        self.theta= 1e-3

    def train(self,plot=True):
       
        # Initialize the policy and value function
        policy = np.zeros(self.num_states)
        V = np.zeros(self.num_states)
        
        mean_V_list = []

        # Outer loop
        delta_history_count = []
        while True:
            # Policy evaluation
            while True:
                delta = 0
                delta_count_per_eval = []
                for s in range(self.num_states):
                    v = V[s]
                    # V[s] = sum(self.env.p(s_new, s, policy[s]) * (r + self.gamma * V[s_new])\
                    #                             for (s_new, r, _) in [self.env.step(policy[s])])
                    V[s] =  sum([self.env.p(s1, s, policy[s]) * (self.env.r(s, policy[s]) + self.gamma * V[s1]) \
                                                            for s1 in range(self.env.num_states)])
                    delta = max(delta, abs(v - V[s]))
                    delta_count_per_eval.append(delta)
                if delta < self.theta:
                    break

                delta_history_count.append(len(delta_count_per_eval))

                mean_V = sum(V)/self.num_states
                mean_V_list.append(mean_V)

            # Policy improvement: update policy using current value function
            policy_stable = True
            for s in range(self.num_states):
                old_action = policy[s]
                policy[s] = policy[s] = np.argmax([sum([self.env.p(s1, s, a) * (self.env.r(s, a) + self.gamma * V[s1]) \
                                            for s1 in range(self.env.num_states)]) for a in range(self.env.num_actions)])
                if old_action != policy[s]:
                    policy_stable = False
            if policy_stable:
                break
         
        # print(f'Total counts of policy evaluations is {len(delta_history_count)}! \n')
        # print(f'Total counts of values stored is {len(mean_V_list)}!')

        if plot is True:
            plt.figure(dpi=200)
            plt.plot(mean_V_list,label='gamma=0.95')
            plt.legend()
            plt.grid('on')
            plt.title('Mean Value vs Evaluations')
            plt.savefig('figures/gridworld/LearningCurve_PolicyIter.png')

        return policy, V, delta_history_count
    
    def plot_example_trajectories(self, num_steps):
        agent =  PolicyIteration(self.env, gamma=0.95)
        policy, _, _ = agent.train(plot=False)
        fig, axs = plt.subplots(nrows=num_steps, figsize=(6, 3*num_steps))
        for i in range(num_steps):
            state = self.env.reset()
            done = False
            traj = [state]
            while not done:
                action = policy[state]
                next_state, _, done= self.env.step(action)
                traj.append(next_state)
                state = next_state
            traj = np.array(traj)

            xs, ys = zip(*(state_to_xy[s] for s in traj))
            axs[i].scatter(xs, ys, s = 50, c='r')
            axs[i].set_xlim([0, 5])
            axs[i].set_ylim([0, 5])

            # for j in range(len(traj) - 1):
            #     start = state_to_xy[traj[j]]
            #     end = state_to_xy[traj[j+1]]
            #     dx = end[0] - start[0]
            #     dy = end[1] - start[1]
            #     axs[i].arrow(start[0], start[1], dx, dy, length_includes_head=True, head_width=0.1, color='black')

            axs[i].grid(color='k', linestyle='-', linewidth=1)
            axs[i].set_title(f'Policy Iter Trajectory {i+1}')
        plt.tight_layout()
        plt.savefig('figures/gridworld/example_trajectory_PolicyIter.png')

    
    def plot_policy(self):
        # Compute state-values and policy
        agent =  PolicyIteration(self.env, gamma=0.95)
        policy, state_values, _ = agent.train(plot=False)
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
        
        plt.savefig('figures/gridworld/policy_stateValue_PolicyIter.png')
        
class ValueIteration:
    def __init__(self, env, gamma):
        self.env = env
        self.num_states = self.env.num_states
        self.num_actions = self.env.num_actions
        self.max_num_steps = self.env.max_num_steps

        self.gamma = gamma #discount factor
        self.theta =  1e-3

    def train(self, plot = True):

        V = np.zeros(self.num_states)
        policy = np.zeros(self.num_states)

        
        value_hist_mean = []

        # Outer loop: iterate over policy and value function until convergence
        while True:
            delta = 0
            # Update the value of each state
            for s in range(self.num_states):
                v = V[s]
                max_v = -np.inf
                max_a = -1
                # Find the action that maximizes the value function
                for a in range(self.num_actions):
                    q = 0
                    
                    # Compute the expected value of each next state
                    for s1 in range(self.num_states):
                        p = self.env.p(s1, s, a)
                        r = self.env.r(s, a)
                        q += p * (r + self.gamma * V[s1])
                    
                    # Update the maximum value and action
                    if q > max_v:
                        max_v = q
                        max_a = a
                
                # Update the value function and optimal policy
                V[s] = max_v
                value_hist_mean.append(V[s]/self.num_states)
                policy[s] = max_a

                 # Update delta
                delta = max(delta, abs(v - V[s]))

            # Check if the value function has converged
            if delta < self.theta:
                break
    
            
        if plot is True:
            plt.figure(dpi=200)
            plt.plot(value_hist_mean,label='gamma=0.95')
            plt.grid('on')
            plt.title('Mean Value vs Iterations')
            plt.legend()
            plt.tight_layout()
            plt.savefig('figures/gridworld/LearningCurve_ValueIter.png')
            
        return policy, V
    
    def plot_example_trajectories(self, num_steps):
        agent =  ValueIteration(self.env, gamma=0.95)
        policy, V = agent.train(False)

        fig, axs = plt.subplots(nrows=num_steps, figsize=(6, 3*num_steps))
        for i in range(num_steps):
            state = self.env.reset()
            done = False
            traj = [state]
            while not done:
                action = policy[state]
                next_state, _, done= self.env.step(action)
                traj.append(next_state)
                state = next_state
            traj = np.array(traj)
            xs, ys = zip(*(state_to_xy[s] for s in traj))
            axs[i].scatter(xs, ys, s = 50, c='r')
            axs[i].set_xlim([0, 5])
            axs[i].set_ylim([0, 5])

            # for j in range(len(traj) - 1):
            #     start = state_to_xy[traj[j]]
            #     end = state_to_xy[traj[j+1]]
            #     dx = end[0] - start[0]
            #     dy = end[1] - start[1]
            #     axs[i].arrow(start[0], start[1], dx, dy, length_includes_head=True, head_width=0.1, color='black')
            axs[i].grid(color='k', linestyle='-', linewidth=1)
            axs[i].set_title(f'Value Iter Trajectory {i+1}')
        plt.tight_layout()
        plt.savefig('figures/gridworld/example_trajectory_ValueIter.png')
    
    def plot_policy(self):
        # Compute state-values and policy
        agent =  ValueIteration(self.env, gamma=0.95)
        policy, state_values = agent.train(False)
        # Plot state-value function
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        axs[0].plot(state_values)
        axs[0].set_xlabel('State')
        axs[0].set_ylabel('State-value')
        axs[0].set_title('State-value function')
        # Plot policy
        axs[1].bar(range(self.num_states), policy)
        axs[1].set_xlabel('State')
        axs[1].set_ylabel('Action')
        axs[1].set_title('Policy')
        
        plt.savefig('figures/gridworld/policy_stateValue_ValueIter.png')
    

class SarsaTD0:
    def __init__(self, env, alpha=0.5, gamma=0.95, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((self.env.num_states, self.env.num_actions))
        self.V = np.zeros(self.env.num_states)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            # Choose a random action
            return random.randint(0, self.env.num_actions-1)
        else:
            # Choose the greedy action
            return np.argmax(self.Q[state])

    def train(self, num_episodes):
        returns = []
        for episode in range(num_episodes):
            state = self.env.reset()
            action = self.choose_action(state)
            total_return = 0
            for step in range(self.env.max_num_steps):
                next_state, reward, done= self.env.step(action)
                next_action = self.choose_action(next_state)
                # SARSA update rule
                self.Q[state][action] += self.alpha * (reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action])
                total_return += reward
                state = next_state
                action = next_action
                if done:
                    break
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
        plt.savefig('figures/gridworld/Learning_curves_SARSA.png')

    def plot_example_trajectories(self, num_trajectories, num_episodes):
        fig, axs = plt.subplots(nrows=num_trajectories, figsize=(6, 3*num_trajectories))
        agent = SarsaTD0(self.env)
        returns, V, Q= agent.train(num_episodes)
        for i in range(num_trajectories):
            state = self.env.reset()
            done = False
            traj = [state]
            while not done:
                action = np.argmax(Q[state])
                next_state, _, done= self.env.step(action)
                traj.append(next_state)
                state = next_state
            traj = np.array(traj)
            xs, ys = zip(*(state_to_xy[s] for s in traj))
            axs[i].scatter(xs, ys, s = 50, c='r')
            axs[i].set_xlim([0, 5])
            axs[i].set_ylim([0, 5])

            # for j in range(len(traj) - 1):
            #     start = state_to_xy[traj[j]]
            #     end = state_to_xy[traj[j+1]]
            #     dx = end[0] - start[0]
            #     dy = end[1] - start[1]
            #     axs[i].arrow(start[0], start[1], dx, dy, length_includes_head=True, head_width=0.1, color='black')
            axs[i].grid(color='k', linestyle='-', linewidth=1)
            axs[i].set_title(f'SARSA Trajectory {i+1}')
        plt.tight_layout()
        plt.savefig('figures/gridworld/example_trajectory_SARSA.png')
        

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
        
        plt.savefig('figures/gridworld/policy_stateValue_SARSA.png')

##########################################################################################

class QLearning:
    def __init__(self, env, alpha=0.5, gamma=0.95, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((self.env.num_states, self.env.num_actions))
        self.V = np.zeros(self.env.num_states)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            # Choose a random action
            return random.randint(0, self.env.num_actions-1)
        else:
            # Choose the greedy action
            return np.argmax(self.Q[state])

    def train(self, num_episodes):
        returns = []
        for episode in range(num_episodes):
            state = self.env.reset()
            total_return = 0
            for step in range(self.env.max_num_steps):
                action = self.choose_action(state)
                next_state, reward, done= self.env.step(action)
                next_action = np.argmax(self.Q[next_state])
                # Q-learning update rule
                self.Q[state][action] += self.alpha * (reward + self.gamma * self.Q[next_state][next_action]\
                                                       - self.Q[state][action])
                total_return += reward
                state = next_state
                if done:
                    break
            returns.append(total_return)

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
        plt.savefig('figures/gridworld/Learning_curves_Qlearning.png')

    def plot_example_trajectories(self, num_trajectories, num_episodes):
        fig, axs = plt.subplots(nrows=num_trajectories, figsize=(6, 3*num_trajectories))
        agent = QLearning(self.env)
        returns, V, Q= agent.train(num_episodes)
        for i in range(num_trajectories):
            state = self.env.reset()
            done = False
            traj = [state]
            while not done:
                action = np.argmax(Q[state])
                next_state, _, done= self.env.step(action)
                traj.append(next_state)
                state = next_state
            traj = np.array(traj)
            xs, ys = zip(*(state_to_xy[s] for s in traj))
            axs[i].scatter(xs, ys, s = 50, c='r')
            axs[i].set_xlim([0, 5])
            axs[i].set_ylim([0, 5])

            # for j in range(len(traj) - 1):
            #     start = state_to_xy[traj[j]]
            #     end = state_to_xy[traj[j+1]]
            #     dx = end[0] - start[0]
            #     dy = end[1] - start[1]
            #     axs[i].arrow(start[0], start[1], dx, dy, length_includes_head=True, head_width=0.1, color='black')
            axs[i].grid(color='k', linestyle='-', linewidth=1)
            axs[i].set_title(f'Q-learning Trajectory {i+1}')
        plt.tight_layout()
        plt.savefig('figures/gridworld/example_trajectory_QLearning.png')


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
        plt.savefig('figures/gridworld/policy_stateValue_QLearning.png')


"""

MAIN LOOP:


"""    
        
    
def main():
    """Policy iteration"""
    
    env = gridworld.GridWorld(hard_version=False)
    env.reset()
    policy_iter = PolicyIteration(env, gamma=0.95)
    policy_iter.plot_example_trajectories(num_steps=5)
    policy_iter.plot_policy()

    """Value iteration"""
    env.reset()
    value_iter = ValueIteration(env, gamma = 0.95)
    
    value_iter.plot_example_trajectories(num_steps=5)
    value_iter.plot_policy()
    
    """SARSA"""
    # by default: alpha=0.5, gamma=0.95, epsilon=0.1
    env.reset()
    alphas = [0.3, 0.5, 0.8]
    epsilons = [0.1, 0.3, 0.5]
    num_episodes = 100
    sarsa = SarsaTD0(env)
    #Learning curves with various alphas and epsilons
    sarsa.plot_learning_curves(alphas, epsilons, num_episodes)
    sarsa.plot_example_trajectories(5,num_episodes)
    sarsa.plot_policy(num_episodes)

    """Q Learning"""
    env.reset()
    alphas = [0.3, 0.5, 0.8]
    epsilons = [0.1, 0.3, 0.5]
    num_episodes = 100
    sarsa = QLearning(env)
    #Learning curves with various alphas and epsilons
    sarsa.plot_learning_curves(alphas, epsilons, num_episodes)
    sarsa.plot_example_trajectories(5,num_episodes)
    sarsa.plot_policy(num_episodes)


if __name__ == '__main__':
    main()


