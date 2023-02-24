import random
import numpy as np
import matplotlib.pyplot as plt
import gridworld

"""Since this is a HW assignment, each method implemented has its own Class. Ideally we can use an abstract class"""

class PolicyIteration:
    def __init__(self, env, gamma):
        self.env = env
        self.num_states = self.env.num_states
        self.num_actions = self.env.num_actions
        self.max_num_steps = self.env.max_num_steps

        self.gamma = gamma #discount factor
        self.theta= 1e-3

    def train(self,plot=True):
        # Initialize policy and value function
        policy =np.random.randint(self.num_actions, size=self.num_states)
        V = np.zeros(self.num_states)
        
        mean_V_list = []

        self.env.reset()

        # Outer loop: iterate over policy and value function until convergence
        delta_history_count = []
        while True:
            # Policy evaluation: update value function using current policy
            while True:

                delta = 0
                delta_count_per_eval = []
    
                for s in range(self.num_states):
                    # print(f'state is {s}')
                    v = V[s]
                    V[s] = sum(self.env.p(s_new, s, policy[s]) * (r + self.gamma * V[s_new])\
                                                for (s_new, r, _) in [self.env.step(policy[s])])
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
                policy[s] = np.argmax([sum(self.env.p(s_new, s, policy[s]) * (r + self.gamma * V[s_new]) \
                                           for (s_new, r, _) in [self.env.step(a)]) for a in range(self.num_actions)])
                if old_action != policy[s]:
                    policy_stable = False
            if policy_stable:
                break

            

        print(f'Total counts of policy evaluations is {len(delta_history_count)}! \n')
        print(f'Total counts of values stored is {len(mean_V_list)}!')

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
        print(f'Policy is {policy}')
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
            print(f'trajectory is {traj}')
            axs[i].scatter(traj[:, 0], traj[:, 1], c=np.arange(len(traj)), cmap='plasma', alpha=0.8)
            axs[i].set_xlim([0, 5])
            axs[i].set_ylim([0, 5])
            axs[i].set_xticks(np.arange(0.5, 5, 1))
            axs[i].set_yticks(np.arange(0.5, 5, 1))
            axs[i].grid(color='k', linestyle='-', linewidth=1)
            axs[i].set_title(f'Trajectory {i+1}')
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

        values = np.zeros(self.num_states)
        policy = np.random.randint(0, self.num_actions, self.num_states)
        
        value_hist_mean = []
        # Outer loop: iterate over policy and value function until convergence
        while True:
            delta = 0
            for s in range(self.num_states):
                v = values[s]
                v_list = np.zeros(self.num_actions)
                for a in range(self.num_actions):
                    v_list[a] = np.sum([self.env.p(s_new, s, policy[s]) * (r + self.gamma * values[s_new]) \
                                         for (s_new, r, _) in [self.env.step(policy[s])]])
                    
                # Update the value function for the state
                values[s] = np.max(v_list)
                policy[s] = np.argmax(v_list)
                delta = max(delta, abs(v - values[s]))

            value_hist_mean.append(sum(values)/self.num_states)
        
            if delta < self.theta:
                    break
            
        print(f'Total counts of values stored is {len(value_hist_mean)}!')
            
        if plot is True:
            plt.figure(dpi=200)
            plt.plot(value_hist_mean,label='gamma=0.95')
            plt.grid('on')
            plt.title('Mean Value vs Iterations')
            plt.legend()
            plt.tight_layout()
            plt.savefig('figures/gridworld/LearningCurve_ValueIter.png')
            
        return policy, values
    
    def plot_example_trajectories(self, num_steps):
        agent =  ValueIteration(self.env, gamma=0.95)
        policy, _ = agent.train(plot=False)

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
            axs[i].scatter(traj[:, 0], traj[:, 1], c=np.arange(len(traj)), cmap='plasma', alpha=0.8)
            axs[i].set_xlim([0, 5])
            axs[i].set_ylim([0, 5])
            axs[i].set_xticks(np.arange(0.5, 5, 1))
            axs[i].set_yticks(np.arange(0.5, 5, 1))
            axs[i].grid(color='k', linestyle='-', linewidth=1)
            axs[i].set_title(f'Trajectory {i+1}')
        plt.tight_layout()
        plt.savefig('figures/gridworld/example_trajectory_ValueIter.png')
    
    def plot_policy(self):
        # Compute state-values and policy
        agent =  PolicyIteration(self.env, gamma=0.95)
        policy, state_values = agent.train(plot=False)
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
        
        plt.savefig('figures/gridworld/policy_stateValue_ValueIter.png')
    

class SarsaTD0:
    def __init__(self, env, alpha=0.5, gamma=0.95, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((self.env.num_states, self.env.num_actions))

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
        self.V = np.zeros(self.num_states)
        for state in range(self.num_states):
            for action in range(self.num_actions):
                self.V[state] += self.Q[state, action] * self.epsilon / self.num_actions + \
                                (1 - self.epsilon) * self.Q[state, np.argmax(self.Q[state, :])] \
                                * (1 - self.epsilon + self.epsilon / self.num_actions)
        return returns

    def plot_learning_curves(self, alphas, epsilons, num_episodes):
        fig, axs = plt.subplots(nrows=len(alphas), ncols=len(epsilons), figsize=(15, 10))
        for i, alpha in enumerate(alphas):
            for j, epsilon in enumerate(epsilons):
                agent = SarsaTD0(self.env, alpha=alpha, gamma=0.95, epsilon=epsilon)
                returns= agent.train(num_episodes)
                axs[i][j].plot(range(num_episodes), returns)
                axs[i][j].set_title(f'alpha={alpha}, epsilon={epsilon}')
        plt.tight_layout()
        # plt.show()
        plt.savefig('figures/gridworld/Learning_curves_SARSA.png')

    def plot_example_trajectories(self, num_trajectories):
        fig, axs = plt.subplots(nrows=num_trajectories, figsize=(6, 3*num_trajectories))
        for i in range(num_trajectories):
            state = self.env.reset()
            done = False
            traj = [state]
            while not done:
                action = np.argmax(self.Q[state])
                next_state, _, done= self.env.step(action)
                traj.append(next_state)
                state = next_state
            traj = np.array(traj)
            axs[i].scatter(traj[:, 0], traj[:, 1], c=np.arange(len(traj)), cmap='plasma', alpha=0.8)
            axs[i].set_xlim([0, 5])
            axs[i].set_ylim([0, 5])
            axs[i].set_xticks(np.arange(0.5, 5, 1))
            axs[i].set_yticks(np.arange(0.5, 5, 1))
            axs[i].grid(color='k', linestyle='-', linewidth=1)
            axs[i].set_title(f'Trajectory {i+1}')
        plt.tight_layout()
        plt.savefig('figures/gridworld/example_trajectory_SARSA.png')
        

    def plot_policy(self):
        # Compute state-values and policy
        state_values = self.V
        policy = np.argmax(self.Q, axis=1)
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


"""

MAIN LOOP:


"""    
        
    
def main():
    """Policy iteration"""
    
    env = gridworld.GridWorld(hard_version=False)
    policy_iter = PolicyIteration(env, gamma=0.95)
    policy_iter.plot_example_trajectories(num_steps=100)
    policy_iter.plot_policy()

    """Value iteration"""
    env.reset()
    value_iter = ValueIteration(env, gamma = 0.95)
    value_iter.plot_example_trajectories(num_steps=100)
    value_iter.plot_policy()
    
    """SARSA"""
    # by default: alpha=0.5, gamma=0.95, epsilon=0.1
    alphas = [0.3, 0.5, 0.8]
    epsilons = [0.1, 0.3, 0.5]
    num_episodes = 100
    sarsa = SarsaTD0(env)
    #Learning curves with various alphas and epsilons
    sarsa.plot_learning_curves(alphas, epsilons, num_episodes)
    sarsa.plot_example_trajectories(num_trajectories=20)
    sarsa.plot_policy()

   

  





if __name__ == '__main__':
    main()


