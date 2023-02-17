import random
import matplotlib
matplotlib.use('Agg')
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
        
        delta = self.one_policy_evaluation()
        delta_history = [delta]
        while delta > tol:
            
            delta = self.one_policy_evaluation()
            delta_history.append(delta)

    def train(self, plot = True):

        V = np.zeros(self.num_states)
        policy = np.zeros(self.num_states)

    def run_policy_improvement(self):
        """
        Runs policy improvement and updates the policy.

        :return: the number of state-action pairs that have been updated and the average value function
        """
         
        while True:
            delta = self.one_policy_evaluation() # evaluate current policy
            if delta < 1e-4:
                break

        stable_policy = True
        for s in range(self.num_states):
            old_action = self.policy[s]

            # Find action that maximizes the Q value
            Q = np.zeros(self.num_actions)
            for a in range(self.num_actions):
                q = 0
                for s_new in range(self.num_states):
                    r = self.env.r(s_new, a)
                    p = self.env.p(s_new, s, a)
                    q += p * (r + self.gamma * self.values[s_new])
                Q[a] = q

            new_action = np.argmax(Q)
            if old_action != new_action:
                stable_policy = False

            self.policy[s] = new_action

        return stable_policy, self.policy, self.values
        
    
def main():
    
    # Create environment
    env = gridworld.GridWorld(hard_version=False)
    pi = PolicyIteration(env, gamma=0.95)

    num_iterations = 100
    mean_values = np.zeros(num_iterations)

    eval_count_history = []
    for i in tqdm(range(num_iterations)):
        eval_count = pi.run_policy_evaluation()
        eval_count_history.append(eval_count)

        mean_values[i] = np.mean(pi.values)
        pi.run_policy_improvement()

    print(eval_count_history)
    # Create plot
    plt.figure(dpi=200)
    plt.plot(eval_count_history,mean_values)
    plt.xlabel("Number of iterations")
    plt.ylabel("Mean of value function")
    plt.show()
    plt.savefig('test_Feb16.png')

if __name__ == '__main__':
    main()


