import matplotlib
matplotlib.use('Agg')
import random
import numpy as np
import matplotlib.pyplot as plt
import gridworld
from tqdm import tqdm

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
        policy = np.zeros(self.num_states)
        V = np.zeros(self.num_states)
        
        mean_V_list = []

        # Outer loop: iterate over policy and value function until convergence
        delta_history_count = []
        while True:
            
            # Policy evaluation: update value function using current policy
            while True:
                delta = 0
                delta_count_per_eval = []
                
                for s in range(self.num_states):
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

            # Reset environment before starting new episode
            s = self.env.reset()

        print(f'Total counts of policy evaluations is {len(delta_history_count)}! \n')
        print(f'Total counts of values stored is {len(mean_V_list)}!')

        if plot is True:
            plt.figure(dpi=200)
            plt.plot(mean_V_list)
            plt.tight_layout()
            plt.grid('on')
            plt.title('Policy iteration')
            plt.savefig('figures/gridworld/policy_iteration.png')

        return policy, V, delta_history_count


class ValueIteration:
    def __init__(self, env, gamma):
        self.env = env
        self.num_states = self.env.num_states
        self.num_actions = self.env.num_actions
        self.max_num_steps = self.env.max_num_steps

        self.gamma = gamma #discount factor
        self.values = np.zeros(self.num_states) #Initialize `values` as zeros
        self.policy = np.random.randint(0, self.num_actions, self.num_states)
        self.theta =  1e-3

    def train(self, plot = True):
        delta = 0
        value_hist_mean = []
        # Outer loop: iterate over policy and value function until convergence
        while True:
        
            for s in range(self.num_states):
                v = self.values[s]
                v_list = np.zeros(self.num_actions)

                for a in range(self.num_actions):
                    # (s_new, r, _) = self.env.step(a)
                    # p = self.env.p(s_new, s, a)

                    v_list[a] = np.sum([self.env.p(s_new, s, self.policy[s]) * (r + self.gamma * self.values[s_new]) \
                                         for (s_new, r, _) in [self.env.step(self.policy[s])]])
                self.values[s] = np.max(v_list)
                self.policy[s] = np.argmax(v_list)
                delta = max(delta, abs(v - self.values[s]))
                s = self.env.reset()

            if delta < self.theta:
                    break
            
            value_hist_mean.append(sum(v_list)/self.num_states)
     

        if plot is True:
            plt.figure(dpi=200)
            plt.plot(value_hist_mean)
            plt.tight_layout()
            plt.savefig('figures/gridworld/value_iteration.png')

        return self.policy, self.values
    
def main():
    env = gridworld.GridWorld(hard_version=False)
    policy_iter = PolicyIteration(env, gamma=0.95)
    policy_iter.train()   

    value_iter = ValueIteration(env, gamma = 0.95)
    value_iter.train()
    

if __name__ == '__main__':
    main()


