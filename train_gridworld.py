import matplotlib
matplotlib.use('Agg')
import random
import numpy as np
import matplotlib.pyplot as plt
import gridworld
from tqdm import tqdm

""""Since this is a HW assignment, each method has its own separate class object for clarity purposes"""


class PolicyIteration:
    def __init__(self, env, gamma):
        self.env = env
        self.num_states = self.env.num_states
        self.num_actions = self.env.num_actions
        self.max_num_steps = self.env.max_num_steps

        self.gamma = gamma #discount factor

        
    def run_policy_evaluation(self, policy, V, tol=1e-3):
  
        done = False
        delta = 0
        delta_history = [delta]
        mean_V_list = []

        while not done:
            for s in range(self.num_states):
                v = V[s]
                a = policy[s]
                (_, _, done) = self.env.step(a)
                # p = self.env.p(s_new, s, a)
               
                #Update V[s]:
                V[s] = np.sum([self.env.p(s_new, s, policy[s]) * (r + self.gamma * V[s_new]) \
                                                for (s_new, r) in [self.env.step(policy[s])[0:2]]])

                delta = max(delta, abs(v - V[s]))
                delta_history.append(delta)

            if delta < tol:
                break

            # Compute the mean value of V and append it to the list
            mean_V = sum(V)/self.num_states
            mean_V_list.append(mean_V)

        return V, len(delta_history), mean_V_list

    def run_policy_improvement(self, policy, V):
        
        new_policy = np.zeros(self.env.n_states, dtype=np.int)
        for s in range(self.num_states):
            
            v_list = np.zeros(self.num_actions)

            for a in range(self.num_actions):
                # (s_new, r, _) = self.env.step(a)
                # p = self.env.p(s_new, s, a)

                v_list[a] = np.sum([self.env.p(s_new, s, policy[s]) * (r + self.gamma * V[s_new]) \
                                         for (s_new, r) in self.env.step(policy[s])[:2]])
                
            new_policy[s] = np.argmax(v_list)

        return new_policy
    
    def train(self, tol=1e-3, max_iters=100, plot=True):
        
        policy = np.zeros(self.env.n_states, dtype=np.int)
        V = np.zeros(self.env.n_states)

        eval_count_history = []

        epoch = 0
        for i in range(max_iters):

            # Evaluate the current policy
            V, eval_count, mean_V_history = self.run_policy_evaluation(policy, V)
            eval_count_history.append(eval_count)

            # Improve the policy
            new_policy = self.run_policy_improvement(policy, V)

            # Check if the policy has converged
            if np.array_equal(new_policy, policy):
                break

            policy = new_policy

            epoch +=1

        print(f'eval count history = {eval_count_history}')

        if plot is True:
            plt.figure(dpi=200)
            plt.plot(mean_V_history)
            plt.tight_layout()
            plt.savefig('figures/gridworld/policy_iteration.png')
            # plt.show()

        # return policy,V

class ValueIteration:
    def __init__(self, env, gamma):
        self.env = env
        self.num_states = self.env.num_states
        self.num_actions = self.env.num_actions
        self.max_num_steps = self.env.max_num_steps

        self.gamma = gamma #discount factor
        self.values = np.zeros(self.num_states) #Initialize `values` as zeros
        self.policy = np.random.randint(0, self.num_actions, self.num_states)

    def train(self, tol=1e-3, plot = True):
        delta = float('inf')
        i = 0
        value_hist = []
        while delta > tol:
            delta = 0
            for s in range(self.num_states):
                v = self.values[s]
                v_list = np.zeros(self.num_actions)
                for a in range(self.num_actions):
                    # (s_new, r, _) = self.env.step(a)
                    # p = self.env.p(s_new, s, a)
                    v_list[a] = np.sum([self.env.p(s_new, s, self.policy[s]) * (r + self.gamma * self.values[s_new]) \
                                         for (s_new, r) in self.env.step(self.policy[s])[:2]])
                self.values[s] = np.max(v_list)
                self.policy[s] = np.argmax(v_list)
                delta = max(delta, abs(v - self.values[s]))
                s = self.env.reset()
            i += 1

            value_hist.append(np.mean(self.values))
        print(f"Converged in {i} iterations")


        if plot:
            plt.figure(dpi=200)
            plt.plot(value_hist)
            plt.tight_layout()
            plt.savefig('figures/gridworld/value_iteration.png')

        return self.policy
def main():
    env = gridworld.GridWorld(hard_version=False)
    policy_iter = PolicyIteration(env, gamma=0.95)
    policy_iter.train()   

    # value_iter = ValueIteration(env, gamma = 0.95)
    # value_iter.train()
    

if __name__ == '__main__':
    main()


