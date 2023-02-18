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
        self.values = np.zeros(self.num_states) #Initialize `values` as zeros
        self.policy = np.random.randint(0, self.num_actions, self.num_states)
        
    def run_policy_evaluation(self, tol=1e-3):
   
        # s = self.env.reset()?
        delta = 0
        delta_history = [delta]
        done = False
        # for _ in range(self.max_num_steps):
        while not done:
            for s in range(self.num_states-1):
                v = self.values[s]
                # a = self.policy[s]
                # (s_new, r, done) = self.env.step(a)
                # p = self.env.p(s_new, s, a)

                #update V[s]:
                self.values[s] = np.sum([self.env.p(s_new, s, self.policy[s]) * (r + self.gamma * self.values[s_new]) \
                                                for (s_new, r) in [self.env.step(self.policy[s])[0:2]]])

                delta = max(delta, abs(v - self.values[s]))
                delta_history.append(delta)

                if delta < tol:
                    break

        return len(delta_history)

    def run_policy_improvement(self):
        update_policy_count = 0
        policy_stable = True

        for s in range(self.num_states-1):
            temp = self.policy[s]
            v_list = np.zeros(self.num_actions)

            for a in range(self.num_actions):
                # (s_new, r, _) = self.env.step(a)
                # p = self.env.p(s_new, s, a)

                v_list[a] = np.sum([self.env.p(s_new, s, self.policy[s]) * (r + self.gamma * self.values[s_new]) \
                                         for (s_new, r) in self.env.step(self.policy[s])[:2]])

                
            self.policy[s] = np.argmax(v_list)

            if temp != self.policy[s]:
                policy_stable = False
                update_policy_count += 1

        if policy_stable == True:
        
            return update_policy_count
    
    def train(self, tol=1e-3, max_iters=100, plot=True):
        
        eval_count = self.run_policy_evaluation(tol)
        eval_count_history = [eval_count]
        policy_change  = self.run_policy_improvement()
        policy_change_history = [policy_change]

        epoch = 0
        val_history= []
        
        for i in tqdm(range(max_iters)):
            epoch += 1
            new_eval_count = self.run_policy_evaluation(tol)
            
            new_policy_change  = self.run_policy_improvement()

            eval_count_history.append(new_eval_count)
            policy_change_history.append(new_policy_change)

            val_history.append(np.mean(self.values))

            if new_policy_change == 0 or epoch >= max_iters:
                break

        print(f'# epoch: {len(policy_change_history)}')
        print(f'eval count = {eval_count_history}')
        print(f'policy change = {policy_change_history}')

        if plot is True:
            plt.figure(dpi=200)
            plt.plot(val_history)
            plt.tight_layout()
            plt.savefig('figures/gridworld/policy_iteration.png')
            # plt.show()


# class ValueIteration:
#     def __init__(self, env, gamma):
#         self.env = env
#         self.num_states = self.env.num_states
#         self.num_actions = self.env.num_actions
#         self.max_num_steps = self.env.max_num_steps

#         self.gamma = gamma #discount factor
#         self.values = np.zeros(self.num_states) #Initialize `values` as zeros
#         self.policy = np.random.randint(0, self.num_actions, self.num_states)

#     def train(self, tol=1e-3, plot = True):
#         delta = float('inf')
#         i = 0
#         value_hist = []
#         while delta > tol:
#             delta = 0
#             for s in range(self.num_states):
#                 v = self.values[s]
#                 v_list = np.zeros(self.num_actions)
#                 for a in range(self.num_actions):
#                     (s_new, r, _) = self.env.step(a)
#                     p = self.env.p(s_new, s, a)
#                     v_list[a] = np.sum(p * (r + self.gamma * self.values[s_new]))
#                 self.values[s] = np.max(v_list)
#                 self.policy[s] = np.argmax(v_list)
#                 delta = max(delta, abs(v - self.values[s]))
#                 s = self.env.reset()
#             i += 1

#             value_hist.append(np.mean(self.values))
#         print(f"Converged in {i} iterations")


#         if plot:
#             plt.figure(dpi=200)
#             plt.plot(value_hist)
#             plt.tight_layout()
#             plt.savefig('figures/gridworld/value_iteration.png')



#         return self.policy
def main():
    env = gridworld.GridWorld(hard_version=False)
    policy_iter = PolicyIteration(env, gamma=0.95)
    policy_iter.train()   

    # value_iter = ValueIteration(env, gamma = 0.95)
    # value_iter.train()
    

if __name__ == '__main__':
    main()


