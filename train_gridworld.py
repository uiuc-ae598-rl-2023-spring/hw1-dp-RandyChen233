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
        self.values = np.zeros(self.num_states) #Initialize `values` as zeros
        self.policy = np.random.randint(0, self.num_actions, self.num_states)

    def one_policy_evaluation(self):
        """
        Runs one iteration of policy evaluation and updates the value function.

        :return: the maximum change in value function
        """
        delta = 0
        for s in range(self.num_states):
            v = self.values[s]
            a = self.policy[s]
            (s_new, r, _) = self.env.step(a)
            p = self.env.p(s_new, s, a)

            """  update V(s): V(s) <- r(s) + gamma * SUM(p(s, s0, a) * V(s')) """
            self.values[s] = np.sum(p * (r + self.gamma * self.values[s_new]))
            delta = max(delta, abs(v - self.values[s]))

        return delta

    def run_policy_evaluation(self, tol = 1e-3):
        """
        Runs policy evaluation until convergence.

        :param tol: the tolerance level for convergence
        :return: the number of iterations of policy evaluation until convergence
        """
        delta = self.one_policy_evaluation()
        delta_history = [delta]

        while delta > tol:
            delta = self.one_policy_evaluation()
            delta_history.append(delta)

        return len(delta_history)

    def run_policy_improvement(self):
        update_policy_count = 0

        for s in range(self.num_states):
            temp = self.policy[s]
            v_list = np.zeros(self.num_actions)
            for a in range(self.num_actions):
                (s_new, r, _) = self.env.step(a)
                p = self.env.p(s_new, s, a)
                v_list[a] = np.sum(p * (r + self.gamma * self.values[s_new]))

            self.policy[s] = np.argmax(v_list)

            if temp != self.policy[s]:
                update_policy_count += 1

        return update_policy_count

    def train(self, tol=1e-3, max_iters=100, plot=True):
        eval_count = self.run_policy_evaluation(tol)
        eval_count_history = [eval_count]
        policy_change = self.run_policy_improvement()
        policy_change_history = [policy_change]

        epoch = 0
        val_history= []

        for i in tqdm(range(max_iters)):
            epoch += 1
            new_eval_count = self.run_policy_evaluation(tol)
            new_policy_change = self.run_policy_improvement()

            eval_count_history.append(new_eval_count)
            policy_change_history.append(new_policy_change)

            val_history.append(np.mean(self.values))

            if new_policy_change == 0:
                break

        print(f'# epoch: {len(policy_change_history)}')
        print(f'eval count = {eval_count_history}')
        print(f'policy change = {policy_change_history}')

        if plot is True:
            plt.figure(dpi=200)
            plt.plot(val_history)
            plt.tight_layout()
            plt.savefig('policy_iteration.png')
            plt.show()

        
def main():
    env = gridworld.GridWorld(hard_version=False)
    agent = PolicyIteration(env, gamma=0.95)
    agent.train()   
    

if __name__ == '__main__':
    main()


