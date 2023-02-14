# HW1 - Dynamic Programming

## What to do

### Algorithms
Your goal is to implement five reinforcement learning algorithms in a "tabular" setting (i.e., assuming small finite state and action spaces).

Two algorithms are *model-based*:
- Policy iteration (Chapter 4.3, Sutton and Barto)
- Value iteration (Chapter 4.4, Sutton and Barto)

Two algorithms are *model-free*:
- SARSA with an epsilon-greedy policy, i.e., on-policy TD(0) to estimate Q (Chapter 6.4, Sutton and Barto)
- Q-learning with an epsilon-greedy policy, i.e., off-policy TD(0) to estimate Q (Chapter 6.5, Sutton and Barto)

One final algorithm, which is also model-free, computes the value function associated with a given policy:
- TD(0) (Chapter 6.1, Sutton and Barto)

### Environments

You will test your algorithms in the following provided environments:
- A simple grid-world, for which an explicit model is available (Example 3.5, Chapter 3.5, Sutton and Barto). The environment is defined in `gridworld.py`. An example of how to use the environment is provided in `test_gridworld.py`.
- A simple pendulum with discretized state and action spaces, for which an explicit model is not available (e.g., http://underactuated.mit.edu/pend.html). The environment is defined in `discrete_pendulum.py`. An example of how to use the environment is provided in `test_discrete_pendulum.py`.

For both environments, you should express the reinforcement learning problem as a Markov Decision Process with an infinite time horizon and a discount factor of $\gamma = 0.95$.

### Results

Please apply policy iteration, value iteration, SARSA, and Q-learning to the grid-world. Please apply TD(0) to learn the value function associated with the optimal policy produced by SARSA and Q-learning.

Please apply only SARSA and Q-learning to the pendulum. Again, please apply TD(0) to learn the value function associated with the optimal policy produced by SARSA and Q-learning.

More specifically, please write code to generate, at a minimum, the following results:
- A plot of the learning curve for each algorithm.
    - For policy iteration and value iteration, plot the mean of the value function $\frac{1}{25}\sum_{s = 1}^{25} V(s)$ versus the number of *value* iterations. For policy iteration, "the number of value iterations" is the number of iterations spent on policy evaluation (which dominates the computation time). For value iteration, "the number of value iterations" is synonymous with the number of iterations of the algorithm.
    - For SARSA and Q-learning, plot the return versus the number of episodes.
- A plot of learning curves for several different values of $\epsilon$ for SARSA and Q-learning.
- A plot of learning curves for several different values of $\alpha$ for SARSA and Q-learning.
- A plot of an example trajectory for each trained agent.
- A plot of the policy that corresponds to each trained agent.
- A plot of the state-value function that corresponds to each trained agent (learned by TD(0) for SARSA and Q-learning).

Include `train_gridworld.py` and `train_pendulum.py` files that can be run to generate all necessary plots and data.

## What to submit

### 1. Initial code

Create a [pull request (PR)](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) to enable a [review of your code](#2-code-review). Your code should be functional - that is, each algorithm should be working and ready to be reviewed for improvements. Remember that you need to write your own code from scratch.

Name your PR "Initial hwX for Firstname Lastname (netid)".

**Due: 10am on Tuesday, February 28**

### 2. Code review

Review the code of at least one colleague. That is, you should:
- Choose a PR that does not already have a reviewer, and [assign yourself as a reviewer]((https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/about-pull-request-reviews)).
- Perform a code review. See [Resources](#resources) for guidance on how to perform code reviews (at a minimum, look at Google's best practices).

The goal of this review process is to arrive at a version of your code that is functional, reasonably efficient, and easy for others to understand. The goal is *not* to make all of our code the same (there are many different ways of doing things). The goal is also *not* to grade the work of your colleagues - your reviews will have no impact on others' grades. Don't forget to remind your colleagues to do the simple things like name their PR and training file correctly!

**Due: 10am on Friday, March 3**

### 3. Final code and results

Improve your own code based on reviews that you receive. Respond to every comment. If you address a comment fully (e.g., by changing your code), you mark it as resolved. If you disagree with or remain uncertain about a comment, engage in follow-up discussion with the reviewer on GitHub. Don't forget to reply to follow-ups on code you reviewed as well.

Submit your repository, containing your final code and a (very brief) report titled `hwX-netid.pdf`, to [Gradescope](https://uiuc-ae598-rl-2023-spring.github.io/resources/assignments/). The report should be formatted using either typical IEEE or AIAA conference/journal paper format and include the following, at a minimum:
- Plots discussed in [Results](#results).
- A very brief discussion of the problem(s), any specific details needed to understand your algorithm implementations (e.g., hyperparameters chosen), and your results.

**Due: 10am on Tuesday, March 7**

## Resources
Here are some resources that may be helpful:
* Google's [best practices for code review](https://google.github.io/eng-practices/review/reviewer/looking-for.html)
* A Microsoft [blog post on code review](https://devblogs.microsoft.com/appcenter/how-the-visual-studio-mobile-center-team-does-code-review/) and [study of the review process](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/05/MS-Code-Review-Tech-Report-MSR-TR-2016-27.pdf)
* A RedHat [blog post on python-specific code review](https://access.redhat.com/blogs/766093/posts/2802001)
* A classic reference on writing code: [The Art of Readable Code (Boswell and Foucher, O'Reilly, 2012)](https://mcusoft.files.wordpress.com/2015/04/the-art-of-readable-code.pdf)

Many other resources are out there - we will happily accept a PR to add more to this list!
