import random
import numpy as np
import matplotlib.pyplot as plt
import discrete_pendulum

def test_x_to_s(env):
    theta = np.linspace(-np.pi * (1 - (1 / env.n_theta)), np.pi * (1 - (1 / env.n_theta)), env.n_theta)
    thetadot = np.linspace(-env.max_thetadot * (1 - (1 / env.n_thetadot)), env.max_thetadot * (1 - (1 / env.n_thetadot)), env.n_thetadot)
    for s in range(env.num_states):
        i = s // env.n_thetadot
        j = s % env.n_thetadot
        s1 = env._x_to_s([theta[i], thetadot[j]])
        if s1 != s:
            raise Exception(f'test_x_to_s: error in state representation: {s} and {s1} should be the same')
    print('test_x_to_s: passed')


def main():
    # Create environment
    #
    #   By default, both the state space (theta, thetadot) and the action space
    #   (tau) are discretized with 31 grid points in each dimension, for a total
    #   of 31 x 31 states and 31 actions.
    #
    #   You can change the number of grid points as follows (for example):
    #
    #       env = discrete_pendulum.Pendulum(n_theta=11, n_thetadot=51, n_tau=21)
    #
    #   Note that there will only be a grid point at "0" along a given dimension
    #   if the number of grid points in that dimension is odd.
    #
    #   How does performance vary with the number of grid points? What about
    #   computation time?
    env = discrete_pendulum.Pendulum(n_theta=15, n_thetadot=21)

    # Apply unit test to check state representation
    test_x_to_s(env)

    # Initialize simulation
    s = env.reset()

    # Create log to store data from simulation
    log = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
        'theta': [env.x[0]],        # agent does not have access to this, but helpful for display
        'thetadot': [env.x[1]],     # agent does not have access to this, but helpful for display
    }

    # Simulate until episode is done
    done = False
    while not done:
        a = random.randrange(env.num_actions)
        (s, r, done) = env.step(a)
        log['t'].append(log['t'][-1] + 1)
        log['s'].append(s)
        log['a'].append(a)
        log['r'].append(r)
        log['theta'].append(env.x[0])
        log['thetadot'].append(env.x[1])

    # Plot data and save to png file
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(log['t'], log['s'])
    ax[0].plot(log['t'][:-1], log['a'])
    ax[0].plot(log['t'][:-1], log['r'])
    ax[0].legend(['s', 'a', 'r'])
    ax[1].plot(log['t'], log['theta'])
    ax[1].plot(log['t'], log['thetadot'])
    ax[1].legend(['theta', 'thetadot'])
    plt.savefig('figures/pendulum/test_discrete_pendulum.png')


if __name__ == '__main__':
    main()
