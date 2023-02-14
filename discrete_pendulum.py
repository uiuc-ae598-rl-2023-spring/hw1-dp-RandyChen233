import random
import numpy as np
import scipy.integrate


class Pendulum():
    def __init__(self, n_theta=31, n_thetadot=31, n_tau=31):
        # Parameters that describe the physical system
        self.params = {
            'm': 1.0,   # mass
            'g': 9.8,   # acceleration of gravity
            'l': 1.0,   # length
            'b': 0.1,   # coefficient of viscous friction
        }

        # Maximum absolute angular velocity
        self.max_thetadot = 15.0

        # Maximum absolute angle to be considered "upright"
        self.max_theta_for_upright = 0.1 * np.pi

        # Maximum absolute angular velocity from which to sample initial condition
        self.max_thetadot_for_init = 5.0

        # Maximum absolute torque
        self.max_tau = 5.0

        # Time step
        self.dt = 0.1

        # Number of grid points in each dimension (should be odd, so that there
        # is always one grid point at "0")
        self.n_theta = n_theta
        self.n_thetadot = n_thetadot
        self.n_tau = n_tau

        # Number of finite states and actions after discretization
        self.num_states = self.n_theta * self.n_thetadot
        self.num_actions = self.n_tau

        # Time horizon
        self.max_num_steps = 100

        # Reset to initial conditions
        self.reset()

    def _x_to_s(self, x):
        # Get theta - wrapping to [-pi, pi) - and thetadot
        theta = ((x[0] + np.pi) % (2 * np.pi)) - np.pi
        thetadot = x[1]
        # Convert to i, j coordinates
        i = (self.n_theta * (theta + np.pi)) // (2 * np.pi)
        j = (self.n_thetadot * (thetadot + self.max_thetadot)) // (2 * self.max_thetadot)
        # Clamp i, j coordinates
        i = max(0, min(self.n_theta - 1, i))
        j = max(0, min(self.n_thetadot - 1, j))
        # Convert to state
        return int(i * self.n_thetadot + j)

    def _a_to_u(self, a):
        return -self.max_tau + ((2 * self.max_tau * a) / (self.n_tau - 1))

    def _dxdt(self, x, u):
        theta_ddot =  (u - self.params['b'] * x[1] + self.params['m'] * self.params['g'] * self.params['l'] * np.sin(x[0])) / (self.params['m'] * self.params['l']**2)
        return np.array([x[1], theta_ddot])

    def step(self, a):
        # Verify action is in range
        if not (a in range(self.num_actions)):
            raise ValueError(f'invalid action {a}')

        # Convert a to u
        u = self._a_to_u(a)

        # Solve ODEs to find new x
        sol = scipy.integrate.solve_ivp(fun=lambda t, x: self._dxdt(x, u), t_span=[0, self.dt], y0=self.x, t_eval=[self.dt])
        self.x = sol.y[:, 0]

        # Convert x to s
        self.s = self._x_to_s(self.x)

        # Get theta - wrapping to [-pi, pi) - and thetadot
        theta = ((self.x[0] + np.pi) % (2 * np.pi)) - np.pi
        thetadot = self.x[1]

        # Compute reward
        if abs(thetadot) > self.max_thetadot:
            # If constraints are violated, then return large negative reward
            r = -100
        elif abs(theta) < self.max_theta_for_upright:
            # If pendulum is upright, then return small positive reward
            r = 1
        else:
            # Otherwise, return zero reward
            r = 0

        # Increment number of steps and check for end of episode
        self.num_steps += 1
        self.t = self.num_steps * self.dt
        done = (self.num_steps >= self.max_num_steps)

        return (self.s, r, done)

    def reset(self):
        # Sample theta and thetadot
        self.x = np.random.uniform([-np.pi, -self.max_thetadot_for_init], [np.pi, self.max_thetadot_for_init])

        # Convert to finite state
        self.s = self._x_to_s(self.x)

        # Reset current time (expressed as number of simulation steps taken so far) to zero
        self.num_steps = 0
        self.t = self.num_steps * self.dt

        return self.s

    def render(self):
        # TODO (we will happily accept a PR to create a graphic visualization of the pendulum)
        pass
