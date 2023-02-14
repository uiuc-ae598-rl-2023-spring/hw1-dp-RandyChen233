import random

class GridWorld():
    """
    The world is a 5 x 5 grid based on Example 3.5 from Sutton 2019. There are 25 states. We index these states as follows:

        0   1   2   3   4
        5   6   7   8   9
        10  11  12  13  14
        15  16  17  18  19
        20  21  22  23  24

    For example, state "1" is cell "A" in Sutton 2019, state "3" is cell "B", and so forth.

    There are 4 actions. We index these actions as follows:

                1 (up)
        2 (left)        0 (right)
                3 (down)

    If you specify hard_version=True, then the action will be selected uniformly at random 10% of the time.
    """

    def __init__(self, hard_version=False):
        self.hard_version = hard_version
        self.num_states = 25
        self.num_actions = 4
        self.last_action = None
        self.max_num_steps = 100
        self.reset()

    def p(self, s1, s, a):
        if self.hard_version:
            return 0.1 * 0.25 * sum([self._p_easy(s1, s, i) for i in range(4)]) + 0.9 * self._p_easy(s1, s, a)
        else:
            return self._p_easy(s1, s, a)

    def _p_easy(self, s1, s, a):
        if s1 not in range(25):
            raise Exception(f'invalid next state: {s1}')
        if s not in range(25):
            raise Exception(f'invalid state: {s}')
        if a not in range(4):
            raise Exception(f'invalid action: {a}')
        # in A
        if s == 1:
            return 1 if s1 == 21 else 0
        # in B
        if s == 3:
            return 1 if s1 == 13 else 0
        # right
        if a == 0:
            if s in [4, 9, 14, 19, 24]:
                return 1 if s1 == s else 0
            else:
                return 1 if s1 == s + 1 else 0
        # up
        if a == 1:
            if s in [0, 1, 2, 3, 4]:
                return 1 if s1 == s else 0
            else:
                return 1 if s1 == s - 5 else 0
        # left
        if a == 2:
            if s in [0, 5, 10, 15, 20]:
                return 1 if s1 == s else 0
            else:
                return 1 if s1 == s - 1 else 0
        # down
        if a == 3:
            if s in [20, 21, 22, 23, 24]:
                return 1 if s1 == s else 0
            else:
                return 1 if s1 == s + 5 else 0

    def r(self, s, a):
        if self.hard_version:
            return 0.1 * 0.25 * sum([self._r_easy(s, i) for i in range(4)]) + 0.9 * self._r_easy(s, a)
        else:
            return self._r_easy(s, a)

    def _r_easy(self, s, a):
        if s not in range(25):
            raise Exception(f'invalid state: {s}')
        if a not in range(4):
            raise Exception(f'invalid action: {a}')
        # in A
        if s == 1:
            return 10
        # in B
        if s == 3:
            return 5
        # right
        if a == 0:
            if s in [4, 9, 14, 19, 24]:
                return -1
            else:
                return 0
        # up
        if a == 1:
            if s in [0, 1, 2, 3, 4]:
                return -1
            else:
                return 0
        # left
        if a == 2:
            if s in [0, 5, 10, 15, 20]:
                return -1
            else:
                return 0
        # down
        if a == 3:
            if s in [20, 21, 22, 23, 24]:
                return -1
            else:
                return 0

    def step(self, a):
        # Store the action (only used for rendering)
        self.last_action = a

        # If this is the hard version of GridWorld, then change the action to
        # one chosen uniformly at random 10% of the time
        if self.hard_version:
            if random.random() < 0.1:
                a = random.randrange(self.num_actions)

        # Compute the next state and reward
        if self.s == 1:
            # We are in the first teleporting state
            self.s = 21
            r = 10
        elif self.s == 3:
            # We are in the second teleporting state
            self.s = 13
            r = 5
        else:
            # We are in neither teleporting state

            # Convert the state to i, j coordinates
            i = self.s // 5
            j = self.s % 5

            # Apply action to i, j coordinates
            if a == 0:      # right
                j += 1
            elif a == 1:    # up
                i -= 1
            elif a == 2:    # left
                j -= 1
            elif a == 3:    # down
                i += 1
            else:
                raise Exception(f'invalid action: {a}')

            # Would the action move us out of bounds?
            if i < 0 or i >= 5 or j < 0 or j >= 5:
                # Yes - state remains the same, reward is negative
                r = -1
            else:
                # No - state changes (convert i, j coordinates back to number), reward is zero
                self.s = i * 5 + j
                r = 0

        # Increment number of steps and check for end of episode
        self.num_steps += 1
        done = (self.num_steps >= self.max_num_steps)

        return (self.s, r, done)

    def reset(self):
        # Choose initial state uniformly at random
        self.s = random.randrange(self.num_states)
        self.num_steps = 0
        self.last_action = None
        return self.s

    def render(self):
        k = 0
        output = ''
        for i in range(5):
            for j in range(5):
                if k == self.s:
                    output += 'X'
                elif k == 1 or k == 3:
                    output += 'o'
                else:
                    output += '.'
                k += 1
            output += '\n'
        if self.last_action is not None:
            print(['right', 'up', 'left', 'down'][self.last_action])
            print()
        print(output)
