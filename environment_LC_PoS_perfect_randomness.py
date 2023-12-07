import numpy as np

class LC_PoS_perfect_randomness():
    def __init__(self, alpha, eta, n_array, block_period_length):
        self.alpha = alpha
        self.eta = eta
        self.n_array = n_array
        self.state_1stPart_length = 2 * (2 * n_array + 1) + 2
        self.p_honest = (1 - alpha) / block_period_length
        self.p_attacker = alpha / block_period_length
        self.state_length = 2 + self.n_array + 3

    def reset(self):
        self.current_state = np.zeros(self.state_length)
        self.previous_state = np.zeros(self.state_length)
        attacker_win, honest_win, match_win = self.run()
        if attacker_win == 1 and honest_win == 0:
            # a
            self.current_state[0] = 1
        elif attacker_win == 0 and honest_win == 1:
            # h
            self.current_state[1] = 1
            # latest
            self.current_state[2 + self.n_array + 2] = 1
        elif attacker_win == 1 and honest_win == 1:
            # a
            self.current_state[0] = 1
            # h
            self.current_state[1] = 1
            # latest
            self.current_state[2 + self.n_array + 2] = 1
        return self.current_state


    def step(self,action):
        self.previous_state = self.current_state
        attacker_win, honest_win, match_win = self.run()
        self.current_state, reward_attacker, reward_honest = self.upcoming_state(self.previous_state,
                                                                                action, self.n_array,
                                                                                attacker_win, honest_win, match_win)
        return self.current_state,  reward_attacker, reward_honest

    def run(self):
        attacker_win = 0
        honest_win = 0
        match_win = 0
        while honest_win == 0 and attacker_win == 0:
            if np.random.rand() <= self.p_attacker:
                attacker_win = 1
            if np.random.rand() <= self.p_honest:
                honest_win = 1
                if np.random.rand() <= self.eta:
                    match_win = 1
        return attacker_win, honest_win, match_win
    def upcoming_state(self, state, action, n_array, attacker_win, honest_win, match_win):
        state = np.array([int(state[i]) for i in range(len(state))])
        a = state[0]
        h = state[1]
        a_future = state[2:2+n_array]
        publish = state[2 + n_array]
        match = state[2 + n_array + 1]
        latest = state[2 + n_array + 2]

        jump = int(np.floor(action / 3))

        if jump > 0:
            rh2 = h - (n_array-jump)
            a = a_future[jump - 1]
            a_future[:jump] = 0
            h = h - rh2
            publish = 0
        elif jump == 0:
            rh2 = 0
        # overwrite
        if action % 3 == 0:
            a_future = np.zeros(n_array, dtype=int)
            publish = 0
            if attacker_win == 1 and honest_win == 0:
                ra = h + 1
                rh = 0
                a = a - h
                h = 0
                latest = 0
            elif attacker_win == 0 and honest_win == 1:
                ra = h + 1
                rh = 0
                a = a - h - 1
                h = 1
                latest = 1
            elif attacker_win == 1 and honest_win == 1:
                ra = h + 1
                rh = 0
                a = a - h
                h = 1
                latest = 1
        # match
        elif action % 3 == 1:
            if attacker_win == 1 and honest_win == 0:
                publish = h
                a = a + 1
                match = 1
                latest = 0
                n_active = min(n_array, h)
                a_future[n_array - n_active:] = a_future[n_array - n_active:] + 1
                ra = 0
                rh = 0
            elif attacker_win == 0 and honest_win == 1 and match_win == 1:
                ra = h
                rh = 0
                publish = 0
                a = a-h
                h = 1
                match = 0
                latest = 1
                a_future = np.zeros(n_array, dtype=int)
            elif attacker_win == 0 and honest_win == 1 and match_win == 0:
                publish = h
                h = h+1
                match = 0
                latest = 1
                a_future[0:n_array-1] = a_future[1:]
                a_future[n_array-1] = 0
                ra = 0
                rh = 0
            elif attacker_win == 1 and honest_win == 1 and match_win == 1:
                ra = h
                rh = 0
                publish = 0
                a = a - h + 1
                h = 1
                match = 0
                latest = 1
                a_future = np.zeros(n_array, dtype=int)
            elif attacker_win == 1 and honest_win == 1 and match_win == 0:
                publish = h
                h = h + 1
                a = a + 1
                match = 0
                latest = 1
                a_future[0:n_array - 1] = a_future[1:] + 1
                a_future[n_array - 1] = 0
                ra = 0
                rh = 0

        # wait
        elif action % 3 == 2:
            if match != 1:
                if attacker_win == 1 and honest_win == 0:
                    a = a + 1
                    latest = 0
                    n_active = min(n_array, h)
                    a_future[n_array - n_active:] = a_future[n_array - n_active:] + 1
                    ra = 0
                    rh = 0
                    ra = 0
                    rh = 0
                elif attacker_win == 0 and honest_win == 1:
                    h = h + 1
                    latest = 1
                    a_future[0:n_array - 1] = a_future[1:]
                    a_future[n_array - 1] = 0
                    ra = 0
                    rh = 0
                elif attacker_win == 1 and honest_win == 1:
                    a = a + 1
                    h = h + 1
                    latest = 1
                    a_future[0:n_array - 1] = a_future[1:] + 1
                    a_future[n_array - 1] = 0
                    ra = 0
                    rh = 0
            elif match == 1:
                if attacker_win == 1 and honest_win == 0:
                    a = a + 1
                    latest = 0
                    match = 1
                    n_active = min(n_array, h)
                    a_future[n_array - n_active:] = a_future[n_array - n_active:] + 1
                    ra = 0
                    rh = 0
                elif attacker_win == 0 and honest_win == 1 and match_win == 1:
                    ra = h
                    rh = 0
                    publish = 0
                    a = a - h
                    h = 1
                    match = 0
                    latest = 1
                    a_future = np.zeros(n_array, dtype=int)
                elif attacker_win == 0 and honest_win == 1 and match_win == 0:
                    h = h + 1
                    match = 0
                    latest = 1
                    delta = a - publish
                    a_future[0:n_array - 1] = a_future[1:]
                    a_future[n_array - 1] = 0
                    ra = 0
                    rh = 0
                elif attacker_win == 1 and honest_win == 1 and match_win == 1:
                    ra = h
                    rh = 0
                    publish = 0
                    a = a - h + 1
                    h = 1
                    match = 0
                    latest = 1
                    a_future = np.zeros(n_array, dtype=int)
                elif attacker_win == 1 and honest_win == 1 and match_win == 0:
                    a = a + 1
                    h = h + 1
                    match = 0
                    latest = 1
                    a_future[0:n_array - 1] = a_future[1:] + 1
                    a_future[n_array - 1] = 0
                    ra = 0
                    rh = 0

        state_1 = []
        for i in range(n_array):
            state_1 = np.concatenate((state_1, [a_future[i]]), 0)
        state_ = np.concatenate(([a, h], state_1, [publish, match, latest]), 0)
        return state_, ra, rh+rh2