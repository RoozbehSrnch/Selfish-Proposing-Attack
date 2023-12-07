import numpy as np

class full_predictable_LC_PoS():
    def __init__(self, alpha, eta, n_array, state_1stPart_length, n_future_block):
        self.alpha = alpha
        self.eta = eta
        self.n_array = n_array
        self.average = 2 * self.alpha - 1
        self.variance = np.sqrt(4 * self.alpha * (1 - self.alpha))
        self.state_1stPart_length = state_1stPart_length
        self.state_length = state_1stPart_length + n_future_block
        self.N_future_block = n_future_block


    def reset(self):
        self.current_state_1stPart = np.zeros(self.state_1stPart_length)
        attacker_win = 0
        honest_win = 0
        if np.random.rand() < self.alpha:
            attacker_win = 1
        else:
            honest_win = 1

        if attacker_win == 1 and honest_win == 0:
            # a
            self.current_state_1stPart[0] = 1
        elif attacker_win == 0 and honest_win == 1:
            # h
            self.current_state_1stPart[1] = 1
            # latest
            self.current_state_1stPart[2 + self.n_array + 2] = 1
        elif attacker_win == 1 and honest_win == 1:
            # a
            self.current_state_1stPart[0] = 1
            # h
            self.current_state_1stPart[1] = 1
            # latest
            self.current_state_1stPart[2 + self.n_array + 2] = 1

        self.current_seq_detail = np.zeros(self.N_future_block)
        self.current_seq_detail_height = np.zeros(self.N_future_block)
        if np.random.rand() < self.alpha:
            self.current_seq_detail[0] = 1
            self.current_seq_detail_height[0] = 1
        else:
            self.current_seq_detail[0] = 0
            self.current_seq_detail_height[0] = -1

        for i in range(1, self.N_future_block):
            if np.random.rand() < self.alpha:
                self.current_seq_detail[i] = 1
                self.current_seq_detail_height[i] = self.current_seq_detail_height[i - 1] + 1
            else:
                self.current_seq_detail_height[i] = self.current_seq_detail_height[i - 1] - 1

        self.current_state_2ndPart = [
            (self.current_seq_detail_height[i] - ((i + 1) * self.average)) / (
                        np.sqrt((i + 1)) * self.variance)
            for i in range(self.N_future_block)]
        self.current_state = np.concatenate((self.current_state_1stPart, self.current_state_2ndPart), 0)
        return self.current_state

    def step(self, action):
        self.previous_state = self.current_state
        self.previous_state_2ndPart = self.current_state_2ndPart
        self.previous_seq_detail = self.current_seq_detail
        self.previous_seq_detail_height = self.current_seq_detail_height
        self.previous_state_1stPart = self.current_state_1stPart

        if action % 3 == 1:
            Match = 1
        else:
            Match = 0

        attacker_win, honest_win, match_win = self.update(Match)
        
        self.current_state_1stPart, reward_attacker, reward_honest = self.upcoming_state(self.previous_state_1stPart,
                                                                                         action, self.n_array,
                                                                                         attacker_win, honest_win,
                                                                                         match_win)
        self.current_state = np.concatenate((self.current_state_1stPart, self.current_state_2ndPart), 0)
        return self.current_state, reward_attacker, reward_honest


    def update(self, Match):
        attacker_win = 0
        honest_win = 0
        match_win = 0
        if self.previous_seq_detail[0] == 1:
            attacker_win = 1
        else:
            honest_win = 1
            if Match == 1 and np.random.rand() <= self.eta:
                match_win = 1

        self.current_seq_detail[:-1] = self.previous_seq_detail[1:]
        self.current_seq_detail_height[:-1] = np.subtract(self.previous_seq_detail_height[1:],
                                                          self.previous_seq_detail_height[0])

        if np.random.rand() < self.alpha:
            self.current_seq_detail[-1] = 1
            self.current_seq_detail_height[-1] = self.current_seq_detail_height[-2] + 1
        else:
            self.current_seq_detail[-1] = 0
            self.current_seq_detail_height[-1] = self.current_seq_detail_height[-2] - 1

        self.current_state_2ndPart = [
            (self.current_seq_detail_height[i] - ((i + 1) * self.average)) / (
                    np.sqrt((i + 1)) * self.variance)
            for i in range(self.N_future_block)]
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
        # override
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

        state_ = np.concatenate(([a, h], a_future, [publish, match, latest]), 0)
        return state_, ra, rh+rh2