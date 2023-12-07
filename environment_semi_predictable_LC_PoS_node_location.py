import numpy as np
import random
import math
import matplotlib.pyplot as plt

class semi_predictable_LC_PoS_node_location():
    def __init__(self, alpha, n_node, n_array, state_1stPart_length, block_period_length, n_future_block, test):
        self.alpha = alpha
        self.n_total = n_node
        self.n_attacker = int(np.ceil(alpha * n_node))
        self.n_honest = self.n_total - self.n_attacker
        self.n_array = n_array
        self.target = 1 / (n_node * block_period_length)
        self.p_honest = (1 - alpha) / block_period_length
        self.p_attacker = alpha / block_period_length
        self.state_1stPart_length = state_1stPart_length
        self.state_length = state_1stPart_length + n_future_block
        self.N_future_block = n_future_block
        self.test = test
        self.circle_r = 0.5
        self.normal_mean = 1
        self.normal_var = 1/25



    def reset(self):
        #print('*****************************************',self.ntH_A)
        self.node_distribution2(self.circle_r, self.test)
        cntr = 0
        index = 0
        self.current_state_2ndPart_detail = np.zeros(self.N_future_block)
        while cntr < self.N_future_block:
            index += 1
            for i in range(self.n_attacker):
                if np.random.rand() < self.target:
                    self.current_state_2ndPart_detail[cntr] = index
                    cntr += 1
                    break

        attacker_win, honest_win, match_win, self.current_state_2ndPart_detail = self.update(
            self.current_state_2ndPart_detail, Match=0)

        self.current_state_1stPart = np.zeros(self.state_1stPart_length)
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

        self.current_state_2ndPart = [
            self.p_attacker * (x - ((index + 1) / self.p_attacker)) / (np.sqrt((index + 1) * (1 - self.p_attacker)))
            for index, x in enumerate(self.current_state_2ndPart_detail)]
        self.current_state = np.concatenate((self.current_state_1stPart, self.current_state_2ndPart), 0)
        return self.current_state

    def step(self, action):
        self.previous_state = self.current_state
        self.previous_state_2ndPart = self.current_state_2ndPart
        self.previous_state_2ndPart_detail = self.current_state_2ndPart_detail
        self.previous_state_1stPart = self.current_state_1stPart

        if action % 3 == 1:
            Match = 1
        else:
            Match = 0

        attacker_win, honest_win, match_win, self.current_state_2ndPart_detail = self.update(
            self.previous_state_2ndPart_detail, Match)
        self.current_state_1stPart, reward_attacker, reward_honest = self.upcoming_state(self.previous_state_1stPart,
                                                                                         action, self.n_array,
                                                                                         attacker_win, honest_win,
                                                                                         match_win)
        self.current_state_2ndPart = [
            self.p_attacker * (x - ((index + 1) / self.p_attacker)) / (np.sqrt((index + 1) * (1 - self.p_attacker)))
            for index, x in enumerate(self.current_state_2ndPart_detail)]
        self.current_state = np.concatenate((self.current_state_1stPart, self.current_state_2ndPart), 0)
        return self.current_state, reward_attacker, reward_honest

    def update(self, state_2ndPart, Match):
        index = 0
        attacker_win = 0
        honest_win = 0
        match_win = 0
        while index < state_2ndPart[0]:
            index += 1
            honest_win, match_win = self.run(Match)
            if honest_win == 1:
                break
        if index == state_2ndPart[0]:
            attacker_win = 1
        if attacker_win == 1:
            done = 0
            index2 = 0
            while done == 0:
                index2 += 1
                if np.random.rand() <= self.p_attacker:
                    new_index = index2
                    done = 1
            state_2ndPart = [x - state_2ndPart[0] for x in state_2ndPart]
            state_2ndPart = np.roll(state_2ndPart, -1)
            state_2ndPart[-1] = state_2ndPart[-2] + new_index

        else:
            state_2ndPart = [x - index for x in state_2ndPart]

        return attacker_win, honest_win, match_win, state_2ndPart

    def run(self, Match):
        honest_win = 0
        match_win = 0
        for i in range(self.n_honest):
            if np.random.rand() < self.target:
                honest_win = 1
                if Match == 1:
                    dist_HH = max(np.random.normal(self.normal_mean * self.distance_HH[self.previous_honest_proposer, i], self.normal_var), 0)
                    dist_HA = np.zeros(self.n_attacker)
                    for j in range(self.n_attacker):
                        dist_HA[j] = max(np.random.normal(self.normal_mean * self.distance_HA[
                            self.previous_honest_proposer, j], self.normal_var), 0) \
                                     + max(np.random.normal(self.normal_mean * self.distance_HA[i, j], self.normal_var), 0)
                    if dist_HH >= min(dist_HA):
                        match_win = 1
                self.previous_honest_proposer = i
                break
        return honest_win, match_win


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

        state_ = np.concatenate(([a, h], a_future, [publish, match, latest]), 0)
        return state_, ra, rh+rh2

    def node_distribution2(self, circle_r, test):
        if test == False:
            self.X_attacker = np.zeros(self.n_attacker)
            self.Y_attacker = np.zeros(self.n_attacker)
            self.X_honest = np.zeros(self.n_honest)
            self.Y_honest = np.zeros(self.n_honest)
            for i in range(self.n_attacker):
                alpha = 2 * math.pi * random.random()
                r = circle_r * math.sqrt(random.random())
                x = r * math.cos(alpha)
                y = r * math.sin(alpha)
                self.X_attacker[i] = x
                self.Y_attacker[i] = y
            for i in range(self.n_honest):
                alpha = 2 * math.pi * random.random()
                r = circle_r * math.sqrt(random.random())
                x = r * math.cos(alpha)
                y = r * math.sin(alpha)
                self.X_honest[i] = x
                self.Y_honest[i] = y
            np.save('X_attacker', self.X_attacker)
            np.save('Y_attacker', self.Y_attacker)
            np.save('X_honest', self.X_honest)
            np.save('Y_honest', self.Y_honest)
            plt.scatter(self.X_attacker, self.Y_attacker)
            plt.scatter(self.X_honest, self.Y_honest)
            plt.show()
        else:
            self.X_attacker = np.load('X_attacker.npy')
            self.Y_attacker = np.load('Y_attacker.npy')
            self.X_honest = np.load('X_honest.npy')
            self.Y_honest = np.load('Y_honest.npy')
            plt.scatter(self.X_attacker, self.Y_attacker)
            plt.scatter(self.X_honest, self.Y_honest)
            plt.savefig('pic_cen.jpg', dpi=300)
            plt.show()

        self.distance_HH = np.zeros((self.n_honest, self.n_honest))
        for i in range(self.n_honest):
            for j in range(i, self.n_honest):
                dist = np.sqrt((self.X_honest[i]-self.X_honest[j])**2 + (self.Y_honest[i]-self.Y_honest[j])**2)
                self.distance_HH[i, j] = dist
                self.distance_HH[j, i] = dist
        self.distance_HA = np.zeros((self.n_honest, self.n_attacker))
        for i in range(self.n_honest):
            for j in range(self.n_attacker):
                dist = np.sqrt((self.X_honest[i]-self.X_attacker[j])**2 + (self.Y_honest[i]-self.Y_attacker[j])**2)
                self.distance_HA[i, j] = dist
        print(self.distance_HH)
        print("#####")
        print(self.distance_HA)