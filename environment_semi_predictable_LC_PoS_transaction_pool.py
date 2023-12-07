import numpy as np


class semi_predictable_LC_PoS_transaction_pool():
    def __init__(self, alpha, eta, n_node, n_array, state_1stPart_length, block_period_length, n_future_block):
        self.alpha = alpha
        self.n_total = n_node
        self.n_attacker = int(np.ceil(alpha * n_node))
        self.n_honest = self.n_total - self.n_attacker
        self.n_array = n_array
        self.state_1stPart_length = state_1stPart_length
        self.target = 1/(n_node*block_period_length)
        self.p_honest = (1 - alpha) / block_period_length
        self.p_attacker = alpha / block_period_length
        self.future_block = n_future_block
        self.fork_block = 30
        self.n_tnx_per_block = 100
        self.n_tnx_per_slot = int(1 * (self.n_tnx_per_block / block_period_length))
        self.tnx_mean_value = 1
        self.max_size_pool = 10 * self.n_tnx_per_block
        self.attacker_fork = np.array([(i + 1) / self.p_attacker for i in range(self.fork_block)])
        self.attacker_fork_length = 0
        self.honest_fork = np.array([(i + 1) / self.p_attacker for i in range(self.fork_block)])
        self.honest_fork_length = 0
        self.slot_number = 0
        self.attacker_publish_fork = np.array([(i + 1) / self.p_attacker for i in range(self.fork_block)])
        self.jump_fork = np.zeros(self.n_array, dtype=int)
        self.publish = 0
        self.latest = 0
        self.match = 0
        self.eta = eta
        self.reserved_reward = 0
        self.cost_per_slot = 0.1 * (self.n_tnx_per_block * self.tnx_mean_value)/(block_period_length/alpha)



    def reset(self):
        print(self.cost_per_slot)
        tnx_fee = np.random.exponential(self.tnx_mean_value, self.max_size_pool)
        tnx_block_attacker = np.zeros(self.max_size_pool)
        tnx_block_honest = np.zeros(self.max_size_pool)
        self.tnx_pool = list(zip(tnx_fee, tnx_block_attacker, tnx_block_honest))
        dtype = [('fee', float), ('blockA', int), ('blockH', int)]
        tnx_pool = np.array(self.tnx_pool, dtype)
        self.tnx_pool = np.sort(tnx_pool, order='fee')

        cntr = 0
        index = 0
        self.current_state_2ndPart_detail = np.zeros(self.future_block)
        while cntr < self.future_block:
            index += 1
            for i in range(self.n_attacker):
                if np.random.rand() < self.target:
                    self.current_state_2ndPart_detail[cntr] = index
                    cntr += 1
                    break

        attacker_win, honest_win, match_win, Nslot, self.current_state_2ndPart_detail = self.update_future_block(
                                                    self.current_state_2ndPart_detail, Match=0)
        self.current_state_1stPart = np.zeros(self.state_1stPart_length)
        # slot
        self.current_state_1stPart[self.state_1stPart_length-1] = 1
        if attacker_win == 1 and honest_win == 0:
            # a
            self.current_state_1stPart[0] = 1
            self.attacker_fork[0] = Nslot
        elif attacker_win == 0 and honest_win == 1:
            # h
            self.current_state_1stPart[1] = 1
            self.honest_fork[0] = Nslot
            # latest
            self.current_state_1stPart[2 + self.n_array + 2] = 1
        elif attacker_win == 1 and honest_win == 1:
            # a
            self.current_state_1stPart[0] = 1
            self.attacker_fork[0] = Nslot
            # h
            self.current_state_1stPart[1] = 1
            self.honest_fork[0] = Nslot
            # latest
            self.current_state_1stPart[2 + self.n_array + 2] = 1
        self.current_state_2ndPart = [
            self.p_attacker * (x - ((index + 1) / self.p_attacker)) / (np.sqrt((index + 1) * (1 - self.p_attacker)))
            for index, x in enumerate(self.current_state_2ndPart_detail)]
        self.current_state = np.concatenate((self.current_state_1stPart, self.current_state_2ndPart), 0)

        self.attacker_fork_length = self.current_state_1stPart[0]
        self.honest_fork_length = self.current_state_1stPart[1]
        self.publish = 0
        self.latest = self.current_state_1stPart[2 + self.n_array + 2]
        self.match = 0
        self.slot_number = Nslot

        return self.current_state

    def tnx_pool_import(self, N_slot):
        # a = []
        for i in range(self.n_tnx_per_slot*N_slot):
            new_fee = np.random.exponential(self.tnx_mean_value)
            dtype = [('fee', float), ('blockA', int), ('blockH', int)]
            new_tnx = np.array([(new_fee, 0, 0)], dtype)
            indx = np.searchsorted(self.tnx_pool['fee'], new_fee)
            self.tnx_pool = np.concatenate((self.tnx_pool[:indx], new_tnx, self.tnx_pool[indx:]))
            # a.append(new_fee)
        # print('$$$$$$$$$$$$$$$$$$$444', np.average(a))
        if len(self.tnx_pool) > self.max_size_pool:
            self.tnx_pool = self.tnx_pool[-self.max_size_pool:]


    def tnx_pool_export_honest(self, n_block):
        self.tnx_pool = self.tnx_pool[np.ceil(self.tnx_pool['blockH']/n_block) != 1]
        for indx in range(len(self.tnx_pool)):
            if self.tnx_pool[indx] ['blockH'] > n_block:
                self.tnx_pool[indx]['blockH'] = self.tnx_pool[indx]['blockH'] - n_block
            if self.tnx_pool[indx] ['blockA'] != 0:
                self.tnx_pool[indx]['blockA'] = 0

    def tnx_pool_export_attacker(self, n_publish, n_new):
        indx = 0
        while True:
            if 0 < self.tnx_pool[indx]['blockA'] <= n_publish:
                self.tnx_pool = np.delete(self.tnx_pool, indx)
            else:
                indx = indx + 1
            if indx == len(self.tnx_pool):
                break
        total_reward = self.reserved_reward
        if n_new > 0:
            total_reward = total_reward + np.sum(self.tnx_pool['fee'][-(min(len(self.tnx_pool), n_new * self.n_tnx_per_block)):])
        normalized_total_reward = total_reward
        self.tnx_pool = self.tnx_pool[0:-(min(len(self.tnx_pool), n_new * self.n_tnx_per_block))]

        for indx in range(len(self.tnx_pool)):
            if n_new == 0 and self.tnx_pool[indx]['blockH'] > n_publish:
                self.tnx_pool[indx]['blockH'] = self.tnx_pool[indx]['blockH'] - n_publish
            elif n_new != 0 and self.tnx_pool[indx]['blockH'] != 0:
                self.tnx_pool[indx]['blockH'] = 0
        return normalized_total_reward


    def tnx_pool_mark_honest(self, honest_block_number):
        indx = len(self.tnx_pool)-1
        counter = 0
        while counter < self.n_tnx_per_block and indx >= 0:
            if self.tnx_pool[indx]['blockH'] == 0:
                self.tnx_pool[indx]['blockH'] = honest_block_number
                counter = counter + 1
            indx = indx - 1

    def tnx_pool_mark_attacker(self, attacker_block_min_number, attacker_block_max_number):
        indx = len(self.tnx_pool)-1
        counter = 0
        attacker_block_number = attacker_block_max_number
        while counter < self.n_tnx_per_block * (attacker_block_max_number-attacker_block_min_number+1) and indx >= 0:
            if self.tnx_pool[indx]['blockA'] == 0:
                self.tnx_pool[indx]['blockA'] = attacker_block_number
                self.reserved_reward = self.reserved_reward + self.tnx_pool[indx]['fee']
                counter = counter + 1
                if counter % self.n_tnx_per_block == 0:
                    attacker_block_number = attacker_block_number - 1
            indx = indx - 1


    def step(self,action):
        self.previous_state = self.current_state
        self.previous_state_2ndPart = self.current_state_2ndPart
        self.previous_state_2ndPart_detail = self.current_state_2ndPart_detail
        self.previous_state_1stPart = self.current_state_1stPart

        if action % 3 == 1:
            Match = 1
        else:
            Match = 0
        attacker_win, honest_win, match_win, Nslot, self.current_state_2ndPart_detail = self.update_future_block(
            self.previous_state_2ndPart_detail, Match)
        self.tnx_pool_import(Nslot)
        if honest_win == 1:
            self.tnx_pool_mark_honest(self.honest_fork_length + 1)
        revenue_attacker = self.upcoming_state(action, attacker_win, honest_win, match_win, Nslot)
        cost_attacker = Nslot * self.cost_per_slot

        profit_attacker = revenue_attacker - cost_attacker

        self.current_state_1stPart = np.concatenate(([self.attacker_fork_length, self.honest_fork_length], self.jump_fork,
                                                     [self.publish, self.match, self.latest]), 0)
        self.current_state_2ndPart = [
            self.p_attacker * (x - ((index + 1) / self.p_attacker)) / (np.sqrt((index + 1) * (1 - self.p_attacker)))
            for index, x in enumerate(self.current_state_2ndPart_detail)]
        self.current_state = np.concatenate((self.current_state_1stPart, self.current_state_2ndPart), 0)
        return self.current_state, profit_attacker

    def upcoming_state(self, action, attacker_win, honest_win, match_win, Nslot):
        a = int(self.attacker_fork_length)
        a_fork = self.attacker_fork
        h = int(self.honest_fork_length)
        h_fork = self.honest_fork
        publish_fork = self.attacker_publish_fork
        jump_inf = self.jump_fork
        publish = int(self.publish)
        latest = self.latest
        match = self.match
        slot = self.slot_number
        n_array = self.n_array

        slot = slot + Nslot
        jump = int(np.floor(action / 3))
        if jump > 0:
            n_honest_block = h - (n_array - jump)
            self.tnx_pool_export_honest(n_honest_block)
            self.reserved_reward = 0
            reset_slot = h_fork[n_honest_block-1]
            # honest information update
            h_old = h
            h_new = h - n_honest_block
            h_fork[:h_new] = h_fork[h_old - h_new:h_old] - reset_slot
            h_fork[h_new:] = [(i+1)/self.p_honest for i in range(h_new, self.fork_block)]
            h = h_new
            # attacker information update
            a_old = a
            a_new = jump_inf[jump - 1]
            a_fork[:a_new] = a_fork[a_old-a_new:a_old] - reset_slot
            a_fork[a_new:] = [(i + 1) / self.p_attacker for i in range(a_new, self.fork_block)]
            a = a_new
            # publish information update
            publish = 0
            publish_fork = [(i + 1) / self.p_attacker for i in range(self.fork_block)]
            # jump information update
            jump_inf[:jump] = 0
            # slot information update
            slot = slot - reset_slot
        # overwrite
        if action % 3 == 0:
            reset_slot = a_fork[h]
            # attacker information update
            n_attacker_block_new = h + 1 - publish
            ra = self.tnx_pool_export_attacker(publish, n_attacker_block_new)
            self.reserved_reward = 0
            a_old = a
            a_new = a - (h+1)
            # print(a_new,a_old,a, h,action)
            a_fork[:a_new] = a_fork[a_old - a_new:a_old] - reset_slot
            a_fork[a_new:] = [(i + 1) / self.p_attacker for i in range(a_new, self.fork_block)]
            a = a_new
            # honest information update
            h_fork = [(i + 1) / self.p_honest for i in range(self.fork_block)]
            h = 0
            # jump information update
            jump_inf = np.zeros(n_array, dtype=int)
            # publish information update
            publish = 0
            publish_fork = [(i + 1) / self.p_attacker for i in range(self.fork_block)]
            # slot information update
            slot = slot - reset_slot
            if attacker_win == 1 and honest_win == 0:
                # attacker information update
                a = a + 1
                a_fork[a-1] = slot
                latest = 0
            elif attacker_win == 0 and honest_win == 1:
                # honest information update
                h = h + 1
                h_fork[h-1] = slot
                latest = 1
            elif attacker_win == 1 and honest_win == 1:
                # attacker information update
                a = a + 1
                a_fork[a - 1] = slot
                # honest information update
                h = h + 1
                h_fork[h - 1] = slot
                latest = 1
        # match
        elif action % 3 == 1:
            self.tnx_pool_mark_attacker(publish+1, h)
            # publish information update
            publish = h
            publish_fork[0:h] = [slot for i in range(h)]
            match = 1
            if attacker_win == 1 and honest_win == 0:
                # attacker information update
                a = a + 1
                a_fork[a - 1] = slot
                # jump information update
                n_active = min(n_array, h)
                jump_inf[n_array - n_active:] = jump_inf[n_array - n_active:] + 1
                latest = 0
                ra = 0
            elif attacker_win == 0 and honest_win == 1 and match_win == 1:
                reset_slot = a_fork[h-1]
                slot = slot - reset_slot
                # attacker information update
                ra = self.tnx_pool_export_attacker(publish, 0)
                self.reserved_reward = 0
                a_old = a
                a_new = a - h
                a_fork[:a_new] = a_fork[a_old - a_new:a_old] - reset_slot
                a_fork[a_new:] = [(i + 1) / self.p_attacker for i in range(a_new, self.fork_block)]
                a = a_new
                # honest information update
                h = 1
                h_fork = [(i + 1) / self.p_honest for i in range(self.fork_block)]
                h_fork[0] = slot
                # publish information update
                publish = 0
                publish_fork = [(i + 1) / self.p_attacker for i in range(self.fork_block)]
                # jump information update
                jump_inf = np.zeros(n_array, dtype=int)
                match = 0
                latest = 1
            elif attacker_win == 0 and honest_win == 1 and match_win == 0:
                # honest information update
                h = h + 1
                h_fork[h-1] = slot
                # jump information update
                jump_inf[0: -1] = jump_inf[1:]
                jump_inf[-1] = 0
                match = 0
                latest = 1
                ra = 0
            elif attacker_win == 1 and honest_win == 1 and match_win == 1:
                reset_slot = a_fork[h - 1]
                slot = slot - reset_slot
                # attacker information update
                ra = self.tnx_pool_export_attacker(publish, 0)
                self.reserved_reward = 0
                a_old = a
                a_new = a - h
                a_fork[:a_new] = a_fork[a_old - a_new:a_old] - reset_slot
                a_fork[a_new] = slot
                a_fork[a_new+1:] = [(i + 1) / self.p_attacker for i in range(a_new+1, self.fork_block)]
                a = a_new + 1
                # honest information update
                h = 1
                h_fork = [(i + 1) / self.p_honest for i in range(self.fork_block)]
                h_fork[0] = slot
                # publish information update
                publish = 0
                publish_fork = [(i + 1) / self.p_attacker for i in range(self.fork_block)]
                # jump information update
                jump_inf = np.zeros(n_array, dtype=int)
                match = 0
                latest = 1
            elif attacker_win == 1 and honest_win == 1 and match_win == 0:
                # attacker information update
                a = a + 1
                a_fork[a - 1] = slot
                # jump information update
                n_active = min(n_array, h)
                jump_inf[n_array - n_active:] = jump_inf[n_array - n_active:] + 1
                jump_inf[0: -1] = jump_inf[1:]
                jump_inf[-1] = 0
                # honest information update
                h = h + 1
                h_fork[h - 1] = slot
                match = 0
                latest = 1
                ra = 0

        # wait
        elif action % 3 == 2:
            if match != 1:
                if attacker_win == 1 and honest_win == 0:
                    # attacker information update
                    a = a + 1
                    a_fork[a - 1] = slot
                    # jump information update
                    n_active = min(n_array, h)
                    jump_inf[n_array - n_active:] = jump_inf[n_array - n_active:] + 1
                    latest = 0
                    ra = 0

                elif attacker_win == 0 and honest_win == 1:
                    # honest information update
                    h = h + 1
                    h_fork[h - 1] = slot
                    # jump information update
                    jump_inf[0: -1] = jump_inf[1:]
                    jump_inf[-1] = 0
                    latest = 1
                    ra = 0

                elif attacker_win == 1 and honest_win == 1:
                    # attacker information update
                    a = a + 1
                    a_fork[a - 1] = slot
                    # jump information update
                    n_active = min(n_array, h)
                    jump_inf[n_array - n_active:] = jump_inf[n_array - n_active:] + 1
                    jump_inf[0: -1] = jump_inf[1:]
                    jump_inf[-1] = 0
                    # honest information update
                    h = h + 1
                    h_fork[h - 1] = slot
                    latest = 1
                    ra = 0

            elif match == 1:
                if attacker_win == 1 and honest_win == 0:
                    # attacker information update
                    a = a + 1
                    a_fork[a - 1] = slot
                    # jump information update
                    n_active = min(n_array, h)
                    jump_inf[n_array - n_active:] = jump_inf[n_array - n_active:] + 1
                    latest = 0
                    ra = 0

                elif attacker_win == 0 and honest_win == 1 and match_win == 1:
                    reset_slot = a_fork[h - 1]
                    slot = slot - reset_slot
                    # attacker information update
                    ra = self.tnx_pool_export_attacker(publish, 0)
                    self.reserved_reward = 0
                    a_old = a
                    a_new = a - h
                    a_fork[:a_new] = a_fork[a_old - a_new:a_old] - reset_slot
                    a_fork[a_new:] = [(i + 1) / self.p_attacker for i in range(a_new, self.fork_block)]
                    a = a_new
                    # honest information update
                    h = 1
                    h_fork = [(i + 1) / self.p_honest for i in range(self.fork_block)]
                    h_fork[0] = slot
                    # publish information update
                    publish = 0
                    publish_fork = [(i + 1) / self.p_attacker for i in range(self.fork_block)]
                    # jump information update
                    jump_inf = np.zeros(n_array, dtype=int)
                    match = 0
                    latest = 1
                elif attacker_win == 0 and honest_win == 1 and match_win == 0:
                    # honest information update
                    h = h + 1
                    h_fork[h - 1] = slot
                    # jump information update
                    jump_inf[0: -1] = jump_inf[1:]
                    jump_inf[-1] = 0
                    match = 0
                    latest = 1
                    ra = 0
                elif attacker_win == 1 and honest_win == 1 and match_win == 1:
                    reset_slot = a_fork[h - 1]
                    slot = slot - reset_slot
                    # attacker information update
                    ra = self.tnx_pool_export_attacker(publish, 0)
                    self.reserved_reward = 0
                    a_old = a
                    a_new = a - h
                    a_fork[:a_new] = a_fork[a_old - a_new:a_old] - reset_slot
                    a_fork[a_new] = slot
                    a_fork[a_new + 1:] = [(i + 1) / self.p_attacker for i in range(a_new + 1, self.fork_block)]
                    a = a_new + 1
                    # honest information update
                    h = 1
                    h_fork = [(i + 1) / self.p_honest for i in range(self.fork_block)]
                    h_fork[0] = slot
                    # publish information update
                    publish = 0
                    publish_fork = [(i + 1) / self.p_attacker for i in range(self.fork_block)]
                    # jump information update
                    jump_inf = np.zeros(n_array, dtype=int)
                    match = 0
                    latest = 1
                elif attacker_win == 1 and honest_win == 1 and match_win == 0:
                    # attacker information update
                    a = a + 1
                    a_fork[a - 1] = slot
                    # jump information update
                    n_active = min(n_array, h)
                    jump_inf[n_array - n_active:] = jump_inf[n_array - n_active:] + 1
                    jump_inf[0: -1] = jump_inf[1:]
                    jump_inf[-1] = 0
                    # honest information update
                    h = h + 1
                    h_fork[h - 1] = slot
                    match = 0
                    latest = 1
                    ra = 0

        self.attacker_fork_length = a
        self.attacker_fork = a_fork
        self.honest_fork_length = h
        self.honest_fork = h_fork
        self.attacker_publish_fork = publish_fork
        self.jump_fork = jump_inf
        self.publish = publish
        self.latest = latest
        self.match = match
        self.slot_number = slot

        return ra

    def update_future_block(self, state_2ndPart, Match):
        index = 0
        attacker_win = 0
        honest_win = 0
        match_win = 0
        while index < state_2ndPart[0]:
            index += 1
            honest_win, match_win = self.run(Match)
            if honest_win == 1:
                break
        Nslot = index
        if state_2ndPart[0] == index:
            attacker_win = 1
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

        return attacker_win, honest_win, match_win, Nslot, state_2ndPart

    def run(self, Match):
        honest_win = 0
        match_win = 0
        for indx in range(self.n_honest):
            if np.random.rand() < self.target:
                honest_win = 1
                if Match == 1:
                    if indx < self.eta:
                        match_win = 1
                break
        return honest_win, match_win
