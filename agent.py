import numpy as np
import torch as T
from deep_q_network import DeepQNetwork
from experience_replay import ReplayBuffer


class Agent():
    def __init__(self, gamma, epsilon, lr, n_actions, n_array, input_dims,
                 mem_size, batch_size, eps_min, eps_dec, replace,
                 algo=None, env_name=None):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.n_array = n_array
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.bound = 10
        self.learn_step_counter = 0
        self.memory = ReplayBuffer(input_dims, mem_size, batch_size, n_actions)
        self.q_eval_attacker = DeepQNetwork(self.lr, self.n_actions, self.input_dims)
        self.q_next_attacker = DeepQNetwork(self.lr, self.n_actions, self.input_dims)
        self.q_eval_honest = DeepQNetwork(self.lr, self.n_actions, self.input_dims)
        self.q_next_honest = DeepQNetwork(self.lr, self.n_actions, self.input_dims)
        self.device = T.device("cuda" if False else "cpu")
    def choose_action(self, state):
        possible_action_set = self.possible_actions(state, self.n_array)
        possible_action_set_index = [i for i, e in enumerate(possible_action_set) if e == 1]
        possible_action_set = T.tensor(possible_action_set, dtype=T.float32).to(self.device)
        if np.random.random() > self.epsilon:
            state_attacker = T.tensor(state, dtype=T.float32).to(self.device)
            actions_attacker = self.q_eval_attacker.forward(state_attacker)
            state_honest = T.tensor(state, dtype=T.float32).to(self.device)
            actions_honest = self.q_eval_honest.forward(state_honest)
            actions_total = actions_attacker/(actions_honest + actions_attacker)
            actions_total = T.mul(actions_total, possible_action_set)
            actions_total = T.nan_to_num(actions_total, nan=float('-inf'))
            action = T.argmax(actions_total).item()

        else:
            action = np.random.choice(possible_action_set_index)
        return action

    def possible_actions(self, state, n_array):
        a = state[0]
        h = state[1]
        a_future = state[2:2 + n_array]
        publish = state[2 + n_array]
        match = state[2 + n_array + 1]
        latest = state[2 + n_array + 2]
        possibleActionSet = np.zeros(3 * (n_array + 1))
        possibleActionSet[0:] = float('nan')
        if (h == 1 and a == 0):
            possibleActionSet[3 * n_array + 2] = 1
            return possibleActionSet
        if match != 1:
            for i in range(n_array):
                if a - a_future[i] >= publish and h >= n_array-i:
                    h_future = n_array - i - 1
                    if a_future[i] == h_future + 1:
                        possibleActionSet[3 * (1 + i)] = 1
                    if (latest == 1) and (a_future[i] >= h_future) and (h_future > 0):
                        possibleActionSet[3 * (1 + i) + 1] = 1
                    possibleActionSet[3 * (1 + i) + 2] = 1
        if a == h + 1:
            possibleActionSet[0] = 1
        if (latest == 1) and (a >= h) and (h > 0):
            possibleActionSet[1] = 1
        if h <= a + self.bound:
            possibleActionSet[2] = 1
        return possibleActionSet

    def store_transition(self, state, action, reward_attacker, reward_honest, state_, possible_action_set):
        self.memory.store(state, action, reward_attacker, reward_honest, state_, possible_action_set)

    def sample_memory(self):
        samples = self.memory.sample_batch()
        states = T.FloatTensor(samples["states"]).to(self.device)
        actions = T.LongTensor(samples["actions"]).to(self.device)
        rewards_attacker = T.FloatTensor(samples["rewards_attacker"]).to(self.device)
        rewards_honest = T.FloatTensor(samples["rewards_honest"]).to(self.device)
        states_ = T.FloatTensor(samples["states_"]).to(self.device)
        possible_action_sets = T.FloatTensor(samples["possible_action_sets"]).to(self.device)
        return states, actions, rewards_attacker, rewards_honest, states_, possible_action_sets

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
           self.q_next_attacker.load_state_dict(self.q_eval_attacker.state_dict())
           self.q_next_honest.load_state_dict(self.q_eval_honest.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon * self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval_attacker.save_checkpoint(name=self.env_name + '_' + self.algo + '_q_eval_attacker')
        self.q_eval_honest.save_checkpoint(name=self.env_name + '_' + self.algo + '_q_eval_honest')
        self.q_next_attacker.save_checkpoint(name=self.env_name + '_' + self.algo + '_q_next_attacker')
        self.q_next_honest.save_checkpoint(name=self.env_name + '_' + self.algo + '_q_next_honest')

    def load_models(self):
        self.q_eval_attacker.load_checkpoint(name=self.env_name + '_' + self.algo + '_q_eval_attacker')
        self.q_eval_honest.load_checkpoint(name=self.env_name + '_' + self.algo + '_q_eval_honest')
        self.q_next_attacker.load_checkpoint(name=self.env_name + '_' + self.algo + '_q_next_attacker')
        self.q_next_honest.load_checkpoint(name=self.env_name + '_' + self.algo + '_q_next_honest')

    def learn(self, counter):
        if counter < self.batch_size:
            return

        self.q_eval_attacker.zero_grad()
        self.q_eval_honest.zero_grad()

        self.replace_target_network()

        states, actions, rewards_attacker, rewards_honest, states_, possible_action_sets= self.sample_memory()

        indices_arranged = np.arange(self.batch_size)

        Q_pred_attacker = self.q_eval_attacker.forward(states)[indices_arranged, actions]
        Q_pred_honest = self.q_eval_honest.forward(states)[indices_arranged, actions]
        q_next_attacker = self.q_next_attacker.forward(states_)
        q_next_honest = self.q_next_honest.forward(states_)
        actions_total = q_next_attacker/(q_next_honest+q_next_attacker)
        actions_total = T.mul(actions_total, possible_action_sets)
        actions_total = T.nan_to_num(actions_total, nan=float('-inf'))
        best_actions = actions_total.max(dim=1)[1]
        Q_next_attacker = q_next_attacker[indices_arranged, best_actions]
        Q_next_honest = q_next_honest[indices_arranged, best_actions]
        Q_target_attacker = rewards_attacker + self.gamma*Q_next_attacker
        Q_target_honest = rewards_honest + self.gamma*Q_next_honest

        elementwise_loss_attacker = self.q_eval_attacker.loss(Q_pred_attacker, Q_target_attacker)
        loss_attacker = T.mean(elementwise_loss_attacker)
        loss_attacker.backward()
        self.q_eval_attacker.optimizer.step()

        elementwise_loss_honest = self.q_eval_honest.loss(Q_pred_honest, Q_target_honest)
        loss_honest = T.mean(elementwise_loss_honest)
        loss_honest.backward()
        self.q_eval_honest.optimizer.step()

        self.learn_step_counter += 1

        self.decrement_epsilon()