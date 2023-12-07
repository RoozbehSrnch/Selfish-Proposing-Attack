# Code adapted from rainbow-is-all-you-need (PrioritizedExperienceReplay)
# https://github.com/Curt-Park/rainbow-is-all-you-need
import numpy as np
from typing import Dict

class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, input_dims: int, max_size: int, batch_size: int, action_size: int):
        self.state_memory = np.zeros([max_size, input_dims], dtype=np.float32)
        self.next_state_memory = np.zeros([max_size, input_dims], dtype=np.float32)
        self.action_memory = np.zeros([max_size], dtype=np.float32)
        self.reward_attacker_memory = np.zeros([max_size], dtype=np.float32)
        self.reward_honest_memory = np.zeros([max_size], dtype=np.float32)
        self.possible_action_set_memory = np.zeros([max_size, action_size], dtype=np.float32)
        self.max_size, self.batch_size = max_size, batch_size
        self.ptr, self.size = 0, 0

    def store(self, state, action, reward_attacker, reward_honest, state_, possible_action_set):
        self.state_memory[self.ptr] = state
        self.next_state_memory[self.ptr] = state_
        self.action_memory[self.ptr] = action
        self.reward_attacker_memory[self.ptr] = reward_attacker
        self.reward_honest_memory[self.ptr] = reward_honest
        self.possible_action_set_memory[self.ptr] = possible_action_set
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(states=self.state_memory[idxs],
                    states_=self.next_state_memory[idxs],
                    actions=self.action_memory[idxs],
                    rewards_attacker=self.reward_attacker_memory[idxs],
                    rewards_honest=self.reward_honest_memory[idxs],
                    possible_action_sets=self.possible_action_set_memory[idxs])

    def __len__(self) -> int:
        return self.size