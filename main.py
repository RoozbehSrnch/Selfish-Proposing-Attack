#Code adapted from course "Modern Reinforcement Learning: Deep Q Agents (PyTorch & TF2)"
#https://www.udemy.com/course/deep-q-learning-from-paper-to-code/
import numpy as np
from agent import Agent
from environment_LC_PoS_perfect_randomness import LC_PoS_perfect_randomness
from environment_semi_predictable_LC_PoS import semi_predictable_LC_PoS
from environment_full_predictable_LC_PoS import full_predictable_LC_PoS
from environment_semi_predictable_LC_PoS_node_location import semi_predictable_LC_PoS_node_location
from environment_semi_predictable_LC_PoS_transaction_pool import semi_predictable_LC_PoS_transaction_pool
import os

if __name__ == '__main__':
    # stake share
    alpha = 1/3
    # communication capability
    eta = 0
    # expected number of slots to propose one block
    block_period_length = 20
    # the number of blocks in the honest fork for which
    # information regarding their subsequent time slots is stored in the state
    n_array = 5
    # number of future blocks whose proposer can be predicted at the current slot
    n_future_block = 50
    # number of proposer nodes
    n_node = 100
    # train or test
    testing = False
    # number of training steps
    number_steps = 2000000
    # number of checkpoints
    n_checkpoint = 3
    # period of storing checkpoints
    index1 = 10000

    # # LC_PoS_perfect_randomness
    # agent = Agent(gamma=0.999999, epsilon=1, lr=0.0001, n_actions=3 * (n_array + 1), n_array=n_array,
    #               input_dims=n_array + 5,
    #               mem_size=700000, batch_size=128, eps_min=0.001, eps_dec=0.99999, replace=10000, algo='Agent',
    #               env_name='LC_PoS_perfect_randomness')
    # env = LC_PoS_perfect_randomness(alpha=alpha, eta=eta, n_array=n_array, block_period_length=block_period_length)

    # semi_predictable_LC_PoS
    agent = Agent(gamma=0.999999, epsilon=1, lr=0.0001, n_actions=3 * (n_array + 1), n_array=n_array,
                  input_dims=n_array + 5 + n_future_block,
                  mem_size=700000, batch_size=128, eps_min=0.001, eps_dec=0.99999, replace=10000, algo='Agent',
                  env_name='semi_predictable_LC_PoS')
    env = semi_predictable_LC_PoS(alpha=alpha, eta=eta, n_node=n_node, n_array=n_array,
                                  state_1stPart_length=n_array + 5, block_period_length=block_period_length,
                                  n_future_block=n_future_block)

    # # full_predictable_LC_PoS
    # agent = Agent(gamma=0.999999, epsilon=1, lr=0.0001, n_actions=3 * (n_array + 1), n_array=n_array,
    #               input_dims=n_array + 5 + n_future_block,
    #               mem_size=700000, batch_size=128, eps_min=0.001, eps_dec=0.99999, replace=10000, algo='Agent',
    #               env_name='full_predictable_LC_PoS')
    # env = full_predictable_LC_PoS(alpha=alpha, eta=eta, n_array=n_array, state_1stPart_length=n_array + 5,
    #                               n_future_block=n_future_block)

    # # semi_predictable_LC_PoS_node_location
    # agent = Agent(gamma=0.999999, epsilon=1, lr=0.0001, n_actions=3 * (n_array + 1), n_array=n_array,
    #               input_dims=n_array + 5 + n_future_block,
    #               mem_size=700000, batch_size=128, eps_min=0.001, eps_dec=0.99999, replace=10000, algo='Agent',
    #               env_name='semi_predictable_LC_PoS_node_location')
    # env = semi_predictable_LC_PoS_node_location(alpha=alpha, n_node=n_node, n_array=n_array,
    #                               state_1stPart_length=n_array + 5, block_period_length=block_period_length,
    #                               n_future_block=n_future_block, test=testing)


    if testing:
        agent.load_models()

    scores_attacker, scores_honest, scores_average = [], [], []
    max_average_index = []

    counter = 0
    state = env.reset()
    while counter < number_steps:
        action = agent.choose_action(state)
        state_, reward_attacker, reward_honest = env.step(action)
        possible_action_set = agent.possible_actions(state_, n_array)
        if not testing:
            agent.store_transition(state, action, reward_attacker, reward_honest, state_, possible_action_set)
            agent.learn(counter)
        state = state_
        counter += 1
        scores_attacker.append(reward_attacker)
        scores_honest.append(reward_honest)
        if counter % index1 == 0:
            avg_reward = np.sum(scores_attacker[-index1:])/np.sum(np.add(scores_attacker[-index1:], scores_honest[-index1:]))
            scores_average.append(avg_reward)
            print('counter:', counter, 'avg_reward:', avg_reward)
            if max(scores_average) == avg_reward and (not testing):
                agent.save_models()
