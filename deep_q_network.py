import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dim):
        super(DeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

        self.optimizer = T.optim.RMSprop(self.parameters(), lr=lr)
        self.device = T.device('cuda' if False else 'cpu')
        self.to(self.device)

    def loss(self, Q_target, Q_pred):
        self.loss_value = F.mse_loss(Q_target, Q_pred, reduction="none")
        return self.loss_value

    def forward(self, state):
        layer1 = F.relu(self.fc1(state))
        layer2 = F.relu(self.fc2(layer1))
        actions = self.fc3(layer2)
        return actions

    def save_checkpoint(self, name):
        print('... saving checkpoint ...')
        checkpoint_file = os.path.join(name)
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, name):
        print('... loading checkpoint ...')
        checkpoint_file = os.path.join(name)
        self.load_state_dict(T.load(checkpoint_file))



