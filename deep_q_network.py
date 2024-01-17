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
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, n_actions)

        self.optimizer = T.optim.RMSprop(self.parameters(), lr=lr)
        self.device = T.device('cuda' if False else 'cpu')
        self.to(self.device)

    def loss(self, Q_target, Q_pred):
        self.loss_value = F.mse_loss(Q_target, Q_pred, reduction="none")
        return self.loss_value

    def forward(self, state):
        layer1 = F.relu(self.fc1(state))
        layer2 = F.relu(self.fc2(layer1))
        layer3 = F.relu(self.fc3(layer2))
        actions = self.fc4(layer3)
        return actions


    def save_checkpoint(self, name, chkpt_dir):
        print('... saving checkpoint ...')
        checkpoint_dir = chkpt_dir
        checkpoint_file = os.path.join(checkpoint_dir, name)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        T.save(self.state_dict(), checkpoint_file)


    def load_checkpoint(self, name, chkpt_dir):
        print('... loading checkpoint ...')
        checkpoint_dir = chkpt_dir
        checkpoint_file = os.path.join(checkpoint_dir, name)
        self.load_state_dict(T.load(checkpoint_file))



