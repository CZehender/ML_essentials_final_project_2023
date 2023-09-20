import torch as T
import torch.nn as nn
import torch.optim as optim

from .feature_engineering import state_to_features

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


class DQN(nn.Module):
    gamma = 0.9
    learning_rate = 0.0005

    def __init__(self, out_action_dim):
        super(DQN, self).__init__()
        self.model_sequential = nn.Sequential(
            nn.Conv2d(4, 1, (4, 4)),  # in_channels, out_channels, kernel size
            nn.Flatten(start_dim=1),
            nn.Linear(14 * 14, 128),  # 14*14 is image size after convolution layer
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, out_action_dim),
            nn.Softmax(dim=1)
        )
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), self.learning_rate)
        self.device = T.device('cpu')  # train with cpu
        self.to(self.device)

    def forward(self, x):
        """
        4 feature channels
        Coins feature in channel 0
        Walls feature in channel 2
        Position of user feature in channel 1
        Position of dangerous area in channel 3
        Args:
            x: input feature
        Returns: model
        """
        x = T.tensor(state_to_features(x)).float().view(-1, 4, 17, 17)  # batch_size, input channels, board_dims
        return self.model_sequential(x.to(self.device)).to('cpu')

    def train(self, old_state, action, new_state, reward):
        if action is not None:
            action_mask = T.zeros(len(ACTIONS), dtype=T.int64)
            action_mask[ACTIONS.index(action)] = 1

            state_action_value = T.masked_select(self.forward(old_state), action_mask.bool())
            next_state_action_value = self.forward(new_state).max().unsqueeze(0)
            # calculate the expected reward with future reward
            expected_state_action_value = (next_state_action_value * self.gamma) + reward

            loss = self.loss(state_action_value.to(self.device), expected_state_action_value.to(self.device))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            self.logger.info("Invalid Action.")