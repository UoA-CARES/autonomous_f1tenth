import torch
import torch.nn as nn
import torch.optim as optim

LIDAR_SIZE = 640

class LidarActor(nn.Module):
    def __init__(self, observation_size, num_actions, learning_rate, max_action):
        super(Actor, self).__init__()

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.max_action = max_action

        self.hidden_size = [128, 64, 32]

        self.h_conv1d_1 = nn.Conv1d(in_channels=1, out_channels=1, stride=3, kernel_size=3)
        self.h_conv1d_2 = nn.Conv1d(in_channels=1, out_channels=1, stride=3, kernel_size=3)
        self.h_conv1d_3 = nn.Conv1d(in_channels=1, out_channels=1, stride=3, kernel_size=3)
        self.h_lidar_linear = nn.Linear(in_features=23, out_features=10)

        self.h_linear_1 = nn.Linear(in_features=18, out_features=self.hidden_size[0])
        self.h_linear_2 = nn.Linear(in_features=self.hidden_size[0], out_features=self.hidden_size[1])
        self.h_linear_3 = nn.Linear(in_features=self.hidden_size[1], out_features=self.hidden_size[2])
        self.h_linear_4 = nn.Linear(in_features=self.hidden_size[2], out_features=num_actions)

        self.optimiser = optim.Adam(self.parameters(), lr=learning_rate)

    def lidar_model_layers(self, state):
        x = torch.relu(self.h_conv1d_1(state))
        x = torch.relu(self.h_conv1d_2(x))
        x = torch.relu(self.h_conv1d_3(x))
        x = torch.tanh(self.h_lidar_linear(x))
        return x.view(x.size(0), -1)

    def main_model_layers(self, state):
        x = torch.relu(self.h_linear_1(state))
        x = torch.relu(self.h_linear_2(x))
        x = torch.relu(self.h_linear_3(x))
        x = torch.tanh(self.h_linear_4(x))
        return x

    def forward(self, state: torch.Tensor):
        if len(state) < 8:
            state = state.reshape([648]).to(self.device)
            
        lidar = state[8:].reshape([1, 640]).to(self.device)
        env_state = state[:8].unsqueeze(0).to(self.device)
        
        lidar = self.lidar_model_layers(lidar)

        x = torch.cat((lidar, env_state), 1)
        x = self.main_model_layers(x)
        return x
    
class Actor(nn.Module):
    def __init__(self, observation_size, num_actions, learning_rate):
        super(Actor, self).__init__()

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.hidden_size = [1024, 1024, 1024]

        self.h_linear_1 = nn.Linear(in_features=observation_size, out_features=self.hidden_size[0])
        self.h_linear_2 = nn.Linear(in_features=self.hidden_size[0], out_features=self.hidden_size[1])
        self.h_linear_3 = nn.Linear(in_features=self.hidden_size[1], out_features=self.hidden_size[2])
        self.h_linear_4 = nn.Linear(in_features=self.hidden_size[2], out_features=num_actions)

        self.optimiser = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = torch.relu(self.h_linear_1(state))
        x = torch.relu(self.h_linear_2(x))
        x = torch.relu(self.h_linear_3(x))
        x = torch.tanh(self.h_linear_4(x))
        return x