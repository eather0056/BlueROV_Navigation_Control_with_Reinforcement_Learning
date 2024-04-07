import torch
import torch.nn as nn
import torch.nn.functional as F


class Q_net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Q_net, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    # def forward(self, s, a):
    #     s = s.reshape(-1, self.state_dim)
    #     a = a.reshape(-1, self.action_dim)
    #     x = torch.cat((s, a), -1) # combination s and a
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return x
    
    def forward(self, imgs_depth, goals, rays, a):  # Modify inputs
        batch_size = imgs_depth.shape[0]  # Get batch size
        s = torch.cat((imgs_depth.view(batch_size, -1), goals.view(batch_size, -1), rays.view(batch_size, -1)), dim=1)  # Flatten and concatenate the input tensors
        a = a.view(batch_size, -1)  # Flatten action tensor
        x = torch.cat((s, a), -1)  # Combine state and action tensors
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x