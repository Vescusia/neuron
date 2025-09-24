import time

from torch import tensor
from torch import nn
import sklearn
import numpy as np


class Embedder:
    def __init__(self, num_champions: int):
        self.num_champions = num_champions
        self.scaler = sklearn.preprocessing.StandardScaler()

    def embed_games(self, games, scale: bool = True):
        # create 2d array for the embedded game
        embedded = np.zeros((len(games), 1 + self.num_champions * 3 + 1), dtype=np.float32)

        # write the ranked score
        embedded[:, 0] = games[:, 0]

        # encode blue side picks
        game_indices = np.arange(len(games))
        pick_indices = games[:, 1:6] + 1  # ranked_score is in embedded[0]
        embedded[game_indices.repeat(5), pick_indices.ravel()] = 1

        # encode red side picks
        game_indices = np.arange(len(games))
        pick_indices = games[:, 6:11] + 1 + self.num_champions  # pick indices are first
        embedded[game_indices.repeat(5), pick_indices.ravel()] = 1

        # encode bans
        game_indices = np.arange(len(games))
        ban_indices = games[:, 11:21] + 1 + 2*self.num_champions  # both pick indices are first
        embedded[game_indices.repeat(10), ban_indices.ravel()] = 1

        # standard scale the whole embedding if desired
        if scale:
            embedded = self.scaler.transform(embedded)

        return embedded

class ResBlock(nn.Module):
    def __init__(self, in_out_features: int, width_factor: int):
        super().__init__()

        self.alpha = nn.Parameter(tensor(0.))
        self.relu = nn.ReLU()

        self.linear_stack = nn.Sequential(
            nn.Linear(in_out_features, in_out_features * width_factor),
            nn.ReLU(),
            nn.Linear(in_out_features * width_factor, 2 * in_out_features * width_factor),
            nn.ReLU(),
            nn.Linear(2 * in_out_features * width_factor, 2 * in_out_features * width_factor),
            nn.ReLU(),
            nn.Linear(2 * in_out_features * width_factor, in_out_features * width_factor),
            nn.ReLU(),
            nn.Linear(in_out_features * width_factor, in_out_features),
        )

    def forward(self, X):
        residual = X
        out = self.linear_stack(X)
        out = self.relu(out)
        out = out * self.alpha + residual
        return out


class ResNet60(nn.Module):
    def __init__(self, num_champions: int, dropout: float, width_factor: int, base_width: int):
        super().__init__()

        self.res_block_stack = nn.Sequential(
            nn.Linear(2 + num_champions*3, base_width),
            nn.ReLU(),
            ResBlock(base_width, width_factor),
            ResBlock(base_width, width_factor),
            ResBlock(base_width, width_factor),
            ResBlock(base_width, width_factor),
            ResBlock(base_width, width_factor),
            ResBlock(base_width, width_factor),
            nn.Dropout(dropout),
            ResBlock(base_width, width_factor),
            ResBlock(base_width, width_factor),
            ResBlock(base_width, width_factor),
            ResBlock(base_width, width_factor),
            ResBlock(base_width, width_factor),
            ResBlock(base_width, width_factor),
            nn.Linear(base_width, 1),
            nn.Sigmoid()
        )

    def forward(self, X):
        out = self.res_block_stack(X)
        return out



