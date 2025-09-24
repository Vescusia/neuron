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
        embedded = np.zeros((len(games), self.num_champions * 3 + 1 + 1), dtype=np.float32)

        # encode blue side picks
        game_indices = np.arange(len(games))
        pick_indices = games[:, 0:5]
        pick_indices = pick_indices[pick_indices != 0] - 1
        embedded[game_indices.repeat(len(pick_indices)), pick_indices.ravel()] = 1

        # encode red side picks
        game_indices = np.arange(len(games))
        pick_indices = games[:, 5:10]
        pick_indices = pick_indices[pick_indices != 0] - 1 + self.num_champions
        embedded[game_indices.repeat(len(pick_indices)), pick_indices.ravel()] = 1

        # encode bans
        game_indices = np.arange(len(games))
        ban_indices = games[:, 10:20]
        ban_indices = ban_indices[ban_indices != 0] - 1 + self.num_champions * 2
        embedded[game_indices.repeat(len(ban_indices)), ban_indices.ravel()] = 1

        # write ranked_score
        embedded[:, -2] = games[:, -2]

        # write win/picking team
        embedded[:, -1] = games[:, -1]

        # standard scale the whole embedding if desired
        if scale:
            embedded = self.scaler.transform(embedded)

        return embedded


class ResNet20(nn.Module):
    class ResBlock(nn.Module):
        def __init__(self, in_out_features: int, width_factor: int, dropout: float):
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
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        def forward(self, X):
            residual = X
            out = self.linear_stack(X)
            out = self.relu(out)
            out = out * self.alpha + residual
            return out


    def __init__(self, num_champions: int, dropout: float, width_factor: int, base_width: int):
        super().__init__()

        self.res_block_stack = nn.Sequential(
            nn.Linear(2 + num_champions*3, base_width),
            nn.ReLU(),
            *[self.ResBlock(base_width, width_factor, dropout) for _ in range(20)],
            nn.Linear(base_width, num_champions),
            nn.ReLU(),
            nn.LogSoftmax(num_champions)
        )

    def forward(self, X):
        out = self.res_block_stack(X)
        return out
