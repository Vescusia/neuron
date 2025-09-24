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
        embedded = np.zeros((len(games), 2 + self.num_champions * 3 + 1), dtype=np.float32)

        # write the patch
        embedded[:, 0] = games[:, 0]

        # write the ranked score
        embedded[:, 1] = games[:, 1]

        # encode blue side picks
        game_indices = np.arange(len(games))
        pick_indices = games[:, 2:7] + 2 - 1  # patch and ranked_score come first, no pick is not possible
        embedded[game_indices.repeat(5), pick_indices.ravel()] = 1

        # encode red side picks
        game_indices = np.arange(len(games))
        pick_indices = games[:, 7:12] + 2 + self.num_champions - 1  # pick indices are first, no pick is not possible
        embedded[game_indices.repeat(5), pick_indices.ravel()] = 1

        # encode bans
        game_indices = np.arange(len(games))
        ban_indices = games[:, 12:22] + 2 + 2 * self.num_champions  # both pick indices are first
        embedded[game_indices.repeat(10), ban_indices.ravel()] = 1

        # standard scale the whole embedding if desired
        if scale:
            embedded = self.scaler.transform(embedded)

        return embedded

    def __call__(self, games):
        return self.embed_games(games)

    def fit(self, games):
        games = self.embed_games(games, scale=False)
        self.scaler.fit(games)


class LinearWide54(nn.Module):
    def __init__(self, num_champions: int):
        super().__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2 + num_champions*3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


class ResBlock(nn.Module):
    def __init__(self, in_out_features: int, bottleneck: int, dropout: float):
        super().__init__()

        self.alpha = nn.Parameter(tensor(0.))

        self.linear_stack = nn.Sequential(
            nn.Linear(in_out_features, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, in_out_features),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, X):
        residual = X
        out = self.linear_stack(X)
        out = out * self.alpha + residual
        return out


class ResNet60(nn.Module):
    def __init__(self, num_champions: int, dropout: float, bottleneck: int, base_width: int):
        super().__init__()

        self.res_block_stack = nn.Sequential(
            nn.Linear(1 + 1 + num_champions*3 + 1, base_width),
            nn.ReLU(),
            ResBlock(base_width, bottleneck, dropout),
            ResBlock(base_width, bottleneck, dropout),
            ResBlock(base_width, bottleneck, dropout),
            ResBlock(base_width, bottleneck, dropout),
            ResBlock(base_width, bottleneck, dropout),
            ResBlock(base_width, bottleneck, dropout),
            nn.Linear(base_width, 1),
            nn.Sigmoid()
        )

    def forward(self, X):
        out = self.res_block_stack(X)
        return out
