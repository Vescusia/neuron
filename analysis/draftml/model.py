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

        for i, game in enumerate(games):
            # write the ranked score
            embedded[i][0] = game[0]

            # encode blue side picks
            for pick in game[1][0:5]:
                embedded[i][pick] = 1
            # encode red side picks
            for pick in game[1][5:10]:
                embedded[i][pick.astype(np.int16) + self.num_champions] = 1

            # encode all bans
            for ban in game[2]:
                embedded[i][ban.astype(np.int16) + 1 + self.num_champions * 2] = 1

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
    def __init__(self, in_out_features: int, width_factor: int):
        super().__init__()

        self.alpha = nn.Parameter(tensor(1.))
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
        out += residual * self.alpha
        out = self.relu(out)
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
