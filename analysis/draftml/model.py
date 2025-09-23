from torch import tensor
from torch import nn
import sklearn
import numpy as np


class Embedder:
    def __init__(self, num_champions: int):
        self.num_champions = num_champions
        self.scaler = sklearn.preprocessing.StandardScaler()

    def embed_game(self, game: list, scale: bool = True):
        # write the ranked score
        embedded = [game[0]]
        # fill with zeros for one-hot-encoding of picks/bans
        embedded += [0] * self.num_champions * 3

        # encode blue side picks
        for pick in game[1][0:5]:
            embedded[pick + 1] = 1
        # encode red side picks
        for pick in game[1][5:10]:
            embedded[int(pick) + 1 + self.num_champions] = 1

        # encode all bans
        for ban in game[2]:
            embedded[int(ban) + 1 + self.num_champions*2] = 1

        # standard scale the whole embedding if desired
        if scale:
            embedded = self.scaler.transform([embedded])

        return np.array(embedded, dtype=np.float32)

    def embed_games(self, games, scale: bool = True):
        return np.array([self.embed_game(game, scale=scale) for game in games])

    def __call__(self, games):
        return self.embed_games(games)

    def fit(self, games):
        games = self.embed_games(games, scale=False)
        self.scaler.fit(games)


class LinearWide54(nn.Module):
    def __init__(self, num_champions: int):
        super().__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1 + num_champions*3, 1024),
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
            nn.Linear(1 + num_champions*3, base_width),
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
