from torch import nn
import sklearn
import numpy as np
from torch.nn.functional import embedding


class NeuralNetwork(nn.Module):
    def __init__(self, num_champions: int):
        super().__init__()

        self.num_champions = num_champions
        self.scaler = sklearn.preprocessing.StandardScaler()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1 + self.num_champions*3, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

    def fit_scaler(self, games):
        games = self.embed_games(games, scale=False)
        self.scaler.fit(games)

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
            embedded[pick + 1 + self.num_champions] = 1

        # encode all bans
        for ban in game[2]:
            embedded[ban + 1 + self.num_champions*2] = 1

        # standard scale the whole embedding if desired
        if scale:
            embedded = self.scaler.transform([embedded])

        return np.array(embedded, dtype=np.float32)

    def embed_games(self, games, scale: bool = True):
        return np.array([self.embed_game(game, scale=scale) for game in games])
