import os
import torch
from torch import nn
import sklearn
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self, num_champions: int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.num_champions = num_champions
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1 + self.num_champions * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

    def fit_scaler(self, games):
        games = self.embed_games(games, scale=False)
        self.scaler.fit(games)

    def embed_game(self, game: list, scale: bool = True):
        embeded = [game[0]]
        embeded += [0] * self.num_champions * 3
        for pick in game[1][0:5]:
            embeded[pick + 1] = 1
        for pick in game[1][5:10]:
            embeded[pick + 1 + self.num_champions] = 1
        for ban in game[2]:
            embeded[ban + 1 + self.num_champions * 2] = 1

        if scale:
            embeded = self.scaler.transform([embeded])
        return embeded

    def embed_games(self, games, scale: bool = True):
        return [self.embed_game(game, scale=scale) for game in games]
