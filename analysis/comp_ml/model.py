from time import process_time_ns

from torch import tensor, nn, cat, unsqueeze
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
        pick_indices = games[:, 0:5] - 1  # None pick (id 0) isn't possible
        embedded[game_indices.repeat(5), pick_indices.ravel()] = 1

        # encode red side picks
        game_indices = np.arange(len(games))
        pick_indices = games[:, 5:10] - 1 + self.num_champions  # pick indices are first, None pick (id 0) isn't possible
        embedded[game_indices.repeat(5), pick_indices.ravel()] = 1

        # encode bans
        ban_indices = games[:, 10:20].ravel()
        game_indices = np.arange(len(games)).repeat(10)[ban_indices != 0]
        embedded[game_indices, ban_indices[ban_indices != 0] - 1 + self.num_champions * 2] = 1  # both pick indices are first, None pick (id 0) isn't possible

        # write patch
        embedded[:, -2] = games[:, -2]

        # write ranked_score
        embedded[:, -1] = games[:, -1]

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
            nn.Linear(num_champions * 3 + 1 + 1, 1024),
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

        self.res_blocks_pre_rank = nn.Sequential(
            nn.Linear(num_champions * 3, base_width),
            nn.ReLU(),
            *[ResBlock(base_width, bottleneck, dropout) for _ in range(3)],
            nn.Linear(base_width, num_champions * 3),
            nn.ReLU(),
        )

        self.res_blocks_post_rank = nn.Sequential(
            nn.Linear(num_champions * 3 + 1 + 1, base_width),
            nn.ReLU(),
            *[ResBlock(base_width, bottleneck, dropout) for _ in range(3)],
            nn.Linear(base_width, 1),
            nn.Sigmoid()
        )

    def forward(self, X):
        champs = X[:, :-2]
        patches = X[:, -2]
        ranks = X[:, -1]
        out = self.res_blocks_pre_rank(champs)
        X = cat([out, unsqueeze(patches, 1), unsqueeze(ranks, 1)], dim=-1)
        out = self.res_blocks_post_rank(X)
        return out
