
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
        game_indices = np.arange(len(games)).repeat(5)
        pick_indices = games[:, 0:5].ravel()
        embedded[game_indices[pick_indices != 0], pick_indices[pick_indices != 0] - 1] = 1

        # encode red side picks
        pick_indices = games[:, 5:10].ravel()
        embedded[game_indices[pick_indices != 0], pick_indices[pick_indices != 0] - 1 + self.num_champions] = 1

        # encode bans
        game_indices = np.arange(len(games)).repeat(10)
        ban_indices = games[:, 10:20].ravel()
        embedded[game_indices[ban_indices != 0], ban_indices[ban_indices != 0] - 1 + 2 * self.num_champions] = 1

        # write ranked_score
        embedded[:, -2] = games[:, -2]

        # write win/picking team
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


class ResNet20(nn.Module):
    class ResBlock(nn.Module):
        def __init__(self, in_out_features: int, bottleneck: int, dropout: float):
            super().__init__()

            self.alpha = nn.Parameter(tensor(0.))

            self.dropout = nn.Dropout(dropout)
            self.linear_stack = nn.Sequential(
                nn.Linear(in_out_features, bottleneck),
                nn.ReLU(),
                nn.Linear(bottleneck, in_out_features),
                nn.ReLU(),
            )

        def forward(self, X):
            residual = X
            out = self.linear_stack(X)
            out = out * self.alpha + residual
            out = self.dropout(out)
            return out

    def __init__(self, num_champions: int, width: int, bottleneck: int, dropout: float, blocks_pre_win: int, blocks_pre_rank: int, blocks_post_rank: int):
        super().__init__()

        self.res_blocks_pre_win = nn.Sequential(
            nn.Linear(num_champions * 3, width),
            nn.ReLU(),
            *[self.ResBlock(width, bottleneck, dropout) for _ in range(blocks_pre_win)],
        )

        self.res_blocks_pre_rank = nn.Sequential(
            nn.Linear(width + 1, width),
            nn.ReLU(),
            *[self.ResBlock(width, width, dropout) for _ in range(blocks_pre_rank)],
        )

        self.res_blocks_post_rank = nn.Sequential(
            nn.Linear(width + 1, width),
            nn.ReLU(),
            *[self.ResBlock(width, bottleneck, dropout) for _ in range(blocks_post_rank)],
            nn.Linear(width, num_champions),
            nn.ReLU(),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, X):
        # split out champions, ranked_scores and wins from the embedded X
        champs = X[:, :-2]
        ranks = X[:, -2]
        wins = X[:, -1]

        out = self.res_blocks_pre_win(champs)

        # add wins
        out = cat((out, unsqueeze(wins, 1)), dim=-1)
        out = self.res_blocks_pre_rank(out)

        # add ranks
        out = cat((out, unsqueeze(ranks, 1)), dim=-1)
        out = self.res_blocks_post_rank(out)

        return out
