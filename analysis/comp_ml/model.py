from torch import tensor, nn, cat, unsqueeze
import sklearn
import numpy as np


class Embedder:
    def __init__(self, num_champions: int):
        self.num_champions = num_champions
        self.scaler = sklearn.preprocessing.StandardScaler()

    def embed_games(self, games, scale: bool = True):
        # create 2d array for the embedded game
        embedded = np.zeros((len(games), self.num_champions * 2 + 1 + 1), dtype=np.float32)

        # encode blue side picks
        game_indices = np.arange(len(games))
        pick_indices = games[:, 0:5] - 1  # None pick (id 0) isn't possible
        embedded[game_indices.repeat(5), pick_indices.ravel()] = 1

        # encode red side picks
        game_indices = np.arange(len(games))
        pick_indices = games[:, 5:10] - 1 + self.num_champions  # pick indices are first, None pick (id 0) isn't possible
        embedded[game_indices.repeat(5), pick_indices.ravel()] = 1

        # write the patch
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


class ResBlock(nn.Module):
    def __init__(self, in_out_features: int, bottleneck: int, dropout: float):
        super().__init__()

        self.alpha = nn.Parameter(tensor(0.))

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
        return out


class ResNet60(nn.Module):
    def __init__(self, num_champions: int, dropout: float, bottleneck: int, base_width: int, pre_rank_blocks: int, post_rank_blocks: int):
        super().__init__()

        self.linear_rank_merger = nn.Sequential(
            nn.Linear(2, base_width),
            nn.ReLU(),
        )

        self.res_blocks_pre_rank = nn.Sequential(
            nn.Linear(num_champions * 2, base_width),
            nn.ReLU(),
            *[ResBlock(base_width, bottleneck, dropout) for _ in range(pre_rank_blocks)],
        )

        self.res_blocks_post_rank = nn.Sequential(
            *[ResBlock(base_width, bottleneck, dropout) for _ in range(post_rank_blocks)],
            nn.Dropout(dropout),
            nn.Linear(base_width, 1),
            nn.Sigmoid()
        )

    def forward(self, X):
        # split out champions and ranks from embedding
        champs = X[:, :-2]
        patches_and_ranks = X[:, -2:]
        
        # run champs through first blocks
        out = self.res_blocks_pre_rank(champs)

        # merge in the rank
        out += self.linear_rank_merger(patches_and_ranks)

        # run through last blocks
        out = self.res_blocks_post_rank(out)
        return out
