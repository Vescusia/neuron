from torch import tensor, nn, unsqueeze
import sklearn
import numpy as np


class Embedder:
    def __init__(self, num_champions: int):
        self.num_champions = num_champions
        self.scaler = sklearn.preprocessing.StandardScaler()

    def embed_games(self, games, scale: bool = True):
        # create 2d array for the embedded game
        embedded = np.zeros((len(games), self.num_champions * 2 + 1), dtype=np.float32)

        # encode blue side picks
        game_indices = np.arange(len(games))
        pick_indices = games[:, 0:5] - 1  # None pick (id 0) isn't possible
        embedded[game_indices.repeat(5), pick_indices.ravel()] = 1

        # encode red side picks
        game_indices = np.arange(len(games))
        pick_indices = games[:, 5:10] - 1 + self.num_champions  # bs picks come first, None pick isn't possible
        embedded[game_indices.repeat(5), pick_indices.ravel()] = 1

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
    def __init__(self, in_out_features: int, bottleneck: int):
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
    def __init__(self, num_champions: int, dropout: float, bottleneck: int, base_width: int, separate_comp_blocks: int, pre_rank_blocks: int, post_rank_blocks: int):
        super().__init__()
        self.num_champions = num_champions

        # separate comp blocks
        self.res_blocks_bs, self.res_blocks_rs = [nn.Sequential(
            nn.Linear(num_champions, base_width),
            nn.ReLU(),
            *[ResBlock(base_width, bottleneck) for _ in range(separate_comp_blocks)],
        ) for _ in range(2)]

        # linear layer for merging both base_width's of the separate comp blocks to one base_width
        self.linear_comp_merger = nn.Sequential(
            nn.Linear(base_width, base_width),
            nn.ReLU(),
        )

        # combined comp blocks before merging in the rank
        self.res_blocks_pre_rank = nn.Sequential(
            *[ResBlock(base_width, bottleneck) for _ in range(pre_rank_blocks)],
        )

        # linear layer for exploding the rank to base_width
        self.linear_rank_merger = nn.Sequential(
            nn.Linear(1, base_width),
            nn.ReLU(),
        )

        # combined comp blocks after merging in the rank
        self.res_blocks_post_rank = nn.Sequential(
            *[ResBlock(base_width, bottleneck) for _ in range(post_rank_blocks)],
            nn.Dropout(dropout),
        )

        # sigmoid
        self.sigmoid = nn.Sequential(
            nn.Linear(base_width, 1),
            nn.Sigmoid()
        )

    def forward(self, X):
        # split out blue side and red side picks as well as the ranked scores
        bs_picks = X[:, :self.num_champions]
        rs_picks = X[:, self.num_champions:-1]
        ranks = X[:, -1]

        # analyze comps separately
        bs_out = self.res_blocks_bs(bs_picks)
        rs_out = self.res_blocks_rs(rs_picks)

        # merge both comps
        out = self.linear_comp_merger(bs_out + rs_out)

        # analyze both comps
        out = self.res_blocks_pre_rank(out)

        # merge in the ranked score
        out += self.linear_rank_merger(unsqueeze(ranks, 1))

        # analyze both comps with ranked score
        out = self.res_blocks_post_rank(out)

        # sigmoid
        out = self.sigmoid(out)

        return out
