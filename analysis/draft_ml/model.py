from torch import nn, unsqueeze
import numpy as np

from analysis import ml_lib


class DraftEmbedder(ml_lib.Embedder):
    def __init__(self, num_champions: int):
        self.num_champions = num_champions

    def embed(self, games):
        # create 2d array for the embedded game
        embedded = np.zeros((len(games), self.num_champions * 3 + 1 + 1), dtype=np.float32)

        # encode blue side picks
        pick_indices = games[:, 0:5].ravel()
        game_indices = np.arange(len(games)).repeat(5)
        embedded[game_indices[pick_indices != 0], pick_indices[pick_indices != 0] - 1] = 1

        # encode red side picks
        pick_indices = games[:, 5:10].ravel()
        game_indices = np.arange(len(games)).repeat(5)
        embedded[game_indices[pick_indices != 0], pick_indices[pick_indices != 0] - 1 + self.num_champions] = 1

        # encode bans
        ban_indices = games[:, 10:20].ravel()
        game_indices = np.arange(len(games)).repeat(10)
        embedded[game_indices[ban_indices != 0], ban_indices[ban_indices != 0] - 1 + 2 * self.num_champions] = 1

        # write ranked_score
        embedded[:, -2] = games[:, -2]

        # write win/picking team
        embedded[:, -1] = games[:, -1]

        return embedded


class ResNet20(nn.Module):
    class ResBlock(nn.Module):
        def __init__(self, in_out_features: int, bottleneck: int):
            super().__init__()

            self.linear_stack = nn.Sequential(
                nn.Linear(in_out_features, bottleneck),
                nn.ReLU(),
                nn.Linear(bottleneck, in_out_features),
                nn.ReLU(),
            )

        def forward(self, X):
            residual = X
            out = self.linear_stack(X)
            out = out + residual
            return out

    def __init__(self, num_champions: int, width: int, bottleneck: int, dropout: float, blocks_individual: int, blocks_pre_win: int, blocks_pre_bans: int, blocks_pre_rank: int, blocks_post_rank: int):
        super().__init__()

        self.num_champions = num_champions

        self.bs_individual = nn.Sequential(
            nn.Linear(num_champions, width),
            nn.ReLU(),
            *[self.ResBlock(width, bottleneck) for _ in range(blocks_individual)],
        )

        self.rs_individual = nn.Sequential(
            nn.Linear(num_champions, width),
            nn.ReLU(),
            *[self.ResBlock(width, bottleneck) for _ in range(blocks_individual)],
        )

        self.res_blocks_pre_win = nn.Sequential(
            *[self.ResBlock(width, bottleneck) for _ in range(blocks_pre_win)],
        )

        self.linear_win_merger = nn.Sequential(
            nn.Linear(1, width),
            nn.ReLU(),
        )

        self.res_blocks_pre_ban = nn.Sequential(
            *[self.ResBlock(width, bottleneck) for _ in range(blocks_pre_bans)],
        )

        self.linear_ban_merger = nn.Sequential(
            nn.Linear(num_champions, width),
            nn.ReLU(),
        )

        self.res_blocks_pre_rank = nn.Sequential(
            *[self.ResBlock(width, width) for _ in range(blocks_pre_rank)],
        )

        self.linear_rank_merger = nn.Sequential(
            nn.Linear(1, width),
            nn.ReLU(),
        )

        self.res_blocks_post_rank = nn.Sequential(
            *[self.ResBlock(width, bottleneck) for _ in range(blocks_post_rank)],
            nn.Linear(width, num_champions),
            nn.Dropout(dropout),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, X):
        # split out picks, bans, ranked_scores and wins from the embedded X
        bs_picks = X[:, :self.num_champions]
        rs_picks = X[:, self.num_champions:2 * self.num_champions]
        bans = X[:, 2 * self.num_champions:3 * self.num_champions]
        ranks = X[:, -2]
        wins = X[:, -1]

        # analyze comps individually and merge together
        out = self.bs_individual(bs_picks) + self.rs_individual(rs_picks)

        # res blocks pre win
        out = self.res_blocks_pre_win(out)

        # merge in wins
        out += self.linear_win_merger(unsqueeze(wins, 1))

        # res blocks pre bans
        out = self.res_blocks_pre_ban(out)

        # merge in bans
        out += self.linear_ban_merger(bans)

        # res blocks pre rank
        out = self.res_blocks_pre_rank(out)

        # merge in ranks
        out += self.linear_rank_merger(unsqueeze(ranks, 1))

        # res blocks post ranks
        out = self.res_blocks_post_rank(out)
        return out
