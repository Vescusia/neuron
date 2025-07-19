"""
Encodes a rank (TIER, DIVISION) into a single-byte integer one way.
"""

import numpy as np

from lib import TIERS, DIVISIONS

_tier_step = 255 // len(TIERS)
_div_step = _tier_step // len(DIVISIONS)


def to_int(tier: str, division: str | None) -> np.uint8:
    """
    When tier is master+, division should be None.

    This is one way! There is no way to consistently decode the returned integer.
    However, it is interpretable as a value (0-255) that linearly scales with tier and division;
    Challenger being ~255 and Iron I being 0
    """
    if division is None:
        assert tier in ["MASTER", "GRANDMASTER", "CHALLENGER"], "Tier should be master+"

        # effectively add one tier to master+ as no divisions exist
        master_addition = _tier_step
    else:
        master_addition = 0

    # decode division and tier strings to int
    division = DIVISIONS.index(division)
    tier = TIERS.index(tier)

    return np.uint8(_tier_step * tier + _div_step * division + master_addition)
