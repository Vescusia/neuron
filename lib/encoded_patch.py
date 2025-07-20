"""
Encodes a patch to a 16-bit integer.

This encoding only respects the major and minor version of the patch.

This promises to be consistent across patch additions. I.e., old patches will retain the same number.
"""

from numpy import uint16

import lib


def _canonize_patch(patch: str) -> str:
    # only respect the major and minor version of the patch
    return '.'.join(patch.split('.')[:2])


# get all patches from data dragon
_all_patches: list[str] = lib.data_dragon.versions_all()
# move the newest patches to the end of the list
_all_patches.reverse()
# remove the pre-season-3 patches (unique format and irrelevant)
_all_patches = _all_patches[_all_patches.index('3.6.14'):]
# reduce patch strings to only minor and major
_all_patches = [_canonize_patch(patch) for patch in _all_patches]

# create the patch-to-int map;
# this has to be consistent
_patch_to_int_map = {patch: uint16(i) for i, patch in enumerate(_all_patches)}
# reverse the map
_int_to_patch_map = {v: k for k, v in _patch_to_int_map.items()}


def to_int(patch: str) -> uint16:
    return _patch_to_int_map[_canonize_patch(patch)]


def to_patch(encoded_patch: uint16) -> str:
    return _int_to_patch_map[encoded_patch]
