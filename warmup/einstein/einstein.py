#%%
import einops
import numpy as np
from fancy_einsum import einsum
from matplotlib import pyplot as plt

from arena.einstein import utils

#%%

ims = np.load('./numbers.npy', allow_pickle=False)

ims2 = einops.rearrange(ims, "b c h w -> c h (b w)")
utils.display_array_as_img(ims2)
# %%

ims3 = einops.repeat(ims, "b c h w -> b c (2 h) w")
utils.display_array_as_img(ims3[0])
# %%

ims4 = einops.repeat(ims[:2], "b c h w -> c (2 h) (b w)")
utils.display_array_as_img(ims4)
# %%

ims5 = einops.repeat(ims, "b c h w -> b c (h 2) w")
utils.display_array_as_img(ims5[0])
# %%

ims6 = einops.rearrange(ims[0], "c h w -> h (c w)")
utils.display_array_as_img(ims6)

# %%
ims7 = einops.rearrange(ims, "(b1 b2) c h w -> c (b1 h) (b2 w)", b1=2)
utils.display_array_as_img(ims7)
# %%

ims8 = einops.reduce(ims, "b c h w -> h (b w)", "max")
utils.display_array_as_img(ims8)
# %%

ims9 = einops.reduce(ims.astype(float), "b c h w -> h (b w)", "mean")
utils.display_array_as_img(ims9)

# %%

ims10 = einops.reduce(ims.astype(float), "b c h w -> h w", "min")
utils.display_array_as_img(ims10)

# %%

ims11 = einops.rearrange(ims[:2], "b c (h2 h1) w -> c h1 (h2 b w)", h2=2)
utils.display_array_as_img(ims11)
# %%

ims12 = einops.rearrange(ims[1], "c h w -> c w h")
utils.display_array_as_img(ims12)
# %%

ims13 = einops.rearrange(ims, "(b1 b2) c h w -> c (b1 w) (b2 h)", b1=2)
utils.display_array_as_img(ims13)
# %%

ims14 = einops.reduce(ims, "(b1 b2) c (h dh) (w dw) -> c (b1 h) (b2 w)", b1=2, dh=2, dw=2, reduction="max")
utils.display_array_as_img(ims14)
# %%

def einsum_trace(mat: np.ndarray):
    """
    Returns the same as `np.trace`.
    """
    return einsum("i i -> ", mat)

def einsum_mv(mat: np.ndarray, vec: np.ndarray):
    """
    Returns the same as `np.matmul`, when `mat` is a 2D array and `vec` is 1D.
    """
    return einsum("i j, j -> i", mat, vec)

def einsum_mm(mat1: np.ndarray, mat2: np.ndarray):
    """
    Returns the same as `np.matmul`, when `mat1` and `mat2` are both 2D arrays.
    """
    return einsum("i j, j k-> i k", mat1, mat2)

def einsum_inner(vec1, vec2):
    """
    Returns the same as `np.inner`.
    """
    return einsum("i, i -> ", vec1, vec2)


def einsum_outer(vec1, vec2):
    """
    Returns the same as `np.outer`.
    """
    return einsum("i, j -> i j", vec1, vec2)


utils.test_einsum_trace(einsum_trace)
utils.test_einsum_mv(einsum_mv)
utils.test_einsum_mm(einsum_mm)
utils.test_einsum_inner(einsum_inner)
utils.test_einsum_outer(einsum_outer)
# %%
