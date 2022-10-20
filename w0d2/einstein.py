#%%
import einops
import numpy as np
from matplotlib import pyplot as plt

import utils
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

