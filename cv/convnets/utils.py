import itertools
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import PIL
import plotly.express as px
import plotly.graph_objects as go
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange, reduce, repeat
from fancy_einsum import einsum
from IPython.display import HTML, display
from PIL import Image, ImageDraw, ImageFont
from plotly.subplots import make_subplots
from torch import nn


def test_conv1d_minimal(conv1d_minimal, n_tests=20):
    import numpy as np

    for _ in range(n_tests):
        b = np.random.randint(1, 10)
        h = np.random.randint(10, 30)
        ci = np.random.randint(1, 5)
        co = np.random.randint(1, 5)
        kernel_size = np.random.randint(1, 10)
        x = t.randn((b, ci, h))
        weights = t.randn((co, ci, kernel_size))
        my_output = conv1d_minimal(x, weights)
        torch_output = t.conv1d(x, weights, stride=1, padding=0)
        t.testing.assert_close(my_output, torch_output)
    print("All tests in `test_conv1d_minimal` passed!")


def test_conv2d_minimal(conv2d_minimal, n_tests=4):
    """
    Compare against torch.conv2d.
    Due to floating point rounding, they can be quite different in float32 but should be nearly identical in float64.
    """
    import numpy as np

    for i in range(n_tests):
        b = np.random.randint(1, 10)
        h = np.random.randint(10, 300)
        w = np.random.randint(10, 300)
        ci = np.random.randint(1, 20)
        co = np.random.randint(1, 20)
        kernel_size = tuple(np.random.randint(1, 10, size=(2,)))
        x = t.randn((b, ci, h, w), dtype=t.float64)
        weights = t.randn((co, ci, *kernel_size), dtype=t.float64)
        my_output = conv2d_minimal(x, weights)
        torch_output = t.conv2d(x, weights)
        t.testing.assert_close(my_output, torch_output)
    print("All tests in `test_conv2d_minimal` passed!")


def test_conv1d(conv1d, n_tests=10):
    import numpy as np

    for i in range(n_tests):
        b = np.random.randint(1, 10)
        h = np.random.randint(10, 300)
        ci = np.random.randint(1, 20)
        co = np.random.randint(1, 20)
        stride = np.random.randint(1, 5)
        padding = np.random.randint(0, 5)
        kernel_size = np.random.randint(1, 10)
        x = t.randn((b, ci, h))
        weights = t.randn((co, ci, kernel_size))
        my_output = conv1d(x, weights, stride=stride, padding=padding)
        torch_output = t.conv1d(x, weights, stride=stride, padding=padding)
        t.testing.assert_close(my_output, torch_output)
    print("All tests in `test_conv1d` passed!")


def test_pad1d(pad1d):
    """Should work with one channel of width 4."""
    x = t.arange(4).float().view((1, 1, 4))
    actual = pad1d(x, 1, 3, -2.0)
    expected = t.tensor([[[-2.0, 0.0, 1.0, 2.0, 3.0, -2.0, -2.0, -2.0]]])
    t.testing.assert_close(actual, expected)
    print("All tests in `test_pad1d` passed!")


def test_pad1d_multi_channel(pad1d):
    """Should work with two channels of width 2."""
    x = t.arange(4).float().view((1, 2, 2))
    actual = pad1d(x, 0, 2, -3.0)
    expected = t.tensor([[[0.0, 1.0, -3.0, -3.0], [2.0, 3.0, -3.0, -3.0]]])
    t.testing.assert_close(actual, expected)
    print("All tests in `test_pad1d_multi_channel` passed!")


def test_pad2d(pad2d):
    """Should work with one channel of 2x2."""
    x = t.arange(4).float().view((1, 1, 2, 2))
    expected = t.tensor(
        [
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [2.0, 3.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ]
        ]
    )
    actual = pad2d(x, 0, 1, 2, 3, 0.0)
    t.testing.assert_close(actual, expected)
    print("All tests in `test_pad2d` passed!")


def test_pad2d_multi_channel(pad2d):
    """Should work with two channels of 2x1."""
    x = t.arange(4).float().view((1, 2, 2, 1))
    expected = t.tensor(
        [
            [
                [[-1.0, 0.0], [-1.0, 1.0], [-1.0, -1.0]],
                [[-1.0, 2.0], [-1.0, 3.0], [-1.0, -1.0]],
            ]
        ]
    )
    actual = pad2d(x, 1, 0, 0, 1, -1.0)
    t.testing.assert_close(actual, expected)
    print("All tests in `test_pad2d_multi_channel` passed!")


def test_conv2d(conv2d, n_tests=5):
    import numpy as np

    for i in range(n_tests):
        b = np.random.randint(1, 10)
        h = np.random.randint(10, 300)
        w = np.random.randint(10, 300)
        ci = np.random.randint(1, 20)
        co = np.random.randint(1, 20)
        stride = tuple(np.random.randint(1, 5, size=(2,)))
        padding = tuple(np.random.randint(0, 5, size=(2,)))
        kernel_size = tuple(np.random.randint(1, 10, size=(2,)))
        x = t.randn((b, ci, h, w), dtype=t.float64)
        weights = t.randn((co, ci, *kernel_size), dtype=t.float64)
        my_output = conv2d(x, weights, stride=stride, padding=padding)
        torch_output = t.conv2d(x, weights, stride=stride, padding=padding)
        t.testing.assert_close(my_output, torch_output)
    print("All tests in `test_conv2d` passed!")


def test_maxpool2d(my_maxpool2d, n_tests=20):
    import numpy as np

    for i in range(n_tests):
        b = np.random.randint(1, 10)
        h = np.random.randint(10, 50)
        w = np.random.randint(10, 50)
        ci = np.random.randint(1, 20)
        stride = (
            None
            if np.random.random() < 0.5
            else tuple(np.random.randint(1, 5, size=(2,)))
        )
        kernel_size = tuple(np.random.randint(1, 10, size=(2,)))
        kH, kW = kernel_size
        padding = np.random.randint(0, 1 + kH // 2), np.random.randint(0, 1 + kW // 2)
        x = t.randn((b, ci, h, w))
        my_output = my_maxpool2d(
            x,
            kernel_size,
            stride=stride,
            padding=padding,
        )
        torch_output = t.max_pool2d(
            x,
            kernel_size,
            stride=stride,  # type: ignore (None actually is allowed)
            padding=padding,
        )
        t.testing.assert_close(my_output, torch_output)
    print("All tests in `test_maxpool2d` passed!")


def test_maxpool2d_module(MaxPool2d, n_tests=20):
    import numpy as np

    for i in range(n_tests):
        b = np.random.randint(1, 10)
        h = np.random.randint(10, 50)
        w = np.random.randint(10, 50)
        ci = np.random.randint(1, 20)
        stride = (
            None
            if np.random.random() < 0.5
            else tuple(np.random.randint(1, 5, size=(2,)))
        )
        kernel_size = tuple(np.random.randint(1, 10, size=(2,)))
        kH, kW = kernel_size
        padding = np.random.randint(0, 1 + kH // 2), np.random.randint(0, 1 + kW // 2)
        x = t.randn((b, ci, h, w))
        my_output = MaxPool2d(
            kernel_size,
            stride=stride,
            padding=padding,
        )(x)

        torch_output = nn.MaxPool2d(
            kernel_size,
            stride=stride,
            padding=padding,
        )(x)
        t.testing.assert_close(my_output, torch_output)
    print("All tests in `test_maxpool2d_module` passed!")


def test_conv2d_module(Conv2d, n_tests=5):
    """
    Your weight should be called 'weight' and have an appropriate number of elements.
    """
    m = Conv2d(4, 5, (3, 3))
    assert isinstance(
        m.weight, t.nn.parameter.Parameter
    ), "Weight should be registered a parameter!"
    assert m.weight.nelement() == 4 * 5 * 3 * 3
    import numpy as np

    for i in range(n_tests):
        b = np.random.randint(1, 10)
        h = np.random.randint(10, 300)
        w = np.random.randint(10, 300)
        ci = np.random.randint(1, 20)
        co = np.random.randint(1, 20)
        stride = tuple(np.random.randint(1, 5, size=(2,)))
        padding = tuple(np.random.randint(0, 5, size=(2,)))
        kernel_size = tuple(np.random.randint(1, 10, size=(2,)))
        x = t.randn((b, ci, h, w))
        my_conv = Conv2d(
            in_channels=ci,
            out_channels=co,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        my_output = my_conv(x)
        torch_output = t.conv2d(x, my_conv.weight, stride=stride, padding=padding)
        t.testing.assert_close(my_output, torch_output)
    print("All tests in `test_conv2d_module` passed!")


def test_relu(ReLU):
    x = t.randn(10) - 0.5
    actual = ReLU()(x)
    expected = F.relu(x)
    t.testing.assert_close(actual, expected)
    print("All tests in `test_relu` passed!")


def test_flatten(Flatten):
    x = t.arange(24).reshape((2, 3, 4))
    assert Flatten(start_dim=0)(x).shape == (24,)
    assert Flatten(start_dim=1)(x).shape == (2, 12)
    assert Flatten(start_dim=0, end_dim=1)(x).shape == (6, 4)
    assert Flatten(start_dim=0, end_dim=-2)(x).shape == (6, 4)
    print("All tests in `test_flatten` passed!")


def test_linear_forward(Linear):
    """Your Linear should produce identical results to torch.nn given identical parameters."""
    x = t.rand((10, 512))
    yours = Linear(512, 64)
    assert yours.weight.shape == (
        64,
        512,
    ), f"Linear layer weights have wrong shape: {yours.weight.shape}, expected shape = (64, 512)"
    assert yours.bias.shape == (
        64,
    ), f"Linear layer bias has wrong shape: {yours.bias.shape}, expected shape = (64,)"
    official = t.nn.Linear(512, 64)
    yours.weight = official.weight
    yours.bias = official.bias
    actual = yours(x)
    expected = official(x)
    t.testing.assert_close(actual, expected)
    print("All tests in `test_linear_forward` passed!")


def test_linear_parameters(Linear):
    m = Linear(2, 3)
    params = dict(m.named_parameters())
    assert len(params) == 2, f"Your model has {len(params)} recognized Parameters"
    assert list(params.keys()) == [
        "weight",
        "bias",
    ], f"For compatibility with PyTorch, your fields should be named weight and bias, not {tuple(params.keys())}"
    print("All tests in `test_linear_parameters` passed!")


def test_linear_no_bias(Linear):

    x = t.rand((10, 512))
    yours = Linear(512, 64, bias=False)

    assert yours.bias is None, "Bias should be None when not enabled."
    assert len(list(yours.parameters())) == 1

    official = nn.Linear(512, 64, bias=False)
    yours.weight = official.weight
    actual = yours(x)
    expected = official(x)
    t.testing.assert_close(actual, expected)
    print("All tests in `test_linear_no_bias` passed!")


def test_embedding(Embedding):
    """Indexing into the embedding should fetch the corresponding rows of the embedding."""
    emb = Embedding(6, 100)
    out = emb(t.tensor([1, 3, 5], dtype=t.int64))
    t.testing.assert_close(out[0], emb.weight[1])
    t.testing.assert_close(out[1], emb.weight[3])
    t.testing.assert_close(out[2], emb.weight[5])

    emb = Embedding(30000, 500)
    t.testing.assert_close(emb.weight.std().item(), 1.0, rtol=0, atol=0.005)


def test_layernorm_mean_1d(LayerNorm):
    """If an integer is passed, this means normalize over the last dimension which should have that size."""
    x = t.randn(20, 10)
    ln1 = LayerNorm(10)
    out = ln1(x)
    max_mean = out.mean(-1).abs().max().item()
    assert max_mean < 1e-5, f"Normalized mean should be about 0, got {max_mean}"
    print(f"All tests in `test_layernorm_mean_1d` passed.")


def test_layernorm_mean_2d(LayerNorm):
    """If normalized_shape is 2D, should normalize over both the last two dimensions."""
    x = t.randn(20, 10)
    ln1 = LayerNorm((20, 10))
    out = ln1(x)
    max_mean = out.mean((-1, -2)).abs().max().item()
    assert max_mean < 1e-5, f"Normalized mean should be about 0, got {max_mean}"
    print(f"All tests in `test_layernorm_mean_2d` passed.")


def test_layernorm_std(LayerNorm):
    """If epsilon is small enough and no elementwise_affine, the output variance should be very close to 1."""
    x = t.randn(20, 10)
    ln1 = LayerNorm(10, eps=1e-11, elementwise_affine=False)
    out = ln1(x)
    var_diff = (1 - out.var(-1, unbiased=False)).abs().max().item()
    assert var_diff < 1e-6, f"Var should be about 1, off by {var_diff}"
    print(f"All tests in `test_layernorm_std` passed.")


def test_layernorm_exact(LayerNorm):
    """Your LayerNorm's output should match PyTorch for equal epsilon, up to floating point rounding error.
    This test uses float64 and the result should be extremely tight.
    """
    x = t.randn(2, 3, 4, 5)
    # Use large epsilon to make sure it fails if they forget it
    ln1 = LayerNorm((5,), eps=1e-2)
    ln2 = t.nn.LayerNorm((5,), eps=1e-2)  # type: ignore
    actual = ln1(x)
    expected = ln2(x)
    t.testing.assert_close(actual, expected)
    print(f"All tests in `test_layernorm_exact` passed.")


def test_layernorm_backward(LayerNorm):
    """The backwards pass should also match PyTorch exactly."""
    x = t.randn(10, 3)
    x2 = x.clone()
    x.requires_grad_(True)
    x2.requires_grad_(True)

    # Without parameters, should be deterministic
    ref = nn.LayerNorm(3, elementwise_affine=False)
    ref.requires_grad_(True)
    ref(x).sum().backward()

    ln = LayerNorm(3, elementwise_affine=False)
    ln.requires_grad_(True)
    ln(x2).sum().backward()
    # Use atol since grad entries are supposed to be zero here
    assert isinstance(x.grad, t.Tensor)
    assert isinstance(x2.grad, t.Tensor)
    t.testing.assert_close(x.grad, x2.grad, atol=1e-5, rtol=1e-5)
    print(f"All tests in `test_layernorm_backward` passed.")


def test_dropout_eval(Dropout):
    dropout = Dropout(p=0.1).eval()
    x = t.randn((3, 4))
    t.testing.assert_close(
        x, dropout(x), msg="Failed on eval mode (shouldn't change tensor)."
    )
    print(f"All tests in `test_dropout_eval` passed.")


def test_dropout_training(Dropout):

    dropout = Dropout(p=0).train()
    x = t.rand((1000, 1000))
    t.testing.assert_close(
        x, dropout(x), msg="Failed on p=0 (dropout shouldn't change tensor)"
    )

    for p in (0.1, 0.5, 0.9):
        dropout = Dropout(p=p).train()
        x = t.rand((1000, 1000))
        x_dropout = dropout(x)

        close_to_zero = t.abs(x_dropout) < 0.001
        fraction_close_to_zero = close_to_zero.sum() / close_to_zero.numel()
        close_to_ratio = t.abs(x_dropout / x - (1 / (1 - p))) < 0.001
        fraction_close_to_ratio = close_to_ratio.sum() / close_to_ratio.numel()

        assert (
            abs(fraction_close_to_zero - p) < 0.01
        ), f"p={p}, Wrong number of values set to zero"
        assert (
            fraction_close_to_zero + fraction_close_to_ratio > 0.995
        ), f"p={p}, Incorrect scaling"

    print(f"All tests in `test_dropout_training` passed.")


def plot_gelu(GELU):
    gelu = GELU()
    x = t.linspace(-5, 5, steps=100)
    px.line(x=x, y=gelu(x), template="ggplot2").show()


def print_param_count(*models, display_df=True, use_state_dict=False):
    """
    display_df: bool
        If true, displays styled dataframe
        if false, returns dataframe
    use_state_dict: bool
        If true, uses model.state_dict() to construct dataframe
            This will include buffers, not just params
        If false, uses model.named_parameters() to construct dataframe
            This misses out buffers (more useful for GPT)
    """
    df_list = []
    gmap_list = []
    for i, model in enumerate(models, start=1):
        iterator = (
            model.state_dict().items() if use_state_dict else model.named_parameters()
        )
        print(
            f"Model {i}, total params = {sum([param.numel() for name, param in iterator])}"
        )
        df = (
            pd.DataFrame(
                [
                    {
                        f"name_{i}": name,
                        f"shape_{i}": tuple(param.shape),
                        f"num_params_{i}": param.numel(),
                    }
                    for name, param in iterator
                ]
            )
            if (i == 1)
            else pd.DataFrame(
                [
                    {
                        f"num_params_{i}": param.numel(),
                        f"shape_{i}": tuple(param.shape),
                        f"name_{i}": name,
                    }
                    for name, param in iterator
                ]
            )
        )
        df_list.append(df)
        gmap_list.append(np.log(df[f"num_params_{i}"]))
    df = df_list[0] if len(df_list) == 1 else pd.concat(df_list, axis=1).fillna(0)
    for i in range(1, len(models) + 1):
        df[f"num_params_{i}"] = df[f"num_params_{i}"].astype(int)
    if len(models) > 1:
        param_counts = [
            df[f"num_params_{i}"].values.tolist() for i in range(1, len(models) + 1)
        ]
        if all(
            [param_counts[0] == param_counts[i] for i in range(1, len(param_counts))]
        ):
            print("All parameter counts match!")
        else:
            print("Parameter counts don't match up exactly.")
    if display_df:
        s = df.style
        for i in range(1, len(models) + 1):
            s = s.background_gradient(
                cmap="viridis", subset=[f"num_params_{i}"], gmap=gmap_list[i - 1]
            )
        with pd.option_context("display.max_rows", 1000):
            display(s)
    else:
        return df


def test_load_pretrained_weights(model, tokenizer):

    model.eval()
    device = next(model.parameters()).device

    def encode(text: str) -> t.Tensor:
        """Return a Tensor of shape (batch=1, seq)."""
        return tokenizer(text, return_tensors="pt")["input_ids"].to(device)

    prompt = "Former President of the United States of America, George"
    input_ids = encode(prompt)
    with t.inference_mode():
        output = model(input_ids)
        logits = output[0, -1] if isinstance(output, t.Tensor) else output.logits[0, -1]
    topk = t.topk(logits, k=10).indices
    next_tokens = tokenizer.batch_decode(topk.reshape(-1, 1))
    print("Prompt: ", prompt)
    print("Your model's top 10 predictions: ", next_tokens)
    assert " Washington" in next_tokens
    assert " Bush" in next_tokens
