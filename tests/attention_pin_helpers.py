"""Stand-in pinned attention modules for the attention pin tests (issue #30).

A ComfyAttention carries ``config`` (the block's JSON) and ``function`` (the
kernel ComfyUI bound, or None). The bound functions here carry ComfyUI's names,
which is how the pack tells the two comfy-kitchen kernels apart; canary seam 39
holds those names against the real tree.
"""
from __future__ import annotations

import functools
import types

import torch


def attention_comfy_kitchen_sol(q, k, v, heads, *, tau=1.0, **kwargs):
    """Named as ComfyUI's Sol binding is; never called here."""
    raise AssertionError("not run")


def attention_comfy_kitchen_int8(q, k, v, heads, **kwargs):
    """Named as ComfyUI's int8 binding is; never called here."""
    raise AssertionError("not run")


class Attention(torch.nn.Module):
    """A ComfyAttention stand-in: the config JSON and the bound kernel."""

    def __init__(self, config=None, function=None):
        super().__init__()
        self.config = config
        self.function = function


def sol(tau=0.7, bound=True):
    return Attention({"attention": "comfy_kitchen_sol", "tau": tau},
                     functools.partial(attention_comfy_kitchen_sol, tau=tau) if bound else None)


def int8(bound=True):
    return Attention({"attention": "comfy_kitchen_int8"},
                     attention_comfy_kitchen_int8 if bound else None)


class Block(torch.nn.Module):
    def __init__(self, self_attn, cross_attn=None):
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.comfy_attention = self_attn
        self.self_attn.q = torch.nn.Linear(2, 2)
        if cross_attn is not None:
            self.cross_attn = torch.nn.Module()
            self.cross_attn.comfy_attention = cross_attn


class CountedModel(torch.nn.Module):
    """A diffusion model whose module walks are counted."""

    def __init__(self, blocks):
        super().__init__()
        self.blocks = torch.nn.ModuleList(blocks)
        self.walks = 0

    def named_modules(self, *args, **kwargs):
        self.walks += 1
        return super().named_modules(*args, **kwargs)


def make_patcher(blocks, object_patches=None):
    model = types.SimpleNamespace(diffusion_model=CountedModel(blocks))
    return types.SimpleNamespace(model=model, object_patches=dict(object_patches or {}))
