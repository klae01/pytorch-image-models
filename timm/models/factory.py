from urllib.parse import urlsplit, urlunsplit
import os
import re

import torch
import torch.nn as nn

from .registry import is_model, is_model_in_modules, model_entrypoint
from .helpers import load_checkpoint
from .layers import set_layer_config
from .hub import load_model_config_from_hf


def parse_model_name(model_name):
    model_name = model_name.replace('hf_hub', 'hf-hub')  # NOTE for backwards compat, to deprecate hf_hub use
    parsed = urlsplit(model_name)
    assert parsed.scheme in ('', 'timm', 'hf-hub')
    if parsed.scheme == 'hf-hub':
        # FIXME may use fragment as revision, currently `@` in URI path
        return parsed.scheme, parsed.path
    else:
        model_name = os.path.split(parsed.path)[-1]
        return 'timm', model_name


def safe_model_name(model_name, remove_source=True):
    def make_safe(name):
        return ''.join(c if c.isalnum() else '_' for c in name).rstrip('_')
    if remove_source:
        model_name = parse_model_name(model_name)[-1]
    return make_safe(model_name)


class WDropout:
    def __init__(self, module: nn.Module):
        self.module = module
        self.__WD_params = {}
        self.progress = False

    def build_dropout(self, rule, dropout):
        for n, p in self.module.named_parameters():
            if p.ndim > 1 and re.match(rule, n):
                self.__WD_params[n] = dropout

    def get_parent_module(self, name):
        cls_parent = ".".join(name.split(".")[:-1])
        cls_child = name.split(".")[-1]
        if cls_parent:
            return self.module.get_submodule(cls_parent), cls_child
        else:
            return self.module, cls_child

    def get_parameter(self, name):
        parent, child = self.get_parent_module(name)
        return getattr(parent, child)

    def set_parameter(self, name, new_params):
        parent, child = self.get_parent_module(name)
        setattr(parent, child, new_params)

    def del_parameter(self, name):
        parent, child = self.get_parent_module(name)
        delattr(parent, child)

    def pre_ff(self):
        assert not self.progress
        self.progress = True
        TR = self.module.training
        for n, dropout in self.__WD_params.items():
            w = self.get_parameter(f"{n}")
            self.set_parameter(f"{n}_raw", w)
            self.del_parameter(f"{n}")
            w = torch.nn.functional.dropout(w, dropout, TR)
            self.set_parameter(f"{n}", w)

    def post_ff(self):
        assert self.progress
        self.progress = False
        for n in self.__WD_params:
            w = self.get_parameter(f"{n}_raw")
            self.set_parameter(f"{n}", w)
            self.del_parameter(f"{n}_raw")


def WDropout_Model(model, dropout):
    WDmodel = WDropout(model)
    WDmodel.build_dropout("^.*$", dropout)
    original_forward = model.forward
    def forward(X):
        nonlocal WDmodel
        WDmodel.pre_ff()
        Y = original_forward(X)
        WDmodel.post_ff()
        return Y

    setattr(model, "forward", forward)


def create_model(
        model_name,
        pretrained=False,
        pretrained_cfg=None,
        checkpoint_path='',
        scriptable=None,
        exportable=None,
        no_jit=None,
        **kwargs):
    """Create a model

    Args:
        model_name (str): name of model to instantiate
        pretrained (bool): load pretrained ImageNet-1k weights if true
        checkpoint_path (str): path of checkpoint to load after model is initialized
        scriptable (bool): set layer config so that model is jit scriptable (not working for all models yet)
        exportable (bool): set layer config so that model is traceable / ONNX exportable (not fully impl/obeyed yet)
        no_jit (bool): set layer config so that model doesn't utilize jit scripted layers (so far activations only)

    Keyword Args:
        drop_rate (float): dropout rate for training (default: 0.0)
        global_pool (str): global pool type (default: 'avg')
        **: other kwargs are model specific
    """
    # Parameters that aren't supported by all models or are intended to only override model defaults if set
    # should default to None in command line args/cfg. Remove them if they are present and not set so that
    # non-supporting models don't break and default args remain in effect.
    drop_connect = kwargs.pop("drop_connect_rate", None)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    model_source, model_name = parse_model_name(model_name)
    if model_source == 'hf-hub':
        # FIXME hf-hub source overrides any passed in pretrained_cfg, warn?
        # For model names specified in the form `hf-hub:path/architecture_name@revision`,
        # load model weights + pretrained_cfg from Hugging Face hub.
        pretrained_cfg, model_name = load_model_config_from_hf(model_name)

    if not is_model(model_name):
        raise RuntimeError('Unknown model (%s)' % model_name)

    create_fn = model_entrypoint(model_name)
    with set_layer_config(scriptable=scriptable, exportable=exportable, no_jit=no_jit):
        model = create_fn(pretrained=pretrained, pretrained_cfg=pretrained_cfg, **kwargs)

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path)

    if drop_connect:
        WDropout_Model(model, drop_connect)

    return model
