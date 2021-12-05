model_config_file = "/content/open_clip/src/training/model_configs/RN101.json"
gpu =0
resume= "/content/rn101.pt"
batch_size = 32
wds_shards= "/content/{00004..00004}.tar"

import hashlib
import os

#os.chdir("/content/open_clip_inference/")
import urllib
import warnings
from typing import Union, List

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop
from tqdm import tqdm


from clip.model import build_model
from clip.tokenizer import SimpleTokenizer as _Tokenizer



import time
import logging
from time import gmtime, strftime
from pathlib import Path
import json

# wandb
import torch
from torch import optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler

from clip.clip import _transform, load
from clip.model import convert_weights, CLIP
from training.train import train, evaluate
from training.data import get_data
from training.params import parse_args
from training.logger import setup_primary_logging, setup_worker_logging
from training.scheduler import cosine_lr

import torch.nn as nn

from clip.clip import tokenize
import webdataset as wds
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
import torchvision.datasets as datasets



shardlist = wds.PytorchShardList(wds_shards, shuffle=False)


def load_model(checkpoint_pt, checkpoint_json,gpu=None):

    def preprocess_txt(text):
        return tokenize([str(text)])[0]

    def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit=True, is_train=False, pretrained=True):
        """Load a CLIP model
        Parameters
        ----------
        name : str
            A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
        device : Union[str, torch.device]
            The device to put the loaded model
        jit : bool
            Whether to load the optimized JIT model (default) or more hackable non-JIT model.
        Returns
        -------
        model : torch.nn.Module
            The CLIP model
        preprocess : Callable[[PIL.Image], torch.Tensor]
            A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
        """
        if name in _MODELS:
            model_path = _download(_MODELS[name])
        elif os.path.isfile(name):
            model_path = name
        else:
            raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            # loading saved state dict
            if jit:
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                jit = False
            state_dict = torch.load(model_path, map_location="cpu")

        if not jit:
            try:
                model = build_model(state_dict or model.state_dict()).to(device)
            except KeyError:
                sd = {k[7:]: v for k,v in state_dict["state_dict"].items()}
                model = build_model(sd).to(device)

            if str(device) == "cpu":
                model.float()
            return model, \
                  _transform(model.visual.input_resolution, is_train=True), \
                  _transform(model.visual.input_resolution, is_train=False)

        # patch the device names
        device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
        device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

        def patch_device(module):
            graphs = [module.graph] if hasattr(module, "graph") else []
            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("prim::Constant"):
                    if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                        node.copyAttributes(device_node)

        model.apply(patch_device)
        patch_device(model.encode_image)
        patch_device(model.encode_text)

        # patch dtype to float32 on CPU
        if str(device) == "cpu":
            float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
            float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
            float_node = float_input.node()

            def patch_float(module):
                graphs = [module.graph] if hasattr(module, "graph") else []
                if hasattr(module, "forward1"):
                    graphs.append(module.forward1.graph)

                for graph in graphs:
                    for node in graph.findAllNodes("aten::to"):
                        inputs = list(node.inputs())
                        for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                            if inputs[i].node()["value"] == 5:
                                inputs[i].node().copyAttributes(float_node)

            model.apply(patch_float)
            patch_float(model.encode_image)
            patch_float(model.encode_text)

            model.float()

        return model, \
              _transform(model.input_resolution.item(), is_train=True), \
              _transform(model.input_resolution.item(), is_train=False)






    with open(model_config_file, 'r') as f:
                model_info = json.load(f)
    model = CLIP(**model_info)
    convert_weights(model)
    #preprocess_train = _transform(model.visual.input_resolution, is_train=True)
    preprocess_val = _transform(model.visual.input_resolution, is_train=False)

    


    if gpu is None:
            model.float()
            checkpoint = torch.load(resume, map_location=torch.device('cpu') )
    else: 
          # Map model to be loaded to specified single gpu.
          loc = "cuda:{}".format(gpu)
          checkpoint = torch.load(resume, map_location=loc)
    start_epoch = checkpoint["epoch"]
    sd = checkpoint["state_dict"]
    #if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
    sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd)


    cudnn.benchmark = True
    cudnn.deterministic = False
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    if gpu is not None:
        loss_img = loss_img.cuda(gpu)
        loss_txt = loss_txt.cuda(gpu)

    #print(model.eval())
    return model.encode_image, model.encode_text, preprocess_val, preprocess_txt  





#m,p = load_model(checkpoint_pt=resume, checkpoint_json= model_config_file, gpu=0)

