import fvcore.common.config
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from functools import partial
from einops import rearrange, reduce
from denoising_diffusion_pytorch.efficientnet import efficientnet_b7, EfficientNet_B7_Weights
from denoising_diffusion_pytorch.resnet import resnet101, ResNet101_Weights
from denoising_diffusion_pytorch.swin_transformer import swin_b, Swin_B_Weights
from denoising_diffusion_pytorch.vgg import vgg16, VGG16_Weights
from denoising_diffusion_pytorch.mask_cond_unet import Unet
import torch_pruning as tp


def pruning(model, pr):
    example_inputs = [torch.randn(1, 3, 64, 64),torch.tensor([0.5124]),torch.rand(1, 3, 256, 256)]
    imp = tp.importance.GroupNormImportance(p=2) 
    ignored_layers = []
    for m in model.modules():
        if 'BatchNorm2d' in m.__class__.__name__:
            ignored_layers.append(m)
        if 'PreNorm' in m.__class__.__name__:
            print(m.fn)
            ignored_layers.append(m.fn)
    print(ignored_layers)
    pruner = tp.pruner.MetaPruner( # We can always choose MetaPruner if sparse training is not required.
        model,
        example_inputs,
        importance=imp,
        pruning_ratio=pr, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        # pruning_ratio_dict = {model.conv1: 0.2, model.layer2: 0.8}, # customized pruning ratios for layers or blocks
        ignored_layers=ignored_layers,
        round_to=8, # It's recommended to round dims/channels to 4x or 8x for acceleration. Please see: https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html
        )   
    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    pruner.step()
    macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"MACs: {base_macs/1e9} G -> {macs/1e9} G, #Params: {base_nparams/1e6} M -> {nparams/1e6} M")
    return model


if __name__ == "__main__":
    # resnet = resnet101(weights=ResNet101_Weights)
    # effnet = efficientnet_b7(weights=EfficientNet_B7_Weights)
    # effnet = efficientnet_b7(weights=None)
    # x = torch.rand(1, 3, 320, 320)
    # y = effnet(x)
    
    model = Unet(dim=128, dim_mults=(1, 2, 4, 4),
                 cond_dim=128,
                 cond_dim_mults=(2, 4, ),
                 channels=1,
                 window_sizes1=[[8, 8], [4, 4], [2, 2], [1, 1]],
                 window_sizes2=[[8, 8], [4, 4], [2, 2], [1, 1]],
                 cfg=fvcore.common.config.CfgNode({'cond_pe': False, 'input_size': [80, 80],
                      'cond_feature_size': (32, 128), 'cond_net': 'vgg',
                      'num_pos_feats': 96})
                 )
    example_inputs = [torch.randn(1, 1, 64, 64),torch.tensor([0.5124]),torch.rand(1, 3, 320, 320)]
    # imp = tp.importance.GroupNormImportance(p=2) 
    # ignored_layers = []
    # for m in model.modules():
    #     if isinstance(m, nn.Embedding) or \
    #        isinstance(m, nn.MaxPool2d) or isinstance(m, nn.AvgPool2d) or \
    #        isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d) or \
    #        isinstance(m, nn.GroupNorm) or 'Norm' in m.__class__.__name__:
    #         ignored_layers.append(m)
    # pruner = tp.pruner.MetaPruner( # We can always choose MetaPruner if sparse training is not required.
    #     model,
    #     example_inputs,
    #     importance=imp,
    #     pruning_ratio=0.9, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
    #     # pruning_ratio_dict = {model.conv1: 0.2, model.layer2: 0.8}, # customized pruning ratios for layers or blocks
    #     ignored_layers=ignored_layers,
    #     round_to=8, # It's recommended to round dims/channels to 4x or 8x for acceleration. Please see: https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html
    #     )   
    #base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    # pruner.step()
    # Print all keys in the model's state dict
    print("Model state dict keys:")
    for key in model.state_dict().keys():
        print(key)
    def calculate_model_size(model):
        total_params = sum(p.numel() for p in model.parameters())
        print(f'Total parameters: {total_params:,}')
        print(f'Total parameters in millions: {total_params/1e6:.2f}M')
        
    print("Model size before pruning:")
    calculate_model_size(model)
    from prune import structured_pruning
    model =structured_pruning(model, 0.9)
    print("Model size after pruning:")
    calculate_model_size(model)
    #macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
    #print(f"MACs: {base_macs/1e9} G -> {macs/1e9} G, #Params: {base_nparams/1e6} M -> {nparams/1e6} M")
    x = torch.rand(1, 1, 64, 64)
    mask = torch.rand(1, 3, 320, 320)
    time = torch.tensor([0.5124])
    with torch.no_grad():
        y = model(x, time, mask)
    pass