from fnmatch import fnmatch
import copy
import math
import numpy as np
import torch
import torch.nn as nn




def replace_module(model, name, new_m):
    r"""Replace the module <name> in <model> with <new_m>
    E.g., 'module.layer1.0.conv1'
    ==> model.__getattr__('module').__getattr__("layer1").__getitem__(0).__setattr__('conv1', new_m)
    """
    obj = model
    segs = name.split(".")
    for ix, s in enumerate(segs):
        if ix == len(segs) - 1:  # the last one
            if s.isdigit():
                obj.__setitem__(int(s), new_m)
            else:
                obj.__setattr__(s, new_m)
            return
        if s.isdigit():
            obj = obj.__getitem__(int(s))
        else:
            obj = obj.__getattr__(s)


def structured_pruning(model, pr=0.4375, reinit=False, layer_prefix='model.'):
    """Structured pruning of a given model.
    Args:
        model: nn.Module, model to prune.
        pr: Pruning ratio. A float to indiate layerwise sparsity.
        reinit: Reinitialize the weights of the pruned model.
    
    Returns:
        new_model: The pruned model.
    """
    weight_types = (nn.Conv2d, nn.Linear)
    embedding_type = (nn.Embedding,)
    norm_types = (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)
    learnable_types = weight_types + embedding_type + norm_types

    # Name of the layers that are prunable.
    target_layers = (
        layer_prefix + 'init_conv_mask.', 
        layer_prefix + 'final_layernorm', 
        layer_prefix + 'embed_tokens*', 
        layer_prefix + 'mm_projector.2*', 
        'lm_head'
    )

    # For first and last layers, only one dimension is shrinked.
    first_layers = (
        layer_prefix + 'embed_tokens',
        layer_prefix + 'mm_projector.2*',
    )
    last_layers = (
        'lm_head',
    )

    # new_model = copy.deepcopy(model)  # @huanwangx: Cannot use this. Don't know why.
    new_model = model

    # Adjust num of heads in attention.
    for name, m in new_model.named_modules():
        if fnmatch(name, layer_prefix + 'layers.*.self_attn'):
            num_heads = m.num_heads - int(np.ceil(m.num_heads * pr))
            num_key_value_heads = m.num_key_value_heads - int(np.ceil(m.num_key_value_heads * pr))
            hidden_size = m.hidden_size - int(np.ceil(m.hidden_size * pr))
            print(f'{name}: num_heads adjusted from {m.num_heads} to {num_heads}')
            print(f'{name}: num_key_value_heads adjusted from {m.num_key_value_heads} to {num_key_value_heads}')
            print(f'{name}: hidden_size adjusted from {m.hidden_size} to {hidden_size}')
            m.num_heads = num_heads
            m.num_key_value_heads = num_key_value_heads
            m.hidden_size = hidden_size

    # Pruning layer by layer.
    old_params, new_params = {}, {}
    for name, m in new_model.named_modules():
        if isinstance(m, learnable_types):
            old_params[name] = m.weight.numel()
            prune_current_layer = any([fnmatch(name, pattern) for pattern in target_layers])
            #print(f"Pruning layer: {name}, Prune current layer: {prune_current_layer}")
            prune_current_layer = True
            if not prune_current_layer:
                new_params[name] = m.weight.numel()
            else:
                w = m.weight
                bias = hasattr(m, "bias") and m.bias is not None
                num_filters = w.shape[0]
                num_channels = w.shape[1] if len(w.shape) > 1 else 0
                num_kept_filters = int(num_filters - np.ceil(pr * num_filters))
                num_kept_channels = int(num_channels - np.ceil(pr * num_channels))
                
                # Handle the first / last layers.
                for pattern in first_layers:
                    if fnmatch(name, pattern):
                        if isinstance(m, norm_types + embedding_type):
                            num_kept_filters = num_filters
                        else:
                            num_kept_channels = num_channels
                for pattern in last_layers:
                    if fnmatch(name, pattern):
                        num_kept_filters = num_filters
                print(f'Current layer to prune: {name} | Shape: {list(w.shape)} | '
                      f'num_kept_filters: {num_kept_filters} '
                      f'num_kept_channels: {num_kept_channels} '
                      f'PR: {pr}'
                )

                # Evenly-spaced pruning.
                kept_filters = []
                space = num_filters // num_kept_filters
                for i in range(num_kept_filters):
                    kept_filters.append(i * space)
                
                kept_channels = []
                if num_channels > 0:
                    space = num_channels // num_kept_channels
                    for i in range(num_kept_channels):
                        kept_channels.append(i * space)
                
                if isinstance(m, nn.Conv2d):
                    new_layer = nn.Conv2d(
                        num_channels,
                        len(kept_filters),
                        m.kernel_size,
                        m.stride,
                        m.padding,
                        m.dilation,
                        m.groups,
                        bias,
                    )
                    kept_weights = m.weight.data[kept_filters][:, kept_channels, :, :]

                    if not reinit:
                        new_layer.weight.data.copy_(
                            kept_weights
                        )  # load weights into the new module
                        if bias:
                            kept_bias = m.bias.data[kept_filters]
                            new_layer.bias.data.copy_(kept_bias)
                    else:
                        print(
                            f"Layer {name} is reinited when building the new model!"
                        )

                elif isinstance(m, nn.Linear):
                    new_layer = nn.Linear(
                        in_features=len(kept_channels),
                        out_features=len(kept_filters),
                        bias=bias,
                    )
                    kept_weights = m.weight.data[kept_filters][:, kept_channels]

                    if not reinit:
                        new_layer.weight.data.copy_(
                            kept_weights
                        )  # load weights into the new module
                        if bias:
                            kept_bias = m.bias.data[kept_filters]
                            new_layer.bias.data.copy_(kept_bias)
                    else:
                        print(
                            f"Layer {name} is reinited when building the new model!"
                        )

                elif isinstance(m, nn.Embedding):
                    # https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
                    new_layer = nn.Embedding(
                        num_embeddings=m.num_embeddings,
                        embedding_dim=num_kept_channels,
                        padding_idx=m.padding_idx,
                    )
                    kept_weights = m.weight.data[kept_filters][:, kept_channels]

                    if not reinit:
                        new_layer.weight.data.copy_(
                            kept_weights
                        )  # load weights into the new module
                        if bias:
                            kept_bias = m.bias.data[kept_filters]
                            new_layer.bias.data.copy_(kept_bias)
                    else:
                        print(
                            f"Layer {name} is reinited when building the new model!"
                        )

                elif isinstance(m, nn.BatchNorm2d):
                    new_layer = nn.BatchNorm2d(
                        len(kept_filter),
                        eps=m.eps,
                        momentum=m.momentum,
                        affine=m.affine,
                        track_running_stats=m.track_running_stats,
                    )

                    # copy bn weight and bias
                    new_layer.weight.data.copy_(m.weight.data[kept_filter])
                    new_layer.bias.data.copy_(m.bias.data[kept_filter])

                    # copy bn running stats
                    new_layer.running_mean.data.copy_(m.running_mean[kept_filter])
                    new_layer.running_var.data.copy_(m.running_var[kept_filter])
                    new_layer.num_batches_tracked.data.copy_(m.num_batches_tracked)

                elif isinstance(m, nn.GroupNorm):
                    kept_filters = list(range(m.num_channels))
                    new_layer = nn.GroupNorm(
                        num_groups=m.num_groups,
                        num_channels=len(kept_filters),
                        eps=m.eps,
                        affine=m.affine,
                    )
                    new_layer.weight.data.copy_(m.weight.data[kept_filters])
                    new_layer.bias.data.copy_(m.bias.data[kept_filters])

                elif isinstance(m, nn.LayerNorm):
                    # See https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
                    new_layer = nn.LayerNorm(
                        len(kept_filters),
                        eps=m.eps,
                        elementwise_affine=m.elementwise_affine,
                    )
                    new_layer.weight.data.copy_(m.weight.data[kept_filters])
                    new_layer.bias.data.copy_(m.bias.data[kept_filters])

                elif isinstance(m, nn.MultiheadAttention):
                    raise NotImplementedError
                
                else:
                    raise NotImplementedError

                # Load the new_layer into the new_model
                new_layer = new_layer.cuda().half()
                replace_module(new_model, name, new_layer)
                new_params[name] = new_layer.weight.numel()

    # Get model params.
    old_total_params = sum(list(old_params.values())) / 1e9
    new_total_params = sum(list(new_params.values())) / 1e9
    old_llm_params, old_vt_params = 0, 0
    new_llm_params, new_vt_params = 0, 0
    for k in old_params:
        if k.startswith(layer_prefix + 'layers'):
            old_llm_params += old_params[k] / 1e9
            new_llm_params += new_params[k] / 1e9
        elif k.startswith(layer_prefix + 'vision_tower'):
            old_vt_params += old_params[k] / 1e9
            new_vt_params += new_params[k] / 1e9
    start_line = '-' * 10 + ' Pruning Summary ' + '-' * 10
    print('\n' + start_line)
    print(f'Layerwise pruning ratio: {pr}')
    print(f'Old total params: {old_total_params:.2f}G, '
          f'New total params: {new_total_params:.2f}G '
          f'(compression ratio: {old_total_params / new_total_params:.2f}x) '
    )
    # print(f'Old   llm params: {old_llm_params:.2f}G, '
    #       f'New   llm params: {new_llm_params:.2f}G '
    #       f'(compression ratio: {old_llm_params / new_llm_params:.2f}x) '
    # )
    # print(f'Old vt params: {old_vt_params:.2f}G, '
    #       f'New vt params: {new_vt_params:.2f}G '
    #       f'(compression ratio: {old_vt_params / new_vt_params:.2f}x) '
    # )
    print('-' * len(start_line) + '\n')
    return new_model


