import time
import torch
import torch.nn as nn
from typing import Sequence
import torch_pruning as tp
from mmdet.registry import MODELS

from mmdet.apis import init_detector
# from mmcv.cnn.bricks.transformer import FFN
from mmdet.models.backbones.swin import WindowMSA, SwinTransformer, SwinBlock
from mmdet.models.layers.transformer import PatchMerging


class WindowMSAPruner(tp.BasePruningFunc):
    def prune_out_channels(self, layer: nn.Module, idxs: list):
        dim = layer.embed_dims
        
        idxs_repeated = idxs + \
            [i+1*dim for i in idxs] + \
            [i+2*dim for i in idxs] + \
            [i+3*dim for i in idxs]

        tp.prune_linear_out_channels(layer.qkv, idxs_repeated)
        tp.prune_linear_out_channels(layer.proj, idxs)
        
        return layer

    def prune_in_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        tp.prune_linear_in_channels(layer.qkv, idxs)
        tp.prune_linear_in_channels(layer.proj, idxs)
        return layer

    def get_out_channels(self, layer):
        return layer.embed_dims

    def get_in_channels(self, layer):
        return layer.embed_dims


class PatchMergingPruner(tp.BasePruningFunc):

    def prune_out_channels(self, layer: nn.Module, idxs: list):
        tp.prune_linear_out_channels(layer.reduction, idxs)
        return layer

    def prune_in_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        dim = int(layer.reduction.in_features / 4)

        idxs_repeated = idxs + \
            [i+dim for i in idxs] + \
            [i+2*dim for i in idxs] + \
            [i+3*dim for i in idxs]

        tp.prune_linear_in_channels(layer.reduction, idxs_repeated)
        tp.prune_layernorm_out_channels(layer.norm, idxs_repeated)
        return layer

    def get_out_channels(self, layer):
        return layer.reduction.out_features

    def get_in_channels(self, layer):
        in_dim = int(layer.reduction.in_features / 4)
        return in_dim



def load_model():
    gd_weights = "/mnt/disks/ext/exps/mini_coco/grounding_dino_swin-t_finetune_custom_dataset/E1/best_coco_bbox_mAP_epoch_2.pth"
    model_cfg = "/mnt/disks/ext/repos/mmdetection/configs/grounding_dino/grounding_dino_swin-t_finetune_32xb1_1x_coco.py"
    model = init_detector(model_cfg, gd_weights)
    model.eval().cpu()
    model.train()
    model = model.backbone
    return model


def load_model_from_cfg():
    # load config of type: [mmengine.config.config.ConfigDict]
    config_dict_path = "/mnt/disks/ext/gd_checkpoints/mmdet_swin_t_backbone_cfg_dict.pth"
    config_dict = torch.load(config_dict_path)
    backbone = MODELS.build(config_dict)
    backbone.load_state_dict( torch.load("/mnt/disks/ext/gd_checkpoints/model_backbone_state_dict.pth") )
    return backbone

def load_from_file():
    model = torch.load("/mnt/disks/ext/gd_checkpoints/swin_loaded_weights_from_gd.pth")
    model.out_indices = (1, 2, 3)
    return model

# model = load_model_from_cfg()
# model = load_from_file()

model = load_model()

for p in model.parameters():
    p.requires_grad_(True)

for m in model.modules():
    if isinstance(m, SwinBlock):
        m.with_cp = False

example_inputs = torch.randn([1, 3, 800, 1333])

start_time = time.time()
# torch.cuda.synchronize()
outputs = model(example_inputs)
# torch.cuda.synchronize()
end_time = time.time()
forward_time = end_time - start_time
print("Base forward_time = ", forward_time)
print("Base fps = ", 1.0 / forward_time)


print(model)

imp = tp.importance.MagnitudeImportance(p=2, group_reduction="mean")
# imp = tp.importance.GroupNormImportance()
# imp = tp.importance.LAMPImportance()
# imp = tp.importance.RandomImportance()
base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
print("Base Macs: %f M, Base Params: %f M"%(base_macs/1e9, base_params/1e6))

num_heads = {}

ignored_layers = []
# All heads should be pruned simultaneously, so we group channels by head.
for m in model.modules():
    if isinstance(m, WindowMSA):
        num_heads[m.qkv] = m.num_heads

output_transform = None

# 0.5 or 0.25
pruning_ratio = 0.5
pruner = tp.pruner.MetaPruner(
                model, 
                example_inputs, 
                global_pruning=False, # If False, a uniform pruning ratio will be assigned to different layers.
                importance=imp, # importance criterion for parameter selection
                iterative_steps=1, # the number of iterations to achieve target pruning ratio
                pruning_ratio=pruning_ratio,
                num_heads=num_heads,
                prune_head_dims=False,
                output_transform=output_transform,
                ignored_layers=ignored_layers,
                customized_pruners={PatchMerging: PatchMergingPruner(), WindowMSA: WindowMSAPruner()},
                root_module_types=(nn.Linear, nn.LayerNorm, PatchMerging, WindowMSA),
                
            )

for g in pruner.step(interactive=True):
    g.prune()

print(model)

# Modify the attention head size and all head size aftering pruning
num_features = []
for m in model.modules():
    if isinstance(m, WindowMSA):
        if m.qkv.in_features not in num_features:
            num_features.append(m.qkv.in_features)

        head_embed_dims = m.embed_dims
        head_embed_dims = head_embed_dims // m.num_heads
        m.scale = head_embed_dims ** -0.5
        m.embed_dims = m.qkv.out_features

for m in model.modules():
    if isinstance(m, SwinTransformer):
        m.num_features = num_features

for m in model.modules():
    if isinstance(m, WindowMSA):
        print("new embed dim: ", m.embed_dims)

example_inputs = torch.randn([1, 3, 800, 1333])


start_time = time.time()
# torch.cuda.synchronize()
test_output = model(example_inputs)
# torch.cuda.synchronize()
end_time = time.time()
forward_time = end_time - start_time
print("Pruned forward_time = ", forward_time)
print("Pruned fps = ", 1.0 / forward_time)

pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
print("Base MACs: %f G, Pruned MACs: %f G"%(base_macs/1e9, pruned_macs/1e9))
print("Base Params: %f M, Pruned Params: %f M"%(base_params/1e6, pruned_params/1e6))


save_as = f"/mnt/disks/ext/gd_checkpoints/gd_backbone_Pruned_{int(pruning_ratio * 100)}.pth"
model.zero_grad()
torch.save(model, save_as)

