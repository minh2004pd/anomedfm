# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
from typing import Union

from models.discrete_unet import DiscreteUNetModel
from models.ema import EMA
from models.unet import UNetModel

MODEL_CONFIGS = {
    "imagenet": {
        "in_channels": 3,
        "model_channels": 192,
        "out_channels": 3,
        "num_res_blocks": 3,
        "attention_resolutions": [2, 4, 8],
        "dropout": 0.1,
        "channel_mult": [1, 2, 3, 4],
        "num_classes": 1000,
        "use_checkpoint": False,
        "num_heads": 4,
        "num_head_channels": 64,
        "use_scale_shift_norm": True,
        "resblock_updown": True,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
    "imagenet_discrete": {
        "in_channels": 3,
        "model_channels": 192,
        "out_channels": 3,
        "num_res_blocks": 4,
        "attention_resolutions": [2, 4, 8],
        "dropout": 0.2,
        "channel_mult": [2, 3, 4, 4],
        "num_classes": 1000,
        "use_checkpoint": False,
        "num_heads": -1,
        "num_head_channels": 64,
        "use_scale_shift_norm": True,
        "resblock_updown": True,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
    "cifar10": {
        "in_channels": 3,
        "model_channels": 128,
        "out_channels": 3,
        "num_res_blocks": 4,
        "attention_resolutions": [2],
        "dropout": 0.3,
        "channel_mult": [2, 2, 2],
        "conv_resample": False,
        "dims": 2,
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": 1,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
    "cifar10_discrete": {
        "in_channels": 3,
        "model_channels": 96,
        "out_channels": 3,
        "num_res_blocks": 5,
        "attention_resolutions": [2],
        "dropout": 0.4,
        "channel_mult": [3, 4, 4],
        "conv_resample": False,
        "dims": 2,
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": -1,
        "num_head_channels": 64,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
    "simple_shape": {
        "in_channels": 1,
        "model_channels": 32,
        "out_channels": 1,
        "num_res_blocks": 2,
        "attention_resolutions": [2],
        "dropout": 0.1,
        "channel_mult": [1, 2],
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": 1,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
    "brats": {
        "in_channels": 4,
        "model_channels": 64,  # Increased capacity slightly for MRI complexity
        "out_channels": 4,
        "num_res_blocks": 3,
        "attention_resolutions": [4, 8],
        "dropout": 0.1,
        "channel_mult": [1, 2, 4, 4], # Deeper net
        "num_classes": 2, # Healthy vs Diseased
        "use_checkpoint": True,
        "num_heads": -1,
        "num_head_channels": 64,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": True,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
    "bratsv2": {
        "in_channels": 4,
        "model_channels": 64,  # Increased capacity slightly for MRI complexity
        "out_channels": 4,
        "num_res_blocks": 3,
        # Attention at 4x, 8x, 16x downsample (i.e. 64x64, 32x32, 16x16).
        # Bottleneck (16x16) attention lets every spatial token see the whole
        # field — important for large peripheral tumours that need global
        # context to be flagged as "non-healthy" by the CFG-guided decode.
        "attention_resolutions": [4, 8, 16],
        "dropout": 0.1,
        # 5 levels: 256 -> 128 -> 64 -> 32 -> 16. The extra 16x16 level doubles
        # the effective receptive field at near-zero compute cost (1.4x params,
        # ~0% step-time slowdown vs the old 4-level UNet) and gives the network
        # a true global-feature stage. UNet skip connections preserve local
        # detail, so a 16x16 bottleneck is not a lossy compression in practice.
        "channel_mult": [1, 2, 4, 4, 4],
        "num_classes": 2, # Healthy vs Diseased
        "use_checkpoint": True,
        "num_heads": -1,
        "num_head_channels": 64,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": True,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
    "brats_healthy": {
        "in_channels": 4,
        "model_channels": 64,
        "out_channels": 4,
        "num_res_blocks": 3,
        "attention_resolutions": [4, 8],
        "dropout": 0.1,
        "channel_mult": [1, 2, 4, 4],
        "num_classes": None,  # Unconditional — no class embedding
        "use_checkpoint": True,
        "num_heads": -1,
        "num_head_channels": 64,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": True,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },

    "mnist": {
        "in_channels": 1,
        "model_channels": 32,
        "out_channels": 1,
        "num_res_blocks": 2,
        "attention_resolutions": [2],
        "dropout": 0.1,
        "channel_mult": [1, 2],
        "num_classes": 10,
        "use_checkpoint": False,
        "num_heads": 1,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
}


def instantiate_model(
    architechture: str, is_discrete: bool, use_ema: bool
) -> Union[UNetModel, DiscreteUNetModel]:
    assert (
        architechture in MODEL_CONFIGS
    ), f"Model architecture {architechture} is missing its config."

    if is_discrete:
        if architechture + "_discrete" in MODEL_CONFIGS:
            config = MODEL_CONFIGS[architechture + "_discrete"]
        else:
            config = MODEL_CONFIGS[architechture]
        model = DiscreteUNetModel(
            vocab_size=257,
            **config,
        )
    else:
        model = UNetModel(**MODEL_CONFIGS[architechture])

    if use_ema:
        return EMA(model=model)
    else:
        return model
