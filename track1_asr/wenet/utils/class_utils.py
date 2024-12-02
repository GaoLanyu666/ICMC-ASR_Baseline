#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright [2023-11-28] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>
import torch
from torch.nn import BatchNorm1d, LayerNorm
# from wenet.paraformer.embedding import ParaformerPositinoalEncoding
from wenet.transformer.norm import RMSNorm
from wenet.transformer.positionwise_feed_forward import (
    GatedVariantsMLP, MoEFFNLayer, PositionwiseFeedForward,
    SpikePositionwiseFeedForward, SpikeLengthPositionwiseFeedForward,
    SpikeAudioLengthPositionwiseFeedForward, MS_MLP)

from wenet.transformer.swish import Swish
from wenet.transformer.subsampling import (
    LinearNoSubsampling,
    EmbedinigNoSubsampling,
    Conv1dSubsampling2,
    Conv2dSubsampling4,
    Conv2dSubsampling4_Spike,
    Conv2dSubsampling4_Spike_OriginMask,
    Conv2dSubsampling4_Spike_T_OriginMask,
    Conv2dSubsampling4_Spike_Length_OriginMask,
    Conv2dSubsampling4_Spike_Audio_Length_OriginMask,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    StackNFramesSubsampling,
)
from wenet.efficient_conformer.subsampling import Conv2dSubsampling2
from wenet.squeezeformer.subsampling import DepthwiseConv2dSubsampling4
from wenet.transformer.embedding import (PositionalEncoding,
                                         RelPositionalEncoding,
                                         RopePositionalEncoding,
                                         WhisperPositionalEncoding,
                                         LearnablePositionalEncoding,
                                         NoPositionalEncoding)
from wenet.transformer.attention import (MultiHeadedAttention, SSA, SSA_BN, SSA_ConvBN, SSA_LN, SSCA_LN,
                                         RelPositionSSA_BN,
                                         RelPositionSSA_ConvBN,
                                         RelPositionSSA_LN,
                                         MultiHeadedCrossAttention, SSCA,
                                         RelPositionMultiHeadedAttention,
                                         RopeMultiHeadedAttention,
                                         ShawRelPositionMultiHeadedAttention)
from wenet.efficient_conformer.attention import (
    GroupedRelPositionMultiHeadedAttention)

WENET_ACTIVATION_CLASSES = {
    "hardtanh": torch.nn.Hardtanh,
    "tanh": torch.nn.Tanh,
    "relu": torch.nn.ReLU,
    "selu": torch.nn.SELU,
    "swish": getattr(torch.nn, "SiLU", Swish),
    "gelu": torch.nn.GELU,
}

WENET_RNN_CLASSES = {
    "rnn": torch.nn.RNN,
    "lstm": torch.nn.LSTM,
    "gru": torch.nn.GRU,
}

WENET_SUBSAMPLE_CLASSES = {
    "linear": LinearNoSubsampling,
    "embed": EmbedinigNoSubsampling,
    "conv1d2": Conv1dSubsampling2,
    "conv2d2": Conv2dSubsampling2,
    "conv2d": Conv2dSubsampling4,
    "conv2d_spike": Conv2dSubsampling4_Spike,
    "conv2d_spike_originmask": Conv2dSubsampling4_Spike_OriginMask,
    "conv2d_spike_t_originmask": Conv2dSubsampling4_Spike_T_OriginMask,
    "conv2d_spike_length_originmask":
    Conv2dSubsampling4_Spike_Length_OriginMask,
    "conv2d_spike_audio_length_originmask":
    Conv2dSubsampling4_Spike_Audio_Length_OriginMask,
    "dwconv2d4": DepthwiseConv2dSubsampling4,
    "conv2d6": Conv2dSubsampling6,
    "conv2d8": Conv2dSubsampling8,
    'paraformer_dummy': torch.nn.Identity,
    'stack_n_frames': StackNFramesSubsampling,
}

WENET_EMB_CLASSES = {
    "embed": PositionalEncoding,
    "abs_pos": PositionalEncoding,
    "rel_pos": RelPositionalEncoding,
    "no_pos": NoPositionalEncoding,
    "abs_pos_whisper": WhisperPositionalEncoding,
    "embed_learnable_pe": LearnablePositionalEncoding,
    # "abs_pos_paraformer": ParaformerPositinoalEncoding,
    'rope_pos': RopePositionalEncoding,
}

WENET_ATTENTION_CLASSES = {
    "selfattn": MultiHeadedAttention,
    "spike_selfattn": SSA,
    "spike_bn_selfattn": SSA_BN,
    "spike_bn_rel_selfattn": RelPositionSSA_BN,
    "spike_ln_selfattn": SSA_LN,
    "spike_ln_rel_selfattn": RelPositionSSA_LN,
    "spike_convbn_selfattn": SSA_ConvBN,
    "spike_convbn_rel_selfattn": RelPositionSSA_ConvBN,
    "rel_selfattn": RelPositionMultiHeadedAttention,
    "grouped_rel_selfattn": GroupedRelPositionMultiHeadedAttention,
    "crossattn": MultiHeadedCrossAttention,
    "spike_crossattn": SSCA,
    "spike_crossattn_ln": SSCA_LN,
    'shaw_rel_selfattn': ShawRelPositionMultiHeadedAttention,
    'rope_abs_selfattn': RopeMultiHeadedAttention,
}

WENET_MLP_CLASSES = {
    'position_wise_feed_forward': PositionwiseFeedForward,
    'spikeconformer_position_wise_feed_forward': SpikePositionwiseFeedForward,
    'spikeconformer_length_position_wise_feed_forward':
    SpikeLengthPositionwiseFeedForward,
    'spikeconformer_audio_length_position_wise_feed_forward':
    SpikeAudioLengthPositionwiseFeedForward,
    'spike_position_wise_feed_forward': MS_MLP,
    'moe': MoEFFNLayer,
    'gated': GatedVariantsMLP
}

WENET_NORM_CLASSES = {
    'layer_norm': LayerNorm,
    'batch_norm': BatchNorm1d,
    'rms_norm': RMSNorm
}
