# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass, field
from typing import Optional

from fairseq import options, utils
from fairseq.models import (
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    DEFAULT_MIN_PARAMS_TO_WRAP,
    Embedding,
    TransformerDecoder,
)
from fairseq.modules import AdaptiveInput, CharacterTokenEmbedder
from fairseq.utils import safe_getattr, safe_hasattr
from omegaconf import II
import torch.nn as nn
from fairseq.models.transformer_lm import TransformerLanguageModel, TransformerLanguageModelConfig, base_lm_architecture, transformer_lm_big, transformer_lm_gpt3_xl
from fairseq import checkpoint_utils, models, optim, utils

import logging
from argparse import Namespace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from fairseq.tasks.language_modeling_position_probe import LanguageModelingPPConfig

DEFAULT_MAX_TARGET_POSITIONS = 1024


@dataclass
class TransformerLanguageModelPositionProbeConfig(TransformerLanguageModelConfig):

    scale_resids: Optional[bool] = field(
        default=False,
        metadata={"help": "Learn a scale coefficient for each residual connection"},
    )

    pretrained_decoder_filename: str = field(
        default="",
        metadata={
            "help": "if not empty load the decoder model from the given checkpoint filename"
        },
    )

    probe_layer_idx: int = field(
        default=-1,
        metadata={
            "help": "the layer that will be probed in the decoder"
        },
    )

    non_linear_probe: bool = field(
        default=False,
        metadata={
            "help": "if set to true use MLP with 2 layers and RELU"}
    )


@register_model("transformer_lmpp", dataclass=TransformerLanguageModelPositionProbeConfig)
class TransformerLanguageModelPositionProbe(TransformerLanguageModel):

    def __init__(self, decoder, position_probe_layers, probe_layer_idx, non_linear_probe):
        super(TransformerLanguageModelPositionProbe, self).__init__(decoder)
        self.decoder.eval()
        self.probe_layer_idx = probe_layer_idx
        self.non_linear_probe = non_linear_probe
        self.position_probe_layer_0 = position_probe_layers[0]
        if non_linear_probe:
            self.position_probe_layer_1 = position_probe_layers[1]
            self.relu = nn.ReLU()

        if len(position_probe_layers) > 2:
            self.position_probe_layer_debug = position_probe_layers[2]

    def forward(self, src_tokens, **kwargs):
        self.decoder.eval()
        with torch.no_grad():
            decoder_out = super(TransformerLanguageModelPositionProbe, self).forward(src_tokens, **kwargs)

        x = decoder_out[1]['inner_states'][self.probe_layer_idx].transpose(0, 1)
        #x = F.dropout(x, 0.2)
        #x = torch.load("positions.pt")
        x = self.position_probe_layer_0(x)
        if self.non_linear_probe:
            #x = F.dropout(x, 0.2)
            x = self.position_probe_layer_1(self.relu(x))

        return x, decoder_out[1]

    @classmethod
    def build_model(cls, args, task):

        def Linear(in_features, out_features, bias=True):
            m = nn.Linear(in_features, out_features, bias)
            nn.init.xavier_uniform_(m.weight)
            if bias:
                nn.init.constant_(m.bias, 0.0)
            return m

        """Build a new model instance."""
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if safe_getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = safe_getattr(
                args, "tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS
            )

        if args.character_embeddings:
            embed_tokens = CharacterTokenEmbedder(
                task.source_dictionary,
                eval(args.character_filters),
                args.character_embedding_dim,
                args.decoder_embed_dim,
                args.char_embedder_highway_layers,
            )
        elif args.adaptive_input:
            embed_tokens = AdaptiveInput(
                len(task.source_dictionary),
                task.source_dictionary.pad(),
                args.decoder_input_dim,
                args.adaptive_input_factor,
                args.decoder_embed_dim,
                options.eval_str_list(args.adaptive_input_cutoff, type=int),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            embed_tokens = cls.build_embedding(
                args, task.source_dictionary, args.decoder_input_dim
            )

        if args.tie_adaptive_weights:
            assert args.adaptive_input
            assert args.adaptive_input_factor == args.adaptive_softmax_factor
            assert (
                    args.adaptive_softmax_cutoff == args.adaptive_input_cutoff
            ), "{} != {}".format(
                args.adaptive_softmax_cutoff, args.adaptive_input_cutoff
            )
            assert args.decoder_input_dim == args.decoder_output_dim

        decoder = TransformerDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True, output_projection=None
        )

        # load weights from checkpoint
        state = checkpoint_utils.load_checkpoint_to_cpu(
            args.pretrained_decoder_filename, load_on_all_ranks=True
        )

        layers_to_delete = []
        new_state_dict = state["model"].copy()
        for layer_name in new_state_dict:
            if layer_name.startswith("decoder."):
                state["model"][layer_name[len("decoder."):]] = state["model"][layer_name]
                layers_to_delete.append(layer_name)

        for layer_name in layers_to_delete:
            del state["model"][layer_name]

        decoder.load_state_dict(
            state["model"], strict=True
        )

        position_probe_layers = []

        if args.non_linear_probe:
            position_probe_layers.append(Linear(args.decoder_output_dim, args.decoder_output_dim*2, bias=False))
            position_probe_layers.append(Linear(args.decoder_output_dim*2, args.tokens_per_sample+decoder.dictionary.nspecial+1, bias=False))
        else:
            position_probe_layers.append(Linear(args.decoder_output_dim, args.tokens_per_sample+decoder.dictionary.nspecial+1, bias=False))

        return cls(decoder, position_probe_layers, int(args.probe_layer_idx), args.non_linear_probe)

        # TorchScript doesn't support super() method so that the scriptable Subclass
        # can't access the base class model in Torchscript.
        # Current workaround is to add a helper function with different name and
        # call the helper function from scriptable Subclass.
    def get_normalized_probs_scriptable(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
    ):
        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

#transformer_lm_wiki103
@register_model_architecture("transformer_lmpp", "transformer_lmpp_gpt")
def transformer_lm_gpt(args):
    base_lm_architecture(args)

@register_model_architecture("transformer_lmpp", "transformer_lmpp_wiki103")
@register_model_architecture("transformer_lmpp", "transformer_lmpp_baevski_wiki103")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = safe_getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 8)
    args.dropout = safe_getattr(args, "dropout", 0.3)
    args.adaptive_input = safe_getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = safe_getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = safe_getattr(
        args, "adaptive_input_cutoff", "20000,60000"
    )
    args.adaptive_softmax_cutoff = safe_getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = safe_getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = safe_getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = safe_getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = safe_getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)


@register_model_architecture("transformer_lmpp", "transformer_lmpp_gpt3_xl")
def transformer_lmpp_gpt3_xl(args):
    # 1.3B params
    transformer_lm_gpt3_xl(args)
