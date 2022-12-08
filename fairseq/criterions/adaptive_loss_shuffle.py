# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

import torch.nn.functional as F
import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.constants import DDP_BACKEND_CHOICES
from omegaconf import II
from fairseq.criterions.adaptive_loss import AdaptiveLoss,AdaptiveLossConfig
from dataclasses import field

@dataclass
class AdaptiveLossShuffleConfig(AdaptiveLossConfig):

    shuffle_idx: int = field(
        default=4,
        metadata={"help": "shuffle the input until this token"},
    )

    no_shuffle: bool = field(
        default=False,
        metadata={"help": "if set to true there is no shuffle"},
    )


@register_criterion("adaptive_loss_shuffle", dataclass=AdaptiveLossShuffleConfig)
class AdaptiveLossShuffle(AdaptiveLoss):

    def __init__(self, task, sentence_avg, shuffle_idx, no_shuffle):
        super().__init__(task, sentence_avg)
        self.shuffle_idx = shuffle_idx
        self.no_shuffle = no_shuffle

    def forward(self, model, sample, reduce=True):
        if not self.no_shuffle:
            sample["net_input"]['src_tokens'][:, :self.shuffle_idx - 1] = \
                sample["net_input"]['src_tokens'].index_select(1, torch.randperm(self.shuffle_idx - 1).
                                                               to(sample["net_input"]['src_tokens'].device))

        assert (
            hasattr(model.decoder, "adaptive_softmax")
            and model.decoder.adaptive_softmax is not None
        )
        adaptive_softmax = model.decoder.adaptive_softmax

        net_output = model(**sample["net_input"]) #18,512,1024
        orig_target = model.get_targets(sample, net_output)

        nsentences = orig_target.size(0)
        orig_target_single_token = orig_target[:, self.shuffle_idx - 2]
        orig_target_single_token = orig_target_single_token.view(-1)

        bsz = orig_target_single_token.size(0)

        net_output_single_token = net_output[0][:, self.shuffle_idx - 2, :]
        logits, target = adaptive_softmax(net_output_single_token, orig_target_single_token)
        assert len(target) == len(logits)

        loss = net_output_single_token.new(1 if reduce else bsz).zero_()

        for i in range(len(target)):
            if target[i] is not None:
                assert target[i].min() >= 0 and target[i].max() <= logits[i].size(1)
                loss += F.cross_entropy(
                    logits[i],
                    target[i],
                    ignore_index=self.padding_idx,
                    reduction="sum" if reduce else "none",
                )
                break

        orig = utils.strip_pad(orig_target_single_token, self.padding_idx)
        ntokens = orig.numel()
        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens
        logging_output = {
            "loss": loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }

        print("\nLoss:::", loss.item())
        return loss, sample_size, logging_output

    @classmethod
    def build_criterion(cls, cfg: AdaptiveLossShuffleConfig, task):
        if cfg.ddp_backend in {"c10d", "pytorch_ddp"}:
            raise Exception(
                "AdaptiveLoss is not compatible with the PyTorch "
                "version of DistributedDataParallel. Please use "
                "`--ddp-backend=legacy_ddp` instead."
            )
        return cls(task, cfg.sentence_avg, cfg.shuffle_idx, cfg.no_shuffle)

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )

        #if sample_size != ntokens:
        #    metrics.log_scalar(
        #        "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
        #    )
        #    metrics.log_derived(
        #        "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        #    )
        #else:
        #    metrics.log_derived(
        #        "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        #    )
