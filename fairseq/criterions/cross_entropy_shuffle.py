# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
from dataclasses import field
import torch
from fairseq.criterions.cross_entropy import CrossEntropyCriterion

@dataclass
class CrossEntropyShuffleCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")

    shuffle_idx: int = field(
        default=4,
        metadata={"help": "shuffle the input until this token"},
    )

    no_shuffle: bool = field(
        default=False,
        metadata={"help": "if set to true there is no shuffle"},
    )

@register_criterion("cross_entropy_shuffle", dataclass=CrossEntropyShuffleCriterionConfig)
class CrossEntropyShuffleCriterion(CrossEntropyCriterion):

    def __init__(self, task, sentence_avg, shuffle_idx, no_shuffle):
        super().__init__(task, sentence_avg)
        self.shuffle_idx = shuffle_idx
        self.no_shuffle = no_shuffle

    def forward(self, model, sample, reduce=True):
        """
        Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if not self.no_shuffle:
            sample["net_input"]['src_tokens'][:, :self.shuffle_idx-1] = \
                sample["net_input"]['src_tokens'].index_select(1, torch.randperm(self.shuffle_idx-1).
                                                               to(sample["net_input"]['src_tokens'].device))

        #todo: maybe we should move here the loss =0?
        loss, sample_size, logging_output = super(CrossEntropyShuffleCriterion, self).forward(model, sample, reduce)
        print("\nLoss:::", loss.item())
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)

        # TODO: this works only in the case when the batch size is 1,
        #       in other cases it will be PAD and we need to make sure it doesn't take loss over the padded token
        if self.shuffle_idx > lprobs.shape[1] - 2:
            zero_loss = torch.tensor(0).to(lprobs.device)
            return zero_loss,  zero_loss

        lprobs = lprobs[:, self.shuffle_idx - 2, :]
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output)[:, self.shuffle_idx - 2].view(-1)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )
        return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(1 if log.get("loss", 0) > 0 else 0 for log in logging_outputs)
        sample_size = sum(1 if log.get("loss", 0) > 0 else 0 for log in logging_outputs)
        #sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )
