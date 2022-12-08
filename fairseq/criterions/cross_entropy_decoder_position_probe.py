# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass
import random

import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
import torch as t
from npe_utils import NPE_Utils
from dataclasses import _MISSING_TYPE, dataclass, field

@dataclass
class DPPCrossEntropyCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")

    dont_report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )


@register_criterion("dpp_cross_entropy_fix", dataclass=DPPCrossEntropyCriterionConfig)
class DPPCrossEntropyCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg, dont_report_accuracy=False):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.positions = [i for i in range(8192)]  # We start from 5 to ignore the special tokens
        self.report_accuracy = not dont_report_accuracy

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        #if False:
        # all tokens are the same probe
        #input_number = random.randint(0,sample["net_input"]['src_tokens'].max().item())
        #print("input:", input_number)
        #sample["net_input"]['src_tokens'] = t.zeros_like(sample["net_input"]['src_tokens']) + input_number
        if False:
            hack = t.zeros_like(sample["net_input"]['src_tokens'])
            i = 0
            while 2*i+1 < len(sample["net_input"]['src_tokens'][0]):
                hack[0][2*i] = sample["net_input"]['src_tokens'][0][i]
                hack[0][2*i + 1] = sample["net_input"]['src_tokens'][0][i]
                i += 1
            sample["net_input"]['src_tokens'] = hack

        net_output = model(**sample["net_input"])

        lprobs, target = get_lprobs_and_target(self.positions, self.padding_idx, model, net_output, sample)

        loss, _ = self.compute_loss(lprobs, target, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(lprobs, target)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)

            abs_diff_sum = self.compute_mean_absolute_difference(lprobs, target)
            logging_output["abs_diff_sum"] = utils.item(abs_diff_sum.data)

        # Print out some examples:
        #lprobs, target = get_lprobs_and_target(self.positions, self.padding_idx, model, net_output, sample)
        #outputs = [[j.item()-5 for j in [i.data for i in lprobs.view(net_output[0].shape).argmax(2)][k]] for k in range(net_output[0].shape[0])]
        #for output in outputs:
        #    print("O: ", output)
        return loss, sample_size, logging_output

    def compute_loss(self, lprobs, target, reduce=True):
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )
        return loss, loss

    def compute_accuracy(self, lprobs, target):
        mask = target.ne(self.padding_idx)
        n_correct = t.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        #print(lprobs.argmax(1).masked_select(mask))
        total = t.sum(mask)
        return n_correct, total

    def two_bos_hack(sample):
        hack = t.zeros_like(sample["net_input"]['src_tokens'])
        hack[0][0] = sample["net_input"]['src_tokens'][0][0]
        hack[0][1] = sample["net_input"]['src_tokens'][0][0]
        hack[0][2] = sample["net_input"]['src_tokens'][0][0]
        hack[0][3:] = sample["net_input"]['src_tokens'][0][2:-1]

        return hack

    def all_double_bos_hack(sample):
        hack = t.zeros_like(sample["net_input"]['src_tokens'])
        for i in range(round(len(sample["net_input"]['src_tokens'][0])/2)):
            hack[0][i] = sample["net_input"]['src_tokens'][0][i]
            hack[0][i+1] = sample["net_input"]['src_tokens'][0][i]
        return hack

    def compute_mean_absolute_difference(self, lprobs, target):
        mask = target.ne(self.padding_idx)
        abs_diff_sum = (target.masked_select(mask) - lprobs.argmax(1).masked_select(mask)).abs().sum()
        return abs_diff_sum


    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

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

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )
            abs_diff_sum = utils.item(
                sum(log.get("abs_diff_sum", 0) for log in logging_outputs)
            )
            metrics.log_scalar("abs_diff_sum", abs_diff_sum)
            metrics.log_derived(
                "mad",
                lambda meters: round(
                    meters["abs_diff_sum"].sum / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


def get_lprobs_and_target(all_positions, padding_idx, model, net_output, sample):
    lprobs = model.get_normalized_probs(net_output, log_probs=True)


    ##### probe hack #####
    batch_size, dim = net_output[0].shape[0:2]
    assert dim <= len(all_positions)
    positions = all_positions[:dim]
    mask = sample['target'].eq(padding_idx)

    # this can only happen in debug!
    #if mask.shape[0] != batch_size:
    #    mask = mask[0].expand(batch_size,dim)

    pos_target = t.tensor(positions).expand(batch_size, dim).to(sample['target'])
    pos_target = pos_target.masked_fill(mask, padding_idx)
    sample['target'] = pos_target + model.decoder.dictionary.nspecial + 1
    #######################

    lprobs = lprobs.view(-1, lprobs.size(-1))
    target = model.get_targets(sample, net_output).view(-1)
    return lprobs, target
