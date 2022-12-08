from typing import List

from fairseq.tasks.language_modeling import *

@dataclass
class LanguageModelingPPConfig(LanguageModelingConfig):
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


@register_task("language_modeling_position_probe", dataclass=LanguageModelingPPConfig)
class LanguageModelingPPTask(LanguageModelingTask):

    def __init__(self, args, dictionary, output_dictionary=None, targets=None):
        super().__init__(args, dictionary, output_dictionary, targets)

    def train_step(
            self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        loss, sample_size, logging_output = super().train_step(sample, model, criterion, optimizer, update_num,
                                                               ignore_grad=ignore_grad)

        #assert (model.position_probe_layers.training)
        assert (not model.decoder.training)

        return loss, sample_size, logging_output
