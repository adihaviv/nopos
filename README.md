# NoPos
This repository contains the code for the analysis and models discussed in the paper "[Transformer Language Models without Positional Encodings Still Learn Positional Information](https://arxiv.org/abs/2203.16634)".

## Requirements and Installation
This repository is a fork of the [Fairseq](https://github.com/facebookresearch/fairseq) repository and has the same requirements.

# NoPos Models
The main models (including the 1.3B parameters model) that were trained without position embeddings (NoPos models) are available in the following [link](https://drive.google.com/drive/folders/1avrK37tzBAVidZSE79b8vQCNcLLpNc-u?usp=sharing).

## Datasets
## Canonical setting - wikitext-103 

To download and preprocess the data, run:
```bash
cd examples/language_model/
bash prepare-wikitext-103.sh
cd ../.

TEXT=examples/language_model/wikitext-103
fairseq-preprocess \
    --only-source \
    --trainpref $TEXT/wiki.train.tokens \
    --validpref $TEXT/wiki.valid.tokens \
    --testpref $TEXT/wiki.test.tokens \
    --destdir data-bin/wikitext-103 \
    --workers 20
```

## Large Scale Setting - The Pile
To reconstruct The Pile subset data we used for the experiments in the paper, see https://github.com/adihaviv/NoPos/tree/main/nopos_experiments/the_pile_construction

The preprocessed dataset can be found at https://www.cs.tau.ac.il/~adihaviv1/nopos_the_pile
    
## Citation

If you find this work helpful, please cite us
```@article{Haviv2022TransformerLM,
  title={Transformer Language Models without Positional Encodings Still Learn Positional Information},
  author={Adi Haviv and Ori Ram and Ofir Press and Peter Izsak and Omer Levy},
  journal={ArXiv},
  year={2022},
  volume={abs/2203.16634}
}
```

This repo is still improving. For any questions, please email adi.haviv@cs.tau.ac.il
