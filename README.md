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

The preprocessed dataset (validation set) can be found at [link](https://drive.google.com/drive/folders/1avrK37tzBAVidZSE79b8vQCNcLLpNc-u?usp=sharing).
    
## Citation

If you find this work helpful, please cite us
```
@inproceedings{haviv-etal-2022-transformer,
    title = "Transformer Language Models without Positional Encodings Still Learn Positional Information",
    author = "Haviv, Adi  and Ram, Ori  and Press, Ofir  and Izsak, Peter  and Levy, Omer",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.99",
    pages = "1382--1390",}
```

This repo is still improving. For any questions, please email adi.haviv@cs.tau.ac.il
