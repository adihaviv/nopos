# How to recreate the pile dataset:
we leverage the The Pile preprocess that was done in Scrolls, 
to access the initial files go to:
https://github.com/oriyor/scrolls_pt_data/tree/main

Create pre-training data:
(1) Download and de-compress the pile chunks from the repo above, 
this differs from the original Pile dataset as they are removing 
specific sources, trimming outliers, replacing special chars, 
output tsv per source and plot stats.

(2) run pile_dataset_creator.py
Unify all the files, flatten them and split to train, dev, test
** Unify all files (00,01,wikipedia)
** split to train, dev(2000), test(2000)
** flatten to txt 

(3) run pile_tokenizer_preprocessor.py 
Encode with the GPT-2 tokenizer by running the 

(4) Preprocess datasets with the fairseq-preprocess
fairseq-preprocess --only-source --trainpref data-bin/pile/pile-tokenized/pile_00_01_wikipedia.train.tokens.bpe --validpref data-bin/pile/pile-tokenized/pile_00_01_wikipedia.valid.tokens.bpe --testpref data-bin/pile/pile-tokenized/pile_00_01_wikipedia.test.tokens.bpe --destdir data-bin/pile/pile-fixed --workers 20  --srcdict data-bin/pile/pile-fixed/dict.txt --fp16
