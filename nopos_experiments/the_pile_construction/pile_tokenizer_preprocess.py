import sys
import itertools
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm
from transformers import GPT2TokenizerFast


def main(input_path, output_path, num_workers):
    print('Reading data from file...')
    paragraphs = read_paragraphs(input_path)
    
    print(f'Processing with {num_workers} threads...')
    stride = len(paragraphs) // num_workers
    shards = [paragraphs[start:start+stride] for start in range(0, len(paragraphs), stride)]
    with Pool(num_workers) as pool:
        doc_shards = pool.map(txt2bow, shards)
    docs = list(itertools.chain.from_iterable(doc_shards))
    
    print('Saving preprocessed data...')
    save_data(output_path, docs)

    
def read_paragraphs(path):
    with open(path) as fin:
        lines = [line.strip() for line in tqdm(fin) if len(line.strip()) > 7]
    return lines


def txt2bow(txts):
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    docs = [tokenizer(x)['input_ids'] for x in tqdm(txts)]
    docs = [' '.join([str(x) for x in doc]) for doc in docs]
    return docs


def save_data(path, docs):
    with open(path, 'w') as fout:
        for doc in tqdm(docs):
            fout.write(doc)
            fout.write('\n')


if __name__ == '__main__':
    split = sys.argv[1]
    main(split, split+'.bpe', 32)
