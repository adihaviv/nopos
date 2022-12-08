import json
import argparse
import os
import random

DEBUG = False

def read_pile(pile_paths, filters):
    examples = []
    log_counter = 0
    yt_counter = 0
    for pile_path in pile_paths:
        with open(pile_path) as f:
            for line in f:
                example = json.loads(line)
                assert len(example['meta']) == 1
                meta = example['meta']['pile_set_name'].lower()
                #print(meta)
                if meta == "youtube":
                  yt_counter += 1
                  print("found one")
                if meta not in filters:
                    examples.append(example['text'])
                log_counter += 1
                if log_counter % 10000 == 0:
                    print("filtered {} youtube examples".format(yt_counter))
                    print("processed {} examples".format(log_counter))

                if DEBUG and log_counter > 12345:
                    log_counter = 0
                    break

    return examples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths",
                        default=r"scrolls_pt_data/scrolls_pt_data/scrolls_corpus/00_filtered.jsonl,"
                                r"scrolls_pt_data/scrolls_pt_data/scrolls_corpus/01_filtered.jsonl,"
                                r"scrolls_pt_data/scrolls_pt_data/scrolls_corpus/wikipedia_filtered.jsonl",
                        type=str,
                        required=False)
    parser.add_argument("--filters",
                        default="github,youtube",
                        type=str,
                        required=False)
    parser.add_argument("--out_dir",
                        default="/home/olab/adi/git/npe/data-bin/pile_no_youtube",
                        type=str,
                        required=False)

    parser.add_argument("--dev_sample_size",
                        default=2000,
                        type=int,
                        required=False)

    return parser.parse_args()


def write_filterd_file(dataset, out_dir, dataset_type):
    print("writing test file")
    counter = len(dataset)
    out_file_path = os.path.join(out_dir, pile_prefix + '.'+dataset_type+'.tokens')
    with open(out_file_path, 'w') as f:
        for item in dataset:
            f.write("%s\n" % item)
            counter -= 1
            if counter % 10000 == 0 and counter > 0:
                print("wrote {} examples in test set".format(counter))
    return out_file_path


if __name__ == '__main__':
    args = parse_args()

    # read all data, filter and split to train/val/test
    examples = read_pile(args.paths.split(","), args.filters.split(","))
    random.shuffle(examples)
    exp_cnt = len(examples)

    val = examples[:args.dev_sample_size]
    test = examples[args.dev_sample_size:2*args.dev_sample_size]
    train = examples[2*args.dev_sample_size:]
    print("train set size:{}, validation set size:{}, test set size:{}".format(len(train), len(val), len(test)))

    out_dir = args.out_dir
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    pile_prefix = "pile_00_01_wikipedia"

    print("writing train file")
    train_file_path = write_filterd_file(train, out_dir, "train")

    print("writing validation file")
    valid_file_path = write_filterd_file(val, out_dir, "valid")

    print("writing test file")
    test_file_path = write_filterd_file(test, out_dir, "test")

    print("all done - you can now run: \n "
          f"fairseq-preprocess --only-source --trainpref {train_file_path} --validpref {valid_file_path} "
          f"--testpref {test_file_path} --destdir {args.out_dir} --workers 20 --srcdict {args.out_dir}/dict.txt"
          f"--bpe gpt2 --fp16")