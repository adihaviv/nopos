#/home/olab/adi/miniconda3/envs/npe/bin/python /home/olab/adi/git/npe/fairseq_cli/validate_lm.py  --task language_modeling /home/olab/adi/git/npe/data-bin/wikitext-103 --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=legacy_ddp --keep-best-checkpoints 5 --max-update 386000 --required-batch-size-multiple 1 --lr 1.0 --t-mult 2 --lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75 --warmup-updates 16000 --warmup-init-lr 1e-07 --stop-min-lr 1e-09 --optimizer nag --min-lr 0.0001 --clip-norm 0.1 --criterion adaptive_loss_shuffle --update-freq 1 --max-tokens 9216 --fp16 --arch transformer_lm_wiki103 --no-token-positional-embeddings --seed 1 --tokens-per-sample 512 --validate-interval-updates 1 --save-dir /home/olab/adi/experiments/npe/lm-baevski-wiki103-512/lm-baevski-wiki103-512-no-token-positional-embeddings --reset-optimizer --sentence-avg --shuffle-idx 5 --no-shuffle > /home/olab/adi/experiments/npe/shuffle_expriments/5_no_shuffle.txt
import argparse
import random
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_path",
                        default=r"shuffle_expriments/shuffle_exp.sh",
                        type=str,
                        required=False)

    parser.add_argument("--sample_size",
                        default=80,
                        type=int,
                        required=False)

    parser.add_argument("--max_idx",
                        default=511,
                        type=int,
                        required=False)

    parser.add_argument("--min_idx",
                        default=6,
                        type=int,
                        required=False)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    sample_idxs = random.sample(range(args.min_idx, args.max_idx), args.sample_size)
    with open(args.out_path, 'w') as f:
        os.chmod(args.out_path, 0o777)
        f.write("#!/bin/bash\n\nexport CUDA_VISIBLE_DEVICES=0\n\n")
        for i, idx in enumerate(sample_idxs):
            job_command = "/home/olab/adi/miniconda3/envs/npe/bin/python " \
                          "/home/olab/adi/git/npe/fairseq_cli/validate_lm.py  --task language_modeling " \
                          "/home/olab/adi/git/npe/data-bin/wikitext-103 --sample-break-mode none " \
                          "--skip-invalid-size-inputs-valid-test --ddp-backend=legacy_ddp --keep-best-checkpoints 5 " \
                          "--max-update 386000 --required-batch-size-multiple 1 --lr 1.0 --t-mult 2 " \
                          "--lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75 --warmup-updates 16000 " \
                          "--warmup-init-lr 1e-07 --stop-min-lr 1e-09 --optimizer nag --min-lr 0.0001 " \
                          "--clip-norm 0.1 --criterion adaptive_loss_shuffle --update-freq 1 --max-tokens 9216 " \
                          "--fp16 --arch transformer_lm_wiki103 --no-token-positional-embeddings --seed 1 " \
                          "--tokens-per-sample 512 --validate-interval-updates 1 " \
                          "--save-dir /home/olab/adi/experiments/npe/lm-baevski-wiki103-512/lm-baevski-wiki103-512-no-token-positional-embeddings " \
                          "--reset-optimizer --sentence-avg --shuffle-idx {0} " \
                          "--no-shuffle > /home/olab/adi/experiments/npe/shuffle_expriments/{0}_no_shuffle.txt"\
                .format(idx)
            f.write("\n"+job_command + "\n")
            f.write(job_command.replace("--no-shuffle", "").replace("_no_", "_") + "\n")
            f.write("echo \"{}/{}\"\n".format(i, len(sample_idxs)))


