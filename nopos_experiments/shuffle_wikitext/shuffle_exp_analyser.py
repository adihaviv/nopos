import argparse
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import  pandas as pd

def parse_line(line):
    stats = line.split("|")
    loss = 1
    for i in range(len(stats)):
        if " loss " in stats[i]:
            loss = float(stats[i].replace("loss", "").strip())

    ppl = 1
    for i in range(len(stats)):
        if " ppl " in stats[i]:
            ppl = float(stats[i].replace("ppl", "").strip())

    return loss, ppl


def read_results(dir):
    files = os.listdir(dir)
    shuffle_losses = np.array([])
    shuffle_ppl = np.array([])
    no_shuffle_losses = np.array([])
    no_shuffle_ppl = np.array([])
    df_arr = []

    for file in files:
        if ".txt" not in file:
            continue

        with open(os.path.join(dir, file), 'r') as f:
            idx = file[:file.find("_")]
            exp = "Baseline" if "no_shuffle" in file else "Shuffled Prefix"
            last_line = f.readlines()[-1]
            if "loss" not in last_line:
                print("check the last line in:", file)
                continue
            loss, ppl = parse_line(last_line)

            df_arr.append([int(idx), float(loss), ppl, exp])

            if "no_shuffle" in file:
                no_shuffle_losses = np.append(no_shuffle_losses, loss)
                no_shuffle_ppl = np.append(no_shuffle_ppl, ppl)
            else:
                shuffle_losses = np.append(shuffle_losses, loss)
                shuffle_ppl = np.append(shuffle_ppl, ppl)

    print("shuffle mean loss:{:.2f}".format(shuffle_losses.mean()))
    print("no shuffle mean loss:{:.2f}".format(no_shuffle_losses.mean()))

    return shuffle_losses, no_shuffle_losses, np.array(df_arr)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir",
                        default=r"/home/olab/adi/experiments/npe/shuffle_expriments",
                        type=str,
                        required=False)

    return parser.parse_args()


def plot_ppl(df):

    df = pd.DataFrame(df_np, columns=['idx', 'Loss', 'ppl', 'set'])
    df[["idx", "ppl"]] = df[["idx", "ppl"]].apply(pd.to_numeric)

    sns.set(style='whitegrid', rc={"grid.linewidth": 1})
    fontsize = 17

    splot = sns.boxplot(data=df, y="ppl", x="set", showfliers=False)
    splot.set_yscale("log")

    plt.ylim(10**1,10**4)
    plt.xlabel('')
    splot.tick_params(labelsize=fontsize)
    plt.ylabel('Perplexity (Log Scale)', fontsize=fontsize);

    plt.tight_layout()
    splot.yaxis.grid(True, clip_on=False)
    sns.despine(left=True, bottom=True)
    plt.savefig('shuffle_exp_ppl.pdf', bbox_inches='tight')

    plt.show()


def plot_loss(df):
    df = pd.DataFrame(df, columns=['idx', 'Loss', 'ppl', 'set'])
    df[["idx", "Loss"]] = df[["idx", "Loss"]].apply(pd.to_numeric)

    sns.set(style='whitegrid', rc={"grid.linewidth": 1})
    fontsize = 17

    splot = sns.boxplot(data=df, y="Loss", x="set", showfliers=False)
    plt.xlabel('')
    splot.tick_params(labelsize=fontsize)
    plt.ylabel('Token-Level Loss', fontsize=fontsize);

    plt.tight_layout()
    splot.yaxis.grid(True, clip_on=False)
    sns.despine(left=True, bottom=True)
    plt.savefig('shuffle_exp_loss.pdf', bbox_inches='tight')

    plt.show()

if __name__ == '__main__':
    args = parse_args()
    shuffle_losses, no_shuffle_losses, df_np = read_results(args.in_dir)

    plot_ppl(df_np)
    plot_loss(df_np)
