import argparse
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import  pandas as pd
import random
def parse_line(line):
    probe_example = line[5:-2].split(",")
    probe_example = [int(i.strip()) for i in probe_example]
    return probe_example


def read_results(file):
    df_arr = []
    with open(file, 'r') as f:
        for line in f.readlines():
            if not line.startswith("O: "):
                continue
            gen_prode = parse_line(line)
            df_arr.append(gen_prode)
    return np.array(df_arr)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir",
                        default=r"/home/olab/adi/git/npe/fairseq_cli/prob_gen_out_512.txt",
                        type=str,
                        required=False)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    probes_pos = read_results(args.in_dir)
    random.shuffle(probes_pos)
    probes_pos = probes_pos[:100,:]
    df = pd.DataFrame(probes_pos, columns=[str(i) for i in range(512)])
    df_std = pd.DataFrame(np.array([float(i) for i in df.std().to_numpy()])).transpose()
    df_mean = pd.DataFrame(np.array([float(i) for i in df.mean().to_numpy()])).transpose()
    df_std_mean = pd.concat([df_std, df_mean], keys=['std', 'mean']).transpose()

    df_std_mean['mean_plus_std'] = df_std_mean.apply(lambda x: (1.96*x['std']) + x['mean'], axis=1)
    df_std_mean['NoPos Probe Predictions\n(mean and conf. interval)'] = df_std_mean.apply(lambda x: x['mean'] - (1.96*x['std']), axis=1).abs()

    sns.set(style='dark', rc={'figure.figsize': (8, 7)})  # rc={"grid.linewidth": 1})
    fontsize = 17
    df_mean = df_std_mean.iloc[:, [1]]
    df_plus = df_std_mean.iloc[:, [2]]
    df_minus = df_std_mean.iloc[:, [3]]

    fig, g = plt.subplots()

    # 100 examples mean and confidence level plot
    lp_mean_plus = sns.lineplot(data=df_plus, legend=False, palette=['b'], linewidth=0)
    lp_mean_minus = sns.lineplot(data=df_minus, palette=['b'], linewidth=0, legend=True)
    lp_mean = sns.lineplot(data=df_mean, palette=['b'], legend=False, linewidth=3)

    plt.fill_between([i for i in range(512)], [i[0] for i in df_minus.to_numpy()], [i[0] for i in df_plus.to_numpy()],
                     color='blue', alpha=.1)

    # single example plot
    probes_pos_values = np.append(probes_pos[0], probes_pos[0][-1])
    single_ex = dict(x=[str(i) for i in range(513)], y=probes_pos_values)
    single_ex_df = pd.DataFrame(single_ex)
    sp_ex = sns.scatterplot(x='x', y='y', data=single_ex_df, s=80, color='g', label="NoPos Probe Single\nExample Predictions")


    # basline plot
    target_pos = np.array([i for i in range(0, 512)])
    my_dict = dict(x=target_pos, y=target_pos)
    data = pd.DataFrame(my_dict)
    lp_baseline = sns.lineplot(x='x', y='y', data=data, linestyle='--', linewidth=3, color='w',#'#a9a9a9',
                      label="Ground Truth")

    #g.set_xlim((0, 520))
    g.set_xticks(range(0, 513, 64))
    g.set_yticks(range(0, 513, 64))
    g.set_xlabel("Target Position", fontsize=fontsize)
    g.set_ylabel("Predicted Position", fontsize=fontsize)
    g.tick_params(labelsize=fontsize)
    lgnd = plt.legend(loc='upper left', ncol=1, fontsize=fontsize - 1, facecolor='grey', framealpha=0.15)
    marker_size = 120
    #lgnd.legendHandles[1]._sizes = [marker_size, marker_size, marker_size, marker_size]
    #lgnd.legendHandles[2].set_linestyle("--")
    plt.tight_layout()

    plt.savefig('pos_gen_100_ex_512.pdf', format='pdf')

    #plt.savefig('probe_pos_multi.pdf', bbox_inches='tight')

    plt.show()