import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


none_color = '#0072B2'
sin_color = '#009E73'
learned_color = '#E69F00'
alibi_color = '#D55E00'


def plot_prob_legend_in():
    sns.set_style("white", {"grid.color": ".1", "grid.linestyle": ":"})
    sns.set(rc={'figure.figsize': (6, 8)})

    fontsize = 17
    df = pd.read_csv("data_plots/probe.csv")
    df = pd.melt(df, ['layer'])
    g = sns.lineplot(x='layer', y='value', hue='variable', style="variable", data=df,
                     dashes=False, markers=["o", "X", "s", ">", ","],
                     palette=[none_color, learned_color, sin_color, alibi_color, '#a9a9a9'],
                     linewidth=2.5, markersize=10)

    g.set_xticks(range(0, 25, 4))
    g.set_xlabel("Layer", fontsize=fontsize)
    g.set_ylabel("Mean Absolute Distance", fontsize=fontsize)
    g.tick_params(labelsize=fontsize)
    g.legend_.legendHandles[4].set_linestyle("--")
    g.lines[4].set_linestyle("--")
    g.legend_.legendHandles[4].set_linestyle("--")

    plt.legend(loc='upper right', ncol=1, fontsize=fontsize)
    g.legend_.legendHandles[4].set_linestyle("--")
    plt.legend(bbox_to_anchor=(1, 0.95), loc='upper right', ncol=1, prop={'size': fontsize})
    g.legend_.legendHandles[4].set_linestyle("--")
    plt.tight_layout()

    plt.savefig('probe.pdf', format='pdf')
    plt.show()


def plot_prob_legend_in_small():
    sns.set_style("whitegrid", {"grid.color": ".1", "grid.linestyle": ":"})
    sns.set(rc={'figure.figsize': (6, 6)})

    fontsize = 17
    df = pd.read_csv("data_plots/probe.csv")
    df = pd.melt(df, ['layer'])
    g = sns.lineplot(x='layer', y='value', hue='variable', style="variable", data=df,
                     dashes=False, markers=["o", "X", "s", ">", ","],
                     palette=[none_color, learned_color, sin_color, alibi_color, '#a9a9a9'],
                     linewidth=2.5, markersize=10)

    g.set_xticks(range(0, 25, 4))
    g.set_xlabel("Layer", fontsize=fontsize)
    g.set_ylabel("Mean Absolute Distance", fontsize=fontsize)
    g.tick_params(labelsize=fontsize)
    g.legend_.legendHandles[4].set_linestyle("--")
    g.lines[4].set_linestyle("--")
    g.legend_.legendHandles[4].set_linestyle("--")

    plt.legend(loc='upper right', ncol=1, fontsize=fontsize)
    g.legend_.legendHandles[4].set_linestyle("--")
    plt.legend(bbox_to_anchor=(1, 0.95), loc='upper right', ncol=1, prop={'size': fontsize})
    g.legend_.legendHandles[4].set_linestyle("--")
    plt.tight_layout()

    plt.savefig('probe_small.pdf', format='pdf')
    plt.show()



def plot_pile_main():
    df = pd.read_csv("data_plots/main_pile.csv")
    df = pd.melt(df, ['method'])

    sns.set(rc={'figure.figsize': (5, 9)})

    # Form a facetgrid using columns
    g = sns.FacetGrid(df, col="variable", sharey=False, margin_titles=True)
    g.map(sns.barplot, 'method', 'value', order=["NoPos", "Learned", "Sinusoidal", "ALiBi"],
          palette=[none_color, learned_color, sin_color, alibi_color]).add_legend()
    g.set_titles(col_template="", row_template="{row_name}", size=16)
    g.axes[0][0].set_ylim(10, 14)
    g.axes[0][0].set_yticks(range(10, 15, 1))

    #remove x labels
    for ax in g.axes[0]:
        ax.set_xlabel("")


    g.axes[0][0].set_ylabel("Perplexity",fontsize=9)

    #add numbers
    for ax in g.axes[0]:
        for p in ax.patches:
            ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=8, color='black', xytext=(0, 5),
                        textcoords='offset points')

            ax.tick_params(labelsize=9)

    plt.tight_layout()

    plt.savefig('pile.pdf', format='pdf')
    plt.show()

def pos_prediction():
    predicted_pos = np.array([0,1,2,3,6,8,11,9,10,10,12,12,9,14,23,18,22,18,13,15,14,18,20,15,
                              18,20,23,30,39,27,29,43,44,61,43,52,46,43,21,33,44,62,43,43,49,
                              43,53,59,43,49,58,62,55,56,57,52,58,58,52,52,58,49,63,63])
    target_pos = np.array([i for i in range(0, 64)])
    sns.set(rc={'figure.figsize': (6, 6)})
    sns.set_style("dark")

    fontsize = 17
    my_dict = dict(x=target_pos, y=predicted_pos, z=target_pos)
    data = pd.DataFrame(my_dict)
    fig, g = plt.subplots()
    g1 = sns.scatterplot(x='x', y='y', data=data, s=70, color=none_color, label="NoPos Probe\nPredictions")
    g2 = sns.lineplot(x='x', y='z', data=data, linestyle='--', linewidth=3, color='#a9a9a9', label="Ground Truth\n(x=y)")

    g.set_xticks(range(0, 65, 8))
    g.set_yticks(range(0, 65, 8))
    g.set_xlabel("Target Position", fontsize=fontsize)
    g.set_ylabel("Predicted Position", fontsize=fontsize)
    g.tick_params(labelsize=fontsize)
    plt.legend(loc='lower right', ncol=1, fontsize=fontsize-2)
    g2.legend_.legendHandles[0].set_linestyle("-")
    plt.tight_layout()

    plt.savefig('pos_gen.pdf', format='pdf')
    plt.show()
