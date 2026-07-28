import matplotlib.pyplot as plt
from typing import Any


def draw_ax(ax, data, task, score, best_epoch):
    epoch = range(1,len(data)+1)

    ax.plot(
        *(epoch, data),
    )
    ax.axvline(
        x=best_epoch,
        color="red",
        linestyle="-",
        linewidth=2,
        label="Best Epoch",
    )

    ax.set_title(
        f"{task} Scores per Epoch ({score})", 
        fontsize=12, 
        fontweight="bold",
    )
    ax.set_xlabel(
        "Epoch", 
        fontsize=10,
    )
    ax.set_ylabel(
        score, 
        fontsize=10,
    )
    ax.grid(
        True, 
        linestyle="--", 
        alpha=0.5,
    )
    ax.legend(
        fontsize=9,
    )


def main(
    records: dict[str, Any], 
    suptitle: str, 
    trn: str="bce", 
    val: str="ndcg@10", 
    figsize: tuple[int]=(7,3),
) -> None:
    trn_nlls = records["trn"]["nll"]
    trn_klds = records["trn"]["kld"]
    val_scores = records["val"]
    best_epoch = records["best_epoch"]

    NROWS = 3
    NCOLS = 1
    WEIGHTS = figsize[0]
    HEIGHTS = figsize[1]

    fig, axes = plt.subplots(
        nrows=NROWS, 
        ncols=NCOLS, 
        figsize=(WEIGHTS*NCOLS, HEIGHTS*NROWS), 
        sharex=True, 
        sharey=False,
    )

    axes = axes.flatten()
    data = [trn_nlls, trn_klds, val_scores]
    task = ["Train", "Train", "Validation"]
    score = [trn, "KLD", val]

    for args in zip(axes, data, task, score):
        draw_ax(*args, best_epoch)

    plt.suptitle(
        t=suptitle,
        fontsize=14,
        fontweight="bold",
    )
    
    plt.tight_layout()
    plt.show()
