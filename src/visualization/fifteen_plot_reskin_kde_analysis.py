import numpy as np
import numpy as np
import os
import hydra
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from matplotlib import patches
import seaborn as sns

from reskin.visualizations.analyse_reskin import get_reskin_reading
from reskin.reskin_calibration.dataset import get_ambient_data


if __name__ == "__main__":
    OmegaConf.register_new_resolver("path", lambda x: os.path.abspath(x))
    hydra.initialize("./conf", version_base=None)
    cfg = hydra.compose("dataset.yaml")
    file_tactile = []
    object_names = cfg.object_names.split(",")
    for object_name in object_names:
        file_tactile.append(
            cfg.data_path + "/" + object_name + "/" + object_name + "_tactile"
        )
    tactile_input_raw = get_reskin_reading(
        file_tactile,
        cfg.binary,
        raw=True,
        raw_ambient=cfg.aggregated_ambient,
        ambient_every_contact=cfg.ambient_every_contact,
    )
    ambient_tactile_input = get_ambient_data(
        file_tactile, binary=True, aggregated=cfg.aggregated_ambient
    )
    # Filter tactile data
    ## delete any elements that are tactile 15 dimensional arrays that contains an element not in the range [-500,500]
    tactile_input_raw = tactile_input_raw[
        (tactile_input_raw > -500).all(axis=1) & (tactile_input_raw < 500).all(axis=1)
    ]
    minimum_data = tactile_input_raw.min()
    maximum_data = tactile_input_raw.max()
    tactile_input = get_reskin_reading(
        file_tactile,
        binary=True,
        raw=False,
        raw_ambient=cfg.aggregated_ambient,
        ambient_every_contact=cfg.ambient_every_contact,
    )
    tactile_input = tactile_input[
        (tactile_input > -500).all(axis=1) & (tactile_input < 500).all(axis=1)
    ]
    print(np.min(tactile_input))

    # create a figure and set of subplots to draw the plots on
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))
    # loop over each dimension of the data and plot the kernel density estimate
    artists = []
    labels = []
    for i in range(15):
        row = i // 4
        col = i % 4
        a = sns.kdeplot(
            data=tactile_input_raw[:, i],
            fill=True,
            ax=axes[row, col],
            label="original data",
            color="orange",
        )
        b = sns.kdeplot(
            data=ambient_tactile_input[:, i],
            fill=True,
            ax=axes[row, col],
            label="ambient data",
            color="blue",
        )
        d = sns.kdeplot(
            data=tactile_input[:, i],
            fill=True,
            ax=axes[row, col],
            label="differential signal",
            color="red",
        )
        axes[row, col].set_title("Dimension {}".format(i + 1), fontdict={"fontsize": 8})
        axes[row, col].set_ylabel("pdf")
        axes[row, col].set_xlim([minimum_data, maximum_data])
    c = sns.kdeplot(
        data=np.linalg.norm(tactile_input_raw, 2, 1),
        fill=True,
        ax=axes[3, 3],
        label="original data norm",
        color="black",
    )
    ambient_tactile_input = np.asarray(ambient_tactile_input, dtype=np.float64)
    e = sns.kdeplot(
        data=np.linalg.norm(ambient_tactile_input, 2, 1),
        fill=True,
        ax=axes[3, 3],
        label="ambient data norm",
        color="green",
    )
    minimum_norm = np.min(np.linalg.norm(tactile_input_raw, 2, 1))
    maximum_norm = np.max(np.linalg.norm(tactile_input_raw, 2, 1))
    if minimum_norm < -2000:
        minimum_norm = -2000
    if maximum_norm > 2000:
        maximum_norm = 2000
    axes[3, 3].set_xlim([minimum_norm, maximum_norm])
    # update the legend with the new labels and colors
    labels = [
        "original data",
        "ambient data",
        "differential signal",
        "original data norm",
        "ambient data norm",
    ]
    legend_labels = {
        "original data": ["orange"],
        "ambient data": ["blue"],
        "differential signal": ["red"],
        "original data norm": ["black"],
        "ambient data norm": ["green"],
    }
    handles = []
    for label in legend_labels:
        colors = legend_labels[label]
        for color in colors:
            handle = patches.Patch(color=color, label=label)
            handles.append(handle)
    plt.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=6)
    plt.subplots_adjust(hspace=1, wspace=0.5)
    plt.show()
