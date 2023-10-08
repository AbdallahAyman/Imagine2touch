import matplotlib.cm as cm
import numpy as np
from reskin.reskin_calibration.dataset import get_ambient_data, subtract_ambient
import numpy as np
import re
import os
import natsort
import hydra
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def get_reskin_reading(
    path,
    binary,
    raw=True,
    raw_ambient=False,
    ambient_every_contact=False,
    handle_negative=True,
):
    """raw: don't subtract ambient from signal
    raw_ambient: ambient data was already processed during collection"""
    regex = re.compile("experiment_.*_reskin$")
    experiments = []
    for p in path:
        for root, dirs, files in os.walk(p):
            for file in files:
                if regex.match(file):
                    experiments.append(p + "/" + file)
    first_path = path[0]
    experiments = np.asarray(natsort.natsorted(experiments))

    # store and pre-process first experiment
    reskin_reading = np.load(experiments[0], allow_pickle=True)
    reskin_reading = np.squeeze(reskin_reading)[
        :, 2
    ]  # extract lists of magnetometers values and temperatures as array of lists
    reskin_reading = list(reskin_reading)  # convert to list of lists then to nd array
    reskin_reading = np.asarray(reskin_reading)
    if binary == "False":
        reskin_reading = np.delete(
            reskin_reading, [3, 7, 11, 15, 19], 1
        )  # eliminate temperatures
    else:
        reskin_reading = np.delete(reskin_reading, [0, 4, 8, 12, 16], 1)
    if raw:
        pass
    else:
        reskin_reading = subtract_ambient(
            reskin_reading,
            get_ambient_data(
                first_path, binary, experiments[0] + "_ambient", aggregated=raw_ambient
            ),
            ambient_every_contact=ambient_every_contact,
            handle_negative=handle_negative,
        )
    if len(experiments) == 1:
        return reskin_reading

    for counter, experiment in enumerate(experiments):
        if counter == 0:
            continue
        else:
            reskin_reading_i = np.load(experiment, allow_pickle=True)
            if reskin_reading_i.shape[0] > 1:
                reskin_reading_i = np.squeeze(reskin_reading_i)[
                    :, 2
                ]  # extract lists of magnetometers values and temperatures as array of lists
            else:
                reskin_reading_i = np.squeeze(reskin_reading_i)[2]
            reskin_reading_i = list(
                reskin_reading_i
            )  # convert to list of lists then to nd array
            reskin_reading_i = np.asarray(reskin_reading_i).reshape(-1, 20)
            if binary == "False":
                reskin_reading_i = np.delete(
                    reskin_reading_i, [3, 7, 11, 15, 19], 1
                )  # eliminate temperatures
            else:
                reskin_reading_i = np.delete(
                    reskin_reading_i, [0, 4, 8, 12, 16], 1
                )  # eliminate temperatures
            if raw:
                pass
            else:
                reskin_reading_i = subtract_ambient(
                    reskin_reading_i,
                    get_ambient_data(
                        path, binary, experiment + "_ambient", aggregated=raw_ambient
                    ),
                    ambient_every_contact=ambient_every_contact,
                    handle_negative=handle_negative,
                )
            reskin_reading = np.vstack((reskin_reading, reskin_reading_i))
    return reskin_reading


if __name__ == "__main__":
    OmegaConf.register_new_resolver("path", lambda x: os.path.abspath(x))
    hydra.initialize("./conf", version_base=None)
    cfg = hydra.compose("dataset.yaml")
    file_tactile = []
    object_names = cfg.object_names.split(",")
    object_names_string = ",".join(object_names)
    for object_name in object_names:
        file_tactile.append(
            cfg.data_path + "/" + object_name + "/" + object_name + "_tactile"
        )

    tactile_input = get_reskin_reading(
        file_tactile,
        binary=True,
        raw=False,
        raw_ambient=cfg.aggregated_ambient,
        ambient_every_contact=cfg.ambient_every_contact,
        handle_negative=cfg.handle_negative_data,
    )

    # Filter tactile data
    ## delete any elements that are tactile 15 dimensional arrays that contains an element not in the range [-200,200]
    tactile_input = tactile_input[
        (tactile_input > -200).all(axis=1) & (tactile_input < 200).all(axis=1)
    ]

    # plot configuration initialization
    num_samples, num_dimensions = tactile_input.shape
    minimum_data = tactile_input.min()
    maximum_data = tactile_input.max()
    z_grid = np.linspace(minimum_data, maximum_data, num_samples)
    colormap = cm.get_cmap("tab10", num_dimensions)
    direction_mappings = {1: "Center", 2: "Top", 3: "Left", 4: "Bottom", 5: "Right"}
    legend_labels = [
        f"{direction_mappings[j]}" for j in range(1, len(direction_mappings) + 1)
    ]
    # Create separate plots for each axis
    for axis in ["x", "y", "z"]:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=30, azim=45 + 180)  # Set consistent viewing angles
        # Plot the KDEs
        j = 1
        l = 1
        k = 1
        m = 1
        for i in range(num_dimensions):
            # Create a meshgrid for x (dimensions) and z (range of values)
            if i % 3 == 0 and i != 0:
                j += 1
            z = tactile_input[:, i]
            mean_z = np.mean(z)
            if axis == "x":
                if i in {0, 3, 6, 9, 12}:
                    kde = gaussian_kde(z)
                    y_estimate = kde.evaluate(z_grid)
                    color = colormap(l / 5)
                    ax.plot3D(
                        np.ones_like(z_grid) * l,
                        z_grid,
                        y_estimate,
                        label=f"{direction_mappings[j]} b_{axis}",
                        color=color,
                    )
                    ax.plot3D(
                        np.linspace(1, 5, 5),
                        np.zeros(5),
                        np.zeros(5),
                        color="black",
                        linewidth=2,
                    )
                    peak_idx = np.argmax(y_estimate)
                    ax.scatter(l, z_grid[peak_idx], c="red", marker="o", s=50)
                    # Create vertices for the translucent polygonal surface
                    verts = [
                        (l, z_val, y_estimate[idx]) for idx, z_val in enumerate(z_grid)
                    ]
                    verts.append((l, z_grid[-1], 0))  # Vertex at the end of the curve
                    verts.append(
                        (l, z_grid[0], 0)
                    )  # Vertex at the beginning of the curve
                    # Plot the polygonal surface with the unique color
                    poly = Poly3DCollection([verts], color=color, alpha=0.3)
                    ax.add_collection3d(poly)
                    ax.plot3D(
                        [l, l],
                        [z_grid[peak_idx], z_grid[peak_idx]],
                        [y_estimate[peak_idx], 0],
                        linestyle="dotted",
                        color="black",
                    )
                    # Set the x-axis tick positions
                    ax.set_xticks(range(1, len(legend_labels) + 1))
                    # Set the x-axis tick labels using the legend labels
                    ax.set_xticklabels(legend_labels)
                    l += 1
            elif axis == "y":
                if i in {1, 4, 7, 10, 13}:
                    kde = gaussian_kde(z)
                    y_estimate = kde.evaluate(z_grid)
                    color = colormap(k / 5)
                    ax.plot3D(
                        np.ones_like(z_grid) * k,
                        z_grid,
                        y_estimate,
                        label=f"{direction_mappings[j]} b_{axis}",
                        color=color,
                    )
                    ax.plot3D(
                        np.linspace(1, 5, 5),
                        np.zeros(5),
                        np.zeros(5),
                        color="black",
                        linewidth=2,
                    )
                    peak_idx = np.argmax(y_estimate)
                    ax.scatter(k, z_grid[peak_idx], c="red", marker="o", s=50)
                    # Create vertices for the translucent polygonal surface
                    verts = [
                        (k, z_val, y_estimate[idx]) for idx, z_val in enumerate(z_grid)
                    ]
                    verts.append((k, z_grid[-1], 0))  # Vertex at the end of the curve
                    verts.append(
                        (k, z_grid[0], 0)
                    )  # Vertex at the beginning of the curve
                    # Plot the polygonal surface with the unique color
                    poly = Poly3DCollection([verts], color=color, alpha=0.3)
                    ax.add_collection3d(poly)
                    ax.plot3D(
                        [k, k],
                        [z_grid[peak_idx], z_grid[peak_idx]],
                        [y_estimate[peak_idx], 0],
                        linestyle="dotted",
                        color="black",
                    )
                    # Set the x-axis tick positions
                    ax.set_xticks(range(1, len(legend_labels) + 1))
                    # Set the x-axis tick labels using the legend labels
                    ax.set_xticklabels(legend_labels)
                    k += 1
            elif axis == "z":
                if i in {2, 5, 8, 11, 14}:
                    kde = gaussian_kde(z)
                    y_estimate = kde.evaluate(z_grid)
                    color = colormap(m / 5)
                    ax.plot3D(
                        np.ones_like(z_grid) * m,
                        z_grid,
                        y_estimate,
                        label=f"{direction_mappings[j]} b_{axis}",
                        color=color,
                    )
                    ax.plot3D(
                        np.linspace(1, 5, 5),
                        np.zeros(5),
                        np.zeros(5),
                        color="black",
                        linewidth=2,
                    )
                    peak_idx = np.argmax(y_estimate)
                    ax.scatter(m, z_grid[peak_idx], c="red", marker="o", s=50)
                    # Create vertices for the translucent polygonal surface
                    verts = [
                        (m, z_val, y_estimate[idx]) for idx, z_val in enumerate(z_grid)
                    ]
                    verts.append((m, z_grid[-1], 0))  # Vertex at the end of the curve
                    verts.append(
                        (m, z_grid[0], 0)
                    )  # Vertex at the beginning of the curve
                    # Plot the polygonal surface with the unique color
                    poly = Poly3DCollection([verts], color=color, alpha=0.3)
                    ax.add_collection3d(poly)
                    peak_idx = np.argmax(y_estimate)
                    ax.plot3D(
                        [m, m],
                        [z_grid[peak_idx], z_grid[peak_idx]],
                        [y_estimate[peak_idx], 0],
                        linestyle="dotted",
                        color="black",
                    )
                    # Set the x-axis tick positions
                    ax.set_xticks(range(1, len(legend_labels) + 1))
                    # Set the x-axis tick labels using the legend labels
                    ax.set_xticklabels(legend_labels)
                    m += 1
        if axis == "x":
            ax.set_ylabel("Range of Values")
            ax.set_zlabel("Probability Estimate")
        elif axis == "y":
            ax.set_ylabel("Range of Values")
            ax.set_zlabel("Probability Estimate")
        elif axis == "z":
            ax.set_ylabel("Range of Values")
            ax.set_zlabel("Probability Estimate")
        ax.set_title(f"Stacked 3D Probability Estimates (KDE) - Axis: {axis}")
        # Set consistent aspect ratio and plot limits
        # ax.set_box_aspect([1, 1, 1])
        ax.set_xlim([0, 5])
        ax.set_ylim([np.min(tactile_input), np.max(tactile_input)])
        ax.set_zlim([0, 0.06])

        #  Export the plot to an image file (e.g., PNG)
        # plt.savefig(
        #     f"{object_names_string}_b_{axis}.png", dpi=2000, bbox_inches="tight"
        # )
        ax.legend()
        plt.tight_layout()  # Ensures the layout is adjusted correctly
        plt.show()
