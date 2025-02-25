import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

SAVE_FILE_PATH = os.path.join('..', 'data', 'reference_points', 'manualy_selected_points.npy')
REFERENCE_POINTS_PATH = os.path.join('..', 'data', 'reference_points', 'key_points_xyz.npy')

points = np.load(REFERENCE_POINTS_PATH)[0]
selected_indices = np.load(SAVE_FILE_PATH) if os.path.exists(SAVE_FILE_PATH) else []


def onclick(event):
    global selected_indices
    if event.inaxes is not None:
        x, y = event.xdata, event.ydata
        distances = np.sqrt((points[:, 0] - x)**2 + (points[:, 1] - y)**2)
        index = np.argmin(distances)
        if index not in selected_indices:
            selected_indices = np.append(selected_indices, index)
            print(f"Added index: {index}")
            ax.scatter(points[index, 0], points[index, 1], color='red')
        else:
            selected_indices = selected_indices[selected_indices != index]
            print(f"Removed index: {index}")
            ax.scatter(points[index, 0], points[index, 1], color='blue')
        plt.draw()


def save_indices(_):
    np.save(SAVE_FILE_PATH, np.array(selected_indices))
    print(f"Selected indices saved to {SAVE_FILE_PATH}")


if __name__ == "__main__":
    fig, ax = plt.subplots()
    fig.set_size_inches(13, 13)

    ax.invert_yaxis()

    ax.scatter(points[:, 0], points[:, 1], color='blue')
    if len(selected_indices) > 0:
        ax.scatter(points[selected_indices, 0], points[selected_indices, 1], color='red')

    fig.canvas.mpl_connect('button_press_event', onclick)

    ax_save = plt.axes([0.81, 0.05, 0.1, 0.075])
    btn_save = Button(ax_save, 'Save')
    btn_save.on_clicked(save_indices)

    plt.show()
