import numpy as np
from matplotlib import pyplot as plt

from dsa2000_cal.common.plot_utils import figs_to_gif


def test_figs_to_gif():
    # Example usage:
    def example_figure_generator():
        x = np.linspace(0, 2 * np.pi, 100)
        for i in range(50):
            fig, ax = plt.subplots()
            ax.plot(x, np.sin(x + i * 0.1))
            ax.set_title(f'Step {i}')
            yield fig
            plt.close(fig)

    # Convert figures to GIF
    fig_generator = example_figure_generator()
    figs_to_gif(fig_generator, 'example_animation.gif', loop=0)
