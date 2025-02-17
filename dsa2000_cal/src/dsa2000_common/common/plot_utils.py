import tempfile

import imageio


def figs_to_gif(fig_generator, gif_path, duration=0.5, loop=0, dpi=80):
    """
    Convert a generator of matplotlib figures to a GIF using a temporary directory, with options for loop and resolution.

    Parameters:
        fig_generator (generator): A generator that yields matplotlib figures. Generator should close figs.
        gif_path (str): Path where the GIF should be saved.
        duration (float): Duration of each frame in the GIF in seconds.
        loop (int): Number of times the GIF should loop (0 for infinite).
        dpi (int): Dots per inch (resolution) of images in the GIF.
    """
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        filenames = []
        for i, fig in enumerate(fig_generator):
            # Save each figure to a temporary file
            filename = f'{tmp_dir}/frame_{i}.png'
            fig.savefig(filename, dpi=dpi)  # Specify DPI for image quality
            filenames.append(filename)

        # Create a GIF using the saved frames
        with imageio.get_writer(gif_path, mode='I', duration=duration, loop=loop) as writer:
            for filename in filenames:
                image = imageio.v2.imread(filename)
                writer.append_data(image)

        # Temporary files are automatically cleaned up when exiting the block

    print(f"GIF saved as {gif_path}")
