"""Utility for converting matplotlib figures to .mp4 video files."""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


def fig_to_rgb(fig):
    """Render a matplotlib Figure to an RGB numpy array.

    Args:
        fig: matplotlib Figure object.

    Returns:
        np.ndarray of shape (H, W, 3) with dtype uint8.
    """
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())
    return buf[:, :, :3].copy()


def save_figures_as_mp4(figures, filepath, fps=2):
    """Save a list of matplotlib Figures as an animated .mp4 video.

    Args:
        figures: List of matplotlib Figure objects (one per frame).
        filepath: Output .mp4 file path.
        fps: Frames per second.
    """
    import imageio.v2 as imageio

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    frames = []
    for fig in figures:
        frame = fig_to_rgb(fig)
        # Ensure even dimensions (required by libx264)
        h, w = frame.shape[:2]
        h = h if h % 2 == 0 else h - 1
        w = w if w % 2 == 0 else w - 1
        frames.append(frame[:h, :w])
        plt.close(fig)

    if not frames:
        return

    writer = imageio.get_writer(
        filepath, fps=fps, codec='libx264',
        output_params=['-pix_fmt', 'yuv420p'],
    )
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def save_static_plot_as_mp4(fig, filepath, duration_seconds=3, fps=2):
    """Save a single static figure as a short .mp4 (repeated frame).

    Args:
        fig: matplotlib Figure object.
        filepath: Output .mp4 file path.
        duration_seconds: How long the video should be.
        fps: Frames per second.
    """
    import imageio.v2 as imageio

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    frame = fig_to_rgb(fig)
    h, w = frame.shape[:2]
    h = h if h % 2 == 0 else h - 1
    w = w if w % 2 == 0 else w - 1
    frame = frame[:h, :w]
    plt.close(fig)

    n_frames = max(1, duration_seconds * fps)
    writer = imageio.get_writer(
        filepath, fps=fps, codec='libx264',
        output_params=['-pix_fmt', 'yuv420p'],
    )
    for _ in range(n_frames):
        writer.append_data(frame)
    writer.close()
