import numpy as np


# HSV->RGB helper removed: using discrete RGB palette mapping below (no interpolation).


def colorize(iteration_counts, max_iterations, gamma=0.6, palette=None):
    # Default discrete palette (clear, non-interpolated colors)
    if palette is None:
        # Gradual progression from very dark navy -> lighter warm pale (10 steps)
        palette = np.array([
            [4, 8, 16],     # near-black navy
            [8, 24, 48],    # very dark navy
            [16, 48, 96],   # deep blue
            [32, 74, 120],  # steel/indigo
            [56, 100, 140], # desaturated blue
            [88, 130, 150], # muted teal
            [120, 160, 150],# soft green-teal
            [150, 185, 145],# pale greenish
            [185, 205, 165],# warm pale
            [220, 230, 200],# very light warm
        ], dtype=np.uint8)

    palette = np.asarray(palette, dtype=np.uint8)
    palette_len = palette.shape[0]

    max_iter_safe = max(1, int(max_iterations))
    normalized = iteration_counts.astype(np.float32) / max_iter_safe

    # Mask for points inside the Mandelbrot set (kept black)
    inside_mask = iteration_counts == max_iterations

    # Gamma styrer, hvor hurtigt farverne skifter gennem paletten:
    # lav gamma = hurtigere skift tæt på Mandelbrot-mængden, høj gamma = langsommere skift i starten.
    scaled = np.power(normalized, gamma)

    # Compute palette indices using floor -> discrete bands (no interpolation)
    indices = np.floor(scaled * palette_len).astype(np.int32)
    # Clamp indices so high normalized values map to the last palette entry
    indices = np.clip(indices, 0, palette_len - 1)

    # Map indices to RGB colors (vectorized indexing)
    rgb_pixels = palette[indices]

    # Set interior points to black
    rgb_pixels[inside_mask] = 0

    return rgb_pixels
