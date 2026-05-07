import numpy as np

def colorize(iteration_counts, max_iterations, gamma=0.6, palette=None):
    if palette is None:
        # Glidende palet fra navy til lys varm farve.
        palette = np.array([
            [4, 8, 16],
            [8, 24, 48],
            [16, 48, 96],
            [32, 74, 120],
            [56, 100, 140],
            [88, 130, 150],
            [120, 160, 150],
            [150, 185, 145],
            [185, 205, 165],
            [220, 230, 200],
        ], dtype=np.uint8)

    palette = np.asarray(palette, dtype=np.uint8)
    palette_len = palette.shape[0]

    max_iter_safe = max(1, int(max_iterations))
    normalized = iteration_counts.astype(np.float32) / max_iter_safe

    # Maske for punkter inde i Mandelbrot-mængden, som skal være sorte.
    inside_mask = iteration_counts == max_iterations

    # Gamma styrer, hvor hurtigt farverne skifter gennem paletten.
    # Lav gamma giver hurtigere skift tæt på mængden; høj gamma giver en langsommere start.
    scaled = np.power(normalized, gamma)

    # Beregner paletindeks med gulvfunktion, så farverne bliver i tydelige bånd.
    indices = np.floor(scaled * palette_len).astype(np.int32)
    # Begrænser indekserne, så høje værdier bruger den sidste farve i paletten.
    indices = np.clip(indices, 0, palette_len - 1)

    # Slår indekserne op som RGB-farver.
    rgb_pixels = palette[indices]

    # Gør punkter inde i mængden sorte.
    rgb_pixels[inside_mask] = 0

    return rgb_pixels
