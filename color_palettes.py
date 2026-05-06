import numpy as np


def colorize(iteration_counts, max_iterations):

    # Normalisér iterationstællinger til 0..1 og undgå division med nul.
    safe_max_iterations = max(1, int(max_iterations))
    normalized_iterations = iteration_counts.astype(np.float32) / safe_max_iterations

    # Fast palette med farvestop.
    color_stops = np.array([
        [66, 30, 15],    # dark brown
        [25, 7, 26],     # violet
        [57, 125, 209],  # blue
        [134, 181, 229], # light blue
        [248, 201, 95],  # yellow-orange
        [255, 170, 0],   # orange
    ], dtype=np.float32)

    number_of_color_stops = color_stops.shape[0]

    # Position i farvestop-rummet, fra 0 til number_of_color_stops - 1.
    palette_position = normalized_iterations * (number_of_color_stops - 1)
    lower_stop_index = np.floor(palette_position).astype(np.int32)
    blend_fraction = palette_position - lower_stop_index

    # Begræns indeks, så vi ikke overskrider arrayets grænser.
    lower_stop_index = np.clip(lower_stop_index, 0, number_of_color_stops - 2)
    upper_stop_index = lower_stop_index + 1

    # Hent stopfarver og interpolér.
    lower_stop_color = color_stops[lower_stop_index]
    upper_stop_color = color_stops[upper_stop_index]
    color_image = lower_stop_color * (1.0 - blend_fraction)[..., None] + upper_stop_color * blend_fraction[..., None]

    # Sæt indre punkter til sort.
    mask_inside = iteration_counts == max_iterations
    color_image[mask_inside] = 0

    return np.clip(color_image, 0, 255).astype(np.uint8)
