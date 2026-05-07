import pygame
import numpy as np
from numba import njit, prange
from color_palettes import colorize
from settings import (
    fixed_max_iterations,
    fps,
    initial_zoom_center_x,
    initial_zoom_center_y,
    initial_zoom_level,
    max_zoom_level,
    pan_speed_pixels,
    screen_height,
    screen_width,
    zoom_multiplier,
)
# Henter startværdier for zoom og centrum fra settings.py.
current_center_x, current_center_y = initial_zoom_center_x, initial_zoom_center_y
current_zoom_level = initial_zoom_level

@njit(parallel=True)
def calculate_mandelbrot_iterations(center_x, center_y, zoom_level, max_iterations, image_width, image_height):
    # Beregner skalaen for det komplekse plan ud fra zoomniveauet.
    coordinate_scale = 4.0 / zoom_level
    iteration_counts = np.zeros((image_height, image_width), dtype=np.int32)
    # Behandler hver række parallelt for at gøre beregningen hurtigere.
    for y_position in prange(image_height):
        # Finder det imaginære koordinat for rækken.
        imaginary_coordinate = center_y + (y_position - image_height / 2) * coordinate_scale / image_width
        for x_position in range(image_width):
            # Finder det reelle koordinat for pixelens position.
            real_coordinate = center_x + (x_position - image_width / 2) * coordinate_scale / image_width
            # Opretter punktet i det komplekse plan.
            complex_point = complex(real_coordinate, imaginary_coordinate)
            # Starter ved 0 og tæller, hvor mange iterationer punktet overlever.
            # Python bruger j til den imaginære enhed, selv om man ofte skriver i i matematik.
            current_value = 0.0j
            iteration_count = 0
            # Fortsætter, indtil punktet flygter, eller vi rammer max antal iterationer.
            while (current_value.real * current_value.real + current_value.imag * current_value.imag <= 4.0) and (iteration_count < max_iterations):
                current_value = current_value * current_value + complex_point
                iteration_count += 1
            # Gemmer iterationstallet for den aktuelle pixel.
            iteration_counts[y_position, x_position] = iteration_count
    return iteration_counts

def main_loop():
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)

    global current_zoom_level, current_center_x, current_center_y
    running = True
    while running:
        elapsed_seconds = clock.tick(fps) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Panorerer med WASD eller piletaster; hastigheden følger zoomniveauet.
        keys = pygame.key.get_pressed()
        coordinate_scale = 4.0 / current_zoom_level
        pan_distance = pan_speed_pixels * elapsed_seconds * coordinate_scale / screen_width
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            current_center_x -= pan_distance
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            current_center_x += pan_distance
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            current_center_y -= pan_distance
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            current_center_y += pan_distance

        # Zoomer automatisk for hver frame, indtil maksimum er nået.
        max_iterations = fixed_max_iterations
        if current_zoom_level < max_zoom_level:
            current_zoom_level *= zoom_multiplier
            current_zoom_level = min(current_zoom_level, max_zoom_level)
        zoom_limit_reached = current_zoom_level >= max_zoom_level
        
        iteration_counts = calculate_mandelbrot_iterations(
            current_center_x,
            current_center_y,
            current_zoom_level,
            max_iterations,
            screen_width,
            screen_height,
        )
        rotated_color_image = np.rot90(colorize(iteration_counts, max_iterations))   # Drejer billedet, så pygame viser det rigtigt.
        image_surface = pygame.surfarray.make_surface(rotated_color_image)
        screen.blit(image_surface, (0, 0))
        
        # Tegner titelbjælken øverst.
        pygame.draw.rect(screen, (0, 0, 0), (0, 0, screen_width, 30))
        current_fps = clock.get_fps()
        zoom_status_text = (
            f"max zoom reached"
            if zoom_limit_reached
            else f"zoom: {current_zoom_level:.2f}"
        )
        title_text_surface = font.render(
            f"Mandelbrot set | W, A, S, D to pan | frame rate: {current_fps:.1f} | fixed iterations: {max_iterations} | {zoom_status_text}",
            True,
            (255, 255, 255),
        )
        screen.blit(title_text_surface, (10, 5))
        
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main_loop()