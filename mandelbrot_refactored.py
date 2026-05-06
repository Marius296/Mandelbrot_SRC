import pygame
import numpy as np
from dataclasses import dataclass
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


@dataclass
class ViewerState:
    center_x: float
    center_y: float
    zoom_level: float
    zoom_limit_reached: bool = False


@njit(parallel=True)
def calculate_mandelbrot_iterations(center_x, center_y, zoom_level, max_iterations, image_width, image_height):
    coordinate_scale = 4.0 / zoom_level
    iteration_counts = np.zeros((image_height, image_width), dtype=np.int32)

    for y_position in prange(image_height):
        imaginary_coordinate = center_y + (y_position - image_height / 2) * coordinate_scale / image_width
        for x_position in range(image_width):
            real_coordinate = center_x + (x_position - image_width / 2) * coordinate_scale / image_width
            complex_point = complex(real_coordinate, imaginary_coordinate)
            current_value = 0.0j
            iteration_count = 0
            while (current_value.real * current_value.real + current_value.imag * current_value.imag <= 4.0) and (iteration_count < max_iterations):
                current_value = current_value * current_value + complex_point
                iteration_count += 1
            iteration_counts[y_position, x_position] = iteration_count

    return iteration_counts


def handle_input(state, elapsed_seconds):
    keys = pygame.key.get_pressed()
    coordinate_scale = 4.0 / state.zoom_level
    pan_distance = pan_speed_pixels * elapsed_seconds * coordinate_scale / screen_width

    if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        state.center_x -= pan_distance
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        state.center_x += pan_distance
    if keys[pygame.K_w] or keys[pygame.K_UP]:
        state.center_y -= pan_distance
    if keys[pygame.K_s] or keys[pygame.K_DOWN]:
        state.center_y += pan_distance

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False

    return True


def update_state(state):
    # Zoom automatically until the configured ceiling is reached.
    if state.zoom_level < max_zoom_level:
        state.zoom_level *= zoom_multiplier
        state.zoom_level = min(state.zoom_level, max_zoom_level)

    state.zoom_limit_reached = state.zoom_level >= max_zoom_level


def render_frame(screen, font, clock, state):
    iteration_counts = calculate_mandelbrot_iterations(
        state.center_x,
        state.center_y,
        state.zoom_level,
        fixed_max_iterations,
        screen_width,
        screen_height,
    )

    rotated_color_image = np.rot90(colorize(iteration_counts, fixed_max_iterations))
    image_surface = pygame.surfarray.make_surface(rotated_color_image)
    screen.blit(image_surface, (0, 0))

    pygame.draw.rect(screen, (0, 0, 0), (0, 0, screen_width, 30))
    current_fps = clock.get_fps()
    zoom_status_text = (
        "max zoom reached"
        if state.zoom_limit_reached
        else f"zoom: {state.zoom_level:.2f}"
    )
    title_text_surface = font.render(
        f"Mandelbrot set | W, A, S, D to pan | frame rate: {current_fps:.1f} | fixed iterations: {fixed_max_iterations} | {zoom_status_text}",
        True,
        (255, 255, 255),
    )
    screen.blit(title_text_surface, (10, 5))


def main():
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)

    state = ViewerState(
        center_x=initial_zoom_center_x,
        center_y=initial_zoom_center_y,
        zoom_level=initial_zoom_level,
    )

    running = True
    while running:
        elapsed_seconds = clock.tick(fps) / 1000.0
        running = handle_input(state, elapsed_seconds)
        update_state(state)
        render_frame(screen, font, clock, state)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()