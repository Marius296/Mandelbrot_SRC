import pygame
import numpy as np
from numba import njit, prange

WIDTH, HEIGHT = 800, 600
MAX_ITER = 64
FPS = 60

center_x, center_y = 0.3508, 0.3456
zoom = 1.0
zoom_factor = 1.02

@njit(parallel=True)
def mandelbrot_numba(center_x, center_y, zoom, max_iter, width, height):
    scale = 4.0 / zoom
    result = np.zeros((height, width), dtype=np.int32)

    for y in prange(height):
        im = center_y + (y - height/2) * scale / width
        for x in range(width):
            re = center_x + (x - width/2) * scale / width
            c = complex(re, im)
            z = 0.0j
            count = 0
            while (z.real*z.real + z.imag*z.imag <= 4.0) and (count < max_iter):
                z = z*z + c
                count += 1
            result[y, x] = count
    return result

def colorize(N):
    colors = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    colors[..., 0] = (N % 8) * 32
    colors[..., 1] = (N % 16) * 16
    colors[..., 2] = (N % 32) * 8
    return colors

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    global zoom
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        N = mandelbrot_numba(center_x, center_y, zoom, MAX_ITER, WIDTH, HEIGHT)
        arr = np.rot90(colorize(N))   # rotate for pygame
        surf = pygame.surfarray.make_surface(arr)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        zoom *= zoom_factor
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()