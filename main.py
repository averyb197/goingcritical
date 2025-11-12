import pygame
from numba import njit
import numpy as np

N = 250
thresh=4
init_dump = 1e10

grid = np.zeros((N, N))
grid[int(N/2), int(N/2)] = init_dump

pygame.init()

cell_size = 3
width, height = N* cell_size, N* cell_size

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Wowowowowow")

@njit(nogil=True)
def topple(grid, point):
    i, j = point
    grid[i, j] -= 4
    if i>0:
        grid[i-1, j] += 1
    if i<grid.shape[0]:
        grid[i+1, j] += 1
    if j >0:
        grid[i, j-1] += 1
    if j<grid.shape[1]:
        grid[i, j+1] += 1
    return grid

@njit(nogil=True)
def get_crit_points(grid):
    return np.argwhere(grid > thresh)

@njit(nogil=True)
def iteration(grid):
    critical_points = get_crit_points(grid)
    if critical_points.size > 0:
        i, j = critical_points[np.random.choice(critical_points.shape[0])]
        topple(grid, (i, j))
    else:
        print("Arrived at steady state")
        return grid
    return grid

@njit(nogil=True)
def batch_iterations(grid, n=100):
    for i in range(n):
        iteration(grid)

def draw_grid(grid, screen, cell_size):
    for i in  range(N):
        for j in range(N):
            color_val = int(min(grid[i, j] * 50, 255))
            color = (color_val, 0, 255-color_val)
            pygame.draw.rect(screen, color, pygame.Rect(j*cell_size, i*cell_size, cell_size, cell_size))

running = True
clock = pygame.time.Clock()
batch_size = 1000

frame = 0
while running:
    screen.fill((0,0,0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    batch_iterations(grid, n=batch_size)
    draw_grid(grid, screen, cell_size)
    pygame.display.flip()
    clock.tick(60)

    frame += 1

pygame.quit()