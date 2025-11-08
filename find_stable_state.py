from numba import njit
import numpy as np
import time
import matplotlib.pyplot as plt

N = 250
thresh=4
init_dump = 1e10

grid = np.zeros((N, N))
grid[int(N/2), int(N/2)] = init_dump


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
    return grid

@njit(nogil=True)
def batch_iterations(grid, n=100):
    for i in range(n):
        iteration(grid)

@njit(nogil=True)
def find_stable_state(grid):
    stable = False
    steps = 0
    batch_size = 1000
    while not stable:
        grid = batch_iterations(grid, batch_size)
        steps += batch_size
        grits = get_crit_points(grid)
        stable = grits == 0
        if stable:
            return grid, steps

start = time.time()
stable_state, steps = find_stable_state(grid)
end = time.time()

print(f"Finished after {steps} steps in {end-start:.1f} seconds")

fig, ax = plt.subplots()
ax.imshow(stable_state, cmap="plasma")



