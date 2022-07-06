import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from copy import copy

from typing import Optional


class Robot:
    def __init__(
        self,
        x: float,
        y: float,
        theta: float,
        v: float,
        omega: float,
        max_v: float,
        max_dv: float,
        max_omega: float,
        max_domega: float,
    ):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.omega = omega

        self.max_v = max_v
        self.max_dv = max_dv
        self.max_omega = max_omega
        self.max_domega = max_domega


class Trajectory:
    def init(self, robot: Robot, v: float, omega: float, dt: float, t: float):

        self.robot = copy(robot)
        self.v = v
        self.omega = omega
        self.dt = dt
        self.unit_time_length = int(1 / dt)
        self.t = t

    def gen_traj_points(self):
        self.t_array = np.arange(0, self.t, self.dt)
        if self.omega == 0:
            self.x_array = self.robot.x + self.v * self.t_array * np.cos(self.robot.theta)
            self.y_array = self.robot.y + self.v * self.t_array * np.sin(self.robot.theta)

        else:
            radius = np.abs(self.v / self.omega)
            clockwise = 1 if self.omega >= 0 else -1
            center_x = -np.sin(self.robot.theta) * radius * clockwise + self.robot.x
            center_y = np.cos(self.robot.theta) * radius * clockwise + self.robot.y

            circle_theta_array = -np.pi / 2 * clockwise + self.robot.theta + self.omega * self.t_array
            self.x_array = center_x + radius * np.cos(circle_theta_array)
            self.y_array = center_y + radius * np.sin(circle_theta_array)

    def set_faisible(self, faisible: bool):
        self.faisible = faisible


class GridMap:
    def __init__(self, grid_size: int):
        self.grid_size = grid_size

        self.data = np.zeros((grid_size, grid_size))

        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(6, 6)
        self.ax.set_xticks(np.arange(0, self.grid_size + 1))
        self.ax.set_yticks(np.arange(0, self.grid_size + 1))
        self.ax.grid(which="major", axis="both", linestyle="--", color="k", linewidth=0.25)

    def gen_map_bool(self, block_rate=0.1, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.data = np.random.rand(self.grid_size, self.grid_size)  # array[y][x]

        self.obst_coor = []
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.data[y][x] >= 1 - block_rate:
                    self.obst_coor.append((x, y))
        self.obst_coor = np.array(self.obst_coor)

        cmap = colors.ListedColormap(["w", "k"])
        bounds = [0, 1 - block_rate, 1]
        norm = colors.BoundaryNorm(bounds, cmap.N)  # type: ignore
        self.ax.imshow(
            self.data,
            cmap,
            norm,
            origin="lower",
            extent=[0, self.grid_size, 0, self.grid_size],
        )

    def draw_traj(self, traj: Trajectory, indice1: Optional[int], indice2: Optional[int], *args, **kwargs):
        self.ax.plot(traj.x_array[indice1:indice2], traj.y_array[indice1:indice2], *args, **kwargs)

    def show(self):
        # self.ax.set_xlim(0, self.grid_size)
        # self.ax.set_ylim(0, self.grid_size)
        plt.show()

    def save(self):
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        plt.savefig("fig.png", dpi=300)


class Simulator:
    def __init__(self, robot: Robot, grid_map: GridMap, dest: tuple):
        self.robot = robot
        self.grid_map = grid_map
        self.dest = dest

        self.grid_map.ax.scatter(robot.x, robot.y, color="r")
        self.grid_map.ax.scatter(dest[0], dest[1], color="b")
