import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Arc


class RobotState:
    def __init__(self, x=0, y=0, theta=0, v=0, omega=0):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.omega = omega


class RobotConstrain:
    def __init__(self, max_v, max_dv, max_omega):
        self.max_v = max_v
        self.max_dv = max_dv
        self.max_omega = max_omega


class GridMap:
    def __init__(self, grid_size):
        self.grid_size = grid_size

        self.data = np.zeros((grid_size, grid_size))

        self.fig, self.ax = plt.subplots()
        self.ax.set_xticks(np.arange(0, self.grid_size + 1))
        self.ax.set_yticks(np.arange(0, self.grid_size + 1))
        self.ax.grid(which='major', axis='both', linestyle='--', color='k', linewidth=0.5)

    def gen_map_bool(self, block_rate=0.1, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.data = np.random.rand(self.grid_size, self.grid_size)
        cmap = colors.ListedColormap(['w', 'k'])
        bounds = [0, 1 - block_rate, 1]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        self.ax.imshow(self.data, cmap, norm, origin='lower', extent=[0, self.grid_size, 0, self.grid_size])

    def draw_trajectory_old(self, x, y, theta, v, omega):

        if (omega == 0):
            x2 = x + v * np.cos(theta)
            y2 = y + v * np.sin(theta)
            self.ax.add_line(Line2D([x, x2], [y, y2], color='g'))

        else:
            radius = np.abs(v / omega)
            clockwise = 1 if omega >= 0 else -1
            center_x = -np.sin(theta) * radius * clockwise + x
            center_y = np.cos(theta) * radius * clockwise + y

            if (clockwise == 1):
                theta1 = (-np.pi / 2 + theta) / np.pi * 180
                theta2 = (-np.pi / 2 + theta + v / radius) / np.pi * 180
            else:
                theta1 = (np.pi / 2 + theta - v / radius) / np.pi * 180
                theta2 = (np.pi / 2 + theta) / np.pi * 180

            self.ax.add_patch(
                Arc((center_x, center_y),
                    radius * 2,
                    radius * 2,
                    angle=0,
                    theta1=theta1,
                    theta2=theta2,
                    edgecolor='g',
                    linewidth=3))

    def gen_trajectory_points(self, x, y, theta, v, omega, dt):
        length = int(1 / dt + 1)
        t_array = np.linspace(0, 1, length)
        theta_array = theta + omega * t_array
        x_array = np.zeros(length)
        x_array[0] = x
        y_array = np.zeros(length)
        y_array[0] = y

        if (omega == 0):
            for i in range(1, length):
                x_array[i] = x_array[i - 1] + v * np.cos(theta_array[i]) * dt
            for i in range(1, length):
                y_array[i] = y_array[i - 1] + v * np.sin(theta_array[i]) * dt

        else:
            for i in range(1, length):
                x_array[i] = x_array[i - 1] - v / omega * (np.sin(theta_array[i]) - np.sin(theta_array[i] + omega * dt))
            for i in range(1, length):
                y_array[i] = y_array[i - 1] + v / omega * (np.cos(theta_array[i]) - np.cos(theta_array[i] + omega * dt))

        return x_array, y_array

    def draw_trajectory(self, x_array, y_array, style='r--'):
        self.ax.plot(x_array, y_array, style)

    def show(self):
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        plt.show()

    def save(self):
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        plt.savefig('fig.png')


class Simulator:
    def __init__(self, robot_state, robot_constrain, grid_map, dest):
        self.robot_state = robot_state
        self.robot_constrain = robot_constrain
        self.grid_map = grid_map
        self.dest = dest

        self.grid_map.ax.scatter(robot_state.x, robot_state.y, color='r')
        self.grid_map.ax.scatter(dest[0], dest[1], color='b')
