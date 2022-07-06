from simulator import GridMap, Robot
import numpy as np
import logging


class BaseAlgo:
    def __init__(self):
        self.logger = logging.getLogger("ROBOT_NAVIGATION_GLOBAL")
        self.logger.setLevel(logging.DEBUG)


class Baseline(BaseAlgo):
    def __init__(self, robot: Robot, grid_map: GridMap, num_v: int, num_omega: int, height_threshold: float, dt: float):
        super().__init__()
        self.robot = robot
        self.grid_map = grid_map
        assert num_v % 2 == 1
        assert num_omega % 2 == 1
        self.num_v = num_v
        self.num_omega = num_omega
        self.height_threshold = height_threshold
        self.dt = dt
        self.time_length = int(1 / dt)

    # def get_obst_dist(self, x, y):
    #     return np.min(np.linalg.norm(np.array([x, y]) - self.grid_map.obst_coor, axis=1))

    def _check_traj_obst(self, x_array, y_array):
        assert x_array.size == y_array.size

        # obstacles in trajectory
        for i in range(x_array.size):
            try:
                if self.grid_map.data[int(y_array[i]), int(x_array[i])] >= self.height_threshold:
                    return False
            except IndexError:
                return False

        # robot cannot stop before it reaches to nearest obstacle
        # if
        return True

    def _gen_dynamic_space(self):

        min_v = max(self.robot.v - self.robot.max_dv, 1)
        max_v = min(self.robot.v + self.robot.max_dv, self.robot.max_v)

        min_omega = max(self.robot.omega - self.robot.max_domega, -self.robot.max_omega)
        max_omega = min(self.robot.omega + self.robot.max_domega, self.robot.max_omega)

        self.logger.warning("min_v:{}, max_v:{}, min_omega:{}, max_omega:{}".format(min_v, max_v, min_omega, max_omega))

        v_space = np.linspace(min_v, max_v, self.num_v) if self.num_v != 1 else np.array(self.robot.v).reshape((1, ))
        omega_space = np.linspace(min_omega, max_omega, self.num_omega) if self.num_omega != 1 else np.array(
            self.robot.omega).reshape((1, ))

        return v_space, omega_space

    def _gen_dynamic_trajectory(self, v_array, omega_array):
        trajectories_x = np.empty((self.num_v, self.num_omega, self.time_length))
        trajectories_y = np.empty((self.num_v, self.num_omega, self.time_length))
        faisible_idx = np.ones((self.num_v, self.num_omega), dtype='bool')
        for i in range(self.num_v):
            for j in range(self.num_omega):
                trajectories_x[i][j], trajectories_y[i][j] = self.grid_map.gen_trajectory_points(
                    self.robot.x, self.robot.y, self.robot.theta, v_array[i], omega_array[j], self.dt, 1)
                faisible_idx[i][j] = self._check_traj_obst(trajectories_x[i][j], trajectories_y[i][j])
        return trajectories_x, trajectories_y, faisible_idx

    def __call__(self):
        v_array, omega_array = self._gen_dynamic_space()
        trajectories_x, trajectories_y, faisible_idx = self._gen_dynamic_trajectory(v_array, omega_array)
        for i in range(self.num_v):
            for j in range(self.num_omega):
                # self.grid_map.draw_trajectory_old(self.robot.x, self.robot.y, self.robot.theta,
                #                                   v_array[i], omega_array[j])
                if faisible_idx[i][j]:
                    self.grid_map.draw_trajectory(trajectories_x[i][j], trajectories_y[i][j], 'g--', linewidth=0.25)
                else:
                    self.grid_map.draw_trajectory(trajectories_x[i][j], trajectories_y[i][j], 'r--', linewidth=0.25)
