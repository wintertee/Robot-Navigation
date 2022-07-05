from simulator import GridMap, RobotState, RobotConstrain
import numpy as np


class BaseAlgo:
    def __init__(self):
        raise NotImplementedError


class Baseline(BaseAlgo):
    def __init__(self, robot_state: RobotState, robot_constrain: RobotConstrain, grid_map: GridMap, res_v: int,
                 res_omega: int, height_threshold: float, dt: float):
        self.robot_state = robot_state
        self.robot_constrain = robot_constrain
        self.grid_map = grid_map
        assert res_v % 2 == 1
        assert res_omega % 2 == 1
        self.res_v = res_v
        self.res_omega = res_omega
        self.height_threshold = height_threshold
        self.dt = dt
        self.time_length = int(1 / dt + 1)

    def check_traj_feasible(self, x_array, y_array):
        assert x_array.size == y_array.size
        for i in range(x_array.size):
            if (self.grid_map.data[int(y_array[i]), int(x_array[i])] >= self.height_threshold):
                return False
        return True

    def _gen_dynamic_space(self):
        min_v = max(self.robot_state.v - self.robot_constrain.max_dv, 0)
        max_v = min(self.robot_state.v + self.robot_constrain.max_dv, self.robot_constrain.max_v)

        min_omega = -self.robot_constrain.max_omega
        max_omega = self.robot_constrain.max_omega

        v_space = np.linspace(min_v, max_v, self.res_v) if self.res_v != 1 else np.array(self.robot_state.v).reshape(
            (1, ))
        omega_space = np.linspace(min_omega, max_omega, self.res_omega) if self.res_omega != 1 else np.array(
            self.robot_state.omega).reshape((1, ))

        return v_space, omega_space

    def _gen_dynamic_trajectory(self, v_array, omega_array):
        trajectories_x = np.empty((self.res_v, self.res_omega, self.time_length))
        trajectories_y = np.empty((self.res_v, self.res_omega, self.time_length))
        faisible_idx = np.ones((self.res_v, self.res_omega), dtype='bool')
        for i in range(self.res_v):
            for j in range(self.res_omega):
                trajectories_x[i][j], trajectories_y[i][j] = self.grid_map.gen_trajectory_points(
                    self.robot_state.x, self.robot_state.y, self.robot_state.theta, v_array[i], omega_array[j], self.dt)
                faisible_idx[i][j] = self.check_traj_feasible(trajectories_x[i][j], trajectories_y[i][j])
        return trajectories_x, trajectories_y, faisible_idx

    def __call__(self):
        v_array, omega_array = self._gen_dynamic_space()
        trajectories_x, trajectories_y, faisible_idx = self._gen_dynamic_trajectory(v_array, omega_array)
        for i in range(self.res_v):
            for j in range(self.res_omega):
                if faisible_idx[i][j]:
                    self.grid_map.draw_trajectory(trajectories_x[i][j], trajectories_y[i][j], style='g--')
                else:
                    self.grid_map.draw_trajectory(trajectories_x[i][j], trajectories_y[i][j], style='r--')
