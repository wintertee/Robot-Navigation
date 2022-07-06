from simulator import GridMap, Robot, Trajectory
import numpy as np
import logging


class BaseAlgo:
    def __init__(self):
        self.logger = logging.getLogger("ROBOT_NAVIGATION_GLOBAL")
        self.logger.setLevel(logging.DEBUG)


class Baseline(BaseAlgo):
    def __init__(
        self,
        robot: Robot,
        grid_map: GridMap,
        num_v: int,
        num_omega: int,
        height_threshold: float,
        dt: float,
    ):
        super().__init__()
        self.robot = robot
        self.grid_map = grid_map
        assert num_v % 2 == 1
        assert num_omega % 2 == 1
        self.num_v = num_v
        self.num_omega = num_omega
        self.height_threshold = height_threshold
        self.dt = dt

    def _check_traj_obst(self, x_array, y_array):
        assert x_array.size == y_array.size

        # obstacles in trajectory
        try:
            for i in range(x_array.size):
                if self.grid_map.data[int(y_array[i]), int(x_array[i])] >= self.height_threshold:
                    return False
        except IndexError:
            # robot will move out of map
            return False

        return True

    def _gen_dynamic_space(self):

        min_v = max(self.robot.v - self.robot.max_dv, 1)
        max_v = min(self.robot.v + self.robot.max_dv, self.robot.max_v)

        min_omega = max(self.robot.omega - self.robot.max_domega, -self.robot.max_omega)
        max_omega = min(self.robot.omega + self.robot.max_domega, self.robot.max_omega)

        self.logger.warning("min_v:{}, max_v:{}, min_omega:{}, max_omega:{}".format(min_v, max_v, min_omega, max_omega))

        v_space = np.linspace(min_v, max_v, self.num_v) if self.num_v != 1 else np.array(self.robot.v).reshape((1,))
        omega_space = (
            np.linspace(min_omega, max_omega, self.num_omega)
            if self.num_omega != 1
            else np.array(self.robot.omega).reshape((1,))
        )

        self.logger.warning(v_space)
        self.logger.warning(omega_space)

        return v_space, omega_space

    def _gen_dynamic_traj(self, v_array, omega_array):
        traj_array = [
            [Trajectory() for j in range(self.num_omega)] for i in range(self.num_v)
        ]  # shape = (num_v, num_omega)
        for i in range(self.num_v):
            for j in range(self.num_omega):
                traj_array[i][j].init(
                    robot=self.robot,
                    v=v_array[i],
                    omega=omega_array[j],
                    dt=self.dt,
                    t=1 + v_array[i] / self.robot.max_dv,  # prediction until robot stop moving
                )
                traj_array[i][j].gen_traj_points()
                faisible = self._check_traj_obst(traj_array[i][j].x_array, traj_array[i][j].y_array)

                traj_array[i][j].set_faisible(faisible)
        return traj_array

    def __call__(self, show_faisible: bool = True, show_non_faisible: bool = False):
        v_array, omega_array = self._gen_dynamic_space()
        traj_array = np.array(self._gen_dynamic_traj(v_array, omega_array)).flatten()
        for traj in traj_array:
            if traj.faisible:
                if show_faisible:
                    self.grid_map.draw_traj(
                        traj,
                        None,
                        traj.unit_time_length,
                        "g-",
                        linewidth=0.5,
                    )
                    self.grid_map.draw_traj(
                        traj,
                        traj.unit_time_length - 1,
                        None,
                        "g:",
                        linewidth=0.25,
                    )
            else:
                if show_non_faisible:
                    self.grid_map.draw_traj(
                        traj,
                        None,
                        traj.unit_time_length,
                        "r-",
                        linewidth=0.5,
                    )
                    self.grid_map.draw_traj(
                        traj,
                        traj.unit_time_length - 1,
                        None,
                        "r:",
                        linewidth=0.25,
                    )
