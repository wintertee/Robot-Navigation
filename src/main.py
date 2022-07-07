from simulator import Simulator, Robot, GridMap
from algorithms import Baseline
import numpy as np

if __name__ == "__main__":
    robot = Robot(
        x=1.5,
        y=5.2,
        theta=-np.pi / 6,
        v=4,
        omega=np.pi / 6,
        max_v=10,
        max_dv=3,
        max_omega=np.pi / 2,
        max_domega=np.pi / 2,
    )
    block_rate = 0.1
    simulator = Simulator(robot, GridMap(grid_size=10), dest=(8.7, 9.2))
    simulator.grid_map.gen_map_bool(block_rate=block_rate, seed=None)

    algo = Baseline(
        robot=simulator.robot,
        grid_map=simulator.grid_map,
        dest=simulator.dest,
        num_v=15,
        num_omega=15,
        height_threshold=1 - block_rate,
        dt=0.1,
    )
    algo(show_faisible=True, show_non_faisible=False)

    simulator.grid_map.save()
