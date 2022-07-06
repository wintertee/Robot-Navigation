from simulator import Simulator, Robot, GridMap
from algorithms import Baseline
import numpy as np
if __name__ == '__main__':
    robot = Robot(x=1.5,
                  y=1.2,
                  theta=np.pi / 6,
                  v=4,
                  omega=np.pi / 6,
                  max_v=10,
                  max_dv=3,
                  max_omega=np.pi / 2,
                  max_domega=np.pi / 2)
    block_rate = 0.1
    simulator = Simulator(robot, GridMap(grid_size=10), dest=(9, 9))
    simulator.grid_map.gen_map_bool(block_rate=block_rate, seed=1)

    algo = Baseline(robot=simulator.robot,
                    grid_map=simulator.grid_map,
                    num_v=9,
                    num_omega=9,
                    height_threshold=1 - block_rate,
                    dt=0.1)
    algo()

    simulator.grid_map.save()
