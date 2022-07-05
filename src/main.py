from simulator import Simulator, RobotConstrain, RobotState, GridMap
from algorithms import Baseline
import numpy as np
if (__name__ == '__main__'):
    simulator = Simulator(RobotState(1, 1, np.pi / 6, 4, np.pi / 6),
                          RobotConstrain(max_v=10, max_dv=5, max_omega=np.pi / 2),
                          GridMap(grid_size=10),
                          dest=(9, 9))
    simulator.grid_map.gen_map_bool(block_rate=0.1, seed=1)

    algo = Baseline(simulator.robot_state, simulator.robot_constrain, simulator.grid_map, 3, 5, 0.9, 0.02)
    algo()

    simulator.grid_map.save()
