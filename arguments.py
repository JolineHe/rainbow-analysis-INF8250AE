__credits__ = ["Intelligent Unmanned Systems Laboratory at Westlake University."]
'''
Specify parameters of the env
'''
import argparse
from typing import Union

import numpy as np

parser = argparse.ArgumentParser("Grid World Environment")

## ==================== User settings ===================='''
# specify the number of columns and rows of the grid world
parser.add_argument("--env-size", type=Union[list, tuple, np.ndarray], default=(10,10) )   

# specify the start state
parser.add_argument("--start-state", type=Union[list, tuple, np.ndarray], default=(7,0))     #column-1, row-1

# specify the target state
parser.add_argument("--target-state", type=Union[list, tuple, np.ndarray], default=(6,9))

# sepcify the forbidden states
parser.add_argument("--forbidden-states", type=list, default=[(5,0),(8,0),(8,1),(8,2),(7,3),(6,2), (1,3), (2, 1), (5, 3),(4,6),(3,5),(5,5), (6,5),(7,7)] )

# sepcify the reward when reaching target
parser.add_argument("--reward-target", type=float, default = 10)

# sepcify the reward when entering into forbidden area
parser.add_argument("--reward-forbidden", type=float, default = -5)

# sepcify the reward for each step
parser.add_argument("--reward-step", type=float, default = -1)
## ==================== End of User settings ====================


## ==================== Advanced Settings ====================
# parser.add_argument("--action-space", type=list, default=[(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)] )  # down, right, up, left, stay        
parser.add_argument("--action-space", type=list, default=[(0, 1), (1, 0), (0, -1), (-1, 0)] )  # down, right, up, left 
parser.add_argument("--debug", type=bool, default=False)
parser.add_argument("--animation-interval", type=float, default = 0.2)
## ==================== End of Advanced settings ====================


parser.add_argument("--agent", type=str, choices=["dqn", "ddqn", "dueling_ddqn", "multistep_dqn", 'prioritized_ddqn', 'distributional_dqn', 'noise_dqn', 'a3c', 'rainbow'], default="distributional_dqn", help="Specify the agent type.")
parser.add_argument("--train", type=bool, default=True)
parser.add_argument("--test", type=bool, default=False)


args = parser.parse_args()     
def validate_environment_parameters(env_size, start_state, target_state, forbidden_states):
    if not (isinstance(env_size, tuple) or isinstance(env_size, list) or isinstance(env_size, np.ndarray)) and len(env_size) != 2:
        raise ValueError("Invalid environment size. Expected a tuple (rows, cols) with positive dimensions.")
    
    for i in range(2):
        assert start_state[i] < env_size[i]
        assert target_state[i] < env_size[i]
        for j in range(len(forbidden_states)):
            assert forbidden_states[j][i] < env_size[i]
try:
    validate_environment_parameters(args.env_size, args.start_state, args.target_state, args.forbidden_states)
except ValueError as e:
    print("Error:", e)