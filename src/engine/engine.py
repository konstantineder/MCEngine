from enum import Enum, auto
from common.packages import *
from models.model import Model
from typing import Union, List, Optional, Sequence

# Enum for simulation schemes
class SimulationScheme(Enum):
    EULER = 0
    MILSTEIN = 1
    ANALYTICAL = 2


# Monte Carlo engine to simulate paths for pre and main sim
class MonteCarloEngine:
    def __init__(self, 
                 simulation_timeline    : torch.Tensor,
                 simulation_type        : SimulationScheme,
                 model                  : Model, 
                 num_paths              : int, 
                 num_steps              : int, 
                 is_pre_simulation      : bool = False
                 ):
        
        self.simulation_type = simulation_type
        self.model = model
        self.num_paths = num_paths
        self.num_steps = num_steps  
        self.simulation_timeline=simulation_timeline

        torch.manual_seed(42 if is_pre_simulation else 43)
    
    def generate_paths(self):

        if self.simulation_type==SimulationScheme.ANALYTICAL:
            return self.model.generate_paths_analytically(self.simulation_timeline,self.num_paths, self.num_steps)
        elif self.simulation_type==SimulationScheme.EULER:
            return self.model.generate_paths_euler(self.simulation_timeline,self.num_paths, self.num_steps)

    
