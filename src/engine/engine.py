from enum import Enum, auto
from common.packages import *

# Enum for simulation schemes
class SimulationScheme(Enum):
    EULER = 1
    MILSTEIN = 2
    ANALYTICAL = 3


# Monte Carlo engine using AAD
class MonteCarloEngine:
    def __init__(self, simulation_timeline,simulation_type, model, num_paths, num_steps, is_pre_simulation=False):
        self.simulation_type = simulation_type
        self.model = model
        self.num_paths = num_paths
        self.num_steps = num_steps  # FIX: Define this before using it
        self.simulation_timeline=simulation_timeline

        torch.manual_seed(42 if is_pre_simulation else 43)
    
    def generate_paths(self):

        if self.simulation_type==SimulationScheme.ANALYTICAL:
            return self.model.generate_paths_analytically(self.simulation_timeline,self.num_paths, self.num_steps)
        elif self.simulation_type==SimulationScheme.EULER:
            return self.model.generate_paths_euler(self.simulation_timeline,self.num_paths, self.num_steps)

    
