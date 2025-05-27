from common.packages import *

# AAD-enabled Black-Scholes model
class Model:
    def __init__(self, calibration_date):
        self.calibration_date = torch.tensor([calibration_date], dtype=torch.float64,device=device)
    
    def get_model_params(self):
        return []

    def generate_paths_analytically(self, timeline, num_paths, num_steps):
        return NotImplementedError(f"Method not implemented")
    
    def generate_paths_euler(self, timeline, num_paths, num_steps):
        return NotImplementedError(f"Method not implemented")
    
    def generate_paths_milstein(self, timeline, num_paths, num_steps):
        return NotImplementedError(f"Method not implemented")
    
    def resolve_request(self, state, requests):
        return NotImplementedError(f"Method not implemented")



    




    
