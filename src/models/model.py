from common.packages import *

# Base model class
class Model:
    def __init__(self, 
                 calibration_date : float # Calibration date of the model
                 ):
        
        self.calibration_date = torch.tensor([calibration_date], dtype=FLOAT,device=device)
        self.model_params=[]
        self.num_assets=1
    
    def get_model_params(self):
        return self.model_params
    
    # If differentiation is enabled, put all model parameters on tape
    # and accumulate adjoints during simulation via AAD
    def requires_grad(self):
        for param in self.model_params:
            param.requires_grad_(True)

    def generate_paths_analytically(self, timeline, num_paths, num_steps):
        return NotImplementedError(f"Method not implemented")
    
    def generate_paths_euler(self, timeline, num_paths, num_steps):
        return NotImplementedError(f"Method not implemented")
    
    def generate_paths_milstein(self, timeline, num_paths, num_steps):
        return NotImplementedError(f"Method not implemented")
    
    def resolve_request(self, state, requests):
        return NotImplementedError(f"Method not implemented")



    




    
