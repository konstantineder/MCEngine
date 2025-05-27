from common.packages import *
from enum import Enum

# Enum for option types
class OptionType(Enum):
    CALL = 1
    PUT = 2

class SettlementType(Enum):
    PHYSICAL=0
    CASH=1

# Abstract (base) class for financial products
class Product:
    def __init__(self, product_id=0):
        self.product_id=product_id

    def get_num_states(self):
        return 1
    
    def get_initial_state(self, num_paths):
        return torch.full((num_paths,), 0, dtype=torch.long, device=device)

    # Abstract method to compute the payoff for the specific product 
    def compute_payoff(self, paths, model):
        raise NotImplementedError

    # Abstract method to compute the pv using an analytic formula  
    def compute_pv_analytically(self, model):
        raise NotImplementedError










    


