from models.model import *
from request_interface.request_interface import *
from typing import List, Any

# Vasicek 1-factor model for stochastic interest rates using Ornstein-Uhlenbeck process
class VasicekModel(Model):
    def __init__(self, 
                 calibration_date     : float,  # Calibration date of the model
                 rate                 : float,  # Short rate at calibration date
                 mean                 : float,  # Long-term mean level the rate reverts to
                 mean_reversion_speed : float,  # Speed at which the rate reverts to the long-term mean
                 volatility           : float   # Volatility of the short rate
                 ):
        
        super().__init__(calibration_date)
        # Collect all model parameters in common PyTorch tensor
        # If AAD is enabled the respective adjoints are accumulated
        self.model_params = [
            torch.tensor(param, dtype=FLOAT, device=device)
            for param in list([rate]) + list([volatility]) + list([mean]) + list([mean_reversion_speed])
        ]

    # Retrieve specific model parameters
    def get_rate(self):
        return torch.stack([self.model_params[0]])

    def get_volatility(self):
        return torch.stack([self.model_params[1]])

    def get_mean(self):
        return torch.stack([self.model_params[2]]) 
    
    def get_mean_reversion_speed(self):
        return torch.stack([self.model_params[3]])     

    # Simulate Monte Carlo paths using analytic formula
    def generate_paths_analytically(self, timeline, num_paths, num_steps):

        r_t = self.get_rate().expand(num_paths).clone()
        a=self.get_mean_reversion_speed()
        theta=self.get_mean()
        sigma=self.get_volatility()
        log_B_t = torch.full((num_paths,), 0., dtype=FLOAT, device=device)
        paths = []

        t_start=self.calibration_date

        for i in range(len(timeline)):
            t_end = timeline[i]
            dt_total = t_end - t_start
            dt = dt_total / num_steps

            for _ in range(num_steps):
                log_B_t+=r_t*dt
                exp_decay = torch.exp(-a * dt)
                mean = r_t * exp_decay + theta*(1-exp_decay)
                variance = (sigma ** 2 / (2 * a)) * (1 - exp_decay ** 2)

                noise = torch.randn(num_paths, device=device, dtype=FLOAT)
                r_t = mean + torch.sqrt(variance) * noise

            # Each time point: [num_paths, 2]
            states = torch.stack([r_t, log_B_t], dim=1)  
            paths.append(states)
            t_start=t_end
        paths = torch.stack(paths, dim=0)      
        paths = paths.permute(1, 0, 2)         

        return paths  
    
    # Simulate Monte Carlo paths using Euler-Mayurama scheme
    def generate_paths_euler(self, timeline, num_paths, num_steps):
        rate=self.get_rate()
        r_t = rate.expand(num_paths).clone()
        a=self.get_mean_reversion_speed()
        theta=self.get_mean()
        sigma=self.get_volatility()
        log_B_t = torch.full((num_paths,), 0., dtype=FLOAT, device=device)
        paths = []

        t_start=self.calibration_date

        for i in range(len(timeline)):
            t_end = timeline[i]
            dt_total = t_end - t_start
            dt = dt_total / num_steps

            for _ in range(num_steps):
                log_B_t+=r_t*dt
                drift = a*(theta - r_t)*dt
                z = torch.randn(num_paths, device=device, dtype=FLOAT)
                diffusion = sigma * torch.sqrt(dt) * z

                r_t += drift + diffusion

            # Each time point: [num_paths, 2]
            states = torch.stack([r_t, log_B_t], dim=1)  
            paths.append(states)
            t_start=t_end
        paths = torch.stack(paths, dim=0)      
        paths = paths.permute(1, 0, 2)         

        return paths  
    
    # Use analytic formula for Zerobond price in Vasicek model
    def compute_bond_price(self, time1, time2, rate):
        dt = time2 - time1
        a=self.get_mean_reversion_speed()
        theta=self.get_mean()
        sigma=self.get_volatility()
        B = (1 - torch.exp(-a * dt)) / a

        term1 = (theta - (sigma**2 / (2 * a**2))) 

        # A(t,T) = exp(alpha(t,T))
        alpha_tt = term1 * (B - dt) - (sigma**2 / (4 * a)) * B**2
        A = torch.exp(alpha_tt)

        return A * torch.exp(-B * rate)
    
    # Resolve requests posed by all products and at each exposure timepoint
    def resolve_request(self, req, state):

        if req.request_type == AtomicRequestType.SPOT:
            return state[:,0]
        elif req.request_type == AtomicRequestType.DISCOUNT_FACTOR:
            time = req.time1
            rate=state[:,0]
            return self.compute_bond_price(self.calibration_date,time,rate)
        elif req.request_type == AtomicRequestType.FORWARD_RATE:
            time1 = req.time1
            time2 = req.time2
            rate = state[:,0]
            return self.compute_bond_price(time1,time2,rate)
        elif req.request_type == AtomicRequestType.LIBOR_RATE:
            time1 = req.time1
            time2 = req.time2
            rate=state[:,0]
            bond_price=self.compute_bond_price(time1,time2,rate)
            return (1/bond_price-1)/(time2-time1)
        elif req.request_type == AtomicRequestType.NUMERAIRE:
            log_B_t=state[:,1]
            return torch.exp(log_B_t)
