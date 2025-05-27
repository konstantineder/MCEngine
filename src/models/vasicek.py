from models.model import *
from request_interface.request_interface import *

# AAD-enabled Black-Scholes model
class VasicekModel(Model):
    def __init__(self, calibration_date, rate, mean, mean_reversion_speed, volatility):
        super().__init__(calibration_date)
        self.rate = torch.tensor([rate], dtype=torch.float64, requires_grad=True, device=device)
        self.theta = torch.tensor([mean], dtype=torch.float64, requires_grad=True, device=device)  # mean reversion speed
        self.a=torch.tensor([mean_reversion_speed], dtype=torch.float64, requires_grad=True, device=device)
        self.sigma = torch.tensor([volatility], dtype=torch.float64, requires_grad=True, device=device)  # volatility 
    
    def get_model_params(self):
        return [self.rate,self.sigma,self.theta, self.a]

    def generate_paths_analytically(self, timeline, num_paths, num_steps):
        #model_state = self.spot.expand(num_paths)

        r_t = self.rate.expand(num_paths).clone()
        log_B_t = torch.full((num_paths,), 0., dtype=torch.float64, device=device)
        paths = []

        t_start=self.calibration_date

        for i in range(len(timeline)):
            t_end = timeline[i]
            dt_total = t_end - t_start
            dt = dt_total / num_steps

            for _ in range(num_steps):
                log_B_t+=r_t*dt
                exp_decay = torch.exp(-self.a * dt)
                mean = r_t * exp_decay + self.theta*(1-exp_decay)
                variance = (self.sigma ** 2 / (2 * self.a)) * (1 - exp_decay ** 2)

                noise = torch.randn(num_paths, device=device, dtype=torch.float64)
                r_t = mean + torch.sqrt(variance) * noise

            # Each time point: [num_paths, 2]
            states = torch.stack([r_t, log_B_t], dim=1)  # [num_paths, 2]
            paths.append(states)
            t_start=t_end
        paths = torch.stack(paths, dim=0)      # [num_times, num_paths, 2]
        paths = paths.permute(1, 0, 2)         # [num_paths, num_times, 2]

        return paths  # [num_paths, num_times, 2]
    
    def generate_paths_euler(self, timeline, num_paths, num_steps):
        r_t = self.rate.expand(num_paths).clone()
        log_B_t = torch.full((num_paths,), 0., dtype=torch.float64, device=device)
        paths = []

        t_start=self.calibration_date

        for i in range(len(timeline)):
            t_end = timeline[i]
            dt_total = t_end - t_start
            dt = dt_total / num_steps

            for _ in range(num_steps):
                log_B_t+=r_t*dt
                drift = self.a*(self.theta - r_t)*dt
                z = torch.randn(num_paths, device=device, dtype=torch.float64)
                diffusion = self.sigma * torch.sqrt(dt) * z

                r_t += drift + diffusion

            # Each time point: [num_paths, 2]
            states = torch.stack([r_t, log_B_t], dim=1)  # [num_paths, 2]
            paths.append(states)
            t_start=t_end
        paths = torch.stack(paths, dim=0)      # [num_times, num_paths, 2]
        paths = paths.permute(1, 0, 2)         # [num_paths, num_times, 2]

        return paths  # [num_paths, num_times, 2]
    
    def compute_bond_price(self, time1, time2, rate):
        dt = time2 - time1
        B = (1 - torch.exp(-self.a * dt)) / self.a

        # Common intermediate term for A
        term1 = (self.theta - (self.sigma**2 / (2 * self.a**2))) # Often denoted as 'theta' in some derivations

        # A(t,T) = exp(alpha(t,T))
        alpha_tt = term1 * (B - dt) - (self.sigma**2 / (4 * self.a)) * B**2
        A = torch.exp(alpha_tt)

        return A * torch.exp(-B * rate)
    
    def resolve_request(self, req, state):

        if req.request_type == ModelRequestType.SPOT:
            return state[:,0]
        elif req.request_type == ModelRequestType.DISCOUNT_FACTOR:
            time = req.time1
            rate=state[:,0]
            return self.compute_bond_price(self.calibration_date,time,rate)
        elif req.request_type == ModelRequestType.FORWARD_RATE:
            time1 = req.time1
            time2 = req.time2
            rate = state[:,0]
            return self.compute_bond_price(time1,time2,rate)
        elif req.request_type == ModelRequestType.LIBOR_RATE:
            time1 = req.time1
            time2 = req.time2
            rate=state[:,0]
            bond_price=self.compute_bond_price(time1,time2,rate)
            return (1/bond_price-1)/(time2-time1)
        elif req.request_type == ModelRequestType.NUMERAIRE:
            log_B_t=state[:,1]
            return torch.exp(log_B_t)
