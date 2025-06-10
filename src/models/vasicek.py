from models.model import *
from request_interface.request_interface import *

# AAD-enabled Black-Scholes model
class VasicekModel(Model):
    def __init__(self, calibration_date, rate, mean, mean_reversion_speed, volatility):
        super().__init__(calibration_date)
        self.model_params = [
            torch.tensor(param, dtype=torch.float64, device=device)
            for param in list([rate]) + list([volatility]) + list([mean]) + list([mean_reversion_speed])
        ]

    def get_rate(self):
        return torch.stack([self.model_params[0]])

    def get_volatility(self):
        return torch.stack([self.model_params[1]])

    def get_mean(self):
        return torch.stack([self.model_params[2]]) 
    
    def get_mean_reversion_speed(self):
        return torch.stack([self.model_params[3]])     

    def generate_paths_analytically(self, timeline, num_paths, num_steps):
        #model_state = self.spot.expand(num_paths)

        r_t = self.get_rate().expand(num_paths).clone()
        a=self.get_mean_reversion_speed()
        theta=self.get_mean()
        sigma=self.get_volatility()
        log_B_t = torch.full((num_paths,), 0., dtype=torch.float64, device=device)
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
        rate=self.get_rate()
        r_t = rate.expand(num_paths).clone()
        a=self.get_mean_reversion_speed()
        theta=self.get_mean()
        sigma=self.get_volatility()
        log_B_t = torch.full((num_paths,), 0., dtype=torch.float64, device=device)
        paths = []

        t_start=self.calibration_date

        for i in range(len(timeline)):
            t_end = timeline[i]
            dt_total = t_end - t_start
            dt = dt_total / num_steps

            for _ in range(num_steps):
                log_B_t+=r_t*dt
                drift = a*(theta - r_t)*dt
                z = torch.randn(num_paths, device=device, dtype=torch.float64)
                diffusion = sigma * torch.sqrt(dt) * z

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
        a=self.get_mean_reversion_speed()
        theta=self.get_mean()
        sigma=self.get_volatility()
        B = (1 - torch.exp(-a * dt)) / a

        # Common intermediate term for A
        term1 = (theta - (sigma**2 / (2 * a**2))) # Often denoted as 'theta' in some derivations

        # A(t,T) = exp(alpha(t,T))
        alpha_tt = term1 * (B - dt) - (sigma**2 / (4 * a)) * B**2
        A = torch.exp(alpha_tt)

        return A * torch.exp(-B * rate)
    
    def resolve_request(self, req, state):

        if req.request_type == RequestType.SPOT:
            return state[:,0]
        elif req.request_type == RequestType.DISCOUNT_FACTOR:
            time = req.time1
            rate=state[:,0]
            return self.compute_bond_price(self.calibration_date,time,rate)
        elif req.request_type == RequestType.FORWARD_RATE:
            time1 = req.time1
            time2 = req.time2
            rate = state[:,0]
            return self.compute_bond_price(time1,time2,rate)
        elif req.request_type == RequestType.LIBOR_RATE:
            time1 = req.time1
            time2 = req.time2
            rate=state[:,0]
            bond_price=self.compute_bond_price(time1,time2,rate)
            return (1/bond_price-1)/(time2-time1)
        elif req.request_type == RequestType.NUMERAIRE:
            log_B_t=state[:,1]
            return torch.exp(log_B_t)
