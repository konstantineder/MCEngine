from model import *
from request_interface.request_interface import *

class HullWhiteModel(Model):
    def __init__(self, calibration_date, rate, speed, mean_reversion, volatility):
        super().__init__(calibration_date)
        self.rate = rate
        self.mean_reversion = mean_reversion  # mean reversion speed
        self.sigma = volatility  # volatility
    
    def get_model_params(self):
        return [self.rate,self.sigma,self.mean_reversion]

    def generate_paths_analytically(self, timeline, num_paths, num_steps):
        #model_state = self.spot.expand(num_paths)

        r_t, log_B_t = self.rate.expand(num_paths).clone()
        paths = []

        t_start=self.calibration_date

        for i in range(len(timeline)):
            t_end = timeline[i]
            dt_total = t_end - t_start
            dt = dt_total / num_steps

            for _ in range(num_steps):
                log_B_t=r_t*dt
                exp_decay = torch.exp(-self.mean_reversion * dt)
                mean = r_t * exp_decay
                variance = (self.sigma ** 2 / (2 * self.mean_reversion)) * (1 - exp_decay ** 2)

                noise = torch.randn(num_paths, device=device, dtype=torch.float64)
                r_t = mean + torch.sqrt(variance) * noise

            paths.append([r_t,log_B_t])

        return torch.stack(paths, dim=1)
    
    def generate_paths_euler(self, timeline, num_paths, num_steps):
        paths = []


        return torch.stack(paths, dim=1)  # shape: [num_paths, len(timeline)]
    
    def compute_bond_price(self,time1, time2, rate):
        B = (1 - torch.exp(-self.mean_reversion * (time2 - time1))) / self.mean_reversion
        A = torch.exp((B - time2 + time1) * rate - (self.sigma**2 / (4 * self.mean_reversion)) * B**2)
        return A * torch.exp(-B * rate)
    
    def resolve_request(self, req, state):

        if req.request_type == RequestType.SPOT:
            return state[0]
        elif req.request_type == RequestType.DISCOUNT_FACTOR:
            time = req.time1
            return self.compute_bond_price(self.calibration_date,time,state[0])
        elif req.request_type == RequestType.FORWARD_RATE:
            time1 = req.time1
            time2 = req.time2
            return self.compute_bond_price(time1,time2,state[0])
        elif req.request_type == RequestType.LIBOR_RATE:
            time1 = req.time1
            time2 = req.time2
            bond_price=self.compute_bond_price(time1,time2,state[0])
            return (1/bond_price-1)/(time2-time1)
        elif req.request_type == RequestType.NUMERAIRE:
            log_B_t=state[1]
            return torch.exp(log_B_t)
