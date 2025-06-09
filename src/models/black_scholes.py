from models.model import *
from request_interface.request_interface import ModelRequestType

# AAD-enabled Black-Scholes model
class BlackScholesModel(Model):
    def __init__(self, calibration_date, spot, rate, sigma):
        super().__init__(calibration_date)
        self.model_params = [
            torch.tensor(param, dtype=torch.float64, device=device)
            for param in list([spot]) + list([sigma]) + list([rate])
        ]

    def get_spot(self):
        return torch.stack([self.model_params[0]])

    def get_volatility(self):
        return torch.stack([self.model_params[1]])

    def get_rate(self):
        return torch.stack([self.model_params[2]])

    def generate_paths_analytically(self, timeline, num_paths, num_steps):
        #model_state = self.spot.expand(num_paths)

        spot = self.get_spot().expand(num_paths).clone()
        sigma=self.get_volatility()
        rate=self.get_rate()
        paths = []

        t_start=self.calibration_date

        for i in range(len(timeline)):
            t_end = timeline[i]
            dt_total = t_end - t_start
            dt = dt_total / num_steps

            for _ in range(num_steps):
                z = torch.randn(num_paths, dtype=torch.float64, device=device)
                drift=rate*dt
                diffusion=sigma*torch.sqrt(dt)*z-0.5*dt*sigma**2
                spot = spot*torch.exp(drift+diffusion)

            paths.append(spot)
            t_start=t_end

        return torch.stack(paths, dim=1)
    
    def generate_paths_euler(self, timeline, num_paths, num_steps):
        spot = self.spot.expand(num_paths).clone()
        sigma = self.get_volatility()
        rate= self.get_rate()
        paths = []

        t_start=self.calibration_date

        for i in range(len(timeline)):
            t_start = timeline[i]
            t_end = timeline[i + 1]
            dt_total = t_end - t_start
            dt = dt_total / num_steps

            for _ in range(num_steps):
                z = torch.randn(num_paths, device=device)
                dS = rate * spot * dt + sigma * spot * torch.sqrt(dt) * z
                spot = spot + dS

            paths.append(spot)
            t_start=t_end

        return torch.stack(paths, dim=1)  # shape: [num_paths, len(timeline)]
    
    def resolve_request(self, req, state):

        if req.request_type == ModelRequestType.SPOT:
            return state
        elif req.request_type == ModelRequestType.DISCOUNT_FACTOR:
            t = req.time1
            rate=self.get_rate()
            return torch.exp(-rate * (t - self.calibration_date))
        elif req.request_type == ModelRequestType.FORWARD_RATE:
            t1 = req.time1
            t2 = req.time2
            rate=self.get_rate()
            return torch.exp(rate * (t2-t1))
        elif req.request_type == ModelRequestType.LIBOR_RATE:
            t1 = req.time1
            t2 = req.time2
            rate=self.get_rate()
            return (torch.exp(rate * (t2-t1))-1)/(t2-t1)
        elif req.request_type == ModelRequestType.NUMERAIRE:
            t = req.time1
            rate=self.get_rate()
            return torch.exp(rate * (t - self.calibration_date))
            # Add more request types as needed