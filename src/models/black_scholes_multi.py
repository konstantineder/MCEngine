from models.model import *
from request_interface.request_interface import ModelRequestType

class BlackScholesMulti(Model):
    def __init__(self, calibration_date, rate, spots, volatilities, correlation_matrix):
        super().__init__(calibration_date)
        self.calibration_date = calibration_date
        self.spots = torch.tensor(spots, dtype=torch.float64, requires_grad=True, device=device)
        self.sigmas = torch.tensor(volatilities, dtype=torch.float64, requires_grad=True, device=device)
        self.rate = torch.tensor(rate, dtype=torch.float64, requires_grad=True, device=device)
        self.correlation_matrix = torch.tensor(correlation_matrix, dtype=torch.float64, device=device)
        self.num_assets=len(spots)
        self.cholesky = {}

    def get_model_params(self):
        return list(self.spots) + list([self.rate]) + list(self.sigmas)

    def compute_cov_matrix(self, delta_t):
        sigma_matrix = torch.outer(self.sigmas, self.sigmas)
        cov_matrix = self.correlation_matrix * sigma_matrix * delta_t
        return cov_matrix

    def compute_cholesky(self, delta_t):
        dt = float(delta_t)
        if dt in self.cholesky:
            return self.cholesky[dt]
        else:
            cov_matrix = self.compute_cov_matrix(delta_t)
            chol = torch.linalg.cholesky(cov_matrix)
            self.cholesky[dt] = chol
            return chol

    def generate_paths_analytically(self, timeline, num_paths, num_steps):
        num_assets = self.num_assets
        paths = []
        spot = self.spots.expand(num_paths, num_assets).clone()

        t_start = self.calibration_date

        for t_end in timeline:
            dt_total = t_end - t_start
            dt = dt_total / num_steps
            cholesky = self.compute_cholesky(dt)

            for _ in range(num_steps):
                z = torch.randn(num_paths, num_assets, dtype=torch.float64, device=device)
                correlated_z = z @ cholesky.T
                L_squared = cholesky.pow(2)
                drift = (self.rate - 0.5 * L_squared.sum(dim=1)) * dt
                diffusion = correlated_z * torch.sqrt(dt)
                spot = spot * torch.exp(drift + diffusion)

            paths.append(spot.clone())
            t_start = t_end

        return torch.stack(paths, dim=1)  # [num_paths, len(timeline), num_assets]

    def generate_paths_euler(self, timeline, num_paths, num_steps):
        num_assets = len(self.spots)
        paths = []
        spot = self.spots.expand(num_paths, num_assets).clone()

        for i in range(len(timeline) - 1):
            t_start = timeline[i]
            t_end = timeline[i + 1]
            dt_total = t_end - t_start
            dt = dt_total / num_steps
            cholesky = self.compute_cholesky(dt)

            for _ in range(num_steps):
                z = torch.randn(num_paths, num_assets, dtype=torch.float64, device=device)
                correlated_z = z @ cholesky.T
                dS = self.rate * spot * dt + self.sigmas * spot * correlated_z * torch.sqrt(dt)
                spot = spot + dS

            paths.append(spot.clone())

        return torch.stack(paths, dim=1)  # [num_paths, len(timeline)-1, num_assets]

    def resolve_request(self, req, state):
        if req.request_type == ModelRequestType.SPOT:
            return state

        elif req.request_type == ModelRequestType.DISCOUNT_FACTOR:
            t = req.time1
            return torch.exp(-self.rate * (t - self.calibration_date))

        elif req.request_type == ModelRequestType.FORWARD_RATE:
            t1, t2 = req.time1, req.time2
            return torch.exp(self.rate * (t2 - t1))

        elif req.request_type == ModelRequestType.LIBOR_RATE:
            t1, t2 = req.time1, req.time2
            return (torch.exp(self.rate* (t2 - t1)) - 1) / (t2 - t1)

        elif req.request_type == ModelRequestType.NUMERAIRE:
            t = req.time1
            return torch.exp(self.rate * (t - self.calibration_date))

        else:
            raise NotImplementedError(f"Request type {req.request_type} not supported.")
