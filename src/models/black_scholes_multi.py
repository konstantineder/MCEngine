from models.model import *
from request_interface.request_interface import AtomicRequestType
from typing import List, Any
from numpy.typing import NDArray

# Black-Scholes Model for multiple assets
class BlackScholesMulti(Model):
    def __init__(self, 
                 calibration_date   : float,        # Calibration date of the model
                 rate               : float,        # Risk-free interest rate
                 spots              : List[float],  # Spot prices of the assets
                 volatilities       : List[float],  # Volatilities of the assets
                 correlation_matrix : NDArray[Any]  # Correlations amonng assets
                 ):
        
        super().__init__(calibration_date)
        # Collect all model parameters in common PyTorch tensor
        # If AAD is enabled the respective adjoints are accumulated
        self.model_params = [
            torch.tensor(param, dtype=torch.float64, device=device)
            for param in list(spots) + list(volatilities) + list([rate])
        ]

        self.correlation_matrix = torch.tensor(correlation_matrix, dtype=torch.float64, device=device)
        self.num_assets=len(spots)

        # Store computed Cholesky matrix for each time delta
        self.cholesky = {}
    
    # Retrieve model parameters
    def get_spot(self):
        return torch.stack(self.model_params[:self.num_assets])

    def get_volatility(self):
        return torch.stack(self.model_params[self.num_assets:2*self.num_assets])

    def get_rate(self):
        return self.model_params[2*self.num_assets]

    # Compute covariance matrix for time delta
    def compute_cov_matrix(self, delta_t):
        S = torch.diag(self.get_volatility())
        cov_matrix = S @ self.correlation_matrix @ S
        cov_matrix = cov_matrix*delta_t
        return cov_matrix

    # Compute Cholesky decomposition of covariance matrix if Cholesky matrix
    # has not yet been computed for current time delta
    # otherwise retrieve stored matrix
    def compute_cholesky(self, delta_t):
        dt = float(delta_t)
        if dt in self.cholesky:
            return self.cholesky[dt]
        else:
            cov_matrix = self.compute_cov_matrix(delta_t)
            chol = torch.linalg.cholesky(cov_matrix)
            self.cholesky[dt] = chol
            return chol

    # Simulate Monte Carlo paths using analytic formula
    def generate_paths_analytically(self, timeline, num_paths, num_steps):
        num_assets = self.num_assets
        paths = []
        spot = self.get_spot().expand(num_paths, num_assets).clone()
        rate = self.get_rate()

        t_start = self.calibration_date

        for t_end in timeline:
            dt_total = t_end - t_start
            dt = dt_total / num_steps
            cholesky = self.compute_cholesky(dt)

            for _ in range(num_steps):
                z = torch.randn(num_paths, num_assets, dtype=torch.float64, device=device)
                correlated_z = z @ cholesky.T
                L_squared = cholesky.pow(2)
                drift = (rate - 0.5 * L_squared.sum(dim=1)) * dt
                diffusion = correlated_z * torch.sqrt(dt)
                spot = spot * torch.exp(drift + diffusion)

            paths.append(spot.clone())
            t_start = t_end

        return torch.stack(paths, dim=1)  # [num_paths, len(timeline), num_assets]

    # Simulate Monte Carlo paths using Euler-Mayurama scheme
    def generate_paths_euler(self, timeline, num_paths, num_steps):
        num_assets = self.num_assets
        paths = []
        spot = self.get_spot().expand(num_paths, num_assets).clone()
        rate = self.get_rate()

        t_start = self.calibration_date

        for t_end in timeline:
            dt_total = t_end - t_start
            dt = dt_total / num_steps
            cholesky = self.compute_cholesky(dt)

            for _ in range(num_steps):
                z = torch.randn(num_paths, num_assets, dtype=torch.float64, device=device)
                correlated_z = z @ cholesky.T
                dS = rate * spot * dt + spot * correlated_z * torch.sqrt(dt)
                spot = spot + dS

            paths.append(spot.clone())

        return torch.stack(paths, dim=1)  # [num_paths, len(timeline)-1, num_assets]

    # Resolve requests posed by each product and at each exposure timepoint
    def resolve_request(self, req, state):
        if req.request_type == AtomicRequestType.SPOT:
            return state

        elif req.request_type == AtomicRequestType.DISCOUNT_FACTOR:
            t = req.time1
            rate=self.get_rate()
            return torch.exp(-rate * (t - self.calibration_date))

        elif req.request_type == AtomicRequestType.FORWARD_RATE:
            t1, t2 = req.time1, req.time2
            rate=self.get_rate()
            return torch.exp(rate * (t2 - t1))

        elif req.request_type == AtomicRequestType.LIBOR_RATE:
            t1, t2 = req.time1, req.time2
            rate=self.get_rate()
            return (torch.exp(rate* (t2 - t1)) - 1) / (t2 - t1)

        elif req.request_type == AtomicRequestType.NUMERAIRE:
            t = req.time1
            rate=self.get_rate()
            return torch.exp(rate * (t - self.calibration_date))

        else:
            raise NotImplementedError(f"Request type {req.request_type} not supported.")
