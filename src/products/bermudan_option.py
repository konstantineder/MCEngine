from products.product import *
from request_interface.request_interface import RequestType, AtomicRequest
import numpy as np
from collections import defaultdict

class BermudanOption:
    def __init__(self, underlying, exercise_dates, strike, option_type):
        super().__init__()
        self.strike = torch.tensor([strike], dtype=torch.float64, device=device)
        self.option_type = option_type
        self.product_timeline = torch.tensor(exercise_dates, dtype=torch.float64, device=device)
        self.modeling_timeline = self.product_timeline
        self.regression_timeline = self.product_timeline
        self.regression_coeffs = [
            [torch.zeros(3, device=device) for _ in range(2)]  # 2 states
            for _ in range(len(self.regression_timeline))      # len(time points)
        ]
        self.num_exercise_rights = 1

        self.numeraire_requests={idx: AtomicRequest(RequestType.NUMERAIRE,t) for idx, t in enumerate(self.modeling_timeline)}
        self.spot_requests={idx: AtomicRequest(RequestType.SPOT) for idx in range(len(self.modeling_timeline))}

        self.underlying_requests={}
        idx=0
        for exercise_date in exercise_dates:
            self.underlying_requests[idx]=underlying.generate_composite_requests(exercise_date)
            idx+=1

    def get_requests(self):
        requests=defaultdict(set)
        for t, req in self.numeraire_requests.items():
            requests[t].add(req)

        for t, req in self.spot_requests.items():
            requests[t].add(req)

        return requests
    
    
    def get_composite_requests(self):
        requests=defaultdict(set)
        for t, req in self.underlying_requests.items():
            requests[t].add(req)

        return requests


    def get_num_states(self):
        return 2
    
    def get_initial_state(self, num_paths):
        return torch.full((num_paths,), 1, dtype=torch.long, device=device)

    def payoff(self, spots, model_params):
        zero = torch.tensor([0.0], device=device)
        if self.option_type == OptionType.CALL:
            return torch.maximum(spots - self.strike, zero)
        else:
            return torch.maximum(self.strike - spots, zero)
    
    def compute_normalized_cashflows(self, time_idx, model_params, resolved_requests, regression_monomials, state=[]):
        spot = resolved_requests[1][self.underlying_requests[time_idx].get_handle()]
        immediate = self.payoff(spot, model_params)

        # Check if time is the last in the product timeline
        if time_idx == len(self.product_timeline) - 1:
            continuation = torch.zeros_like(immediate)
        else:
            A = regression_monomials.get_regression_matrix(spot)
            coeffs_matrix = torch.stack([self.regression_coeffs[time_idx][s] for s in state.tolist()], dim=0)  # [num_paths, 3]

            continuation = (A * coeffs_matrix).sum(dim=1)

        # Per-path exercise decision
        should_exercise = (immediate > continuation.squeeze()) & (state > 0)

        # Accumulate cashflows
        numeraire=resolved_requests[0][self.numeraire_requests[time_idx].handle]
        cashflows = immediate * should_exercise.float()/numeraire

        # Update remaining rights
        state -= should_exercise.int()

        return state, cashflows
    
class AmericanOption(BermudanOption):
    def __init__(self, underlying, maturity, num_exercise_dates, strike, option_type):
        exercise_dates=np.linspace(0.,maturity,num_exercise_dates) if num_exercise_dates>1 else [maturity]
        super().__init__(underlying=underlying, exercise_dates=exercise_dates,
                         strike=strike,
                         option_type=option_type)
