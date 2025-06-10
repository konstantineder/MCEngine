from products.product import *
from maths.maths import compute_degree_of_truth
from request_interface.request_interface import RequestType, AtomicRequest
from collections import defaultdict

# Binary (digital) option
class BinaryOption(Product):
    def __init__(self, maturity, strike, payment_amount, option_type):
        super().__init__()
        self.maturity = torch.tensor([maturity], dtype=torch.float64,device=device)
        self.strike = torch.tensor([strike], dtype=torch.float64,device=device)
        self.option_type = option_type
        self.payment_amount = torch.tensor([payment_amount], dtype=torch.float64,device=device)
        self.product_timeline=torch.tensor([maturity], dtype=torch.float64,device=device)
        self.modeling_timeline=self.product_timeline
        self.regression_timeline=torch.tensor([], dtype=torch.float64,device=device)

        self.numeraire_requests={0: AtomicRequest(RequestType.NUMERAIRE,maturity)}
        self.spot_requests={0: AtomicRequest(RequestType.SPOT)}

    def get_requests(self):
        requests=defaultdict(set)
        for t, req in self.numeraire_requests.items():
            requests[t].add(req)

        for t, req in self.spot_requests.items():
            requests[t].add(req)


        return requests

    # Operator overloading of the payoff computation method
    # This method computes the payoff of a binary option based on the option type    
    def payoff(self, spots, model):
        is_strike_above_barrier = compute_degree_of_truth(spots - self.strike,True,1)
        if self.option_type == OptionType.CALL:
            return self.payment_amount*is_strike_above_barrier
        else:
            return self.payment_amount*(1-is_strike_above_barrier)

    # Black-Scholes closed-form pricing for European options (see chapter 1)
    def compute_pv_analytically(self, model):
        spot=model.spot
        rate=model.rate
        sigma=model.sigma

        norm = torch.distributions.Normal(0.0, 1.0)

        d2 = (torch.log(spot / self.strike) + (rate - 0.5 * sigma**2) * self.maturity) / (sigma * torch.sqrt(self.maturity))
        if self.option_type == OptionType.CALL:
            return self.payment_amount * torch.exp(-rate * self.maturity) * norm.cdf(d2)
        else:
            return self.payment_amount * torch.exp(-rate * self.maturity) * norm.cdf(-d2)
    
    def get_initial_state(self, num_paths):
        return torch.full((num_paths,), 0, dtype=torch.long, device=device)
    
    def compute_normalized_cashflows(self, time_idx, model, resolved_requests, regression_monomials=None,state=None):
        spots=resolved_requests[self.spot_requests[time_idx].handle]
        cfs = self.payoff(spots,model)

        numeraire=resolved_requests[self.numeraire_requests[time_idx].handle]
        normalized_cfs=cfs/numeraire

        return state, normalized_cfs