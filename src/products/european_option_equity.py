from products.product import *
from math import pi
from request_interface.request_interface import ModelRequestType, ModelRequest
from collections import defaultdict

# AAD-compatible European option
class EuropeanOption(Product):
    def __init__(self, maturity, strike, option_type):
        super().__init__()
        self.maturity = torch.tensor([maturity], dtype=torch.float64,device=device)
        self.strike = torch.tensor([strike], dtype=torch.float64,device=device)
        self.option_type = option_type
        self.product_timeline=torch.tensor([maturity], dtype=torch.float64,device=device)
        self.modeling_timeline=self.product_timeline
        self.regression_timeline=torch.tensor([], dtype=torch.float64,device=device)

        self.numeraire_requests={0: ModelRequest(ModelRequestType.NUMERAIRE,maturity)}
        self.spot_requests={0: ModelRequest(ModelRequestType.SPOT)}
    
    def get_requests(self):
        requests=defaultdict(set)
        for t, req in self.numeraire_requests.items():
            requests[t].add(req)

        for t, req in self.spot_requests.items():
            requests[t].add(req)


        return requests

    def payoff(self, spots, model):
        zero = torch.tensor([0.0], dtype=torch.float64, device=device)
        if self.option_type == OptionType.CALL:
            return torch.maximum(spots - self.strike, zero)
        else:
            return torch.maximum(self.strike - spots, zero)

    
    def compute_normalized_cashflows(self, time_idx, model, resolved_requests, regression_monomials=None, state=None):
        spots=resolved_requests[self.spot_requests[time_idx].handle]
        cfs = self.payoff(spots,model)

        numeraire=resolved_requests[self.numeraire_requests[time_idx].handle]
        normalized_cfs=cfs/numeraire

        return state, normalized_cfs

    def compute_pv_analytically(self, model):
        spot = model.get_spot()
        rate = model.get_rate()
        sigma = model.get_volatility()

        d1 = (torch.log(spot / self.strike) + (rate + 0.5 * sigma ** 2) * self.maturity) / (sigma * torch.sqrt(self.maturity))
        d2 = d1 - sigma * torch.sqrt(self.maturity)

        norm = torch.distributions.Normal(0.0, 1.0)

        if self.option_type == OptionType.CALL:
            return spot * norm.cdf(d1) - self.strike * torch.exp(-rate * self.maturity) * norm.cdf(d2)
        else:
            return self.strike * torch.exp(-rate * self.maturity) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        
    def compute_dVegadSigma_analytically(self, model):
        spot = model.get_spot()
        rate = model.get_rate()
        sigma = model.get_volatility()
        T = self.maturity

        d1 = (torch.log(spot / self.strike) + (rate+0.5 * sigma ** 2) * T) / (sigma * torch.sqrt(T))
        d2 = d1 - sigma * torch.sqrt(T)

        pdf_d1 = torch.exp(-0.5 * d1 ** 2) / torch.sqrt(torch.tensor(2.0 * pi, device=device))  # φ(d1)

        # Vomma (Volga) formula: S * φ(d1) * sqrt(T) * d1 * d2 / σ
        vomma = spot * pdf_d1 * torch.sqrt(T) * d1 * d2 / sigma

        return vomma
    
    def compute_dDeltadSpot_analytically(self, model):
        spot = model.get_spot()
        rate = model.get_rate()
        sigma = model.get_volatility()
        T = self.maturity

        d1 = (torch.log(spot / self.strike) + (rate+0.5 * sigma ** 2) * T) / (sigma * torch.sqrt(T))
        d2 = d1 - sigma * torch.sqrt(T)

        pdf_d1 = torch.exp(-0.5 * d1 ** 2) / torch.sqrt(torch.tensor(2.0 * pi, device=device))  # φ(d1)

        # Vomma (Volga) formula: S * φ(d1) * sqrt(T) * d1 * d2 / σ
        gamma = pdf_d1/(spot*sigma*torch.sqrt(T))

        return gamma