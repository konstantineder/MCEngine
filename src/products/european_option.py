from products.product import *
from math import pi
from request_interface.request_interface import RequestType, CompositeRequest, AtomicRequest
from collections import defaultdict

# AAD-compatible European option
class EuropeanOption(Product):
    def __init__(self, underlying, exercise_date, strike, option_type):
        super().__init__()
        self.exercise_date = torch.tensor([exercise_date], dtype=torch.float64,device=device)
        self.strike = torch.tensor([strike], dtype=torch.float64,device=device)
        self.option_type = option_type
        self.product_timeline=torch.tensor([exercise_date], dtype=torch.float64,device=device)
        self.modeling_timeline=self.product_timeline
        self.regression_timeline=torch.tensor([], dtype=torch.float64,device=device)
        self.underlying=underlying

        self.numeraire_requests={0: AtomicRequest(RequestType.NUMERAIRE,exercise_date)}
        self.underlying_request={0: underlying.get_composite_requests(exercise_date)}
    
    def get_requests(self):
        requests=defaultdict(set)
        for t, req in self.numeraire_requests.items():
            requests[t].add(req)

        return requests
    
    def get_composite_requests(self,observation_date=None):
        requests=defaultdict(set)
        for t, req in self.underlying_request.items():
            requests[t].add(req)

        return requests

    def payoff(self, spots, model):
        zero = torch.tensor([0.0], dtype=torch.float64, device=device)
        if self.option_type == OptionType.CALL:
            return torch.maximum(spots - self.strike, zero)
        else:
            return torch.maximum(self.strike - spots, zero)

    
    def compute_normalized_cashflows(self, time_idx, model, resolved_requests, regression_monomials=None, state=None):
        spots=resolved_requests[1][self.underlying_request[time_idx].get_handle()]
        cfs = self.payoff(spots,model)

        numeraire=resolved_requests[0][self.numeraire_requests[time_idx].handle]
        normalized_cfs=cfs/numeraire

        return state, normalized_cfs

    def compute_pv_analytically(self, model):
        spot = model.get_spot()
        rate = model.get_rate()
        sigma = model.get_volatility()

        d1 = (torch.log(spot / self.strike) + (rate + 0.5 * sigma ** 2) * self.exercise_date) / (sigma * torch.sqrt(self.exercise_date))
        d2 = d1 - sigma * torch.sqrt(self.exercise_date)

        norm = torch.distributions.Normal(0.0, 1.0)

        if self.option_type == OptionType.CALL:
            return spot * norm.cdf(d1) - self.strike * torch.exp(-rate * self.exercise_date) * norm.cdf(d2)
        else:
            return self.strike * torch.exp(-rate * self.exercise_date) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        
    def compute_pv_bond_option_analytically(self, model):
        a=model.get_mean_reversion_speed()
        rate=model.get_rate()
        sigma=model.get_volatility()
        calibration_date=model.calibration_date

        bond_price_at_exercise_date=model.compute_bond_price(calibration_date,self.exercise_date,rate)
        bond_price_at_underlying_maturity=model.compute_bond_price(calibration_date,self.underlying.maturity,rate)

        B_TS=(1-torch.exp(-a*(self.underlying.maturity-self.exercise_date)))/a
        sigma_tilde=sigma*torch.sqrt((1-torch.exp(-2*a*(self.exercise_date-calibration_date)))/(2*a))*B_TS

        d1 = (torch.log(bond_price_at_underlying_maturity / (bond_price_at_exercise_date * self.strike)) + 0.5 * sigma_tilde ** 2) / sigma_tilde
        d2 = d1 - sigma_tilde

        norm = torch.distributions.Normal(0.0, 1.0)

        if self.option_type == OptionType.CALL:
            return bond_price_at_underlying_maturity * norm.cdf(d1) - self.strike * bond_price_at_exercise_date* norm.cdf(d2)
        else:
            return self.strike * bond_price_at_exercise_date * norm.cdf(-d2) - bond_price_at_underlying_maturity * norm.cdf(-d1)
        
    def compute_dVegadSigma_analytically(self, model):
        spot = model.get_spot()
        rate = model.get_rate()
        sigma = model.get_volatility()
        T = self.exercise_date

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
        T = self.exercise_date

        d1 = (torch.log(spot / self.strike) + (rate+0.5 * sigma ** 2) * T) / (sigma * torch.sqrt(T))
        d2 = d1 - sigma * torch.sqrt(T)

        pdf_d1 = torch.exp(-0.5 * d1 ** 2) / torch.sqrt(torch.tensor(2.0 * pi, device=device))  # φ(d1)

        # Vomma (Volga) formula: S * φ(d1) * sqrt(T) * d1 * d2 / σ
        gamma = pdf_d1/(spot*sigma*torch.sqrt(T))

        return gamma