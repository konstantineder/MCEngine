from products.product import *
from request_interface.request_interface import ModelRequestType, ModelRequest
from collections import defaultdict

# AAD-compatible European option
class EuropeanBondOption(Product):
    def __init__(self, maturity, underlying_maturity, strike, option_type, settlement_type=SettlementType.CASH):
        super().__init__()
        self.maturity = torch.tensor([maturity], dtype=torch.float64,device=device)
        self.underlying_maturity = torch.tensor([underlying_maturity], dtype=torch.float64,device=device)
        self.strike = torch.tensor([strike], dtype=torch.float64,device=device)
        self.option_type = option_type
        self.settlement_type=settlement_type
        self.product_timeline=torch.tensor([maturity], dtype=torch.float64,device=device)
        self.modeling_timeline=self.product_timeline
        self.regression_timeline=torch.tensor([], dtype=torch.float64,device=device)

        self.numeraire_requests={0: ModelRequest(ModelRequestType.NUMERAIRE,maturity)}
        self.forward_requests={0: ModelRequest(ModelRequestType.FORWARD_RATE,maturity,underlying_maturity)}
    
    def get_requests(self):
        requests=defaultdict(set)
        for t, req in self.numeraire_requests.items():
            requests[t].add(req)

        for t, req in self.forward_requests.items():
            requests[t].add(req)


        return requests

    def payoff(self, forwards, model):
        zero = torch.tensor([0.0], device=device)
        if self.option_type == OptionType.CALL:
            return torch.maximum(forwards - self.strike, zero)
        else:
            return torch.maximum(self.strike - forwards, zero)

    
    def compute_normalized_cashflows(self, time_idx, model, resolved_requests, regression_monomials=None,state=None):
        spots=resolved_requests[self.forward_requests[time_idx].handle]
        cfs = self.payoff(spots,model)

        numeraire=resolved_requests[self.numeraire_requests[time_idx].handle]
        normalized_cfs=cfs/numeraire

        return state, normalized_cfs

    def compute_pv_analytically(self, model):
        a = model.a
        rate = model.rate
        sigma = model.sigma
        calibration_date=model.calibration_date

        bond_price_at_exercise_date=model.compute_bond_price(calibration_date,self.maturity,rate)
        bond_price_at_underlying_maturity=model.compute_bond_price(calibration_date,self.underlying_maturity,rate)

        B_TS=(1-torch.exp(-a*(self.underlying_maturity-self.maturity)))/a
        sigma_tilde=sigma*torch.sqrt((1-torch.exp(-2*a*(self.maturity-calibration_date)))/(2*a))*B_TS

        d1 = (torch.log(bond_price_at_underlying_maturity / (bond_price_at_exercise_date * self.strike)) + 0.5 * sigma_tilde ** 2) / sigma_tilde
        d2 = d1 - sigma_tilde

        norm = torch.distributions.Normal(0.0, 1.0)

        if self.option_type == OptionType.CALL:
            return bond_price_at_underlying_maturity * norm.cdf(d1) - self.strike * bond_price_at_exercise_date* norm.cdf(d2)
        else:
            return self.strike * bond_price_at_exercise_date * norm.cdf(-d2) - bond_price_at_underlying_maturity * norm.cdf(-d1)