from products.product import *
from maths.maths import compute_degree_of_truth
from request_interface.request_interface import AtomicRequestType, AtomicRequest
import numpy as np
from collections import defaultdict

# Enum for barrier option types
class BarrierOptionType(Enum):
    DOWNANDOUT = "Down-And-Out"
    UPANDOUT = "Up-And-Out"
    DOWNANDIN = "Down-And-In"
    UPANDIN = "Up-And-In"     

class BarrierOption(Product):
    def __init__(self, strike, barrier, barrier_option_type, startdate, maturity, option_type, num_modeling_timesteps=1, use_brownian_bridge=True, use_seed=12345):
        super().__init__()
        self.strike = torch.tensor([strike], dtype=torch.float64,device=device)
        self.barrier = torch.tensor([barrier], dtype=torch.float64,device=device)
        self.maturity=torch.tensor([maturity], dtype=torch.float64,device=device)
        self.product_timeline=torch.tensor([maturity], dtype=torch.float64,device=device)
        self.modeling_timeline=torch.linspace(startdate, maturity,num_modeling_timesteps, dtype=torch.float64,device=device)
        self.regression_timeline=torch.tensor([], dtype=torch.float64,device=device)
        self.barrier_option_type = barrier_option_type
        self.option_type = option_type
        self.use_brownian_bridge = use_brownian_bridge
        self.use_seed = use_seed
        self.rng = np.random.default_rng(use_seed)

        self.numeraire_requests={idx: AtomicRequest(AtomicRequestType.NUMERAIRE,t) for idx, t in enumerate(self.modeling_timeline)}
        self.spot_requests={idx: AtomicRequest(AtomicRequestType.SPOT) for idx in range(len(self.modeling_timeline))}

    def get_atomic_requests(self):
        requests=defaultdict(list)
        for t, req in self.numeraire_requests.items():
            requests[t].append(req)

        for t, req in self.spot_requests.items():
            requests[t].append(req)

        return requests

    def compute_payoff_without_brownian_bridge(self, paths, model):
        spots_at_maturity = paths[:,-1]
        max_spot = torch.max(paths, dim=1).values
        min_spot = torch.min(paths, dim=1).values

        is_max_below_barrier = compute_degree_of_truth(self.barrier-max_spot,True)
        is_min_above_barrier =compute_degree_of_truth(min_spot - self.barrier,True)

        zero=torch.tensor([0.0], device=device)

        if self.barrier_option_type == BarrierOptionType.UPANDOUT:
            return torch.maximum(spots_at_maturity - self.strike, zero) * is_max_below_barrier if self.option_type == OptionType.CALL else torch.maximum(self.strike - spots_at_maturity, zero) * is_max_below_barrier
        if self.barrier_option_type == BarrierOptionType.DOWNANDOUT:
            return torch.maximum(spots_at_maturity - self.strike, zero) * is_min_above_barrier if self.option_type == OptionType.CALL else torch.maximum(self.strike - spots_at_maturity, zero) * is_min_above_barrier
        if self.barrier_option_type == BarrierOptionType.UPANDIN:
            return torch.maximum(spots_at_maturity - self.strike, zero) * (1 - is_max_below_barrier) if self.option_type == OptionType.CALL else torch.maximum(self.strike - spots_at_maturity, zero) * (1 - is_max_below_barrier)
        if self.barrier_option_type == BarrierOptionType.DOWNANDIN:
            return torch.maximum(spots_at_maturity - self.strike, zero) * (1 - is_min_above_barrier) if self.option_type == OptionType.CALL else torch.maximum(self.strike - spots_at_maturity, zero) * (1 - is_min_above_barrier)
        
    def compute_payoff_with_brownian_bridge(self, spots, model):
        sigma = model.get_volatility()
        spots_at_maturity = spots[:,-1]
        max_spot = torch.max(spots, dim=1).values  # Max spot across each path
        min_spot = torch.min(spots, dim=1).values
        num_spots = spots.shape[1]

        # Precompute Brownian bridge crossing probabilities
        log_spot_barrier = torch.log(spots / self.barrier)
        log_spot_barrier_next = torch.log(spots[:, 1:] / self.barrier)
        bridge_probs = torch.exp(-2 * log_spot_barrier[:, :-1] * log_spot_barrier_next / (sigma ** 2 * (self.maturity / num_spots)))

        is_max_below_barrier = compute_degree_of_truth(self.barrier-max_spot,True)
        is_min_above_barrier =compute_degree_of_truth(min_spot - self.barrier,True)

        zero=torch.tensor([0.0], device=device)

        vanilla_payoff = torch.maximum(spots_at_maturity - self.strike, zero) if self.option_type == OptionType.CALL else torch.maximum(self.strike - spots_at_maturity, zero)

        # One draw per interval
        rdns = self.rng.uniform(0, 1, size=bridge_probs.shape)
        rdns = torch.tensor(rdns, dtype=bridge_probs.dtype, device=bridge_probs.device)
        hit_probs = compute_degree_of_truth(bridge_probs - rdns,True)
        hit_barrier = 1 - torch.prod(1 - hit_probs, dim=1)

        
        if self.barrier_option_type == BarrierOptionType.UPANDOUT:
            payoff = vanilla_payoff * is_max_below_barrier * (1-hit_barrier)

        elif self.barrier_option_type == BarrierOptionType.DOWNANDOUT:
            payoff = vanilla_payoff * is_min_above_barrier * (1-hit_barrier)

        elif self.barrier_option_type == BarrierOptionType.UPANDIN:
            payoff = vanilla_payoff * (1-is_max_below_barrier)* hit_barrier

        elif self.barrier_option_type == BarrierOptionType.DOWNANDIN:
            payoff = vanilla_payoff * (1-is_min_above_barrier)* hit_barrier

        else:
            raise NotImplementedError(f"Barrier type {self.barrier_option_type} not supported.")

        return payoff

            
    def payoff(self, spots, model):
            if self.use_brownian_bridge==True:
                return self.compute_payoff_with_brownian_bridge(spots, model)
            else:
                return self.compute_payoff_without_brownian_bridge(spots, model)


    def compute_pv_analytically(self, model):
            spot=model.get_spot()
            rate=model.get_rate()
            sigma=model.get_volatility()

            d1_spot_strike = (torch.log(spot / self.strike) + (rate + 0.5 * sigma**2) * self.maturity) / (sigma * torch.sqrt(self.maturity))
            d1_spot_barrier = (torch.log(spot / self.barrier) + (rate + 0.5 * sigma**2) * self.maturity) / (sigma * torch.sqrt(self.maturity))
            d1_barrier_strike = (torch.log(self.barrier**2 / (self.strike*spot)) + (rate + 0.5 * sigma**2) * self.maturity) / (sigma * torch.sqrt(self.maturity))
            d1_barrier_spot = (torch.log(self.barrier / spot) + (rate + 0.5 * sigma**2) * self.maturity) / (sigma * torch.sqrt(self.maturity))
            d2_spot_strike = d1_spot_strike - sigma * torch.sqrt(self.maturity)
            d2_spot_barrier = d1_spot_barrier - sigma * torch.sqrt(self.maturity)
            d2_barrier_strike = d1_barrier_strike - sigma * torch.sqrt(self.maturity)
            d2_barrier_spot = d1_barrier_spot - sigma * torch.sqrt(self.maturity)

            norm = torch.distributions.Normal(0.0, 1.0)

            is_spot_below_barrier=spot < self.barrier
            if self.barrier_option_type==BarrierOptionType.UPANDOUT:
                if self.option_type == OptionType.CALL:
                    term1=norm.cdf(d1_spot_strike)-norm.cdf(d1_spot_barrier)
                    term2=norm.cdf(d1_barrier_strike)-norm.cdf(d1_barrier_spot)
                    term3=norm.cdf(d2_spot_strike)-norm.cdf(d2_spot_barrier)
                    term4=norm.cdf(d2_barrier_strike)-norm.cdf(d2_barrier_spot)

                    term_spot=spot * (term1-(self.barrier/spot)**(1+2*rate/(sigma**2))*term2)
                    term_strike=self.strike * torch.exp(-rate * self.maturity) * (term3-(spot/self.barrier)**(1-2*rate/(sigma**2))*term4)

                    return is_spot_below_barrier*(term_spot-term_strike)
                else:
                    return NotImplementedError(f"Analytical method for up-and-out {self.option_type} not yet implemented")
            else:
                return NotImplementedError(f"Analytical method for {self.barrier_option_type} not yet implemented")

    
    def get_initial_state(self, num_paths):
        return torch.full((num_paths,), 0, dtype=torch.long, device=device)
    
    def compute_normalized_cashflows(self, time_idx, model, resolved_requests, regression_monomials=None, state=None):
        spots=torch.stack([resolved_requests[0][self.spot_requests[idx].handle] for idx in range(len(self.modeling_timeline))], dim=1)
        cfs = self.payoff(spots,model)

        numeraire=resolved_requests[0][self.numeraire_requests[len(self.product_timeline)-1].handle]
        normalized_cfs=cfs/numeraire

        return state, normalized_cfs