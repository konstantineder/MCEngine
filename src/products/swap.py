from products.product import *
from products.bond import Bond
from request_interface.request_interface import RequestType, CompositeRequest, AtomicRequest
from collections import defaultdict

class InterestRateSwap(Product):
    def __init__(self, startdate, enddate, notional, fixed_rate, tenor_fixed, tenor_float):
        super().__init__()
        self.startdate = startdate
        self.enddate = enddate
        self.notional = notional
        self.fixed_rate = fixed_rate
        self.tenor_fixed = tenor_fixed
        self.tenor_float = tenor_float
        self.fixed_leg = Bond(startdate=startdate,maturity=enddate,notional=notional,tenor=tenor_fixed,fixed_rate=fixed_rate)
        self.floating_leg = Bond(startdate=startdate,maturity=enddate,notional=notional,tenor=tenor_float)

        fixed_times = {float(t.item()) for t in self.fixed_leg.modeling_timeline}
        float_times = {float(t.item()) for t in self.floating_leg.modeling_timeline}
        all_times = sorted(fixed_times.union(float_times))
        self.product_timeline = torch.tensor(all_times, dtype=torch.float64)
        self.modeling_timeline = self.product_timeline

        self.regression_timeline = torch.tensor([], dtype=torch.float64, device=device)

    def __eq__(self, other):
        return (
            isinstance(other, InterestRateSwap) and
            self.startdate == other.startdate and
            self.enddate == other.enddate and
            self.notional == other.notional and
            self.fixed_rate == other.fixed_rate and
            self.tenor_fixed == other.tenor_fixed and
            self.tenor_float == other.tenor_float
        )

    def __hash__(self):
        return hash((
            self.startdate,
            self.enddate,
            self.notional,
            self.fixed_rate,
            self.tenor_fixed,
            self.tenor_float
        ))

    def get_requests(self):
        requests = defaultdict(list)

        fixed_reqs = self.fixed_leg.get_requests()
        for t, reqs in fixed_reqs.items():
            for req in reqs:
                requests[t].append(req)

        float_reqs = self.floating_leg.get_requests()
        for t, reqs in float_reqs.items():
            for req in reqs:
                requests[t].append(req)

        return requests

    def get_atomic_requests(self):
        # Update time1 for composite requests with the observation_date
        atomic_requests = defaultdict(list)

        fixed_reqs = self.fixed_leg.get_atomic_requests()
        for t, reqs in fixed_reqs.items():
            for req in reqs:
                atomic_requests[t].append(req)

        float_reqs = self.floating_leg.get_atomic_requests()
        for t, reqs in float_reqs.items():
            for req in reqs:
                atomic_requests[t].append(req)

        return atomic_requests
    
    def generate_composite_requests(self, observation_date):
        swap=InterestRateSwap(observation_date,self.enddate,self.notional,self.fixed_rate,self.tenor_fixed,self.tenor_float)
        return CompositeRequest(swap)

    def get_composite_requests(self):
        return defaultdict(set)
            
    def get_value(self, resolved_atomic_requests):
        """
        Present value of floating - fixed leg, discounted
        Floating leg: use resolved LIBOR composite requests
        Fixed leg: use fixed_rate
        """
        fixed_value = self.fixed_leg.get_value(resolved_atomic_requests)
        float_value = self.floating_leg.get_value(resolved_atomic_requests)

        return fixed_value - float_value

    def compute_normalized_cashflows(self, time_idx, model, resolved_requests, regression_monomials=None, state=None):
        total_value = torch.zeros_like(resolved_requests[0][0], dtype=torch.float64, device=device)

        time = self.modeling_timeline[time_idx]

        fixed_cashflow = torch.zeros_like(total_value)
        float_cashflow = torch.zeros_like(total_value)

        if time in self.fixed_leg.modeling_timeline:
            fixed_time_idx = (self.fixed_leg.modeling_timeline == time).nonzero(as_tuple=True)[0].item()
            _, fixed_cashflow = self.fixed_leg.compute_normalized_cashflows(
                fixed_time_idx, model, resolved_requests, regression_monomials, state
            )
            total_value += fixed_cashflow

        if time in self.floating_leg.modeling_timeline:
            float_time_idx = (self.floating_leg.modeling_timeline == time).nonzero(as_tuple=True)[0].item()
            _, float_cashflow = self.floating_leg.compute_normalized_cashflows(
                float_time_idx, model, resolved_requests, regression_monomials, state
            )
            total_value -= float_cashflow

        return state, fixed_cashflow - float_cashflow

