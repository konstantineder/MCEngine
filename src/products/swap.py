from products.product import *
from request_interface.request_interface import RequestType, CompositeRequest, AtomicRequest
from collections import defaultdict

class InterestRateSwap(Product):
    def __init__(self, fixed_rate, startdate, enddate, tenor):
        super().__init__()
        self.fixed_rate = torch.tensor([fixed_rate], dtype=torch.float64, device=device)
        self.startdate = torch.tensor([startdate], dtype=torch.float64, device=device)
        self.enddate = torch.tensor([enddate], dtype=torch.float64, device=device)
        self.tenor = torch.tensor([tenor], dtype=torch.float64, device=device)

        self.payment_dates = []
        self.composite_requests = {}   # LIBOR forward rates as composite requests
        self.numeraire_requests = {}   # For discounting each payment
        self.handles = []

        # Build the payment schedule and requests
        date = startdate + tenor
        idx = 0
        while date < enddate:
            self.composite_requests[idx] = AtomicRequest(RequestType.LIBOR_RATE, date - tenor, date)
            self.numeraire_requests[idx] = AtomicRequest(RequestType.NUMERAIRE, date)
            self.payment_dates.append(date)
            date += tenor
            idx += 1

        # Final payment at maturity
        self.composite_requests[idx] = AtomicRequest(RequestType.LIBOR_RATE, date - tenor, enddate)
        self.numeraire_requests[idx] = AtomicRequest(RequestType.DISCOUNT_FACTOR, enddate)
        self.payment_dates.append(enddate)

        self.payment_dates = torch.tensor(self.payment_dates, dtype=torch.float64, device=device)
        self.modeling_timeline = self.payment_dates
        self.regression_timeline = torch.tensor([], dtype=torch.float64, device=device)

    def __eq__(self, other):
        return (
            isinstance(other, InterestRateSwap) and
            torch.equal(self.startdate, other.startdate) and
            torch.equal(self.enddate, other.enddate) and
            torch.equal(self.tenor, other.tenor) and
            torch.equal(self.fixed_rate, other.fixed_rate)
        )

    def __hash__(self):
        return hash((
            self.startdate.item(),
            self.enddate.item(),
            self.tenor.item(),
            self.fixed_rate.item()
        ))

    def get_requests(self):
        requests = defaultdict(set)
        for t, req in self.numeraire_requests.items():
            requests[t].add(req)
        return requests

    def get_atomic_requests(self, observation_date):
        # Update time1 for composite requests with the observation_date
        requests = defaultdict(set)
        for t, req in self.composite_requests.items():
            req.time1 = observation_date
            requests[t].add(req)
        return requests

    def get_composite_requests(self):
        return CompositeRequest(self)

    def get_value(self, resolved_atomic_requests):
        """
        Present value of floating - fixed leg, discounted
        Floating leg: use resolved LIBOR composite requests
        Fixed leg: use fixed_rate
        """
        total_value = torch.tensor([0.0], dtype=torch.float64, device=device)

        for t in self.composite_requests.keys():
            libor_req = self.composite_requests[t]
            discount_req = self.numeraire_requests[t]

            libor_rate = resolved_atomic_requests[libor_req.handle]
            discount = resolved_atomic_requests[discount_req.handle]

            delta = self.tenor  # assuming fixed accrual period

            float_leg = libor_rate * delta
            fixed_leg = self.fixed_rate * delta

            cashflow_value = (float_leg - fixed_leg) * discount
            total_value += cashflow_value

        return total_value

    def compute_normalized_cashflows(self, time_idx, model, resolved_requests,regression_monomials=None, state=None):
        libor_rate = resolved_requests[self.libor_requests[time_idx].handle]
        numeraire= resolved_requests[self.numeraire_requests[time_idx].handle]

        dt=self.tenor

        float_leg = libor_rate * dt
        fixed_leg = self.fixed_rate * dt
        cashflow = (float_leg - fixed_leg)/numeraire

        return state, cashflow