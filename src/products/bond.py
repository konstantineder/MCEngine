from products.product import *
from math import pi
from request_interface.request_interface import RequestType, CompositeRequest, AtomicRequest
from collections import defaultdict

# AAD-compatible European option
class Bond(Product):
    def __init__(self, startdate, maturity, notional, tenor, fixed_rate=None):
        super().__init__()
        self.startdate = torch.tensor([startdate], dtype=torch.float64,device=device)
        self.maturity = torch.tensor([maturity], dtype=torch.float64,device=device)
        self.notional = torch.tensor([notional], dtype=torch.float64, device=device)
        self.tenor = torch.tensor([tenor], dtype=torch.float64, device=device)
        self.fixed_rate = fixed_rate
        self.composite_req_handle=None

        self.payment_dates = []
        self.libor_requests = {}   # LIBOR forward rates as composite requests
        self.composite_requests = {}
        self.numeraire_requests = {}   # For discounting each payment

        # Build the payment schedule and requests
        date = startdate + tenor
        idx = 0

        if fixed_rate is not None:
            while date < maturity:
                self.numeraire_requests[idx] = AtomicRequest(RequestType.NUMERAIRE, date)
                self.composite_requests[idx] = AtomicRequest(request_type=RequestType.FORWARD_RATE,time1=startdate,time2=date)
                self.payment_dates.append(date)
                date += tenor
                idx += 1

            # Final payment at maturity
            self.numeraire_requests[idx] = AtomicRequest(RequestType.DISCOUNT_FACTOR, maturity)
            self.composite_requests[idx] = AtomicRequest(request_type=RequestType.FORWARD_RATE,time1=startdate,time2=maturity)
            self.payment_dates.append(maturity)
        else:
            while date < maturity:
                self.libor_requests[idx] = AtomicRequest(RequestType.LIBOR_RATE, date - tenor, date)
                self.numeraire_requests[idx] = AtomicRequest(RequestType.NUMERAIRE, date)
                self.composite_requests[idx] = AtomicRequest(request_type=RequestType.FORWARD_RATE,time1=startdate,time2=date-tenor)
                self.payment_dates.append(date)
                date += tenor
                idx += 1

            # Final payment at maturity
            self.libor_requests[idx] = AtomicRequest(RequestType.LIBOR_RATE, date - tenor, maturity)
            self.numeraire_requests[idx] = AtomicRequest(RequestType.DISCOUNT_FACTOR, maturity)
            self.composite_requests[idx] = AtomicRequest(request_type=RequestType.FORWARD_RATE,time1=startdate,time2=date-tenor)
            self.composite_requests[idx + 1] = AtomicRequest(request_type=RequestType.FORWARD_RATE,time1=startdate,time2=maturity)
            self.payment_dates.append(maturity)

        self.payment_dates = torch.tensor(self.payment_dates, dtype=torch.float64, device=device)
        self.modeling_timeline = self.payment_dates
        self.regression_timeline = torch.tensor([], dtype=torch.float64, device=device)

    def __eq__(self, other):
        return (
            isinstance(other, Bond) and
            torch.equal(self.startdate, other.startdate) and
            torch.equal(self.maturity, other.maturity) and
            torch.equal(self.tenor, other.tenor) and
            torch.equal(self.fixed_rate, other.fixed_rate)
        )

    def __hash__(self):
        return hash((
            self.startdate.item(),
            self.maturity.item(),
            self.tenor.item(),
            self.fixed_rate,
        ))
    
    def get_requests(self):
        requests=defaultdict(list)

        for t, req in self.numeraire_requests.items():
            requests[t].append(req)

        if self.fixed_rate is None:
            for t, req in self.libor_requests.items():
                requests[t].append(req)

        return requests
    
    def get_atomic_requests(self):
        requests=defaultdict(list)

        for t, req in self.composite_requests.items():
            requests[t].append(req)

        return requests
    
    def generate_composite_requests(self, observation_date):
        bond = Bond(observation_date,self.maturity.item(),self.notional.item(),self.tenor.item(),self.fixed_rate)
        return CompositeRequest(bond)
    
    def get_composite_requests(self):
        return defaultdict(set)
    
    def get_value(self, resolved_atomic_requests):
        if self.fixed_rate is not None:
            return self.get_value_fixed(resolved_atomic_requests)
        else:
            return self.get_value_float(resolved_atomic_requests)

    def get_value_fixed(self, resolved_atomic_requests):
        total_value = torch.zeros_like(resolved_atomic_requests[0], dtype=torch.float64, device=device)

        prev_time=self.startdate
        for t in self.numeraire_requests.keys():
            discount_req = self.composite_requests[t]

            discount = resolved_atomic_requests[discount_req.handle]

            time=self.modeling_timeline[t]
            dt=time - prev_time

            cashflow_value = self.notional * self.fixed_rate * dt * discount
            total_value += cashflow_value

            prev_time=time

        return total_value
    
    def get_value_float(self, resolved_atomic_requests):
        total_value = torch.zeros_like(resolved_atomic_requests[0], dtype=torch.float64, device=device)

        for t in self.numeraire_requests.keys():
            discount_req = self.composite_requests[t]
            discount_next_req = self.composite_requests[t+1]

            discount = resolved_atomic_requests[discount_req.handle]
            discount_next = resolved_atomic_requests[discount_next_req.handle]

            cashflow_value = self.notional * discount - discount_next
            total_value += cashflow_value

        return total_value
    
    def compute_normalized_cashflows(self, time_idx, model, resolved_requests,regression_monomials=None, state=None):
        if self.fixed_rate is not None:
            return self.compute_normalized_cashflows_fixed(time_idx, model, resolved_requests,regression_monomials, state)
        else:
            return self.compute_normalized_cashflows_float(time_idx, model, resolved_requests,regression_monomials, state)
    
    def compute_normalized_cashflows_fixed(self, time_idx, model, resolved_requests,regression_monomials=None, state=None):
        numeraire= resolved_requests[0][self.numeraire_requests[time_idx].handle]

        prev_time=self.startdate
        if time_idx > 0:
            prev_time = self.payment_dates[time_idx - 1]

        dt=self.payment_dates[time_idx] - prev_time

        fixed_leg = self.fixed_rate * dt
        cashflow = fixed_leg / numeraire

        return state, cashflow
    
    def compute_normalized_cashflows_float(self, time_idx, model, resolved_requests,regression_monomials=None, state=None):
        libor_rate = resolved_requests[0][self.libor_requests[time_idx].handle]
        numeraire= resolved_requests[0][self.numeraire_requests[time_idx].handle]

        prev_time=self.startdate
        if time_idx > 0:
            prev_time = self.payment_dates[time_idx - 1]

        dt=self.payment_dates[time_idx] - prev_time

        float_leg = libor_rate * dt
        cashflow = float_leg/numeraire

        return state, cashflow
