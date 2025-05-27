from products.product import *
from request_interface.request_interface import ModelRequestType, ModelRequest
from collections import defaultdict

class InterestRateSwap(Product):
    def __init__(self, fixed_rate, startdate, enddate, tenor):
        super().__init__()
        self.fixed_rate = torch.tensor([fixed_rate], dtype=torch.float64,device=device)
        self.startdate = torch.tensor(startdate, dtype=torch.float64,device=device)
        self.enddate=torch.tensor(enddate, dtype=torch.float64,device=device)
        self.tenor=torch.tensor(tenor, dtype=torch.float64,device=device)

        self.libor_requests = {}
        self.numeraire_requests = {}
        self.product_timeline = []

        date = startdate + tenor
        idx=0
        while date < enddate:
            self.libor_requests[idx]=ModelRequest(ModelRequestType.LIBOR_RATE, date - tenor, date)
            self.numeraire_requests[idx]=ModelRequest(ModelRequestType.NUMERAIRE, date)
            self.product_timeline.append(date)
            date += tenor
            idx+=1

        # Last discount factor at enddate
        self.numeraire_requests[idx]=ModelRequest(ModelRequestType.DISCOUNT_FACTOR, enddate)
        self.libor_requests[idx]=ModelRequest(ModelRequestType.LIBOR_RATE, date - tenor, enddate)
        self.product_timeline.append(enddate)
        self.product_timeline=torch.tensor(self.product_timeline, dtype=torch.float64)
        self.modeling_timeline=self.product_timeline


        self.regression_timeline=torch.tensor([], dtype=torch.float64,device=device)
    
    def get_requests(self):
        requests=defaultdict(set)
        for t, req in self.numeraire_requests.items():
            requests[t].add(req)

        for t, req in self.libor_requests.items():
            requests[t].add(req)


        return requests

    def compute_normalized_cashflows(self, time_idx, model, resolved_requests,regression_monomials=None, state=None):
        libor_rate = resolved_requests[self.libor_requests[time_idx].handle]
        numeraire= resolved_requests[self.numeraire_requests[time_idx].handle]

        dt=self.tenor

        float_leg = libor_rate * dt
        fixed_leg = self.fixed_rate * dt
        cashflow = (float_leg - fixed_leg)/numeraire

        return state, cashflow