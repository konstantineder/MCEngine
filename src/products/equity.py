from products.product import *
from math import pi
from request_interface.request_interface import AtomicRequestType, CompositeRequest, AtomicRequest
from collections import defaultdict

# Implementation of Equity
class Equity(Product):
    def __init__(self, id):
        super().__init__()
        self.id=id
        self.composite_req_handle=None

        self.spot_requests={0: AtomicRequest(AtomicRequestType.SPOT)}

    def __eq__(self, other):
        return isinstance(other, Equity) and self.id == other.id

    def __hash__(self):
        return hash(self.id)
    
    def get_atomic_requests_for_underlying(self):
        requests=defaultdict(list)

        for t, req in self.spot_requests.items():
            requests[t].append(req)

        return requests
    
    def generate_composite_requests_for_date(self, obervation_date):
        equity=Equity(self.id)
        return CompositeRequest(equity)
    
    def get_value(self, resolved_atomic_requests):
        return resolved_atomic_requests[self.spot_requests[0].handle]
