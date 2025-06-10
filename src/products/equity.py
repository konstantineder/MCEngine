from products.product import *
from math import pi
from request_interface.request_interface import RequestType, CompositeRequest, AtomicRequest
from collections import defaultdict

# AAD-compatible European option
class Equity(Product):
    def __init__(self, id):
        super().__init__()
        self.id=id
        self.composite_req_handle=None

        self.spot_requests={0: AtomicRequest(RequestType.SPOT)}

    def __eq__(self, other):
        return isinstance(other, Equity) and self.id == other.id

    def __hash__(self):
        return hash(self.id)
    
    def get_atomic_requests(self):
        requests=defaultdict(set)

        for t, req in self.spot_requests.items():
            requests[t].add(req)

        return requests
    
    def get_composite_requests(self, observation_date=None):
        equity=Equity('')
        return CompositeRequest(equity)

    
    def get_value(self, resolved_atomic_requests):
        return resolved_atomic_requests[self.spot_requests[0].handle]
