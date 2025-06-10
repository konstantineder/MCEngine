from products.product import *
from math import pi
from request_interface.request_interface import RequestType, CompositeRequest, AtomicRequest
from collections import defaultdict

# AAD-compatible European option
class Bond(Product):
    def __init__(self, maturity):
        super().__init__()
        self.maturity = torch.tensor([maturity], dtype=torch.float64,device=device)
        self.composite_req_handle=None

        self.numeraire_requests={0: AtomicRequest(RequestType.NUMERAIRE,self.maturity)}
        self.composite_request={0: AtomicRequest(request_type=RequestType.FORWARD_RATE,time1=None,time2=self.maturity)}

    def __eq__(self, other):
        return (isinstance(other, Bond) and 
                self.maturity.item() == other.maturity.item())

    def __hash__(self):
        return hash(self.maturity.item())
    
    def get_requests(self):
        requests=defaultdict(set)

        for t, req in self.numeraire_requests.items():
            requests[t].add(req)

        return requests
    
    def get_atomic_requests(self,observation_date):
        requests=defaultdict(set)

        for t, req in self.composite_request.items():
            req.time1=observation_date
            requests[t].add(req)

        return requests
    
    def get_composite_requests(self):
        return CompositeRequest(self)
    
    def get_value(self, resolved_atomic_requests):
        return resolved_atomic_requests[self.composite_request[0].handle]
