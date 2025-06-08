from enum import Enum 
from common.packages import *
from collections import defaultdict

class ModelRequestType(Enum):
    SPOT=1
    DISCOUNT_FACTOR = 2
    NUMERAIRE = 3
    FORWARD_RATE = 4
    LIBOR_RATE = 5

class ModelRequest:
    def __init__(self, request_type, time1=None, time2=None):
        self.request_type = request_type
        self.time1 = time1
        self.time2 = time2
        self.handle = None  # Set during deduplication

    def key(self):
        return (self.request_type, self.time1, self.time2)

    def __eq__(self, other):
        return self.key() == other.key()

    def __hash__(self):
        return hash(self.key())
    


class RequestInterface:
    def __init__(self, model):
        self.model = model  # The model that can resolve requests
        self.num_requests=0
        self.all_requests=defaultdict(set)
        

    def _request_key(self, req):
        """Generate a hashable key for a request."""
        return (req.request_type, req.time1, req.time2)
    
    def collect_and_index_requests(self, products, simulation_timeline, exposure_requests, exposure_timeline):
        """Collect, deduplicate per time point, and assign handles to all requests."""
        from collections import defaultdict

        all_requests = defaultdict(set)
        time_to_index = {float(t): idx for idx, t in enumerate(simulation_timeline)}
        request_key_to_handle = {}  # key: (time_index, req) -> handle

        counter = 0

        def register(req, time_index):
            nonlocal counter
            key = (time_index, req)
            if key not in request_key_to_handle:
                request_key_to_handle[key] = counter
                counter += 1
            req.handle = request_key_to_handle[key]

        # From products
        for prod in products:
            requests = prod.get_requests()
            for t, reqs in requests.items():
                time_index = time_to_index[float(prod.modeling_timeline[t])]
                for req in reqs:
                    all_requests[time_index].add(req)
                    register(req, time_index)

        # From external exposure requests
        for t, exp_reqs in exposure_requests.items():
            time_index = time_to_index[float(exposure_timeline[t])]
            for exp_req in exp_reqs:
                all_requests[time_index].add(exp_req)
                register(exp_req, time_index)

        self.num_requests = counter
        self.all_requests = all_requests

    def resolve_requests(self, paths):
        """Resolve all requests once paths have been simulated."""
        resolved_requests = {}

        for t, reqs in self.all_requests.items():
            for req in reqs:
                state = paths[:, t]
                result = self.model.resolve_request(req, state)  # could be [num_paths] or [num_paths, num_assets]
                resolved_requests[req.handle] = result

        return resolved_requests
