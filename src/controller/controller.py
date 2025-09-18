from common.packages import *
import numpy as np
from typing import Union, List, Optional, Sequence
from engine.engine import MonteCarloEngine, SimulationScheme
from request_interface.request_interface import RequestInterface, AtomicRequest, AtomicRequestType
from metrics.metric import MetricType, Metric
from models.model import Model
from products.product import Product
from collections import defaultdict
from maths.regression import PolyomialRegression

# Container for simulation results
# - Contains both metric outputs and corresponding derivatives if differentiation is enabled
# - If computation of second order derivatives are enabled these are stored as well
# - Results are converted into numpy arrays
class SimulationResults:
    def __init__(self, results, derivatives, second_derivatives):
        self.results = self._to_numpy_nested(results)
        self.derivatives = self._to_numpy_nested(derivatives)
        self.second_derivatives=self._to_numpy_nested(second_derivatives)

    def _to_numpy_nested(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self._to_numpy_nested(x) for x in obj)
        else:
            return obj

    # Get Metric outputs for each product and metric
    def get_results(self, prod_idx, metric_idx):
        return self.results[prod_idx][metric_idx]

    # Get first order derivatives for each metric output
    # For this the label 'differentiate' in the simulation 
    # controller needs to be enabled
    def get_derivatives(self, prod_idx, metric_idx):
        return self.derivatives[prod_idx][metric_idx]
    
    # Get second order derivatives for each metric output
    # For this the label 'requires_higher_order_derivatives'
    # in the simulation controller needs to be enables
    def get_second_derivatives(self, prod_idx, metric_idx):
        return self.second_derivatives[prod_idx][metric_idx]

# Simulation Controller to perform Monte Carlo simulation
# and compute metric outputs for each product in a portfolio
class SimulationController:
    def __init__(self, 
                 portfolio           : Sequence[Product],      # Portfolio containing all products to be evaluated
                 model               : Model,                  # Model used to simulate paths
                 metrics             : Sequence[Metric],       # Collection of metrics used to compute outputs
                 num_paths_mainsim   : int,                    # Number of Monte Carlo paths used for main simulation
                 num_paths_presim    : int,                    # Number of Monte Carlo paths used for pre simulation
                 num_steps           : int,                    # Number of time steps used for path simulation
                 simulation_scheme   : SimulationScheme,       # Set Simulation Scheme: Schemes currently provided: Analytical, Euler, Milstein
                 differentiate       : bool = False,           # Turn differentiation on or off
                 exposure_timeline   : Union[List[float], np.ndarray, None] = None,     # Set timeline for exposure simulation
                 regression_function: Optional[PolyomialRegression] = None):  # Set regression function used for LSM algorithm
        
        if exposure_timeline is None:
            exposure_timeline = []
        
        if regression_function is None:
            regression_function = PolyomialRegression(degree=2)

        self.portfolio = portfolio
        self.model = model
        self.metrics = metrics
        self.num_paths_presim = num_paths_presim
        self.num_paths_mainsim = num_paths_mainsim
        self.num_steps = num_steps
        self.simulation_scheme = simulation_scheme
        self.differentiate = differentiate
        self.exposure_timeline = torch.tensor(exposure_timeline, dtype=FLOAT, device=device)
        self.regression_function=regression_function
        self.requires_higher_order_derivatives=False

        # Label products in portfolio
        # Used to store and retrieve regression coefficients
        prod_id=0
        for prod in portfolio:
            prod.product_id=prod_id
            prod_id+=1

        # Set differentiation mode
        # If enabled, differentiation of metric outputs is performed via AAD
        if differentiate:
            self.model.requires_grad()

        # Pre-allocate array to store regression coefficients
        # for each product and exposure timepoint
        self.regression_coeffs = [
            [  
                [torch.zeros(self.regression_function.get_degree(), device=device) for _ in range(prod.get_num_states())]
                for _ in self.exposure_timeline
            ]
            for prod in portfolio
        ]

        # Set up requests for each exposure timepoint 
        self.numeraire_requests = {idx: AtomicRequest(AtomicRequestType.NUMERAIRE, t) for idx, t in enumerate(exposure_timeline)}
        self.spot_requests = {idx: AtomicRequest(AtomicRequestType.SPOT) for idx in range(len(exposure_timeline))}

        # Collect timelines of each product and unify with exposure timeline
        # Remove duplicates and sort timeline
        # Store as common simulation timellne
        prod_times = {float(t.item()) for prod in self.portfolio for t in prod.modeling_timeline}
        exposure_times = {t for t in exposure_timeline}
        all_times = sorted(prod_times.union(exposure_times))
        self.simulation_timeline = torch.tensor(all_times, dtype=FLOAT, device=device)

        # Decide whether regression is required
        # If no product in the portfolio needs to be constructed and
        # no exposure metric needs to be evaluated, the regression step in the preprocessing phase can be skipped
        self.requires_regression = any(len(prod.regression_timeline) > 0 for prod in self.portfolio) or len(exposure_timeline) > 0

    # Collect and return all requests for exposure computation 
    def _get_requests(self):
        requests=defaultdict(set)
        for t, req in self.numeraire_requests.items():
            requests[t].add(req)

        for t, req in self.spot_requests.items():
            requests[t].add(req)

        return requests
    
    # Enable computation of second order derivatives
    def compute_higher_derivatives(self):
        self.requires_higher_order_derivatives=True

    # Perfrom preprocessing
    # - Collect and Index requests for each product and exposure timepoint
    # - Perform rergression of products need to be built or exposure metrics need to be computated otherwise skip
    def perform_prepocessing(self, products : Sequence[Product], request_interface : RequestInterface):
        
        request_interface.collect_and_index_requests(self.portfolio,self.simulation_timeline, self._get_requests(), self.exposure_timeline)
        if self.requires_regression:
            self._perform_regression(products, request_interface)

    # Perform regression of each constructible product and expsoure timepoint
    def _perform_regression(self, products : Sequence[Product], request_interface : RequestInterface):

        # Set up Monte Carlo engine to simulate paths for pre-simulation
        engine = MonteCarloEngine(
            simulation_timeline=self.simulation_timeline,
            simulation_type=self.simulation_scheme,
            model=self.model,
            num_paths=self.num_paths_presim,
            num_steps=self.num_steps,
            is_pre_simulation=True
        )

        # Generate pre-simulation paths
        paths = engine.generate_paths()

        # Resolve all atomic and underlying requests
        resolved_requests = request_interface.resolve_requests(paths)

        # Perform regression for each product
        for prod in products:
            self._perform_regression_for_product(prod, resolved_requests)

    # Perform regression for a specific product via Dynamic Porgramming using the LSM algorithm
    def _perform_regression_for_product(self, product : Product, resolved_requests : List[dict]):

        regression_timeline = set(product.regression_timeline.tolist()).union(self.exposure_timeline.tolist())
        regression_timeline = torch.tensor(sorted(regression_timeline), dtype=FLOAT, device=device)

        product_timeline = product.product_timeline
        product_regression_timeline = product.regression_timeline
        num_states = product.get_num_states()
        num_paths = self.num_paths_presim

        # Store date at which the last product cashflow has been computed (in reverse order)
        # starting at the last timepoint on the product timeline
        last_cf_index_computed = len(product_timeline)

        # Store last computed cashflows for each product state
        cf_cache = defaultdict(lambda: torch.zeros((num_paths, num_states), device=device))
        cf_cache[last_cf_index_computed] = torch.zeros((num_paths, num_states), device=device)

        # Perform regression via backwards induction and applying the LSM algorithm at each timepoint
        for reg_idx, t_reg in reversed(list(enumerate(regression_timeline))):
            total_cfs = torch.zeros((num_paths, num_states), device=device)

            product_time_idx = int(torch.searchsorted(product_timeline, t_reg).item())

            if product_time_idx >= len(product_timeline):
                continue
            t_next_idx = product_time_idx + 1 if product_timeline[product_time_idx] == t_reg else product_time_idx

            if t_next_idx < last_cf_index_computed:
                if num_states == 1:
                    total_cfs[:, 0] = cf_cache[last_cf_index_computed][:, 0]
                    for idx in range(t_next_idx, last_cf_index_computed):
                        _, cfs = product.compute_normalized_cashflows(
                            idx, self.model, resolved_requests, self.regression_function
                        )
                        total_cfs[:, 0] += cfs

                    cf_cache[t_next_idx] = total_cfs.clone()
                    last_cf_index_computed = t_next_idx
                else:
                    for state in range(num_states):
                        current_state = torch.full((num_paths,), state, dtype=torch.long, device=device)
                        for idx in range(t_next_idx, len(product.product_timeline)):
                            current_state, cfs = product.compute_normalized_cashflows(
                                idx, self.model, resolved_requests, self.regression_function, current_state
                            )
                            total_cfs[:, state] += cfs

                    cf_cache[t_next_idx] = total_cfs.clone()
                    last_cf_index_computed = t_next_idx
            else:
                total_cfs = cf_cache[t_next_idx]


            if t_reg in product_regression_timeline:
                i_t = product_timeline.tolist().index(t_reg)
                numeraire = resolved_requests[0][product.numeraire_requests[i_t].handle]
                explanatory = resolved_requests[0][product.spot_requests[i_t].handle]
            else:
                i_t = self.exposure_timeline.tolist().index(t_reg)
                numeraire = resolved_requests[0][self.numeraire_requests[i_t].handle]
                explanatory = resolved_requests[0][self.spot_requests[i_t].handle]

            normalized_cfs = numeraire.unsqueeze(1) * total_cfs

            A = self.regression_function.get_regression_matrix(explanatory)

            for s in range(num_states):
                Y = normalized_cfs[:, s]
                coeffs = (torch.linalg.pinv(A) @ Y.unsqueeze(-1)).squeeze(-1)
                
                if t_reg in product_regression_timeline:
                    product_reg_idx = torch.searchsorted(product_regression_timeline, t_reg)
                    product.regression_coeffs[product_reg_idx][s] = coeffs

                if t_reg in self.exposure_timeline:
                    exp_reg_idx = torch.searchsorted(self.exposure_timeline, t_reg)
                    self.regression_coeffs[product.product_id][exp_reg_idx][s] = coeffs

    def _evaluate_product(self, product : Product, resolved_requests : List[dict]):
        prod_state=product.get_initial_state(self.num_paths_mainsim)
        num_paths=self.num_paths_mainsim
        exposures=[]
        t_start=0
        cfs = torch.zeros(num_paths, dtype=FLOAT, device=device)

        any_pv=any(m.metric_type==MetricType.PV for m in self.metrics)

        if len(self.exposure_timeline)==0 and any_pv:
            while t_start < len(product.product_timeline):
                    prod_state, new_cfs=product.compute_normalized_cashflows(t_start, self.model, resolved_requests, self.regression_function, prod_state)
                    cfs+=new_cfs
                    t_start+=1

        else:
            for i, t in enumerate(self.exposure_timeline):
                while t_start < len(product.product_timeline) and product.product_timeline[t_start] <= t:
                    prod_state, new_cfs=product.compute_normalized_cashflows(t_start, self.model, resolved_requests, self.regression_function,prod_state)
                    cfs+=new_cfs
                    t_start+=1

                explanatory=resolved_requests[0][self.spot_requests[i].handle]
                # Compute continuation value: A @ coeffs_per_path
                A = self.regression_function.get_regression_matrix(explanatory)

                coeffs_matrix = torch.stack([
                    self.regression_coeffs[product.product_id][i][s] for s in prod_state
                ], dim=0)  # shape: [num_paths, 3]

                numeraire= resolved_requests[0][self.numeraire_requests[i].handle]
                exposure = (A * coeffs_matrix).sum(dim=1)/numeraire
                exposures.append(exposure)

                if any_pv:
                    while t_start < len(product.product_timeline):
                        prod_state, new_cfs=product.compute_normalized_cashflows(t_start, self.model, resolved_requests, self.regression_function, prod_state)
                        cfs+=new_cfs
                        t_start+=1


        results=[]

        for metric in self.metrics:
            eval=metric.evaluate(exposures, cfs)
            results.append(eval)

        return results

    # Compute metric outputs in main simulation phase
    def evaluate_products(self, resolved_requests : List[dict]):
        results=[]
        for product in self.portfolio:
            result=self._evaluate_product(product,resolved_requests)
            results.append(result)

        # # Improve performance by parallelizing product evalutation in product dimension
        # import concurrent.futures

        # results_by_id = {}

        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     futures = {
        #         executor.submit(self.evaluate_product, product, resolved_requests): product.product_id
        #         for product in self.portfolio
        #     }

        #     for future in concurrent.futures.as_completed(futures):
        #         product_id = futures[future]
        #         results_by_id[product_id] = future.result()

        # If differentiation is enabled, compute and store first order derivatives 
        # via algorithmic adjoint differentiation (AAD) using PyTorch 'autograd' method
        # for each metric output for each product
        grads = []
        higher_grads = []
        if self.differentiate:
            model_params = self.model.get_model_params()
            for prod in results:
                grads_per_metric = []
                for metric in prod:
                    grads_per_eval = []
                    for eval in metric:
                        _grads = torch.autograd.grad(
                            eval,
                            model_params,
                            retain_graph=True,
                            create_graph=self.requires_higher_order_derivatives,
                            allow_unused=True
                        )
                        grads_per_eval.append(_grads)
                    grads_per_metric.append(grads_per_eval)
                grads.append(grads_per_metric)

            # If higher order derivatives are enabled, the second order derivatives
            # are computed and stored
            if self.requires_higher_order_derivatives:
                for prod_grads in grads:
                    higher_grads_per_metric = []
                    for metric_grads in prod_grads:
                        evaluation_grads = []
                        for eval_grads in metric_grads:
                            higher_grads_per_eval = []
                            for grad in eval_grads:
                                _higher_grads = torch.autograd.grad(
                                    grad,
                                    model_params,
                                    retain_graph=True,
                                    allow_unused=True
                                )
                                higher_grads_per_eval.append(_higher_grads)
                            evaluation_grads.append(higher_grads_per_eval)
                        higher_grads_per_metric.append(evaluation_grads)
                    higher_grads.append(higher_grads_per_metric)

        return SimulationResults(results, grads, higher_grads)

    # Perform entire simulation to simulate metric outputs
    def run_simulation(self):

        # Instatiate request interface to index, collect and solve requests by the model
        request_interface = RequestInterface(self.model)

        # Perform preprocessing:
        # - Collect and index requests for each product and exposure timepoint
        # - If products or exposure metrics need to be constructed a regression will be performed
        #   using the Longstaff-Schwartz algorithm 
        self.perform_prepocessing(self.portfolio, request_interface)
        
        # Instantiate Monte Carlo engine to simulate paths in the main simulation phase
        main_engine = MonteCarloEngine(
            simulation_timeline=self.simulation_timeline,
            simulation_type=self.simulation_scheme,
            model=self.model,
            num_paths=self.num_paths_mainsim,
            num_steps=self.num_steps,
            is_pre_simulation=False
        )

        # Generate Monte Carlo paths
        paths=main_engine.generate_paths()

        # Resolve product and exposure requests 
        resolved_requests = request_interface.resolve_requests(paths)

        return self.evaluate_products(resolved_requests)