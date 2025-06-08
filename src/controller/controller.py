
from itertools import groupby
from operator import itemgetter
import torch
import numpy as np
from engine.engine import MonteCarloEngine
from request_interface.request_interface import RequestInterface, ModelRequest, ModelRequestType
from metrics.metric import MetricType, Metric
from enum import Enum, auto
from collections import defaultdict
from common.packages import *
from maths.monomials import Polynomials

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

    def get_results(self, prod_idx, metric_idx):
        return self.results[prod_idx][metric_idx]

    def get_derivatives(self, prod_idx, metric_idx):
        return self.derivatives[prod_idx][metric_idx]
    
    def get_second_derivatives(self, prod_idx, metric_idx):
        return self.second_derivatives[prod_idx][metric_idx]

class SimulationController:
    def __init__(self, portfolio, model, 
                 metrics, 
                 num_paths_mainsim,
                 num_paths_presim, 
                 num_steps, 
                 simulation_scheme,
                 differentiate=False,
                 exposure_timeline=[],
                 regression_monomials=Polynomials(degree=2)):
        self.portfolio = portfolio
        self.model = model
        self.metrics = metrics
        self.num_paths_presim = num_paths_presim
        self.num_paths_mainsim = num_paths_mainsim
        self.num_steps = num_steps
        self.simulation_scheme = simulation_scheme
        self.differentiate = differentiate
        self.exposure_timeline = torch.tensor(exposure_timeline, dtype=torch.float64)
        self.regression_monomials=regression_monomials
        self.requires_higher_order_derivatives=False

        prod_id=0
        for prod in portfolio:
            prod.product_id=prod_id
            prod_id+=1

        self.regression_coeffs = [
            [  # For each time point in this product
                [torch.zeros(3, device=device) for _ in range(prod.get_num_states())]
                for _ in self.exposure_timeline
            ]
            for prod in portfolio
        ]

        self.numeraire_requests = {idx: ModelRequest(ModelRequestType.NUMERAIRE, t) for idx, t in enumerate(exposure_timeline)}
        self.spot_requests = {idx: ModelRequest(ModelRequestType.SPOT) for idx in range(len(exposure_timeline))}

        prod_times = {float(t.item()) for prod in self.portfolio for t in prod.modeling_timeline}
        exposure_times = {float(t.item()) for t in exposure_timeline}
        all_times = sorted(prod_times.union(exposure_times))
        self.simulation_timeline = torch.tensor(all_times, dtype=torch.float64)


        self.requires_regression = any(len(prod.regression_timeline) > 0 for prod in self.portfolio) or len(exposure_timeline) > 0


    def _get_requests(self):
        requests=defaultdict(set)
        for t, req in self.numeraire_requests.items():
            requests[t].add(req)

        for t, req in self.spot_requests.items():
            requests[t].add(req)


        return requests
    
    def compute_higher_derivatives(self):
        self.requires_higher_order_derivatives=True

    def perform_prepocessing(self, products, request_interface):
        request_interface.collect_and_index_requests(self.portfolio,self.simulation_timeline, self._get_requests(), self.exposure_timeline)
        if self.requires_regression:
            self._perform_regression(products, request_interface)

    def _perform_regression(self, products, request_interface):

        engine = MonteCarloEngine(
            simulation_timeline=self.simulation_timeline,
            simulation_type=self.simulation_scheme,
            model=self.model,
            num_paths=self.num_paths_presim,
            num_steps=self.num_steps,
            is_pre_simulation=True
        )

        paths = engine.generate_paths()
        resolved_requests = request_interface.resolve_requests(paths)

        for prod in products:
            self._perform_regression_for_product(prod, resolved_requests)

    def _perform_regression_for_product(self, product, resolved_requests):

        regression_timeline = set(product.regression_timeline.tolist()).union(self.exposure_timeline.tolist())
        regression_timeline = torch.tensor(sorted(regression_timeline), dtype=torch.float64)

        product_timeline = product.product_timeline
        product_regression_timeline = product.regression_timeline
        num_states = product.get_num_states()
        num_paths = self.num_paths_presim

        last_cf_index_computed = len(product_timeline)
        cf_cache = defaultdict(lambda: torch.zeros((num_paths, num_states), device=device))
        cf_cache[last_cf_index_computed] = torch.zeros((num_paths, num_states), device=device)

        for reg_idx, t_reg in reversed(list(enumerate(regression_timeline))):
            total_cfs = torch.zeros((num_paths, num_states), device=device)

            product_time_idx = torch.searchsorted(product_timeline, t_reg).item()

            if product_time_idx >= len(product_timeline):
                continue
            t_next_idx = product_time_idx + 1 if product_timeline[product_time_idx] == t_reg else product_time_idx

            if t_next_idx < last_cf_index_computed:
                if num_states == 1:
                    total_cfs[:, 0] = cf_cache[last_cf_index_computed][:, 0]
                    for idx in range(t_next_idx, last_cf_index_computed):
                        _, cfs = product.compute_normalized_cashflows(
                            idx, self.model.get_model_params(), resolved_requests, self.regression_monomials
                        )
                        total_cfs[:, 0] += cfs

                    cf_cache[t_next_idx] = total_cfs.clone()
                    last_cf_index_computed = t_next_idx
                else:
                    for state in range(num_states):
                        current_state = torch.full((num_paths,), state, dtype=torch.long, device=device)
                        for idx in range(t_next_idx, len(product.product_timeline)):
                            current_state, cfs = product.compute_normalized_cashflows(
                                idx, self.model, resolved_requests, self.regression_monomials, current_state
                            )
                            total_cfs[:, state] += cfs

                    cf_cache[t_next_idx] = total_cfs.clone()
                    last_cf_index_computed = t_next_idx
            else:
                total_cfs = cf_cache[t_next_idx]


            if t_reg in product_regression_timeline:
                i_t = product_timeline.tolist().index(t_reg)
                numeraire = resolved_requests[product.numeraire_requests[i_t].handle]
                explanatory = resolved_requests[product.spot_requests[i_t].handle]
            else:
                i_t = self.exposure_timeline.tolist().index(t_reg)
                numeraire = resolved_requests[self.numeraire_requests[i_t].handle]
                explanatory = resolved_requests[self.spot_requests[i_t].handle]

            normalized_cfs = numeraire.unsqueeze(1) * total_cfs

            A = self.regression_monomials.get_regression_matrix(explanatory)

            for s in range(num_states):
                Y = normalized_cfs[:, s]
                coeffs = torch.linalg.lstsq(A, Y.unsqueeze(1)).solution.squeeze()
                
                if t_reg in product_regression_timeline:
                    product_reg_idx = torch.searchsorted(product_regression_timeline, t_reg)
                    product.regression_coeffs[product_reg_idx][s] = coeffs

                if t_reg in self.exposure_timeline:
                    exp_reg_idx = torch.searchsorted(self.exposure_timeline, t_reg)
                    self.regression_coeffs[product.product_id][exp_reg_idx][s] = coeffs

    def evaluate_product(self, product, resolved_requests):
        prod_state=product.get_initial_state(self.num_paths_mainsim)
        num_paths=self.num_paths_mainsim
        exposures=[]
        t_start=0
        cfs = torch.zeros(num_paths, dtype=torch.float64, device=device)

        any_pv=any(m.metric_type==MetricType.PV for m in self.metrics)

        if len(self.exposure_timeline)==0 and any_pv:
            while t_start < len(product.product_timeline):
                    prod_state, new_cfs=product.compute_normalized_cashflows(t_start, self.model, resolved_requests, self.regression_monomials, prod_state)
                    cfs+=new_cfs
                    t_start+=1

        else:
            for i, t in enumerate(self.exposure_timeline):
                while t_start < len(product.product_timeline) and product.product_timeline[t_start] <= t:
                    prod_state, new_cfs=product.compute_normalized_cashflows(t_start, self.model, resolved_requests, self.regression_monomials,prod_state)
                    cfs+=new_cfs
                    t_start+=1

                explanatory=resolved_requests[self.spot_requests[i].handle]
                # Compute continuation value: A @ coeffs_per_path
                A = self.regression_monomials.get_regression_matrix(explanatory)

                coeffs_matrix = torch.stack([
                    self.regression_coeffs[product.product_id][i][s] for s in prod_state
                ], dim=0)  # shape: [num_paths, 3]

                numeraire= resolved_requests[self.numeraire_requests[i].handle]
                exposure = (A * coeffs_matrix).sum(dim=1)/numeraire
                exposures.append(exposure)

                if any_pv:
                    while t_start < len(product.timeline):
                        prod_state, new_cfs=product.compute_normalized_cashflows(t_start, self.model, resolved_requests, self.regression_monomials, prod_state)
                        cfs+=new_cfs
                        t_start+=1


        results=[]

        for metric in self.metrics:
            eval=metric.evaluate(exposures, cfs)
            results.append(eval)

        return results

    def evaluate_products(self, resolved_requests):
        results=[]
        for product in self.portfolio:
            result=self.evaluate_product(product,resolved_requests)
            results.append(result)

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
                            create_graph=True,
                        )
                        grads_per_eval.append(_grads)
                    grads_per_metric.append(grads_per_eval)
                grads.append(grads_per_metric)

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
                                )
                                higher_grads_per_eval.append(_higher_grads)
                            evaluation_grads.append(higher_grads_per_eval)
                        higher_grads_per_metric.append(evaluation_grads)
                    higher_grads.append(higher_grads_per_metric)

        return SimulationResults(results, grads, higher_grads)


    def run_simulation(self):
        request_interface = RequestInterface(self.model)

        self.perform_prepocessing(self.portfolio, request_interface)
        
        main_engine = MonteCarloEngine(
            simulation_timeline=self.simulation_timeline,
            simulation_type=self.simulation_scheme,
            model=self.model,
            num_paths=self.num_paths_mainsim,
            num_steps=self.num_steps,
            is_pre_simulation=False
        )

        paths=main_engine.generate_paths()
        resolved_requests = request_interface.resolve_requests(paths)

        return self.evaluate_products(resolved_requests)