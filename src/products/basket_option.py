from products.product import *
from request_interface.request_interface import RequestType, AtomicRequest
from collections import defaultdict

class BasketOptionType(Enum):
    ARITHMETIC=0
    GEOMETRIC=1

class BasketOption(Product):
    def __init__(self,maturity,weights,strike,option_type,basket_option_type=BasketOptionType.ARITHMETIC, use_variation_reduction=False):
        super().__init__()
        self.maturity = torch.tensor([maturity], dtype=torch.float64,device=device)
        self.strike = torch.tensor([strike], dtype=torch.float64,device=device)
        self.weights=torch.tensor(weights, dtype=torch.float64,device=device)
        self.option_type = option_type
        self.product_timeline=torch.tensor([maturity], dtype=torch.float64,device=device)
        self.modeling_timeline=self.product_timeline
        self.regression_timeline=torch.tensor([], dtype=torch.float64,device=device)
        self.basket_option_type=basket_option_type
        self.use_variation_reduction=use_variation_reduction

        self.numeraire_requests={0: AtomicRequest(RequestType.NUMERAIRE,maturity)}
        self.spot_requests={0: AtomicRequest(RequestType.SPOT)}
    
    def get_requests(self):
        requests=defaultdict(set)
        for t, req in self.numeraire_requests.items():
            requests[t].add(req)

        for t, req in self.spot_requests.items():
            requests[t].add(req)


        return requests
    
    def payoff(self, spots, model):
        if self.use_variation_reduction:
            return self.payoff_variation_reduction(spots,model)
        else:
            return self.compute_payoff(spots,self.basket_option_type)

    def compute_payoff(self, spots, basket_option_type):
            # spots: shape [num_paths, num_assets]
            zero = torch.tensor(0.0, dtype=torch.float64, device=device)
            weights = self.weights

            if basket_option_type == BasketOptionType.ARITHMETIC:
                basket = (spots * weights).sum(dim=1)  # weighted arithmetic mean
            else:  # GEOMETRIC
                log_spots = torch.log(spots + 1e-10)  # prevent log(0)
                log_basket = (log_spots * weights).sum(dim=1)
                basket = torch.exp(log_basket)

            if self.option_type == OptionType.CALL:
                return torch.maximum(basket - self.strike, zero)
            else:
                return torch.maximum(self.strike - basket, zero)
            
    def payoff_variation_reduction(self, spots, model):
        payoff_classical=self.compute_payoff(spots,self.basket_option_type)
        payoff_reduction=self.compute_payoff(spots,BasketOptionType.GEOMETRIC)

        correction_term=self.compute_pv_analytically(model)

        return payoff_classical-payoff_reduction+correction_term
        
    def compute_normalized_cashflows(self, time_idx, model, resolved_requests, regression_monomials=None, state=None):
        spots = resolved_requests[self.spot_requests[time_idx].handle]
        cfs = self.payoff(spots, model)
        numeraire = resolved_requests[self.numeraire_requests[time_idx].handle]
        normalized_cfs = cfs / numeraire
        return state, normalized_cfs

    def compute_pv_analytically(self, model):
        # Assumes geometric basket under Black-Scholes

        S = model.get_spot()                  # Tensor of shape [n]
        r = model.get_rate()                 # Scalar tensor
        sigmas=model.get_volatility()
        T = self.maturity        # Scalar float
        K = self.strike          # Scalar float
        n = len(S)
        w=self.weights
        
        # 1. Geometric mean of initial prices
        log_S = torch.log(S)
        log_S_bar = log_S.mean()
        F_S_bar = torch.exp(log_S_bar)

        # 2. Covariance matrix: Σ = corr * σ_i * σ_j
        cov_matrix = model.compute_cov_matrix(T)  # shape [n, n]

        # 3. Compute basket variance: σ² = 1/n² * wᵀ Σ w, with w_i = 1/n
        basket_variance = torch.dot(w, torch.mv(cov_matrix, w))
        sigma = torch.sqrt(basket_variance)       # Effective basket volatility

        # 4. Adjusted forward price F
        sum_sigma_squared = torch.sum(sigmas ** 2)
        F = F_S_bar * torch.exp((r - 0.5 * sum_sigma_squared / n + 0.5 * sigma**2) * T)

        # 5. Black-Scholes d1, d2
        sigma_sqrt_T = sigma * torch.sqrt(T)
        d1 = (torch.log(F / K) + 0.5 * sigma**2 * T) / sigma_sqrt_T
        d2 = d1 - sigma_sqrt_T

        norm = torch.distributions.Normal(0.0, 1.0)

        # 6. Present value
        if self.option_type == OptionType.CALL:
            pv = torch.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
        else:
            pv = torch.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

        return pv
     
    