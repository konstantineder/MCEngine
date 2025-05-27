import torch

class Monomials:
    def __init__(self, degree):
        self.degree=degree

class Polynomials(Monomials):
    def __init__(self, degree):
        super().__init__(degree)

    def get_regression_matrix(self, explanatory_variables):
        return torch.stack([explanatory_variables**k for k in range(self.degree+1)],dim=1)