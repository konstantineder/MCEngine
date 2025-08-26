import torch

class RegressionFunction:
    def __init__(self, degree):
        self.degree=degree

    def get_degree(self):
        return self.degree + 1

class PolyomialRegression(RegressionFunction):
    def __init__(self, degree):
        super().__init__(degree)

    def get_regression_matrix(self, explanatory_variables):
        return torch.stack([explanatory_variables**k for k in range(self.degree+1)],dim=1)