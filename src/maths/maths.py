from common.packages import *

def symmetric_linear_smoothing(x,is_fuzzy,eps):
    if not is_fuzzy:
        return (x > 0).float()
    return torch.clamp((x + eps) / (2 * eps), min=0.0, max=1.0)
    
def compute_degree_of_truth(x,is_fuzzy,eps=0.05):
    return symmetric_linear_smoothing(x,is_fuzzy,eps)

def sigmoid_smoothing(x, beta=500):
    return torch.sigmoid(beta * x)

