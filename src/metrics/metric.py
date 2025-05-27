from common.packages import *
from enum import Enum

# Enum for metric types 
# Currently only present value
# In the future may be extended to include EE, PFE, etc.
class MetricType(Enum):
    PV = 0
    CE = 1
    EPE = 2
    ENE = 3
    PFE = 4
    EEPE = 5
    CVA = 6

# Metric computation engine
class Metric:
    class EvaluationType(Enum):
         ANALYTICAL=0
         NUMERICAL=1

    def __init__(self, metric_type, evaluation_type):
        self.metric_type=metric_type
        self.evaluation_type=evaluation_type
        self.regression_coeffs=[]

    def evaluate_analytically(self, *args, **kwargs):
        raise NotImplementedError("Analytical evaluation not implemented.")
    
    def evaluate_numerically(self, *args, **kwargs):
        raise NotImplementedError("Numerical evluation not implemented.")


    def evaluate(self,exposures,cfs):
        if self.evaluation_type==Metric.EvaluationType.NUMERICAL:
            return self.evaluate_numerically(exposures,cfs)
        else:
            return self.evaluate_analytically(exposures,cfs)
        
