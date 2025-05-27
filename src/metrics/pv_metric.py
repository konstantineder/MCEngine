from metrics.metric import *

# Metric representing present value
class PVMetric(Metric):
    def __init__(self, evaluation_type=Metric.EvaluationType.NUMERICAL):
        super().__init__(metric_type=MetricType.PV, evaluation_type=evaluation_type)

    def evaluate_analytically(self, *args, **kwargs):
        raise NotImplementedError("Analytical PV not implemented.")

    def evaluate_numerically(self, exposures,cfs):
        return [cfs.mean()]