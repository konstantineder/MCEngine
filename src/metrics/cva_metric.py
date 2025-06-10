from metrics.metric import *

class CVAMetric(Metric):
    def __init__(self, exposure_timeline, default_probability_curve, recovery_rate, evaluation_type=Metric.EvaluationType.NUMERICAL):
        super().__init__(metric_type=MetricType.CVA, evaluation_type=evaluation_type)
        self.exposure_timeline = torch.tensor(exposure_timeline, device=device)
        self.default_prob_curve = default_probability_curve  # Function or list
        self.recovery_rate = recovery_rate
        #self.numeraire_requests = [AtomicRequest(RequestType.NUMERAIRE, t) for t in exposure_timeline]


    def evaluate_analytically(self, *args, **kwargs):
        raise NotImplementedError("Analytical CVA not implemented.")

    def evaluate_numerically(self, explanatory_variables, time_idx, nettingset_idx, state, cfs, continuation, resolved_requests):
        #numeraire = resolved_requests[self.numeraire_requests[time_idx].handle]
        expected_exposure = torch.relu(continuation)
        discounted_exposure = expected_exposure #* numeraire

        pd = self.default_prob_curve(time_idx)  # scalar or tensor
        cva = (1 - self.recovery_rate) * discounted_exposure.mean() * pd
        return [cva]

    def finalize(self, evaluations):
        return sum(evaluations)