


# class BermudanSwaption(Product):
#     def __init__(self, maturity,exercise_dates,underlying_swap):
#         super().__init__()
#         self.underlying_swap=underlying_swap
#         self.exercise_dates=torch.tensor(exercise_dates, dtype=torch.float64, device=device)

#         self.regression_timeline = self.exercise_dates
#         self.regression_coeffs = [
#             [torch.zeros(3, device=device) for _ in range(2)]  # 2 states
#             for _ in range(len(self.regression_timeline))      # len(time points)
#         ]
#         self.num_exercise_rights = 1

#         self.libor_requests = {}
#         self.numeraire_requests = {}
#         self.product_timeline = []

#         date = startdate + tenor
#         idx=0
#         while date < enddate:
#             self.libor_requests[idx]=ModelRequest(ModelRequestType.LIBOR_RATE, date - tenor, date)
#             self.numeraire_requests[idx]=ModelRequest(ModelRequestType.NUMERAIRE, date)
#             self.product_timeline.append(date)
#             date += tenor
#             idx+=1

#         # Last discount factor at enddate
#         self.numeraire_requests[idx]=ModelRequest(ModelRequestType.DISCOUNT_FACTOR, enddate)
#         self.libor_requests[idx]=ModelRequest(ModelRequestType.LIBOR_RATE, date - tenor, enddate)
#         self.product_timeline.append(enddate)
#         self.product_timeline=torch.tensor(self.product_timeline, dtype=torch.float64)
#         self.modeling_timeline=self.product_timeline
    
#     def get_requests(self):
#         requests=defaultdict(set)
#         for t, req in self.numeraire_requests.items():
#             requests[t].add(req)

#         for t, req in self.libor_requests.items():
#             requests[t].add(req)


#         return requests
    
#     def _make_exercise_decision(self,time_idx,resolved_requests,state):
#         spot = resolved_requests[self.spot_requests[time_idx].handle]
#         immediate = self.payoff(spot, model_params)

#         # Check if time is the last in the product timeline
#         if time_idx == len(self.product_timeline) - 1:
#             continuation = torch.zeros_like(immediate)
#         else:
#             A = torch.stack([torch.ones_like(spot), spot, spot ** 2], dim=1)
#             coeffs_matrix = torch.stack([self.regression_coeffs[time_idx][s] for s in state.tolist()], dim=0)  # [num_paths, 3]

#             continuation = (A * coeffs_matrix).sum(dim=1)

#         # Per-path exercise decision
#         should_exercise = (immediate > continuation.squeeze()) & (state > 0)

#         # Accumulate cashflows
#         numeraire=resolved_requests[self.numeraire_requests[time_idx].handle]
#         cashflows = immediate * should_exercise.float()/numeraire

#         # Update remaining rights
#         state -= should_exercise.int()

#         return state

#     def compute_normalized_cashflows(self, time_idx, model, resolved_requests, state):
        
#         if self.product_timeline[time_idx] in self.exercise_dates:
#             state = self._make_exercise_decision(time_idx,resolved_requests,state)

#         return state, cashflow