from context import *

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from controller.controller import SimulationController
from models.black_scholes import BlackScholesModel
from metrics.pv_metric import PVMetric  
from products.european_option_equity import EuropeanOption, OptionType
from products.bermudan_option import BermudanOption
from engine.engine import SimulationScheme

# --- Hyperparameters ---
gamma = 0.90
epsilon = 0.1
epsilon_min = 0.1
epsilon_decay = 0.1
batch_size = 64
replay_capacity = 10000
learning_rate = 1e-3
update_target_every = 10
num_episodes = 10000
exercise_dates = torch.linspace(0.0, 1.0, steps=5)[1:]

K = 100  # Strike price
r = 0.05
sigma = 0.2
S0 = 100

# --- Q-Network ---
class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),  # state = (t, S_t)
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Q values for [hold, exercise]
        )

    def forward(self, state):
        return self.net(state)

# --- Environment: Simulates GBM path ---
def simulate_gbm_path(S0, exercise_dates, sigma, r):
    path = []
    prev_date=0.0
    spot=S0
    for idx in range(len(exercise_dates)):
        next_date=exercise_dates[idx]
        dt=next_date-prev_date
        dW = np.random.normal(0, np.sqrt(dt))
        spot = spot * np.exp((r - 0.5 * sigma**2) * dt + sigma * dW)
        path.append(spot)
        prev_date=next_date
    return path

# --- Epsilon-greedy policy ---
def select_action(q_net, state_tensor, epsilon):
    if random.random() < epsilon:
        return random.choice([0, 1])  # hold or exercise
    else:
        with torch.no_grad():
            return q_net(state_tensor).argmax().item()

# --- Main training loop ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
q_net = QNetwork().to(device)
target_net = QNetwork().to(device)
target_net.load_state_dict(q_net.state_dict())
optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)

replay_buffer = deque(maxlen=replay_capacity)

for episode in range(num_episodes):
    S_path = simulate_gbm_path(S0, exercise_dates, sigma, r)
    done = False

    for idx, t in enumerate(exercise_dates):
        S_t = S_path[idx]
        state = torch.tensor([idx, S_t / K], dtype=FLOAT).to(device)
        action = select_action(q_net, state.unsqueeze(0), epsilon)

        is_last = idx == len(exercise_dates) - 1

        if action == 1:  # Chose to exercise early
            reward = max(K - S_t, 0)
            next_state = None
            done = True

        elif is_last:  # Final timestep, no more exercise opportunity
            reward = 0  # Held to the end but chose not to exercise
            next_state = None
            done = True

        else:  # Hold and continue
            reward = 0
            S_next = S_path[idx + 1]
            next_state = torch.tensor([idx + 1, S_next / K], dtype=FLOAT).to(device)
            done = False

        replay_buffer.append((state, action, reward, next_state, done))

        if done:
            break

    # Sample minibatch
    if len(replay_buffer) >= batch_size:
        batch = random.sample(replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_tensor = torch.stack(states)
        actions_tensor = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
        rewards_tensor = torch.tensor(rewards, dtype=FLOAT).unsqueeze(1).to(device)

        next_q_values = torch.zeros(batch_size, 1).to(device)
        
        non_final_mask = torch.tensor([s is not None for s in next_states], dtype=torch.bool).to(device)

        if non_final_mask.any():
            non_final_next_states = torch.stack([s for s in next_states if s is not None])
            with torch.no_grad():
                next_q_values[non_final_mask] = target_net(non_final_next_states.to(device)).max(1, keepdim=True)[0]


        targets = rewards_tensor + gamma * next_q_values

        predictions = q_net(states_tensor).gather(1, actions_tensor)
        loss = nn.MSELoss()(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

    # Update target network
    if episode % update_target_every == 0:
        target_net.load_state_dict(q_net.state_dict())

print("Training complete. You can now evaluate Q*(s, a).")

def compute_present_value(q_net, num_paths=100000):
    q_net.eval()
    total_discounted_payoff = 0.0

    with torch.no_grad():
        for _ in range(num_paths):
            S_path = simulate_gbm_path(S0, exercise_dates, sigma, r)

            for idx, t in enumerate(exercise_dates):
                S_t = S_path[idx]
                state = torch.tensor([idx, S_t / K], dtype=FLOAT).unsqueeze(0).to(device)
                q_values = q_net(state)
                action = q_values.argmax().item()

                if action == 1:  # Exercise
                    payoff = max(K - S_t, 0)
                    discount = np.exp(-r * t)
                    total_discounted_payoff += payoff * discount
                    break  # stop simulation for this path
            else:
                # Never exercised, payoff is 0
                total_discounted_payoff += 0.0

    return total_discounted_payoff / num_paths


pv = compute_present_value(q_net, num_paths=10000)
print(f"Estimated Present Value of the American Option: {pv:.4f}")


model = BlackScholesModel(0, S0, r, sigma)
#product = BinaryOption(T,strike,10,OptionType.CALL)
bo = BermudanOption(1.0,np.linspace(0.0, 1.0, 5)[1:],K,OptionType.PUT)
eo = EuropeanOption(1.0,K,OptionType.PUT)
#portfolio=[BarrierOption(strike, 120,BarrierOptionType.UPANDOUT,0,T,OptionType.CALL,True,10)]
portfolio = [bo,eo]
metrics=[PVMetric()]
# Compute analytical price (if available)

sc=SimulationController(portfolio, model, metrics, 1000000, 100000, 1, SimulationScheme.ANALYTICAL, True)

sim_results=sc.run_simulation()
print(sim_results.get_results(0,0)[0])
print(sim_results.get_results(1,0)[0])