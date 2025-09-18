from context import *

import torch
import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
from itertools import product as cartesian_product
from models.black_scholes import *
from metrics.pv_metric import *
from products.european_option import *
from engine.engine import *
from products.equity import Equity


if __name__ == "__main__":
    # # --- GPU device setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
        # Numerical Differentiation
    def second_partial_diff(function, args, arg1_num, arg2_num, shift1=1e-4, shift2=1e-4):
        # Modify the argument at position `arg_num` by `shift`
        args[arg1_num] = args[arg1_num] + shift1
        args[arg2_num] = args[arg2_num] + shift2
        fplusplus = function(args)  # Call function with the modified argument

        args[arg2_num] = args[arg2_num] - 2*shift2
        fplusminus = function(args)  # Call function with the modified argument

        args[arg1_num] = args[arg1_num] - 2*shift1
        fminusminus = function(args)  # Call function with the modified argument

        args[arg2_num] = args[arg2_num] + 2*shift2
        fminusplus = function(args)  # Call function with the modified argument

        second_derivative = (fplusplus-fplusminus-fminusplus + fminusminus) / (4 * shift1*shift2)
        return second_derivative
    
    
    def partial_diff2(function, args, arg_num, shift=1e-5):
        # Modify the argument at position `arg_num` by `shift`
        f=function(args)

        args[arg_num] = args[arg_num] + shift
        fplus = function(args)  # Call function with the modified argument

        args[arg_num] = args[arg_num] - 2*shift
        fminus = function(args)  # Call function with the modified argument

        second_derivative = (fplus + fminus - 2*f) / (2 * shift**2)
        return second_derivative
    

    # # --- Relative error function ---
    def rel_err(x, y, eps=1e-4):
        denom = abs(x) + abs(y)
        if denom < eps:
            return 0.0
        return 2 * abs(x - y) / denom

    def compute_prices_for_grid(param_grid, num_paths, steps):
        results = []

        for T, S0, sigma, rate, strike in param_grid:


            spot = torch.tensor([S0], dtype=FLOAT, requires_grad=True, device=device)
            Trate = torch.tensor([rate], dtype=FLOAT, requires_grad=True, device=device)
            Tsigma =torch.tensor([sigma], dtype=FLOAT, requires_grad=True, device=device)
            time = torch.tensor(T, dtype=FLOAT, device=device)
            Tstrike = torch.tensor(strike, dtype=FLOAT, device=device)

            d1 = (torch.log(spot / Tstrike) + (Trate + 0.5 * Tsigma ** 2) * time) / (Tsigma * torch.sqrt(time))
            d2 = d1 - Tsigma * torch.sqrt(time)

            norm = torch.distributions.Normal(0.0, 1.0)

            pv=spot * norm.cdf(d1) - Tstrike * torch.exp(-Trate * time) * norm.cdf(d2)
            params=[spot,Tsigma,Trate]
            grads = torch.autograd.grad(pv, params, retain_graph=True,create_graph=True)
            secondgrads=[]
            for grad in grads:
                secondgrad = torch.autograd.grad(grad, params,retain_graph=True)
                secondgrads.append(secondgrad)
            
            model = BlackScholesModel(0, S0, rate, sigma)
            #product = BinaryOption(T,strike,10,OptionType.CALL)
            underlying = Equity("id")
            product = EuropeanOption(underlying,T,strike,OptionType.CALL)

            vomma_analytic=product.compute_dVegadSigma_analytically(model)

            results.append({
                "spot": S0,
                "vola": sigma,
                "rate": rate,
                "vomma": vomma_analytic,
                "time to maturity": T,
                "Vomma": secondgrads[1][1],
            })

        return pd.DataFrame(results)
    
    # Define parameter grid
    S0_vals = np.linspace(10, 300, 30)
    sigma_vals = np.linspace(0.3,1,30)
    r_vals = [0.05]
    strikes = [100]
    T_vals = [2.0]

    # Cartesian product of all combinations
    # defining the parameter grid
    param_grid = list(cartesian_product(T_vals,S0_vals, sigma_vals, r_vals, strikes))

    num_paths = 100000
    steps = 1
    
    # Simulate option prices and store in data frame.
    # Since only spot price and time to maturity are varied
    # the rate and volatility will be filtered out
    df_results_spot_maturity=compute_prices_for_grid(param_grid,num_paths,steps).drop(columns=["rate", "time to maturity"])
    
    # def compute_pv_analytically_wrapper(args):
    #     spot, rate, vola = args
    #     model_deriv = BlackScholesModel(0, spot, rate, vola)
    #     #product_deriv = BarrierOption(100, 120,BarrierOptionType.UPANDOUT,0.0,2.0,OptionType.CALL,True,10)
    #     product_deriv = EuropeanOption(2.0,100,OptionType.CALL)
    #     #product_deriv=BinaryOption(2.0,100,10,OptionType.CALL)
        
    #     return float(product_deriv.compute_pv_analytically(model_deriv))

    # Define X, Y and Z data
    X = df_results_spot_maturity["spot"].astype(float).to_numpy()
    Y = df_results_spot_maturity["vola"].astype(float).to_numpy()
    Zs = {
        "vomma": df_results_spot_maturity["vomma"].astype(float).to_numpy(),
        "Vomma": df_results_spot_maturity["Vomma"].astype(float).to_numpy(),
    }
    #T = df_results_spot_maturity["rel. error (sim)"].astype(float).to_numpy()


    # Set fixed parameters for Delta and Vega plotting
    sigma_fixed = 0.3

    # Create plots
    fig = plt.figure(figsize=(14, 12))

    # Plot for Analytical Price and Simulation
    for i, (title, Z) in enumerate(Zs.items(), 1):
        ax = fig.add_subplot(2, 2, i, projection='3d')
        ax.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor='none')
        ax.set_xlabel("Spot (S₀)")
        ax.set_ylabel("Vola (sigma)")
        ax.set_zlabel(title)
        ax.set_title(title)

    # deriv_param_grid = list(cartesian_product(S0_vals, r_vals, [0.3]))

    # analytic_vanna = np.array([second_partial_diff(compute_pv_analytically_wrapper, [spot, rate, sigma], 0,0) 
    #                        for spot, rate, sigma in deriv_param_grid])

    # analytic_volga = np.array([partial_diff2(compute_pv_analytically_wrapper, [spot, rate, sigma], 0) 
    #                       for spot, rate, sigma in deriv_param_grid])

    # # Plot Delta and Vega below, comparing analytical and simulated values
    # df_delta_vega = df_results_spot_maturity[np.isclose(df_results_spot_maturity["vola"], 0.3)]

    # # Delta Plot
    # ax_delta = fig.add_subplot(2, 2, 3)
    # ax_delta.plot(S0_vals, analytic_vanna, marker='o', linestyle='-', label='Analytic Delta')
    # ax_delta.plot(df_delta_vega["spot"], df_delta_vega["Vanna"], marker='x', linestyle='--', label='Simulated Delta')
    # ax_delta.set_title(f"Delta vs Spot (T={sigma_fixed}, σ={sigma_fixed})")
    # ax_delta.set_xlabel("Spot Price (S₀)")
    # ax_delta.set_ylabel("Delta")
    # ax_delta.grid(True)
    # ax_delta.legend()

    # # Vega Plot
    # ax_vega = fig.add_subplot(2, 2, 4)
    # ax_vega.plot(S0_vals, analytic_volga, marker='o', linestyle='-', label='Analytic Vega')
    # ax_vega.plot(df_delta_vega["spot"], df_delta_vega["vomma"], marker='x', linestyle='--', label='Simulated Vega')
    # ax_vega.set_title(f"Vega vs Spot (T={sigma_fixed}, σ={sigma_fixed})")
    # ax_vega.set_xlabel("Spot Price (S₀)")
    # ax_vega.set_ylabel("Vega")
    # ax_vega.grid(True)
    # ax_vega.legend()

    # plt.tight_layout()
    plt.show()



    