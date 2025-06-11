from context import *

import torch
import numpy as np
import matplotlib.pyplot as plt
from controller.controller import SimulationController
from models.vasicek import *
from metrics.pfe_metric import *
from metrics.epe_metric import *
from metrics.ene_metric import *
from products.swap import InterestRateSwap, IRSType
from engine.engine import *


if __name__ == "__main__":
    # # --- GPU device setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup model and product


    model = VasicekModel(calibration_date=0.,rate=0.03,mean=0.05,mean_reversion_speed=0.02,volatility=0.02)
    exercise_dates = [3.0]
    maturity = 3.0
    strike = 100.0

    irs1 = InterestRateSwap(startdate=0.0,enddate=2.0,notional=1.0,fixed_rate=0.03,tenor_fixed=0.25,tenor_float=0.25, irs_type=IRSType.RECEIVER)
    irs2 = InterestRateSwap(startdate=0.0,enddate=2.0,notional=1.0,fixed_rate=0.03,tenor_fixed=0.5,tenor_float=0.25, irs_type=IRSType.RECEIVER)
    #product = BarrierOption(strike, 120,BarrierOptionType.UPANDOUT,0.0,2.0,OptionType.CALL,10)
    #product = BermudanOption(maturity=maturity, exercise_dates=exercise_dates, strike=strike, option_type=OptionType.CALL)

    portfolio=[irs1, irs2]

    # Metric timeline for EE
    exposure_timeline = np.linspace(0, 3.,100)
    epe_metric = EPEMetric()
    ene_metric = ENEMetric()
    pfe_metric = PFEMetric(0.9)

    metrics=[epe_metric, ene_metric, pfe_metric]

    num_paths_mainsim=10000
    num_paths_presim=10000
    num_steps=1
    sc=SimulationController(portfolio, model, metrics, num_paths_mainsim, num_paths_presim, num_steps, SimulationScheme.EULER, False, exposure_timeline)

    sim_results=sc.run_simulation()

    ees_irs1=sim_results.get_results(0,0)
    enes_irs1=sim_results.get_results(0,1)
    ees_irs2=sim_results.get_results(1,0)
    pfes_irs1=sim_results.get_results(0,2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot for IRS1
    ax1.plot(exposure_timeline, ees_irs1, label='EPE (IRS1)', color='red')
    ax1.plot(exposure_timeline, enes_irs1, label='ENE (IRS1)', color='orange')
    ax1.plot(exposure_timeline, pfes_irs1, label='PFE (IRS1)', color='blue', linestyle='--')
    ax1.set_title('Exposure Metrics for IRS1')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Exposure')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # Plot for comparison between IRS1 and IRS2 EE
    ax2.plot(exposure_timeline, ees_irs1, label='EE (IRS1)', color='red')
    ax2.plot(exposure_timeline, ees_irs2, label='EE (IRS2)', color='green')
    ax2.set_title('Expected Exposure Comparison (IRS1 vs IRS2)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Exposure')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.show()