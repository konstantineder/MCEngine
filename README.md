Copyright (c) 2025 Dr. Konstantin K. C. Eder

# Monte Carlo Simulation Engine

A flexible, extensible Monte Carlo simulation engine for pricing and risk analytics of financial derivatives.

## Features

- **Metrics Calculation**
  - Present Value (**PV**)
  - Credit Exposure (**CE**)
  - Expected Exposure (**EE**)
  - Effective Expected Exposure (**EEPE**)
  - Potential Future Exposure (**PFE**)
  - Expected Loss (**EL**)
  - xVA metrics (e.g., CVA, DVA, FVA – under development)

- **Sensitivity Analysis**
  - Efficient and scalable **adjoint algorithmic differentiation (AAD)** using [PyTorch](https://pytorch.org/)

- **Financial Products**
  - European Equity Options
  - European Bond Options
  - Binary Options
  - Bermudan Options
  - American Options
  - Barrier Options
  - Basket Option
  - Interest Rate Swaps

- **Models**
  - **Black-Scholes Model**
  - **Black-Scholes Multi-asset Model**
  - Stochastic interest rate models:
    - **Vasicek**
    - **Hull-White**
  
## In Progress

- [ ] Extend the **request interface** to support composite requests
- [ ] Implement valuation of **Bermudan Swaptions**
- [ ] Add **Merton** jump-diffusion model
- [ ] Integrate **Jarrow–Turnbull (JWT)** credit risk model
- [ ] Incorporate **machine learning-based valuation techniques**

## Architecture

- Object-oriented design based on:
  - Modular simulation controller
  - Metric interfaces
  - Regression-based continuation value estimation
  - Request-response model evaluation interface
- Regression and exposure calculation built on unified and extensible **timeline architecture**

## License

This codebase is available for **personal, non-commercial use only**. Please refer to the [LICENSE](LICENSE) file for full details.
