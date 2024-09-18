# Monte Carlo Portfolio Simulation

This repository contains a Python implementation of a Monte Carlo simulation to forecast portfolio returns using historical stock data. The project allows users to simulate multiple market scenarios and evaluate the potential future values of a portfolio.

## Features

- **Historical Stock Data**: Automatically pulls data from `Yahoo Finance` using the `yfinance` library.
- **Covariance Matrix & Cholesky Decomposition**: Simulates correlated stock movements based on historical data.
- **Portfolio Simulation**: Simulates multiple scenarios over a specified time period with customizable parameters (number of simulations, projected days, starting amount).
- **Performance Evaluation**: Provides probabilistic insights on the likelihood of returns based on simulation results.
- **Visualization**: Plots simulated portfolio returns over time for easy visualization of scenarios.

