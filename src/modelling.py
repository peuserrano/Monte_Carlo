import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf

class MonteCarlo:

    def __init__(self, stocks_list, simulations, projected_days, starting_amount):
        """
        Initializes the MonteCarlo simulation class.
        Args:
            stocks_list (list): List of stock tickers.
            simulations (int): Number of simulation scenarios.
            projected_days (int): Number of days to project.
            starting_amount (float): Starting amount of the portfolio.
        """
        self.stocks_list = [stock + '.SA' for stock in stocks_list]
        self.simulations = simulations
        self.projected_days = projected_days
        self.starting_amount = starting_amount
        self.stock_returns = None
        self.final_amount = None
        self.portfolio_returns = None

    def pulling_stock_data(self):
        """
        Pulls historical stock data and calculates daily returns.
        Returns:
            pandas.DataFrame: Stock daily returns.
        """
        current_date = dt.datetime.now()
        starting_date = current_date - dt.timedelta(days=252)

        stock_prices = yf.download(self.stocks_list, start=starting_date, end=current_date)['Adj Close']
        self.stock_returns = stock_prices.pct_change().dropna()

        return self.stock_returns

    def covariance_matrix(self):
        """
        Calculates the covariance matrix for the stock returns.
        Returns:
            numpy.ndarray: Covariance matrix.
        """
        if self.stock_returns is None:
            raise ValueError("Stock returns not available. Call pulling_stock_data() first.")
        return self.stock_returns.cov()

    def get_portfolio_weights(self):
        """
        Returns the portfolio weights (equally weighted in this case).
        Returns:
            numpy.ndarray: Array of portfolio weights.
        """
        return np.full(len(self.stocks_list), 1/len(self.stocks_list))

    def returns_matrix(self):
        """
        Generates a matrix of expected returns for each stock.
        Returns:
            numpy.ndarray: Returns matrix for all projected days.
        """
        stock_mean_returns = self.stock_returns.mean(axis=0).to_numpy()
        return stock_mean_returns * np.ones(shape=(self.projected_days, len(self.stocks_list)))

    def l_matrix(self):
        """
        Calculates the Cholesky decomposition of the covariance matrix.
        Returns:
            numpy.ndarray: Cholesky decomposition matrix.
        """
        return LA.cholesky(self.covariance_matrix())

    def simulating_scenarios(self):
        """
        Runs the Monte Carlo simulation over multiple scenarios.
        Returns:
            numpy.ndarray: Simulated portfolio returns for all scenarios.
        """
        portfolio_returns = np.zeros([self.projected_days, self.simulations])
        final_amount = np.zeros(self.simulations)

        for s in range(self.simulations):
            rpdf = np.random.normal(size=(self.projected_days, len(self.stocks_list)))
            synthetic_returns = self.returns_matrix() + np.inner(rpdf, self.l_matrix())
            portfolio_returns[:, s] = np.cumprod(np.inner(self.get_portfolio_weights(), synthetic_returns) + 1) * self.starting_amount
            final_amount[s] = portfolio_returns[-1, s]

        self.portfolio_returns = portfolio_returns
        self.final_amount = final_amount

        return self.portfolio_returns

    def plotting_scenarios(self):
        """
        Plots the results of the Monte Carlo simulation.
        """
        if self.portfolio_returns is None:
            raise ValueError("Portfolio returns not available. Call simulating_scenarios() first.")
        plt.plot(self.portfolio_returns, linewidth=1)
        plt.ylabel('Wealth')
        plt.xlabel('Time (Days)')
        plt.title('Monte Carlo Simulation of Portfolio Returns')
        plt.show()

    def performance_evaluation(self):
        """
        Evaluates the performance of the simulated portfolios.
        """
        if self.final_amount is None:
            raise ValueError("Final amounts not available. Call simulating_scenarios() first.")
        
        perc_99_amount = np.percentile(self.final_amount, 1)
        perc_95_amount = np.percentile(self.final_amount, 5)
        median_amount = np.percentile(self.final_amount, 50)
        profitable_scenarios = np.sum(self.final_amount > self.starting_amount) / len(self.final_amount) * 100

        print(f'''
        Invested Amount: {self.starting_amount};
        Selected Portfolio: {self.stocks_list};
        {profitable_scenarios:.2f}% of the simulated portfolios returned a profit.

        With 50% probability, the final amount will be greater than {median_amount:.2f};
        With 95% probability, the final amount will be greater than {perc_95_amount:.2f};
        With 99% probability, the final amount will be greater than {perc_99_amount:.2f}.
        ''')

