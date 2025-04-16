# distribution_visualizer.py
import math
import random
import numpy as np
import matplotlib.pyplot as plt

class DistributionVisualizer:
    """
    A visualization utility for demonstrating concepts in statistical distributions:
    - Scatter plot with best-fit line
    - Standard normal PDF
    - Positively skewed histogram
    - CDF of the standard normal distribution
    """

    @staticmethod
    def generate_uniform_averages(seed=1, data_len=10000, samples=50, low=0, high=100):
        random.seed(seed)
        return [sum(random.uniform(low, high) for _ in range(samples)) / samples for _ in range(data_len)]

    @staticmethod
    def best_fit_line(x, y):
        x_bar = sum(x) / len(x)
        y_bar = sum(y) / len(y)
        m = sum((x[i] - x_bar) * (y[i] - y_bar) for i in range(len(x))) / sum((x[i] - x_bar)**2 for i in range(len(x)))
        b = y_bar - m * x_bar
        return m, b

    @staticmethod
    def standard_normal_pdf(x):
        return math.exp(-x**2 / 2) / math.sqrt(2 * math.pi)

    @staticmethod
    def standard_normal_cdf(x):
        # Approximation to the CDF of the standard normal distribution
        a1, a2, a3, a4, a5, p = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429, 0.3275911
        sign = 1 if x >= 0 else -1
        x = abs(x) / math.sqrt(2.0)
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * math.exp(-x * x))
        return 0.5 * (1.0 + sign * y)

    @staticmethod
    def visualize():
        fig, ax = plt.subplots(2, 2)

        # Subplot 1: Scatter plot with best-fit line
        x1 = DistributionVisualizer.generate_uniform_averages(seed=1, data_len=100, samples=1, low=0, high=100)
        noise = DistributionVisualizer.generate_uniform_averages(seed=2, data_len=100, samples=1, low=0, high=20)
        y1 = [x / 1.5 + n for x, n in zip(x1, noise)]

        ax[0, 0].scatter(x1, y1, c='red', s=50, edgecolor='blue')
        m, b = DistributionVisualizer.best_fit_line(x1, y1)
        ax[0, 0].plot([min(x1), max(x1)], [m * min(x1) + b, m * max(x1) + b], label='Best Fit Line')
        ax[0, 0].legend()
        ax[0, 0].set_title("Scatter Plot + Best Fit")

        # Subplot 2: Standard normal PDF
        x2 = np.arange(-3, 3, 0.1)
        y2 = [DistributionVisualizer.standard_normal_pdf(val) for val in x2]
        ax[0, 1].plot(x2, y2)
        ax[0, 1].set_title("Standard Normal PDF")

        # Subplot 3: Positively skewed histogram
        np.random.seed(0)
        skewed_data = np.random.gamma(shape=2, scale=250, size=10000)
        ax[1, 0].hist(skewed_data, bins=50, density=True, color='blue')
        ax[1, 0].set_title("Skewed Distribution")

        # Subplot 4: Cumulative distribution
        x4 = np.arange(-3, 3, 0.1)
        y4 = [DistributionVisualizer.standard_normal_cdf(val) for val in x4]
        ax[1, 1].plot(x4, y4)
        ax[1, 1].set_title("Standard Normal CDF")

        fig.suptitle("Distribution Visualizations")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    DistributionVisualizer.visualize()
