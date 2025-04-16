# distribution_visualizer.py
import math
import random
import numpy as np
import matplotlib.pyplot as plt

class DistributionVisualizer:
    """
    A visualization utility for demonstrating concepts in statistical distributions:
    - Scatter plot with noise and curved regression
    - Bell-shaped curve for a normal PDF
    - Bimodal histogram
    - S-curve approximation of a custom CDF
    """

    @staticmethod
    def generate_noisy_parabola(seed=1, count=100):
        random.seed(seed)
        x = [i for i in range(count)]
        y = [(0.05 * (xi - 50) ** 2 + random.uniform(-20, 20)) for xi in x]
        return x, y

    @staticmethod
    def standard_normal_pdf(x):
        return math.exp(-x**2 / 2) / math.sqrt(2 * math.pi)

    @staticmethod
    def custom_cdf(x):
        return 1 / (1 + math.exp(-2 * x))  # Sigmoid for variety

    @staticmethod
    def visualize():
        fig, ax = plt.subplots(2, 2)

        # Subplot 1: Noisy parabolic curve
        x1, y1 = DistributionVisualizer.generate_noisy_parabola()
        ax[0, 0].scatter(x1, y1, c='orange', s=20, alpha=0.7)
        ax[0, 0].set_title("Noisy Parabolic Curve")

        # Subplot 2: Normal PDF (bell shape)
        x2 = np.linspace(-4, 4, 100)
        y2 = [DistributionVisualizer.standard_normal_pdf(val) for val in x2]
        ax[0, 1].plot(x2, y2, color='green')
        ax[0, 1].set_title("Normal PDF")

        # Subplot 3: Bimodal histogram
        np.random.seed(1)
        data = np.concatenate([
            np.random.normal(loc=30, scale=5, size=5000),
            np.random.normal(loc=70, scale=5, size=5000)
        ])
        ax[1, 0].hist(data, bins=50, color='purple', alpha=0.7, density=True)
        ax[1, 0].set_title("Bimodal Distribution")

        # Subplot 4: Custom sigmoid CDF
        x4 = np.linspace(-6, 6, 100)
        y4 = [DistributionVisualizer.custom_cdf(val) for val in x4]
        ax[1, 1].plot(x4, y4, color='blue')
        ax[1, 1].set_title("Sigmoid CDF")

        fig.suptitle("Updated Distribution Visualizations")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    DistributionVisualizer.visualize()