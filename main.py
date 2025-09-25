import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from banana_distribution import BananaDistribution
from mh_sampler import metropolis_hastings
from hmc_sampler import hamiltonian_monte_carlo

N_SAMPLES = 1000
BURN_IN = 200
MH_STEP_SIZE = 1.0
HMC_STEP_SIZE = 0.2
HMC_PATH_LEN = 1.5

banana_dist = BananaDistribution(b=0.5, a=2.0)

print("Running Metropolis-Hastings Sampler...")
mh_samples, mh_acceptance = metropolis_hastings(banana_dist, N_SAMPLES, BURN_IN, MH_STEP_SIZE)
print(f"MH Final Acceptance Rate: {mh_acceptance[-1]:.2f}")

print("\nRunning Hamiltonian Monte Carlo Sampler...")
hmc_samples, hmc_acceptance = hamiltonian_monte_carlo(banana_dist, N_SAMPLES, BURN_IN, HMC_PATH_LEN, HMC_STEP_SIZE)
print(f"HMC Final Acceptance Rate: {hmc_acceptance[-1]:.2f}")

x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 10, 200)
X, Y = np.meshgrid(x, y)
Z = np.exp(banana_dist.log_prob(np.stack([X, Y], axis=-1)))

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
fig.suptitle('MCMC Sampler Comparison on Banana Distribution', fontsize=16)

def setup_plot(ax, title):
    ax.contour(X, Y, Z, levels=10, cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    line, = ax.plot([], [], 'r-', alpha=0.5, lw=1)
    points, = ax.plot([], [], 'bo', alpha=0.3, markersize=3)
    text = ax.text(0.05, 0.95, '', transform=ax.transAxes, va='top', fontsize=10)
    return line, points, text

line_mh, points_mh, text_mh = setup_plot(axes[0], 'Metropolis-Hastings (MH)')
line_hmc, points_hmc, text_hmc = setup_plot(axes[1], 'Hamiltonian Monte Carlo (HMC)')

plt.xlim(-5, 5)
plt.ylim(-5, 10)

def update(frame):
    line_mh.set_data(mh_samples[:frame+1, 0], mh_samples[:frame+1, 1])
    points_mh.set_data(mh_samples[:frame+1, 0], mh_samples[:frame+1, 1])
    text_mh.set_text(f'Sample: {frame+1}/{N_SAMPLES}\nAcceptance: {mh_acceptance[frame]:.2f}')
    
    line_hmc.set_data(hmc_samples[:frame+1, 0], hmc_samples[:frame+1, 1])
    points_hmc.set_data(hmc_samples[:frame+1, 0], hmc_samples[:frame+1, 1])
    text_hmc.set_text(f'Sample: {frame+1}/{N_SAMPLES}\nAcceptance: {hmc_acceptance[frame]:.2f}')
    
    return line_mh, points_mh, text_mh, line_hmc, points_hmc, text_hmc

frames = tqdm(range(N_SAMPLES), desc="Creating animation")
ani = FuncAnimation(fig, update, frames=frames, blit=True, interval=20)

print("\nSaving animation to mcmc_comparison.gif...")
ani.save('mcmc_comparison.gif', writer='imagemagick', fps=30)
print("Done!")

plt.show()
