import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import RegularGridInterpolator

# --- 1. CONFIGURATION ---
# Removed 'mpl.use("pgf")' so the plot window appears interactively.

# Configure matplotlib to use LaTeX for all text and set font sizes
mpl.rcParams.update({
    "text.usetex": True,            # Use LaTeX for rendering text
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"], # Standard LaTeX font
    "font.size": 20,                # Base font size
    "axes.labelsize": 30,           # Axis label font size
    "axes.titlesize": 24,           
    "xtick.labelsize": 24,          
    "ytick.labelsize": 24,     
    "legend.fontsize": 20, 
    # "text.latex.preamble": r"\usepackage{bm} \usepackage{amsmath} \boldmath",
    # "axes.labelweight": "bold",    
})

# --- 2. LOGIC (Value Iteration) ---
A = np.array([[0.7, 0.8], [-0.4, 1.4]])
B = np.array([[1], [1]])

x_min, x_max = -4, 4
grid_res = 100
u_min, u_max = -1, 1
gamma = 0.95

x1 = np.linspace(x_min, x_max, grid_res)
x2 = np.linspace(x_min, x_max, grid_res)
X1, X2 = np.meshgrid(x1, x2)
states = np.column_stack([X1.ravel(), X2.ravel()])

Q = np.eye(2)
R = 0.1

V = np.zeros((grid_res, grid_res))
policy = np.zeros((grid_res, grid_res))
u_options = np.linspace(u_min, u_max, 21)

print("Running Value Iteration...")
for i in range(50):
    interp = RegularGridInterpolator((x1, x2), V, bounds_error=False, fill_value=None)
    costs = []
    
    for u in u_options:
        next_states = (A @ states.T).T + (B * u).T
        
        immediate_cost = np.sum((states @ Q) * states, axis=1) + R * u**2
        future_val = interp(next_states)
        
        # Penalty for going out of bounds
        mask_out = (next_states[:,0] < x_min) | (next_states[:,0] > x_max) | \
                   (next_states[:,1] < x_min) | (next_states[:,1] > x_max)
        future_val[mask_out] += 1000
        
        costs.append(immediate_cost + gamma * future_val)
    
    costs = np.array(costs)
    best_idx = np.argmin(costs, axis=0)
    V = np.min(costs, axis=0).reshape(grid_res, grid_res)
    policy = u_options[best_idx].reshape(grid_res, grid_res)

print("Calculation complete. Generating plot...")

# --- 3. PLOTTING ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot surface
surf = ax.plot_surface(
    -X1, -X2, policy, 
    cmap=cm.coolwarm, 
    alpha=0.9, 
    rstride=2,
    cstride=2,
    linewidth=0,
    antialiased=True
)

# --- AXIS LABELS & ROTATION ---
# Disable auto-rotation so labels stay "upright" (horizontal to screen)
ax.xaxis.set_rotate_label(False)
ax.yaxis.set_rotate_label(False)
ax.zaxis.set_rotate_label(False)

# Set labels with explicit rotation=0
ax.set_xlabel(r'$x_1$', labelpad=20, rotation=0)
ax.set_ylabel(r'$x_2$', labelpad=20, rotation=0)
ax.set_zlabel(r'$u$', labelpad=15, rotation=0)

# Commented out title as requested
# ax.set_title(r'\textbf{Optimal Controller Policy} $\pi^*(x)$')

# --- TICKS ---
# Match X/Y ticks to your reference image
ax.set_xticks([-4, -2, 0, 2, 4])
ax.set_yticks([-4, -2, 0, 2, 4])

# Revert Z ticks to automatic (removed manual set_zticks)
ax.tick_params(axis='both', which='major', pad=10)

# Set view angle
ax.view_init(elev=30, azim=-55)

plt.tight_layout()
plt.show()