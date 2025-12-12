import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import RegularGridInterpolator

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

for i in range(50):
    interp = RegularGridInterpolator((x1, x2), V, bounds_error=False, fill_value=None)
    costs = []
    
    for u in u_options:
        next_states = (A @ states.T).T + (B * u).T
        
        immediate_cost = np.sum((states @ Q) * states, axis=1) + R * u**2
        future_val = interp(next_states)
        
        mask_out = (next_states[:,0] < x_min) | (next_states[:,0] > x_max) | \
                   (next_states[:,1] < x_min) | (next_states[:,1] > x_max)
        future_val[mask_out] += 1000
        
        costs.append(immediate_cost + gamma * future_val)
    
    costs = np.array(costs)
    best_idx = np.argmin(costs, axis=0)
    V = np.min(costs, axis=0).reshape(grid_res, grid_res)
    policy = u_options[best_idx].reshape(grid_res, grid_res)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(
    -X1, -X2, policy, 
    cmap=cm.coolwarm, 
    alpha=0.7, 
    rstride=5, 
    cstride=5
)

# ax.contour(
#     -X1, -X2, policy, 
#     levels=[0], 
#     colors="k", 
#     linestyles="dashed", 
#     linewidths=2.5
# )

# ax.set_xlabel('$x_1$', fontsize=14)
# ax.set_ylabel('$x_2$', fontsize=14)
# ax.set_zlabel('$u$', fontsize=14)
ax.set_title("Optimal Controller", fontsize=26)
ax.tick_params(axis='both', labelsize=20)

# ax.invert_xaxis()
# ax.invert_yaxis()

ax.view_init(elev=30, azim=-55)

plt.show()