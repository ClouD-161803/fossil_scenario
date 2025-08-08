
# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import timeit

from fossil import plotting
from fossil.scenapp import ScenApp, Result
from fossil import domains
from fossil import certificate
from fossil import main
from fossil import analysis
from experiments.benchmarks import models
from fossil.consts import *
from multiprocessing import Pool
import random
import numpy as np
import torch
import scipy
import copy

# Create a modified Barr1 model with correct functions for integration
class Barr1Modified(models.Barr1):
    n_vars = 2
    time_horizon = 50
    num_time_samples = 100
    
    def f_torch_wrapper(self, t, x):
        """Wrapper for scipy.integrate.solve_ivp compatibility"""
        if isinstance(x, np.ndarray):
            x_tensor = torch.tensor(x, dtype=torch.float32)
            result = self.f_torch(t, x_tensor)
            return np.array(result)
        return self.f_torch(t, x)
    
    def generate_trajs(self, x_0, time_horizon=None):
        """Override to use our wrapper function"""
        if time_horizon is None:
            time_horizon = self.time_horizon
            
        trajs = []
        for elem in x_0:
            trajs.append(scipy.integrate.solve_ivp(
                self.f_torch_wrapper, 
                (0, time_horizon), 
                elem.numpy(),
                t_eval=np.linspace(0., time_horizon, self.num_time_samples)
            ))
            
        state_trajs = [traj["y"] for traj in trajs]
        times = [traj["t"] for traj in trajs]
        nexts = [traj[:, 1:] for traj in state_trajs]
        state_trajs = [traj[:, :-1] for traj in state_trajs]
        times = [time[1:] - time[:-1] for time in times]
        
        return times, state_trajs, nexts

# Set seed for reproducibility
claudio_seed = 42
random.seed(claudio_seed)
np.random.seed(claudio_seed)
torch.manual_seed(claudio_seed)


def solve(opts):
    PAC = ScenApp(opts)
    result = PAC.solve()
    return result


class UnsafeDomain(domains.Set):
    dimension = 2

    def generate_domain(self, x):
        x1, x2 = x
        return x1 + x2**2 <= 0

    def generate_data(self, batch_size):
        points = []
        limits = [[-2, -2], [0, 2]]
        while len(points) < batch_size:
            dom = domains.square_init_data(limits, batch_size)
            idx = torch.nonzero(dom[:, 0] + dom[:, 1] ** 2 <= 0)
            points += dom[idx][:, 0, :]
        return torch.stack(points[:batch_size])
    
    def check_containment(self, x):
        return x[:,0] + x[:,1]**2 <= 0

    def plot(self, fig, ax, label=None):
        if self.dimension != 2:
            raise NotImplementedError("Plotting only supported for 2D sets")
        colour, label = "red", "Unsafe" 
        
        y = np.linspace(-2,2,50)
        x = -y**2
        ax.plot(x,y, colour, linewidth=2, label=label)
        return fig, ax

def test_lnn(args):
    # Define domains for BarrierAlt certificate
    XD = domains.Rectangle(tuple([-2, -2]), tuple([2, 2]))
    XI = domains.Rectangle(tuple([0.25, -1]), tuple([1, 1]))
    XU = UnsafeDomain()

    n_trajectory_data = 100
    n_background_data = 1000
    num_runs = 1
    
    # Define sets for BarrierAlt certificate
    sets = {
        certificate.XD: XD,
        certificate.XI: XI,
        certificate.XU: XU,
    }
    
    # Generate state data for all required regions
    state_data = {
        certificate.XD: XD._generate_data(n_background_data)(),
        certificate.XI: XI._generate_data(n_background_data)(),
        certificate.XU: XU._generate_data(n_background_data)(),
    }
    
    init_data = [XI._generate_data(n_trajectory_data)() for i in range(num_runs)]

    # Use our modified Barr1 model
    system = Barr1Modified
    
    # Generate trajectories with our custom model
    all_data = []
    trajectory_indices = []
    
    for init_datum in init_data:
        model_instance = system()
        traj_data = model_instance.generate_trajs(init_datum)
        all_data.append(traj_data)
        
        # Create indices for trajectories
        traj_length = traj_data[1][0].shape[1]
        traj_indices = []
        current_idx = 0
        for i in range(len(traj_data[1])):
            indices = list(range(current_idx, current_idx + traj_length))
            traj_indices.append(indices)
            current_idx += traj_length
        trajectory_indices.append({"lie": traj_indices})

    data = [{"states_only": state_data,
             "full_data":
             {"times": all_datum[0],
              "states": all_datum[1],
              "derivs": all_datum[2]},
             "indices": trajectory_indices[i]
            } for i, all_datum in enumerate(all_data)]
    
    # Define NN parameters
    activations = [ActivationType.SIGMOID]
    hidden_neurons = [5] * len(activations)
    
    opts = [ScenAppConfig(
        N_VARS=2,
        SYSTEM=system,
        DOMAINS=sets,
        DATA=datum,
        N_DATA=n_trajectory_data,
        N_TEST_DATA=n_background_data,
        CERTIFICATE=CertificateType.BARRIERALT,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        ACTIVATION=tuple(activations),
        N_HIDDEN_NEURONS=(hidden_neurons[0],),
        SYMMETRIC_BELT=True,
        VERBOSE=0,
        SCENAPP_MAX_ITERS=2000,
        VERIFIER=VerifierType.SCENAPPNONCONVEX,
        SEED=claudio_seed,
        BETA=(0.1,),
        # MAX_JUMPS=1,
        # USE_APRIORI_JUMPS=False
    ) for datum in data]
    
    with Pool(processes=num_runs) as pool:
        res = pool.map(solve, opts)
    
    if args.plot:
        axes = plotting.benchmark(
            system(), res[-1].cert,
            domains=opts[-1].DOMAINS,
            xrange=[-2, 2], yrange=[-2, 2]
        )
        for ax, name in axes:
            plotting.save_plot_with_tags(ax, opts[-1], name)

    if args.record:
        for i, result in enumerate(res):
            rec = analysis.Recorder()
            rec.record(opts[i], result, 0)


if __name__ == "__main__":
    args = main.parse_benchmark_args()
    test_lnn(args)
