# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import timeit
import random
import numpy as np
import torch
import logging
import argparse

# Configure logging at the root level to reduce debug messages
logging.basicConfig(level=logging.WARNING)

from fossil.scenapp import ScenApp, Result
from fossil import plotting
from fossil import domains
from fossil import certificate
from fossil import main
from fossil import analysis
from experiments.scenapp_tests.benchmarks import models
from fossil.consts import (
    ActivationType,
    ScenAppConfig,
    CertificateType,
    TimeDomain,
    VerifierType,
)
from multiprocessing import Pool

import time
# claudio_seed = int(time.time()) % 100000000
claudio_seed = 42
print(f"Using seed: {claudio_seed}")
random.seed(claudio_seed)
np.random.seed(claudio_seed)
torch.manual_seed(claudio_seed)


def solve(opts):

    PAC = ScenApp(opts)
    result = PAC.solve()
    return result

def test_lnn(args):
    XD = domains.Rectangle(tuple([-5, -5]), tuple([5, 5]))
    XI = domains.Rectangle(tuple([-1, 4]), tuple([1, 4.5]))
    XU = domains.Rectangle(tuple([-5,-2]), tuple([-4,2]))

    n_trajectory_data = 1000
    n_background_data = 1000
    max_iters = 100
    use_apriori_jumps = True
    max_jumps = 3
    
    sets = {
        certificate.XD: XD,
        certificate.XI: XI,
        certificate.XU: XU,
    }
    state_data = {
        certificate.XD: XD._generate_data(n_background_data)(),
        certificate.XI: XI._generate_data(n_background_data)(),
        certificate.XU: XU._generate_data(n_background_data)(),
    }
    
    num_runs = 1
    init_data = [XI._generate_data(n_trajectory_data)() for i in range(num_runs)]
    
    system = models.Spiral
    system.time_horizon = 50
    all_data = [system().generate_trajs(init_datum) for init_datum in init_data]
    data = [{"states_only": state_data, 
             "full_data": {"times": all_datum[0], 
                          "states": all_datum[1],
                          "derivs": all_datum[2]}} 
            for all_datum in all_data]
    
    activations = [ActivationType.SIGMOID, ActivationType.SIGMOID]
    # activations = [ActivationType.RELU]
    hidden_neurons = [5] * len(activations)
    opts = [ScenAppConfig(
        N_VARS=2,
        SYSTEM=system,
        DOMAINS=sets,
        DATA=datum,
        N_DATA=n_trajectory_data,
        N_TEST_DATA=n_trajectory_data,
        BETA=(0.01,),
        CERTIFICATE=CertificateType.BARRIERALT,
        TIME_DOMAIN=TimeDomain.DISCRETE,
        # VERIFIER=VerifierType.DREAL,
        ACTIVATION=tuple(activations),
        N_HIDDEN_NEURONS=(hidden_neurons[0],),
        SYMMETRIC_BELT=True,
        VERBOSE=0,
        SCENAPP_MAX_ITERS=max_iters,
        VERIFIER=VerifierType.SCENAPPNONCONVEX,
        SEED=claudio_seed,
        MAX_JUMPS=max_jumps,
        USE_APRIORI_JUMPS=use_apriori_jumps,
        # CONVEX_NET=True,
    ) for datum in data]
    
    # Setting VERBOSE=0 for all options to reduce debug output
    for opt in opts:
        opt.VERBOSE = 0  # Set all instances to non-verbose mode

    with Pool(processes=num_runs) as pool:
        res = pool.map(solve, opts)
    # res = [solve(opt) for opt in opts]

    opts = ScenAppConfig(
        N_VARS=2,
        SYSTEM=system,
        DOMAINS=sets,
        DATA=data[-1],
        N_DATA=n_trajectory_data,
        N_TEST_DATA=n_trajectory_data,
        BETA=(0.01,),
        CERTIFICATE=CertificateType.BARRIERALT,
        TIME_DOMAIN=TimeDomain.DISCRETE,
        # VERIFIER=VerifierType.DREAL,
        ACTIVATION=tuple(activations),
        N_HIDDEN_NEURONS=(hidden_neurons[0],),
        SYMMETRIC_BELT=True,
        VERBOSE=0,
        SCENAPP_MAX_ITERS=100,  # Reduced from 2500 to 100 to limit debug output
        VERIFIER=VerifierType.SCENAPPNONCONVEX,
        SEED=claudio_seed,
        MAX_JUMPS=max_jumps,
        USE_APRIORI_JUMPS=use_apriori_jumps,
        # CONVEX_NET=True,
    )
    
    # Force specific contour levels to avoid "Contour levels must be increasing" error
    custom_levels = [-0.1, 0, 0.1]  # Ensure increasing levels
    axes = plotting.benchmark(
        system(), res[-1].cert, 
        domains=opts.DOMAINS, 
        xrange=[-5, 5], yrange=[-5, 5],
        levels=[custom_levels]  # Pass custom levels
    )
    for ax, name in axes:
        plotting.save_plot_with_tags(ax, opts, name + "_unsafe")


if __name__ == "__main__":
    # Customize argument parsing to add verbosity control
    parser = argparse.ArgumentParser(description='Barrier certificate for Spiral unsafe example')
    parser.add_argument('--plot', action='store_true', help='Whether to plot the results')
    parser.add_argument('--record', action='store_true', help='Whether to record results')
    parser.add_argument('--verbose', type=int, default=0, choices=[0, 1, 2], 
                        help='Verbosity level: 0=WARNING, 1=INFO, 2=DEBUG')
    
    args = parser.parse_args()
    
    # Set logging level based on verbosity argument
    if args.verbose == 0:
        logging.getLogger().setLevel(logging.WARNING)
    elif args.verbose == 1:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.DEBUG)
    
    test_lnn(args)
