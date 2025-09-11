# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import timeit
import random
import numpy as np
import torch

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
from functools import partial
from multiprocessing import Pool

import time
# claudio_seed = int(time.time()) % 100000000
claudio_seed = 54308950
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
    XU = domains.Rectangle(tuple([-5,-1]), tuple([-4.5,1]))

    n_trajectory_data = 1000
    n_background_data = 5000
    max_iters = 20
    use_apriori_jumps = True
    max_jumps = 4
    
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
    activations = [ActivationType.SIGMOID, ActivationType.SIGMOID]
    #activations = [ActivationType.RELU]
    hidden_neurons = [5] * len(activations)
    
    system = models.Spiral
    system.time_horizon = 100
    
    num_runs = 1

    init_data = [XI._generate_data(n_trajectory_data)() for j in range(num_runs)]
    
    all_data = [system().generate_trajs(init_datum) for init_datum in init_data]
    data = [{"states_only": state_data, 
             "full_data": {"times":all_datum[0],
                          "states":all_datum[1],
                          "derivs":all_datum[2]}} 
            for all_datum in all_data]
    
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
        #VERIFIER=VerifierType.DREAL,
        ACTIVATION=tuple(activations),
        N_HIDDEN_NEURONS=(hidden_neurons[0],),
        SYMMETRIC_BELT=True,
        VERBOSE=0,
        SCENAPP_MAX_ITERS=max_iters,
        VERIFIER=VerifierType.SCENAPPNONCONVEX,
        SEED=claudio_seed,
        # MAX_JUMPS=max_jumps,
        # USE_APRIORI_JUMPS=use_apriori_jumps,
        #CONVEX_NET=True,
    ) for datum in data]

    with Pool(processes=num_runs) as pool:
        res = pool.map(solve, opts)

    if args.plot:
        axes = plotting.benchmark(
            system(), res[-1].cert, domains=opts[-1].DOMAINS, xrange=[-5, 5], yrange=[-5, 5]
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
