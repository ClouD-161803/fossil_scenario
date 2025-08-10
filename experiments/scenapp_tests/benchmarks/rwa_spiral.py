# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from experiments.scenapp_tests.benchmarks import models
from fossil import domains
from fossil import plotting
from fossil import certificate
from fossil import main, control
from fossil import analysis
from fossil.consts import (
    ActivationType,
    ScenAppConfig,
    CertificateType,
    TimeDomain,
    VerifierType,
)
import random
import numpy as np
import torch
from multiprocessing import Pool
from fossil.scenapp import ScenApp, Result

# Set seed for reproducibility
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
    n_vars = 2
    
    system = models.Spiral
    system.time_horizon = 100

    XD = domains.Rectangle(tuple([-5, -5]), tuple([5, 5]))
    XI = domains.Rectangle(tuple([-1, 4]), tuple([1, 4.5]))
    SU = domains.Rectangle(tuple([-5,-1]), tuple([-4.5,1]))
    XG = domains.Sphere(tuple([0,0]),1.0)

    XS = domains.SetMinus(XD, SU)
    SD = domains.SetMinus(XS, XG)

    n_trajectory_data = 300
    n_background_data = 1000
    max_iters = 50
    use_apriori_jumps = False
    max_jumps = 4
    num_runs = 1
    
    sets = {
        certificate.XD: XD,
        certificate.XI: XI,
        certificate.XS_BORDER: XS,
        certificate.XS: XS,
        certificate.XG: XG,
        certificate.XG_BORDER: XG,
    }

    state_data = {
        certificate.XD: SD._generate_data(n_background_data)(),
        certificate.XI: XI._generate_data(n_background_data)(),
        certificate.XS_BORDER: XS._sample_border(n_background_data)(),
        certificate.XG: XG._generate_data(n_background_data)(),
        certificate.XG_BORDER: XG._sample_border(n_background_data)()
    }
    
    init_data = [XI._generate_data(n_trajectory_data)() for i in range(num_runs)]
    all_data = [system().generate_trajs(init_datum) for init_datum in init_data]
    data = [{"states_only": state_data, 
             "full_data": {"times":all_datum[0],
                          "states":all_datum[1],
                          "derivs":all_datum[2]}} 
            for all_datum in all_data]
    
    activations = [ActivationType.SIGMOID, ActivationType.SIGMOID]
    hidden_neurons = [5] * len(activations)

    opts = [ScenAppConfig(
        N_VARS=2,
        SYSTEM=system,
        DOMAINS=sets,
        DATA=datum,
        N_DATA=n_trajectory_data,
        N_TEST_DATA=n_trajectory_data,
        CERTIFICATE=CertificateType.RWS,
        TIME_DOMAIN=TimeDomain.DISCRETE,
        ACTIVATION=tuple(activations),
        N_HIDDEN_NEURONS=(hidden_neurons[0],),
        SYMMETRIC_BELT=True,
        VERBOSE=0,
        SCENAPP_MAX_ITERS=max_iters,
        VERIFIER=VerifierType.SCENAPPNONCONVEX,
        SEED=claudio_seed,
        MAX_JUMPS=max_jumps,
        USE_APRIORI_JUMPS=use_apriori_jumps,
    ) for datum in data]
    
    with Pool(processes=num_runs) as pool:
        res = pool.map(solve, opts)

    if args.plot:
        custom_levels = [-0.1, 0, 0.1]
        axes = plotting.benchmark(
            system(), res[-1].cert, 
            domains=sets,
            xrange=[-5, 5], yrange=[-5, 5],
            levels=[custom_levels]
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
