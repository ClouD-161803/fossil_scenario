from fossil import plotting
from fossil.scenapp import ScenApp, Result
from fossil import domains
from fossil import certificate
from fossil import main
from experiments.scenapp_tests.benchmarks import models
from fossil.consts import (
    ActivationType,
    ScenAppConfig,
    CertificateType,
    TimeDomain,
    VerifierType,
)
from multiprocessing import Pool
import random
import numpy as np
import torch

claudio_seed = 42
random.seed(claudio_seed)
np.random.seed(claudio_seed)
torch.manual_seed(claudio_seed)


def solve(opts):
    PAC = ScenApp(opts)
    result = PAC.solve()
    return result


def test_lnn(args):
    XD = domains.Rectangle((0.1, 0.1), (0.5, 1))
    XI = domains.Rectangle((0.1, 0.1), (0.4, 0.55))
    XU = domains.Rectangle((0.45, 0.6), (0.5, 1))

    n_trajectory_data = 10
    n_background_data = 1000
    num_runs = 1

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
    init_data = [XI._generate_data(
        n_trajectory_data)() for i in range(num_runs)]

    system = models.DC_Motor
    all_data = [system().generate_trajs(
        init_datum) for init_datum in init_data]

    data = [{"states_only": state_data,
             "full_data":
             {"times": all_datum[0],
              "states": all_datum[1],
              "derivs": all_datum[2]}} for all_datum in all_data]
    
    activations = [ActivationType.SIGMOID]
    #activations = [ActivationType.RELU]
    hidden_neurons = [5] * len(activations)
    opts = [ScenAppConfig(
        N_VARS=2,
        SYSTEM=system,
        DOMAINS=sets,
        DATA=datum,
        N_DATA=n_trajectory_data,
        BETA=(0.01,),
        CERTIFICATE=CertificateType.BARRIERALT,
        TIME_DOMAIN=TimeDomain.DISCRETE,
        ACTIVATION=tuple(activations),
        N_HIDDEN_NEURONS=(hidden_neurons[0],),
        SYMMETRIC_BELT=True,
        VERBOSE=0,
        SCENAPP_MAX_ITERS=2500,
        VERIFIER=VerifierType.SCENAPPNONCONVEX,
        SEED=claudio_seed,
    ) for datum in data]
    with Pool(processes=num_runs) as pool:
        res = pool.map(solve, opts)
    
    if args.plot:
        axes = plotting.benchmark(
            system(), res[-1].cert,
            domains=opts[-1].DOMAINS,
            xrange=[0.1, 1], yrange=[0.1, 1]
        )
        for ax, name in axes:
            plotting.save_plot_with_tags(ax, opts[-1], name)
        
    for cfg in opts:
        main.run_benchmark(
            cfg,
            record=args.record,
            plot=args.plot,
            concurrent=args.concurrent,
            repeat=args.repeat,
        )


if __name__ == "__main__":
    args = main.parse_benchmark_args()
    test_lnn(args)
