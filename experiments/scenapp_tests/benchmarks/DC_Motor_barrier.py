from fossil import plotting
from fossil.scenapp import ScenApp, Result
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
import random
import numpy as np
import torch

# Set seed for reproducibility
claudio_seed = 42
random.seed(claudio_seed)
np.random.seed(claudio_seed)
torch.manual_seed(claudio_seed)


def solve(opts):
    PAC = ScenApp(opts)
    result = PAC.solve()
    return result


def test_lnn(args):
    # Define domains for BarrierAlt certificate
    XD = domains.Rectangle(tuple([0.1, 0.1]), tuple([0.5, 1]))
    XI = domains.Rectangle(tuple([0.1, 0.1]), tuple([0.4, 0.55]))
    
    # Define the unsafe region
    XU = domains.Rectangle(tuple([0.45, 0.6]), tuple([0.5, 1]))

    n_trajectory_data = 10
    n_background_data = 50
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

    system = models.DC_Motor
    system.time_horizon = 100  # Set time horizon
    all_data = [system().generate_trajs(init_datum) for init_datum in init_data]

    data = [{"states_only": state_data,
             "full_data":
             {"times": all_datum[0],
              "states": all_datum[1],
              "derivs": all_datum[2]}} for all_datum in all_data]
    
    # Define NN parameters
    activations = [ActivationType.SIGMOID, ActivationType.SIGMOID]
    hidden_neurons = [5] * len(activations)
    
    opts = [ScenAppConfig(
        N_VARS=2,
        SYSTEM=system,
        DOMAINS=sets,
        DATA=datum,
        N_DATA=n_trajectory_data,
        N_TEST_DATA=n_background_data,
        CERTIFICATE=CertificateType.BARRIERALT,  # Use BarrierAlt certificate
        TIME_DOMAIN=TimeDomain.DISCRETE,
        ACTIVATION=tuple(activations),
        N_HIDDEN_NEURONS=(hidden_neurons[0],),
        SYMMETRIC_BELT=True,
        VERBOSE=0,
        SCENAPP_MAX_ITERS=2500,
        VERIFIER=VerifierType.SCENAPPNONCONVEX,
        SEED=claudio_seed,
        MAX_JUMPS=5,
        USE_APRIORI_JUMPS=True,
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

    if args.record:
        for i, result in enumerate(res):
            rec = analysis.Recorder()
            rec.record(opts[i], result, 0)


if __name__ == "__main__":
    args = main.parse_benchmark_args()
    test_lnn(args)
