#!/usr/bin/env python3
# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys
import time
import random
import numpy as np
import torch

from fossil.scenapp import ScenApp, Result
from fossil import plotting
from fossil import domains
from fossil import certificate
from fossil import main as fossil_main
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


def run_barr_spiral_with_j(args, j_value, summary_path):
    """
    Run barr_spiral benchmark with a specific J value (MAX_JUMPS)
    
    Args:
        args: Command line arguments
        j_value: Value for MAX_JUMPS parameter
        summary_path: Path to the summary file for this run
        
    Returns:
        Result object from running the benchmark
    """
    if args.fixed_seed:
        claudio_seed = args.fixed_seed
    else:
        claudio_seed = int(time.time()) % 100000000
    
    print(f"Using seed: {claudio_seed} for J={j_value}")
    random.seed(claudio_seed)
    np.random.seed(claudio_seed)
    torch.manual_seed(claudio_seed)

    XD = domains.Rectangle(tuple([-5, -5]), tuple([5, 5]))
    XI = domains.Rectangle(tuple([-1, 4]), tuple([1, 4.5]))
    XU = domains.Rectangle(tuple([-5,-1]), tuple([-4.5,1]))
    
    sets = {
        certificate.XD: XD,
        certificate.XI: XI,
        certificate.XU: XU,
    }
    
    state_data = {
        certificate.XD: XD._generate_data(args.n_background_data)(),
        certificate.XI: XI._generate_data(args.n_background_data)(),
        certificate.XU: XU._generate_data(args.n_background_data)(),
    }
    
    activations = [ActivationType.SIGMOID, ActivationType.SIGMOID]
    hidden_neurons = [5] * len(activations)
    
    system = models.Spiral
    system.time_horizon = args.time_horizon
    
    init_data = XI._generate_data(args.n_trajectory_data)()
    
    all_data = system().generate_trajs(init_data)
    data = {"states_only": state_data, 
            "full_data": {"times": all_data[0],
                         "states": all_data[1],
                         "derivs": all_data[2]}}
    
    config = ScenAppConfig(
        N_VARS=2,
        SYSTEM=system,
        DOMAINS=sets,
        DATA=data,
        N_DATA=args.n_trajectory_data,
        N_TEST_DATA=args.n_trajectory_data,
        BETA=(args.beta,),
        CERTIFICATE=CertificateType.BARRIERALT,
        TIME_DOMAIN=TimeDomain.DISCRETE,
        ACTIVATION=tuple(activations),
        N_HIDDEN_NEURONS=(hidden_neurons[0],),
        SYMMETRIC_BELT=True,
        VERBOSE=0,
        SCENAPP_MAX_ITERS=args.scenapp_max_iters,
        VERIFIER=VerifierType.SCENAPPNONCONVEX,
        SEED=claudio_seed,
        MAX_JUMPS=j_value,
        USE_APRIORI_JUMPS=True,
    )
    
    PAC = ScenApp(config)
    result = PAC.solve()
    
    os.makedirs("results_varying_j", exist_ok=True)
    
    if args.plot:
        axes = plotting.benchmark(
            system(), result.cert, domains=config.DOMAINS, xrange=[-5, 5], yrange=[-5, 5]
        )
        for ax, name in axes:
            filename = f"J{j_value}_{name}"
            plotting.save_plot_with_tags(ax, config, filename)
    
    if args.record:
        rec = analysis.Recorder()
        rec.record(config, result, 0)
    
    with open(summary_path, "a") as f:
        f.write(f"J={j_value}, Epsilon={result.res}, A-Post-Epsilon={result.a_post_res}, " + 
                f"Iters={result.stats.iters}\n")
    
    print(f"Finished run with J={j_value}")
    print(f"Epsilon: {result.res}")
    print(f"A-Posteriori Epsilon: {result.a_post_res}")
    print(f"Iterations: {result.stats.iters}")
    print("-" * 50)
    
    return result


def parse_args():
    """
    Parse command line arguments specific to this script.
    
    We're defining our own argument parser rather than using fossil_main.parse_benchmark_args()
    because we need additional parameters specific to our varying J experiments.
    """
    parser = argparse.ArgumentParser(description="Run barr_spiral benchmark with varying J values")
    parser.add_argument("--beta", type=float, default=0.01, help="Beta parameter for scenario approach")
    parser.add_argument("--n_trajectory_data", type=int, default=200, help="Number of trajectory data points")
    parser.add_argument("--n_background_data", type=int, default=500, help="Number of background data points")
    parser.add_argument("--scenapp_max_iters", type=int, default=2500, help="Maximum iterations for ScenApp")
    parser.add_argument("--time_horizon", type=int, default=100, help="Time horizon for the system")
    parser.add_argument("--max_j", type=int, default=10, help="Maximum value of J to start with")
    parser.add_argument("--fixed_seed", type=int, help="Fixed seed for reproducibility (optional)")
    parser.add_argument("--plot", action="store_true", help="Generate plots for each run")
    parser.add_argument("--record", action="store_true", help="Record detailed results for each run")
    return parser.parse_args()


def main():
    args = parse_args()
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    summary_filename = f"summary_{timestamp}.txt"
    full_summary_path = f"results_varying_j/{summary_filename}"
    
    seed_info = f", fixed_seed={args.fixed_seed}" if args.fixed_seed else ""
    
    print(f"Starting barr_spiral runs with varying J values from {args.max_j} down to 1")
    print(f"Parameters: beta={args.beta}, n_trajectory_data={args.n_trajectory_data}, " +
          f"n_background_data={args.n_background_data}, max_iters={args.scenapp_max_iters}, " +
          f"time_horizon={args.time_horizon}{seed_info}")
    print("-" * 50)
    
    os.makedirs("results_varying_j", exist_ok=True)
    with open(full_summary_path, "w") as f:
        f.write("Summary of barr_spiral runs with varying J values\n")
        f.write(f"Parameters: beta={args.beta}, n_trajectory_data={args.n_trajectory_data}, " +
                f"n_background_data={args.n_background_data}, max_iters={args.scenapp_max_iters}, " +
                f"time_horizon={args.time_horizon}{seed_info}\n")
        f.write("-" * 50 + "\n")
    
    results = []
    for j in range(args.max_j, 0, -1):
        print(f"Running with J={j}")
        result = run_barr_spiral_with_j(args, j, full_summary_path)
        results.append((j, result))
    
    print("Running with J=-1 (no limit)")
    result = run_barr_spiral_with_j(args, -1, full_summary_path)
    results.append((-1, result))
    
    print("\nSummary of all runs:")
    print("-" * 50)
    print("J\tEpsilon\t\tA-Post-Epsilon\tIters")
    print("-" * 50)
    for j, result in results:
        print(f"{j}\t{result.res:.6f}\t{result.a_post_res:.6f}\t{result.stats.iters}")
    
    print(f"\nResults saved to {full_summary_path}")


if __name__ == "__main__":
    main()
