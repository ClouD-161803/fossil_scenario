"""Example: Verification-only run using CSV trajectory datasets.

This script assumes you have prepared two CSV files:
  data/train_trajs.csv  (used to build internal DATA structure if needed)
  data/test_trajs.csv   (used for a posteriori risk bound evaluation)

Columns required: id,time,x1,...,xN,(optional dx1,...,dxN)
If derivatives are provided, set CSV_HAS_DERIVS=True; otherwise they will be
estimated with finite differences.
"""
from fossil.consts import ScenAppConfig, CertificateType, TimeDomain
from fossil.scenapp import ScenApp

# Minimal dummy system class just to satisfy references when VERIFY_ONLY
# (No synthetic trajectory generation will be used in this path.)
class DummySystem:
    def __call__(self):
        return self
    def generate_trajs(self, x):
        raise RuntimeError("Synthetic generation not used in VERIFY_ONLY mode")


def main():
    cfg = ScenAppConfig(
        SYSTEM=DummySystem,            # Not used when VERIFY_ONLY
        CERTIFICATE=CertificateType.LYAPUNOV,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        N_VARS=2,
        N_HIDDEN_NEURONS=(8,),
        ACTIVATION=(),                 # default stays as in learner if empty
        BETA=(1e-3,),
        N_DATA=0,                      # not used in verification-only
        N_TEST_DATA=0,                 # overridden by CSV test size
        USE_CSV_DATA=True,
        VERIFY_ONLY=True,
        TRAIN_CSV=["data/train_trajs.csv"],
        TEST_CSV=["data/test_trajs.csv"],
        CSV_HAS_DERIVS=True,
    )

    app = ScenApp(cfg)
    result = app.solve()
    print("A posteriori risk bound (epsilon):", result.a_post_res)

if __name__ == "__main__":
    main()
