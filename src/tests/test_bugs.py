# Tests based on real world bugs
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import polars as pl
from sim_dags.probability import p, p_array


@dataclass
class Parameters:
    """Parameters for realistic DAG."""

    seed: int
    alpha: int = 2
    num_r: int = 5
    num_a: int = 4
    num_o: int = 5
    pr: npt.NDArray[np.float64] = field(init=False)
    pa_r: npt.NDArray[np.float64] = field(init=False)
    po_ar: npt.NDArray[np.float64] = field(init=False)
    pn_oar: npt.NDArray[np.float64] = field(init=False)

    def __post_init__(self) -> None:
        """Initialise parameters."""
        rng = np.random.default_rng(self.seed)
        self.pr = rng.dirichlet(np.repeat(self.alpha, self.num_r))
        self.pa_r = rng.dirichlet(np.repeat(self.alpha, self.num_a), size=self.num_r)
        self.po_ar = rng.dirichlet(
            np.repeat(self.alpha, self.num_o), size=(self.num_a, self.num_r)
        )
        self.pn_oar = rng.uniform(size=(self.num_o, self.num_a, self.num_r))


def sim_dag(size: int, params: Parameters, seed: int) -> pl.DataFrame:
    """Simulate from DAG."""
    rng = np.random.default_rng(seed)
    r = rng.choice(params.num_r, p=params.pr, size=size)
    a = rng.multinomial(1, pvals=params.pa_r[r]).argmax(axis=1)
    o = rng.multinomial(1, pvals=params.po_ar[a, r]).argmax(axis=1)
    n = rng.binomial(1, p=params.pn_oar[o, a, r])

    return pl.DataFrame({"r": r, "a": a, "o": o, "n": n})


def test_p_array_duplicates() -> None:
    """Testing a bug where duplicates caused p_array to fail on MultiIndex conversion."""  # noqa: E501
    params = Parameters(12346)
    sim = sim_dag(100, params, 12345)

    p(sim, "n|o")
    p_array(sim, "n|o")

    p(sim, "n|o,a")
    p_array(sim, "n|o,a")

    p(sim, "n|o,a,r")
    p_array(sim, "n|o,a,r")
