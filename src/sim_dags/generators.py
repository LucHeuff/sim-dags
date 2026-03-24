from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandera.polars as pa
import polars as pl


@dataclass()
class SimpleDAGParams:
    """Parameters for generating simple DAGs."""

    seed: int
    nz: int = 4
    nx: int = 3
    alpha: int = 2
    px: npt.NDArray[np.float64] = field(init=False)
    px_z: npt.NDArray[np.float64] = field(init=False)
    pz: npt.NDArray[np.float64] = field(init=False)
    pz_x: npt.NDArray[np.float64] = field(init=False)
    pz_xy: npt.NDArray[np.float64] = field(init=False)
    py_x: npt.NDArray[np.float64] = field(init=False)
    py_xz: npt.NDArray[np.float64] = field(init=False)
    p_do_x: npt.NDArray[np.int64] = field(init=False)
    schema: pa.DataFrameSchema = field(init=False)

    def __post_init__(self) -> None:
        """Initialise random generator and parameters."""
        rng = np.random.default_rng(self.seed)
        self.px = rng.dirichlet(np.repeat(self.alpha, self.nx))
        self.px_z = rng.dirichlet(np.repeat(self.alpha, self.nx), size=self.nz)
        self.pz = rng.dirichlet(np.repeat(self.alpha, self.nz))
        self.pz_x = rng.dirichlet(np.repeat(self.alpha, self.nz), size=self.nx)
        self.pz_xy = rng.dirichlet(np.repeat(self.alpha, self.nz), size=(self.nx, 2))
        self.py_x = rng.uniform(size=self.nx)
        self.py_xz = rng.uniform(size=(self.nx, self.nz))
        self.p_do_x = np.repeat(1 / self.nx, self.nx)
        # Schema checks for output
        self.schema = pa.DataFrameSchema(
            columns={
                "x": pa.Column(int, pa.Check.isin(list(range(self.nx)))),
                "z": pa.Column(int, pa.Check.isin(list(range(self.nz)))),
                "y": pa.Column(int, pa.Check.isin(list(range(2)))),
            },
            strict=True,
        )
        self.do_schema = self.schema.remove_columns(["x"]).add_columns(
            {"do(x)": pa.Column(int, pa.Check.isin(list(range(self.nx))))}
        )

        # Sanity checks
        assert self.px.shape == (self.nx,), "P(x) wrong shape"
        assert self.px_z.shape == (self.nz, self.nx), "P(x|z) wrong shape"
        assert self.pz.shape == (self.nz,), "P(z) wrong shape"
        assert self.pz_x.shape == (self.nx, self.nz), "P(z|x) wrong shape"
        assert self.pz_xy.shape == (self.nx, 2, self.nz), "P(z|x, y) wrong shape"
        assert self.py_x.shape == (self.nx,), "P(y|x) wrong shape"
        assert self.py_xz.shape == (self.nx, self.nz), "P(y|x,z) wrong shape"
        assert self.p_do_x.shape == (self.nx,), "P(do(x)) wrong shape"


def _get_do_x(
    params: SimpleDAGParams, *, do_x: bool
) -> tuple[str, pa.DataFrameSchema, np.ndarray]:
    """Retrieve relevant parameters depending on do_x."""
    x_name = "do(x)" if do_x else "x"
    schema = params.do_schema if do_x else params.schema
    px = params.p_do_x if do_x else params.px

    return x_name, schema, px


GenerateFunction = Callable[[int, SimpleDAGParams], pl.DataFrame]


def generate_pipe(
    size: int, params: SimpleDAGParams, seed: int, *, do_x: bool = False
) -> pl.DataFrame:
    """Generate samples from Pipe DAG."""
    rng = np.random.default_rng(seed)
    x_name, schema, px = _get_do_x(params, do_x=do_x)

    x = rng.choice(params.nx, p=px, size=size)
    z = rng.multinomial(1, pvals=params.pz_x[x]).argmax(axis=1)
    y = rng.binomial(n=1, p=params.py_xz[x, z])
    df = pl.DataFrame({x_name: x, "z": z, "y": y})
    schema.validate(df)
    return df


def generate_fork(
    size: int, params: SimpleDAGParams, seed: int, *, do_x: bool = False
) -> pl.DataFrame:
    """Generate samples from Fork DAG."""
    rng = np.random.default_rng(seed)
    x_name, schema, _ = _get_do_x(params, do_x=do_x)

    z = rng.choice(params.nz, p=params.pz, size=size)
    x = (
        rng.choice(params.nx, p=params.p_do_x, size=size)
        if do_x
        else rng.multinomial(1, pvals=params.px_z[z]).argmax(axis=1)
    )
    y = rng.binomial(n=1, p=params.py_xz[x, z])
    df = pl.DataFrame({x_name: x, "z": z, "y": y})
    schema.validate(df)
    return df


def generate_collider(
    size: int, params: SimpleDAGParams, seed: int, *, do_x: bool = False
) -> pl.DataFrame:
    """Generate samples from Collider DAG."""
    rng = np.random.default_rng(seed)
    x_name, schema, px = _get_do_x(params, do_x=do_x)

    x = rng.choice(params.nx, p=px, size=size)
    y = rng.binomial(n=1, p=params.py_x[x])
    z = rng.multinomial(1, pvals=params.pz_xy[x, y]).argmax(axis=1)
    df = pl.DataFrame({x_name: x, "z": z, "y": y})
    schema.validate(df)
    return df
