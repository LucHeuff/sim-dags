This package contains tools for simulating DAGs and validating these against estimands derived using _do_-calculus.

> ⚠ Note that these simulators are *only* intended for validating DAGs! Since DAGs and _do_-calculus do not require high fidelity simulations,
this package doesn't provide those. If you are trying to simulate those, I would recommend using [`pymc`](https://www.pymc.io/welcome.html) instead.
See [here](https://tomicapretto.com/posts/2024-11-01_pymc-data-simulation/#good-practices) for a guide to simulating data using `pymc` models.

# Installation

Install this package using `uv`:
```bash
uv add git+https://github.com/LucHeuff/sim-dags.git
```

# Contents

## `DAGSimulator`

The core of this package revolves around the `DAGSimulator`, which constructs and validates a DAG from a list of `Distribution`, and allows sampling from the graph, optionally with interventions.
Since DAG validation is non-parameteric, to keep the validation as simple as possible the only available distributions are the `Categorical` and the `Binomial`.
For example, to generate a `DAGSimulator` for the DAG $X \rightarrow Z \rightarrow Y$ one would simply use:

```python
from sim_dags import Binomial, Categorical, DAGSimulator

distributions = [
    Categorical("X", 4),            # Categorical variable with 4 levels
    Categorical("Z", 3, ["X"]),     # Categorical variable with 3 levels and X as ancestor
    Binomial("Y", ["Z"]),           # Binomial variable with Z as ancestor
]
dag_simulator = DAGSimulator(distributions)

```
`DAGSimulator` has a `sample` method to perform sampling from the DAG, returning samples in the form of a `polars.DataFrame`. 

```python
size = 100
seed = 12345
observations = dag_simulator.sample(size, seed)
```
To sample interventions, use the `do` argument of `sample`. This argument expects a dictionary of variables to intervene on, and how to intervene on them.
For example, `{"X": True}` will intervene on $X$ by uniformly sampling from $X$'s catgories. `{"Z": 1}` will set all $Z$ to the value 1. 
```python
interventions = dag_simulator.sample(size, seed, do={"X": True, "Z": 1})

```
For more examples see [`example_generators.py`](https://github.com/LucHeuff/sim-dags/blob/main/src/sim_dags/example_generators.py).

## Validation

Once you have simulated your samples, one way to validate your estimands is by comparing them to simulated interventions.
For example, if you have calculated $P(y|\mathrm{do}(x)) = \sum_z P(y|x,z)P(z)$, you can compare this in simulations by estimating both quantities
from simulations and then calculating the Euclidian distance between these. You would expect this distance to shrink to 0 with larger sample sizes.

This package provides convenience functions to graph these validations through the `iterate_simulations` and `plot_simulations` functions.

`iterate_simulations` iterates over simulations and compares these using a provided `CompareFunction`, with the following signature:
```python
def compare(size: int, seed: int) -> pl.DataFrame: ...
    "Compare estimands with interventions."
```
Where the resulting `pl.DataFrame` is expected to have the following columns:
- `estimand`: string with names of estimands
- `value`: some validation score for this estimand (e.g. Euclidian distance)

You can implement this function however you see fit, but the package provides `build_compare_function` for convenience.
This function requires 4 arguments:
- `intervention_sim`: a function that generates intervention samples from a `size` and a `seed`,
- `intervention`: a function that calculates the interventional distribution from `samples`,
- `observation_sim`: a function that generates observational samples from a `size` and a `seed`
- `estimands`: a function or list of functions that calculate estimands from `samples`,

For example, to implement the comparison from the example above, use
```python
from sim_dags import build_compare_function, iterate_simulation, plot_simulations, p, p_array, to_df

compare = build_compare_function(
        intervention_sim=lambda size, seed: dag_simulator.sample(
            size, seed, do={"x": True}, rename_do=False # not renaming makes sure the results can be easily combined with estimands
        ),
        intervention=lambda samples: p(samples, "y|x"), # represents P(y|do(x))
        observation_sim=dag_simulator.sample,
        estimands = [
            lambda samples: p(samples, "y|x"), # including P(y|x) to show that this estimand is incorrect
            lambda samples: to_df(
                (p_array(samples, "y|x,z") * p_array(samples, "z"))
                .sum(dim="z")
                .rename(sum_)
            ),
        ],
)

sims = iterate_simulations(compare, n_sizes=3, n_seeds=2) 
plot_simulations(sims).save("demo.png")
```
> Lambda functions are used here for brevity, but the functions can of course be defined seperately as well.

Here the function under `intervention` is applied to the intervention sample, which is generated using `intervention_sim`. 
The estimands are applied to the observation sample, which is generated using `observation_sim`.
The terminology 'intervention' and 'observation' is just to make clearer where which function are being applied, there is nothing stopping you from also applying interventions with the `observation` function.

`sims` will contain a `polars.DataFrame` with mean and standard deviation of the Euclidian distance (or whatever metric you provide yourself) caculated over each of the `n_seeds`.
This will be repeated for `n_sizes`, where the next size is always an order of magnitude larger than previous. By default, `iterate_samples` will start with a samples size of 100,
so with `n_sizes=3` the estimands will be compared at sample sizes of 100, 100 and 10 000.

The `plot_simulations` function is designed to graph these results into an [`altair`](https://altair-viz.github.io/) chart, which can then be used like any other `altair` chart.

For more examples see [`example_validation.py`](https://github.com/LucHeuff/sim-dags/blob/main/src/sim_dags/example_validation.py), including how you would manually write a `CompareFunction`.

## Probability

In order to validate results calculated using _do_-calculus, probability distributions need to be estimated from the simulated samples.
This is what the `p` and `p_array` functions are for. These both calculate (conditional) distributions from a given `polars.DataFrame`.
`p` returns a `polars.DataFrame`, while `p_array` returns a `xarray.DataArray`, which is more convenient when calculating products of multiple distributions.

`p` and `p_array` have (almost) the same signatures, with the following arguments:
- `data`: a `polars.DataFrame` from which the distribution is to be calculated
- `query`: the desired (conditional) probability in string form, e.g. "x" (for $P(x)$), "y|x" (for P(y|x)) or "y|do(x), z" (for $P(y|do(x), z)),
- `name`: Optionally allows renaming of the column containing the distribution. By default, the column will be named "P(query)" (e.g for `query`="y|x", the column will be "P(y|x)")
- `include_zeros`: whether you want to include cases that do not appear in the data, meaning they have zero probability. Turned off by default, and only available for `p`.

To convert `xarray.DataArray` into `polars.DataFrame`, the `to_df` convenience function is provided.

