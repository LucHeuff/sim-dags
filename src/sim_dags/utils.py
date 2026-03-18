import polars as pl
import xarray as xr


def to_df(arr: xr.DataArray | xr.Dataset) -> pl.DataFrame:  # pragma: no cover
    """Convert xr.Dataset or xr.DataArray to pl.DataFrame."""
    return pl.from_pandas(arr.to_dataframe(), include_index=True)
