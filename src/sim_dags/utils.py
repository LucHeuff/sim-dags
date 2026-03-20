import altair as alt
import polars as pl
import xarray as xr


def to_df(arr: xr.DataArray | xr.Dataset) -> pl.DataFrame:  # pragma: no cover
    """Convert xr.Dataset or xr.DataArray to pl.DataFrame."""
    return pl.from_pandas(arr.to_dataframe(), include_index=True)


Chart = (
    alt.Chart
    | alt.LayerChart
    | alt.FacetChart
    | alt.VConcatChart
    | alt.HConcatChart
    | alt.RepeatChart
)

AXIS_LABELS = 13
AXIS_TITLE = 15
LEGEND_TITLE = 16
LEGEND_LABELS = 15
TITLE = 18


def default_chart_config(chart: Chart) -> Chart:
    """Apply default configuration to a chart."""
    return (
        chart.configure_axis(grid=False)
        .configure_title(fontSize=TITLE, fontWeight=500)
        .configure_view(stroke=None)
        .configure_legend(titleFontSize=LEGEND_TITLE, labelFontSize=LEGEND_LABELS)
        .configure_axisY(
            titleAngle=0,
            titleAlign="left",
            titleY=-3,
            titleX=3,
            titleFontWeight="normal",
            titleFontStyle="italic",
            titleFontSize=AXIS_TITLE,
            labelFontSize=AXIS_LABELS,
        )
        .configure_axisX(
            labelAngle=0,
            titleFontSize=AXIS_TITLE,
            labelFontSize=AXIS_LABELS,
            titleFontWeight="normal",
            titleFontStyle="italic",
        )
    )
