class SimDagError(Exception):
    """Base exception for package."""


class VariableDoesNotExistError(SimDagError):
    """Raised when variable does not exist in probability calculation."""


class InvalidPriorShapeError(SimDagError):
    """Raised when the provided prior has an incorrect shape."""


class InvalidPriorDistributionError(SimDagError):
    """Raised when the provided prior has an invalid distribution."""


class InvalidGridStepsError(SimDagError):
    """Raised when trying to set grid steps to an invalid value."""


class IllegalColumnNameError(SimDagError):
    """Raised when a column name is used that is also used internally."""


class UnknownDoVariableError(SimDagError):
    """Raised when trying to intervene on a variable that is not in the DAG."""


class InvalidDoValueError(SimDagError):
    """Raised when trying to set an intervention variable outside available values."""  # noqa: E501


class UnknownDistributionError(SimDagError):
    """Raised when there is no generator implemented for this distribution."""


class MissingDistributionError(SimDagError):
    """Raised when a variable does not have an associated distribution."""
