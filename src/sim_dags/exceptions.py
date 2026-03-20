class SimDagError(Exception):
    """Base exception for package."""


class VariableDoesNotExistError(SimDagError):
    """Raised when variable does not exist in probability calculation."""


class InvalidPriorError(SimDagError):
    """Raised when the provided prior has an incorrect shape."""


class InvalidGridStepsError(SimDagError):
    """Raised when trying to set grid steps to an invalid value."""
