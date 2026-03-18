class SimDagError(Exception):
    """Base exception for package."""


class VariableDoesNotExistError(SimDagError):
    """Raised when variable does not exist in probability calculation."""
