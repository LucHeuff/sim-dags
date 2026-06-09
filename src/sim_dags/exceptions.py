class SimDAGError(Exception):
    """Base exception for package."""


class VariableDoesNotExistError(SimDAGError):
    """Raised when variable does not exist in probability calculation."""


class VariableNotInDAGError(SimDAGError):
    """Raised when a variable does not appear in the DAG."""


class VariableNotBinomialError(SimDAGError):
    """Raised when the variable doesn't seem to be Binomial."""


class InvalidPriorShapeError(SimDAGError):
    """Raised when the provided prior has an incorrect shape."""


class InvalidPriorDistributionError(SimDAGError):
    """Raised when the provided prior has an invalid distribution."""


class InvalidGridStepsError(SimDAGError):
    """Raised when trying to set grid steps to an invalid value."""


class IllegalColumnNameError(SimDAGError):
    """Raised when a column name is used that is also used internally."""


class UnknownDoVariableError(SimDAGError):
    """Raised when trying to intervene on a variable that is not in the DAG."""


class InvalidDoValueError(SimDAGError):
    """Raised when trying to set an intervention variable outside available values."""  # noqa: E501


class UnknownDistributionError(SimDAGError):
    """Raised when there is no generator implemented for this distribution."""


class MissingNodeError(SimDAGError):
    """Raised when a node appears in edges, but not in nodes."""


class NodeDoesNotExistError(SimDAGError):
    """Raised when a requested node does not exist."""


class DuplicateVariableError(SimDAGError):
    """Raised when trying to construct a DAGSimulator with duplicate variables."""


class NotADAGError(SimDAGError):
    """Raised when provided graph is not a DAG."""


class NoDisjointSetsError(SimDAGError):
    """Raised when sets are not disjoint when they should be."""
