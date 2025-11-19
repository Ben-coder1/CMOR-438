import warnings
from ml.metrics_and_evaluations.evaluation.supervised.loss_functions import Loss
from ml.utils.activations import Activation


def _validate_activation(func, approved=None):
    """
    Validate that the provided object is an Activation.

    Parameters
    ----------
    func : str or Activation
        Either the name of a pre-approved activation or an Activation instance.
    approved : dict, optional
        Mapping of approved names to Activation objects.

    Returns
    -------
    Activation
        A validated Activation object.

    Raises
    ------
    ValueError
        If func is not a string or Activation, or is unsupported.
    """
    if approved is None:
        approved = {}

    # Case 1: string lookup
    if isinstance(func, str):
        if func not in approved:
            raise ValueError(f"Unsupported activation name: {func}")
        return approved[func]

    # Case 2: Activation object
    if isinstance(func, Activation):
        return func

    raise ValueError("Activation must be a string or an Activation instance.")


def _validate_loss_fn(loss_fn, approved_losses=None):
    """
    Validate that the provided object is a Loss instance or a string key
    referring to an approved loss function.

    Parameters
    ----------
    loss_fn : Union[str, Loss]
        Either a Loss instance or the name of a loss in approved_losses.
    approved_losses : dict, optional
        Dictionary of approved loss functions (e.g., APPROVED_LOSSES).
        If provided, also checks membership.

    Returns
    -------
    Loss
        The validated Loss object.

    Raises
    ------
    TypeError
        If loss_fn is neither a Loss instance nor a string.
    ValueError
        If a string is provided but not found in approved_losses.
    """

    # Handle string lookup
    if isinstance(loss_fn, str):
        if approved_losses is None:
            raise ValueError("String loss_fn provided but no approved_losses registry given.")
        if loss_fn not in approved_losses:
            raise ValueError(
                f"Loss '{loss_fn}' is not in approved losses: {list(approved_losses.keys())}"
            )
        return approved_losses[loss_fn]

    # Handle Loss instance
    if isinstance(loss_fn, Loss):
        if approved_losses is not None and loss_fn.name not in approved_losses:
            warnings.warn(
                f"Loss '{loss_fn.name}' is valid but not in approved losses. "
                "Proceeding anyway.",
                UserWarning
            )
        return loss_fn

    # Otherwise invalid type
    raise TypeError(f"Expected a Loss instance or string, got {type(loss_fn).__name__}")
