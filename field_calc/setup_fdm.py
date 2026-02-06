from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np


@dataclass
class FDMConfig:
    fdm_func: Callable
    stopping_criterion: float
    b_min: np.ndarray | None = None
    b_max: np.ndarray | None = None



def solve_fdm(field, boundary, dielectric, config: FDMConfig):

    """
    Perform Finite Difference Method (FDM) iterations until convergence.

    Parameters
    ----------
    field : numpy.ndarray
        Initial potential distribution (0th iteration).
    B : numpy.ndarray
        Boundary condition mask.
    D : numpy.ndarray
        Dielectric constant matrix.
    stopping_criterion : float, optional
        Convergence threshold (default from args.conv).

    Returns
    -------
    field : numpy.ndarray
        Potential distribution after convergence.
    """

    n_iter = 0
    delta = np.float32(500.0)

    while delta > config.stopping_criterion:
        print(f"iteration No = {n_iter}, delta = {delta:.6e}")

        if config.b_min is None or config.b_max is None:
            new_field = config.fdm_func(field, boundary, dielectric)
        else:
            new_field = config.fdm_func(
                field,
                boundary,
                dielectric,
                config.b_min,
                config.b_max
            )

        delta = np.max(np.abs(new_field - field))
        field = new_field
        n_iter += 1
        
    return field
