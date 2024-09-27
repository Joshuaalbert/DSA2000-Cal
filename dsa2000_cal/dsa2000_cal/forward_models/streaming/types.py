from typing import NamedTuple, Any

import jax


class StepReturn(NamedTuple):
    cal_params: Any
    calibration_solver_state: Any
    solver_aux: Any
    vis_residual: jax.Array
    image_pb_cor: jax.Array
    image_psf: jax.Array
