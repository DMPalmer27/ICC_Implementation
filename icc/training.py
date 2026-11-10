"""
File: training.py
Author: Daniel Palmer (d.m.palmer@wustl.edu)
Description: Least-squares linear regression trained by gradient descent THROUGH the scheme,
    from the client's side (docs/GD_LINEAR_REGRESSION.md sections 4 and 10.2).

    Training is live: each step's query is built from the previous step's decoded gradient.
    One step, at the case-study format:

        w_t = [0.5, -0.125, 0.625], b_t = 0.25        client's float model (never sent)
        w^_t = [8, -2, 10], b^_t = 4                   x 2^f_w = 16: the public query coefficients
        P + 1 queries (query.least_squares_gradient)   -> server -> one decode for all of them
        g^ = lift_balanced(...)                        exact integer gradient at scale 2^(2 f_X + f_w)
        w_{t+1} = clip(w_t - (eta / n_s) * g^ / 2^(2 f_X + f_w))

    WHAT THE TRAINER CAN SEE, which is the point of the module's shape. train_linear_regression
    takes a Client and a Server and nothing else: there is no argument through which the
    plaintext data could reach it, and the Client keeps only its key after upload. The plain-
    text references (integer GD, float GD) live in tools/gd_reference.py and drive the SAME
    loop, run_gd, through a different gradient function, so that the bit-for-bit comparison
    between the scheme and plaintext integer GD compares the gradients and nothing else.

    WHAT THE PROVIDER SEES: the weight codes w^_t, as coefficients of every query. They are a
    function of the data, a channel Theorem 1 does not cover (section 2 of the plan); the
    Trajectory records them exactly, as the provider's view of the queries.

    Design choices, all from the plan:
      * the update uses the float w_t, not w^_t, so quantisation error does not accumulate;
      * the gradient is a SUM over samples; eta / n_s absorbs the mean;
      * the step size, run length and initial point are public and fixed before training --
        no early stopping, since a stopping time read off the decoded gradient would publish
        one more data-dependent number;
      * clipping w to the declared A_w (and b to A_b) keeps every query inside the budget q
        was sized for, so a refused query is a bug and raises.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from query import least_squares_gradient
from quantize import FixedPointSpec, lift_balanced, to_integer

# A gradient function maps the public codes of one step to the exact integer gradient,
# one entry per component (features, then the bias if present), plus (eval s, decode s).
GradientFn = Callable[[np.ndarray, Optional[int]], Tuple[List[int], float, float]]


@dataclass(frozen=True)
class GDSchedule:
    """
    Everything about a training run that is public and fixed before it starts.

    :param eta: Learning rate; the update is w <- w - (eta / n_s) * gradient
    :param steps: Number of steps T, fixed in advance
    :param w_spec: Weight format (f_w, A_w); A_w is the declared clip
    :param f_X: Fractional bits of the stored data, which sets the gradient's scale
    :param b_spec: Intercept format (f_b = f_w, A_b), or None for no intercept
    :param rounding: 'nearest' or 'stochastic' for quantising the model each step
    """
    eta: float
    steps: int
    w_spec: FixedPointSpec
    f_X: int
    b_spec: Optional[FixedPointSpec] = None
    rounding: str = "nearest"

    def __post_init__(self):
        if self.b_spec is not None and self.b_spec.f != self.w_spec.f:
            raise ValueError("the intercept must share the weights' fractional bits")

    @property
    def scale(self) -> float:
        """:return: 2^(2 f_X + f_w), the scale of every decoded gradient component"""
        return 2.0 ** (2 * self.f_X + self.w_spec.f)


@dataclass
class Trajectory:
    """
    One training run. w_codes is exactly what the provider saw; the rest is client-side.

    w has T + 1 rows (w[0] is the initial point) and, with an intercept, the bias as its
    last column; w_codes and grad have T rows.
    """
    w_codes: np.ndarray
    w: np.ndarray
    grad: List[List[int]]
    t_eval: np.ndarray
    t_decode: np.ndarray
    meta: dict = field(default_factory=dict)


def quantise_model(w: np.ndarray, schedule: GDSchedule, n_features: int,
                   rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, Optional[int]]:
    """
    The public codes of one step: the weights at f_w, and the bias (if any) at the same f_w.

    :param w: Float model, length P (+1 with an intercept)
    :param schedule: Run schedule
    :param n_features: P
    :param rng: Client-secret generator, needed only for stochastic rounding
    :return: (weight codes, bias code or None)
    """
    w_codes = to_integer(w[:n_features], schedule.w_spec, schedule.rounding, rng)
    if schedule.b_spec is None:
        return w_codes, None
    return w_codes, int(to_integer(w[n_features:], schedule.b_spec, schedule.rounding, rng)[0])


def gd_update(w: np.ndarray, grad: List[int], schedule: GDSchedule, n_samples: int,
              n_features: int) -> np.ndarray:
    """
    One client-side update, in float: descale, step, clip to the declared box.

    Shared by the scheme and every plaintext reference, so that their trajectories can be
    compared bit for bit: identical integer gradients give identical floats.

    :param w: Current float model
    :param grad: Exact integer gradient, one entry per component
    :param schedule: Run schedule
    :param n_samples: n_s
    :param n_features: P
    :return: Next float model
    """
    g = np.array([float(v) for v in grad]) / schedule.scale
    out = w - (schedule.eta / n_samples) * g
    out[:n_features] = np.clip(out[:n_features], -schedule.w_spec.clip, schedule.w_spec.clip)
    if schedule.b_spec is not None:
        out[n_features:] = np.clip(out[n_features:], -schedule.b_spec.clip,
                                   schedule.b_spec.clip)
    return out


def run_gd(schedule: GDSchedule, n_samples: int, n_features: int, gradient_fn: GradientFn,
           w0: Optional[np.ndarray] = None,
           rng: Optional[np.random.Generator] = None) -> Trajectory:
    """
    The training loop, independent of where the gradient comes from.

    :param schedule: Run schedule
    :param n_samples: n_s
    :param n_features: P
    :param gradient_fn: Maps (weight codes, bias code) to the exact integer gradient
    :param w0: Initial model; zero (public) by default
    :param rng: Client-secret generator for stochastic rounding
    :return: Trajectory
    """
    width = n_features + (schedule.b_spec is not None)
    w = np.zeros(width) if w0 is None else np.asarray(w0, dtype=float).copy()
    ws, codes, grads = [w.copy()], [], []
    t_eval, t_decode = np.zeros(schedule.steps), np.zeros(schedule.steps)
    for t in range(schedule.steps):
        w_codes, b_code = quantise_model(w, schedule, n_features, rng)
        grad, t_eval[t], t_decode[t] = gradient_fn(w_codes, b_code)
        codes.append(list(w_codes) + ([] if b_code is None else [b_code]))
        grads.append(grad)
        w = gd_update(w, grad, schedule, n_samples, n_features)
        ws.append(w.copy())
    return Trajectory(w_codes=np.array(codes, dtype=np.int64), w=np.array(ws), grad=grads,
                      t_eval=t_eval, t_decode=t_decode)


def scheme_gradient(client, server, n_samples: int, n_features: int,
                    schedule: GDSchedule) -> GradientFn:
    """
    The gradient through the scheme: P (+1) queries to the server, one decode.

    :param client: Client holding the key of the uploaded data
    :param server: Server holding the encoded data
    :param n_samples: n_s
    :param n_features: P
    :param schedule: Run schedule
    :return: GradientFn
    :raises RuntimeError: if the server refuses any component, with its reason
    """
    q = client.context.q
    n_components = n_features + (schedule.b_spec is not None)

    def gradient(w_codes, b_code):
        t0 = time.perf_counter()
        responses = [server.compute_request(
            least_squares_gradient(n_samples, n_features, w_codes, schedule.w_spec.f, j,
                                   bias_code=b_code, f_X=schedule.f_X))
            for j in range(n_components)]
        t1 = time.perf_counter()
        refused = [r for r in responses if not r.ok]
        if refused:
            raise RuntimeError(
                f"server refused a gradient query ({refused[0].status.value}): "
                f"{refused[0].reason}. Clipping to the declared box should make this "
                f"impossible, so the budget and the schedule disagree.")
        decoded = client.decode_results(responses)
        grad = [int(lift_balanced(np.array(int(d.value), dtype=object), q)) for d in decoded]
        return grad, t1 - t0, time.perf_counter() - t1

    return gradient


def train_linear_regression(client, server, schedule: GDSchedule, n_samples: int,
                            n_features: int,
                            rng: Optional[np.random.Generator] = None) -> Trajectory:
    """
    Trains through the scheme. Takes only the client and the server: the plaintext never
    reaches the trainer.

    :param client: Client that uploaded the data and holds the key
    :param server: Server holding the encoded data
    :param schedule: Public run schedule
    :param n_samples: n_s
    :param n_features: P
    :param rng: Client-secret generator for stochastic rounding
    :return: Trajectory
    """
    return run_gd(schedule, n_samples, n_features,
                  scheme_gradient(client, server, n_samples, n_features, schedule), rng=rng)
