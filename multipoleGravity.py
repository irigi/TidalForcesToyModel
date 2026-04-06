from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import os

from scipy.integrate import solve_ivp
import plotly.graph_objects as go
import matplotlib.pyplot as plt


# =========================
# Linear algebra utilities
# =========================


def stf(a: np.ndarray) -> np.ndarray:
    """Symmetric trace-free projection for arrays with trailing (..., 3, 3) axes."""
    a = 0.5 * (a + np.swapaxes(a, -1, -2))
    tr = np.trace(a, axis1=-2, axis2=-1)[..., None, None]
    eye = np.eye(3)
    return a - (tr / 3.0) * eye


def hat(omega: np.ndarray) -> np.ndarray:
    """Convert (..., 3) vectors to (..., 3, 3) skew matrices."""
    omega = np.asarray(omega)
    out = np.zeros(omega.shape[:-1] + (3, 3), dtype=float)
    out[..., 0, 1] = -omega[..., 2]
    out[..., 0, 2] = omega[..., 1]
    out[..., 1, 0] = omega[..., 2]
    out[..., 1, 2] = -omega[..., 0]
    out[..., 2, 0] = -omega[..., 1]
    out[..., 2, 1] = omega[..., 0]
    return out


def axl(a: np.ndarray) -> np.ndarray:
    """Inverse of hat() for skew-symmetric (..., 3, 3) matrices."""
    a = np.asarray(a)
    return np.stack([
        a[..., 2, 1],
        a[..., 0, 2],
        a[..., 1, 0],
    ], axis=-1)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.where(n > 0.0, n, 1.0)
    return q / n


def quat_multiply(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Hamilton product. Quaternions are [w, x, y, z]."""
    w1, x1, y1, z1 = np.moveaxis(q, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(p, -1, 0)
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def quat_derivative_body_to_inertial(q: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
    """q maps body-frame vectors into inertial-frame vectors."""
    omega_quat = np.concatenate([np.zeros(omega_body.shape[:-1] + (1,)), omega_body], axis=-1)
    return 0.5 * quat_multiply(q, omega_quat)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert normalized quaternions [w, x, y, z] to rotation matrices, body -> inertial."""
    q = quat_normalize(q)
    w, x, y, z = np.moveaxis(q, -1, 0)

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    R = np.empty(q.shape[:-1] + (3, 3), dtype=float)
    R[..., 0, 0] = 1.0 - 2.0 * (yy + zz)
    R[..., 0, 1] = 2.0 * (xy - wz)
    R[..., 0, 2] = 2.0 * (xz + wy)

    R[..., 1, 0] = 2.0 * (xy + wz)
    R[..., 1, 1] = 1.0 - 2.0 * (xx + zz)
    R[..., 1, 2] = 2.0 * (yz - wx)

    R[..., 2, 0] = 2.0 * (xz - wy)
    R[..., 2, 1] = 2.0 * (yz + wx)
    R[..., 2, 2] = 1.0 - 2.0 * (xx + yy)
    return R


def normalize_rows(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    out = np.zeros_like(v)
    mask = (n[..., 0] > 1e-14)
    out[mask] = v[mask] / n[mask]
    if fallback is None:
        fallback = np.zeros(v.shape[-1])
        fallback[2] = 1.0
    out[~mask] = fallback
    return out



def safe_angle_deg_from_dot(dot_value: np.ndarray, axis_like: bool = False) -> np.ndarray:
    """Return angle in degrees from dot product. If axis_like=True, identifies v and -v."""
    x = np.abs(dot_value) if axis_like else dot_value
    x = np.clip(x, -1.0, 1.0)
    return np.degrees(np.arccos(x))


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    q = random_quaternions(rng, 1)[0]
    return quat_to_matrix(q[None, :])[0]


# =========================
# Model parameters
# =========================


@dataclass
class QuadrupoleParameters:
    G: float
    mass: np.ndarray
    radius: np.ndarray
    gamma: np.ndarray
    omega2: np.ndarray
    J0: np.ndarray
    quad_coeff: np.ndarray
    inertia_shape_coeff: np.ndarray
    rot_flattening_coeff: np.ndarray
    relax_strength: np.ndarray
    relax_time: np.ndarray
    colors: Tuple[str, ...] = ("royalblue", "orangered")
    tidal_quad_coeff: np.ndarray | None = None
    rot_quad_coeff: np.ndarray | None = None
    tidal_inertia_shape_coeff: np.ndarray | None = None
    rot_inertia_shape_coeff: np.ndarray | None = None
    bulge_response_coeff: np.ndarray | None = None
    bulge_relax_time: np.ndarray | None = None

    def __post_init__(self):
        self.mass = np.asarray(self.mass, dtype=float)
        self.radius = np.asarray(self.radius, dtype=float)
        self.gamma = np.asarray(self.gamma, dtype=float)
        self.omega2 = np.asarray(self.omega2, dtype=float)
        self.J0 = np.asarray(self.J0, dtype=float)
        self.quad_coeff = np.asarray(self.quad_coeff, dtype=float)
        self.inertia_shape_coeff = np.asarray(self.inertia_shape_coeff, dtype=float)
        self.rot_flattening_coeff = np.asarray(self.rot_flattening_coeff, dtype=float)
        self.relax_strength = np.asarray(self.relax_strength, dtype=float)
        self.relax_time = np.asarray(self.relax_time, dtype=float)

        n = self.mass.size
        def _vec_or_default(value, default):
            if value is None:
                arr = np.asarray(default, dtype=float)
            else:
                arr = np.asarray(value, dtype=float)
            if arr.shape == ():
                arr = np.full(n, float(arr), dtype=float)
            if arr.shape != (n,):
                raise ValueError(f"Expected shape {(n,)}, got {arr.shape}.")
            return arr

        self.tidal_quad_coeff = _vec_or_default(self.tidal_quad_coeff, self.quad_coeff)
        self.rot_quad_coeff = _vec_or_default(self.rot_quad_coeff, np.zeros(n))
        # In the split-bulge closure the tide-driven tensor S contributes to the
        # external quadrupole, while the separately relaxing rotational bulge B
        # carries the shape dependence of the spin inertia.  Keeping S inside
        # J(S,B) would reintroduce a conservative spin--shape coupling into the
        # S equation without the reciprocal term in the reduced spin balance.
        self.tidal_inertia_shape_coeff = _vec_or_default(self.tidal_inertia_shape_coeff, np.zeros(n))
        self.rot_inertia_shape_coeff = _vec_or_default(self.rot_inertia_shape_coeff, self.inertia_shape_coeff + 2.0 * self.rot_flattening_coeff)
        self.bulge_response_coeff = _vec_or_default(self.bulge_response_coeff, np.zeros(n))
        default_tauB = 2.0 * self.gamma / np.maximum(self.omega2 ** 2, 1e-14)
        self.bulge_relax_time = _vec_or_default(self.bulge_relax_time, default_tauB)

        if self.relax_strength.ndim == 1 and self.relax_strength.size == 0:
            self.relax_strength = np.zeros((n, 0), dtype=float)
        if self.relax_time.ndim == 1 and self.relax_time.size == 0:
            self.relax_time = np.zeros((n, 0), dtype=float)
        if self.relax_strength.ndim != 2 or self.relax_time.ndim != 2:
            raise ValueError("relax_strength and relax_time must be rank-2 arrays with shape (n_bodies, n_relax).")
        if self.relax_strength.shape != self.relax_time.shape:
            raise ValueError("relax_strength and relax_time must have the same shape.")
        if self.relax_strength.shape[0] != n:
            raise ValueError("Relaxation arrays must have leading dimension n_bodies.")

    @property
    def n_bodies(self) -> int:
        return int(self.mass.size)

    @property
    def n_relax(self) -> int:
        if self.relax_strength.ndim != 2:
            return 0
        return int(self.relax_strength.shape[1])

    @property
    def has_split_bulge(self) -> bool:
        return bool(
            np.any(np.abs(self.rot_quad_coeff) > 0.0)
            or np.any(np.abs(self.rot_inertia_shape_coeff) > 0.0)
            or np.any(np.abs(self.bulge_response_coeff) > 0.0)
        )


# =========================
# Packing / unpacking state
# =========================


def pack_state(x, v, q, omega, S, W, Z=None, D=None) -> np.ndarray:
    """Legacy packer without the split rotational bulge state."""
    pieces = [
        np.asarray(x).reshape(-1),
        np.asarray(v).reshape(-1),
        np.asarray(q).reshape(-1),
        np.asarray(omega).reshape(-1),
        np.asarray(S).reshape(-1),
        np.asarray(W).reshape(-1),
    ]
    if Z is not None:
        pieces.append(np.asarray(Z).reshape(-1))
    if D is not None:
        pieces.append(np.asarray([D], dtype=float))
    return np.concatenate(pieces)


def pack_state_full(x, v, q, omega, S, W, B=None, Z=None, D=None, has_split_bulge: bool | None = None) -> np.ndarray:
    pieces = [
        np.asarray(x).reshape(-1),
        np.asarray(v).reshape(-1),
        np.asarray(q).reshape(-1),
        np.asarray(omega).reshape(-1),
        np.asarray(S).reshape(-1),
        np.asarray(W).reshape(-1),
    ]
    include_B = (B is not None) if has_split_bulge is None else bool(has_split_bulge)
    if include_B:
        if B is None:
            raise ValueError("B must be provided when has_split_bulge is True.")
        pieces.append(np.asarray(B).reshape(-1))
    if Z is not None:
        pieces.append(np.asarray(Z).reshape(-1))
    if D is not None:
        pieces.append(np.asarray([D], dtype=float))
    return np.concatenate(pieces)


def unpack_state(y: np.ndarray, n: int, n_relax: int = 0):
    """Legacy unpacker returning the original 8-tuple.

    For states that include the split rotational-bulge tensor, use
    unpack_state_full().
    """
    k = 0
    x = y[k:k + 3 * n].reshape(n, 3)
    k += 3 * n
    v = y[k:k + 3 * n].reshape(n, 3)
    k += 3 * n
    q = y[k:k + 4 * n].reshape(n, 4)
    k += 4 * n
    omega = y[k:k + 3 * n].reshape(n, 3)
    k += 3 * n
    S = y[k:k + 9 * n].reshape(n, 3, 3)
    k += 9 * n
    W = y[k:k + 9 * n].reshape(n, 3, 3)
    k += 9 * n
    # Ignore any split-bulge block if present.
    remaining = y.size - k
    if remaining == 9 * n * n_relax + 1 + 9 * n:
        k += 9 * n
    if n_relax > 0:
        Z = y[k:k + 9 * n * n_relax].reshape(n, n_relax, 3, 3)
        k += 9 * n * n_relax
    else:
        Z = np.zeros((n, 0, 3, 3), dtype=float)
    if k < y.size:
        D = float(y[k])
    else:
        D = 0.0
    return x, v, q, omega, S, W, Z, D


def unpack_state_full(y: np.ndarray, n: int, n_relax: int = 0, has_split_bulge: bool = False):
    k = 0
    x = y[k:k + 3 * n].reshape(n, 3)
    k += 3 * n
    v = y[k:k + 3 * n].reshape(n, 3)
    k += 3 * n
    q = y[k:k + 4 * n].reshape(n, 4)
    k += 4 * n
    omega = y[k:k + 3 * n].reshape(n, 3)
    k += 3 * n
    S = y[k:k + 9 * n].reshape(n, 3, 3)
    k += 9 * n
    W = y[k:k + 9 * n].reshape(n, 3, 3)
    k += 9 * n
    if has_split_bulge:
        B = y[k:k + 9 * n].reshape(n, 3, 3)
        k += 9 * n
    else:
        B = np.zeros((n, 3, 3), dtype=float)
    if n_relax > 0:
        Z = y[k:k + 9 * n * n_relax].reshape(n, n_relax, 3, 3)
        k += 9 * n * n_relax
    else:
        Z = np.zeros((n, 0, 3, 3), dtype=float)
    if k < y.size:
        D = float(y[k])
    else:
        D = 0.0
    return x, v, q, omega, S, W, B, Z, D


# =========================
# Physics core
# =========================


def pairwise_geometry(x: np.ndarray):
    """r_ij = x_j - x_i."""
    r = x[None, :, :] - x[:, None, :]
    d2 = np.sum(r * r, axis=-1)
    mask = ~np.eye(x.shape[0], dtype=bool)
    d = np.zeros_like(d2)
    d[mask] = np.sqrt(d2[mask])

    inv_d = np.zeros_like(d)
    inv_d[mask] = 1.0 / d[mask]
    inv_d3 = inv_d ** 3
    inv_d5 = inv_d ** 5
    inv_d7 = inv_d ** 7

    n = np.zeros_like(r)
    n[mask] = r[mask] * inv_d[mask, None]
    return r, d, n, inv_d3, inv_d5, inv_d7, mask


def effective_shape_spin_coeff(params: QuadrupoleParameters) -> np.ndarray:
    """Energy-consistent coefficient for the S:STF(omega omega) coupling.

    The spin kinetic energy already depends on S through J(S). Any additional
    rotational-flattening coupling must enter through the same kinetic term;
    otherwise the shape equation does work on the spin without the reciprocal
    change in the spin angular momentum, and the bookkeeping drifts.

    Writing
        T_spin = 0.5 * omega^T (J0 I - c_eff S) omega
    gives the shape forcing
        W_dot ... - 0.5 * c_eff * STF(omega omega).
    To preserve the originally intended extra rotational-flattening strength
    rot_flattening_coeff in W_dot, use
        c_eff = inertia_shape_coeff + 2 * rot_flattening_coeff.
    """
    return params.inertia_shape_coeff + 2.0 * params.rot_flattening_coeff


def current_body_inertia(S_body: np.ndarray, params: QuadrupoleParameters, B_body: np.ndarray | None = None) -> np.ndarray:
    """Linearized body-frame spin inertia.

    In the one-tensor closure this reduces to J(S) = J0 I - c_eff S.
    In the split extension it becomes
        J(S,B) = J0 I - cJ^(t) S - cJ^(r) B.
    """
    eye = np.eye(3)
    S_body = stf(S_body)
    if B_body is None:
        B_body = np.zeros_like(S_body)
    else:
        B_body = stf(B_body)

    if params.has_split_bulge:
        # In the split closure the tide-driven quadrupole S contributes to the
        # external field, while the explicit rotational bulge B carries the
        # shape dependence of the spin inertia. This keeps the reduced model
        # variationally consistent without adding a separate spin--shape
        # generalized potential.
        J_body = (
            params.J0[:, None, None] * eye[None, :, :]
            - params.rot_inertia_shape_coeff[:, None, None] * B_body
        )
    else:
        coeff = effective_shape_spin_coeff(params)
        J_body = params.J0[:, None, None] * eye[None, :, :] - coeff[:, None, None] * S_body
    return 0.5 * (J_body + np.swapaxes(J_body, -1, -2))


def total_body_quadrupole(S_body: np.ndarray, params: QuadrupoleParameters, B_body: np.ndarray | None = None) -> np.ndarray:
    S_body = stf(S_body)
    if B_body is None:
        B_body = np.zeros_like(S_body)
    else:
        B_body = stf(B_body)
    if params.has_split_bulge:
        return params.tidal_quad_coeff[:, None, None] * S_body + params.rot_quad_coeff[:, None, None] * B_body
    return params.quad_coeff[:, None, None] * S_body





def solve_body_inertia(J_body: np.ndarray, rhs_vec: np.ndarray, J0_scale: np.ndarray) -> np.ndarray:
    """Solve J x = rhs with finite-value guards and a positive eigenvalue floor."""
    J_sym = 0.5 * (J_body + np.swapaxes(J_body, -1, -2))
    J0_arr = np.asarray(J0_scale, dtype=float)
    floor = 1.0e-10 * np.maximum(J0_arr, 1.0)[:, None]

    eye = np.eye(3)
    finite_matrix = np.all(np.isfinite(J_sym), axis=(-2, -1))
    finite_rhs = np.all(np.isfinite(rhs_vec), axis=-1)
    bad = ~(finite_matrix & finite_rhs)

    J_work = np.array(J_sym, copy=True)
    rhs_work = np.array(rhs_vec, copy=True)
    if np.any(bad):
        J_work[bad] = J0_arr[bad, None, None] * eye[None, :, :]
        rhs_work[bad] = 0.0

    evals, evecs = np.linalg.eigh(J_work)
    evals_safe = np.maximum(evals, floor)
    rhs_body = np.einsum("aij,aj->ai", np.swapaxes(evecs, -1, -2), rhs_work)
    sol_body = rhs_body / evals_safe
    sol = np.einsum("aij,aj->ai", evecs, sol_body)
    if np.any(bad):
        sol[bad] = 0.0
    return sol



def split_bulge_target_and_effective(
    omega: np.ndarray,
    E_body: np.ndarray,
    params: QuadrupoleParameters,
    B_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (B_target, B_effective, instantaneous_mask) for the split bulge.

    Self-consistency requires the same split mode that contributes to both the
    spin inertia and the external quadrupole to respond to both reciprocal
    forcings. Writing
        J = J0 I - cJ^(r) B,
        I_quad^(r) = cQ^(r) B,
    the relaxational target is taken to be
        B_* = -eta STF(omega omega) - lambda_B E,
        lambda_B = eta * cQ^(r) / cJ^(r).
    The recoverable branch energy is *not* |B-B_*|^2. The linear couplings to
    the conservative forcings are already accounted for in the spin kinetic
    energy through J(B) and in the orbital potential through I_quad(B), so the
    remaining branch self-energy is the absolute quadratic term ~ |B|^2 while
    the dissipation rate is governed by the lag |B-B_*|^2. For cJ^(r)=0 the
    tidal part is disabled.
    """
    eta = np.asarray(params.bulge_response_coeff, dtype=float)
    cjr = np.asarray(params.rot_inertia_shape_coeff, dtype=float)
    cqr = np.asarray(params.rot_quad_coeff, dtype=float)
    lambda_B = np.zeros_like(eta, dtype=float)
    active = np.abs(cjr) > 1.0e-14
    lambda_B[active] = eta[active] * cqr[active] / cjr[active]
    B_target = -eta[:, None, None] * spin_flattening_tensor(omega) - lambda_B[:, None, None] * E_body
    instant_mask = params.bulge_relax_time <= 1.0e-12
    if np.any(instant_mask):
        B_eff = np.where(instant_mask[:, None, None], B_target, B_state)
    else:
        B_eff = B_state
    return B_target, B_eff, instant_mask

def spin_flattening_tensor(omega_body: np.ndarray) -> np.ndarray:
    """STF centrifugal/rotational-flattening forcing tensor."""
    return stf(np.einsum("ai,aj->aij", omega_body, omega_body))


def dissipation_rate_from_state(
    W_body: np.ndarray,
    S_body: np.ndarray,
    Z_body: np.ndarray,
    params: QuadrupoleParameters,
) -> float:
    rate = 2.0 * np.sum(params.gamma[:, None, None] * W_body * W_body)
    if params.n_relax > 0:
        delta = S_body[:, None, :, :] - Z_body
        rate += np.sum((params.relax_strength[:, :, None, None] / params.relax_time[:, :, None, None]) * delta * delta)
    return float(rate)


def rotational_bulge_free_energy(
    omega_body: np.ndarray,
    E_body: np.ndarray,
    B_body: np.ndarray,
    params: QuadrupoleParameters,
) -> float:
    """Stored free energy of the split rotational bulge.

    For the first-order branch

        tau_B D_t B + B = B_*,

    with

        B_* = -eta STF(omega omega) - lambda_B E,

    the recoverable mechanical energy is the branch self-energy

        E_B = sum_a [ cJ^(r)_a / (4 eta_a) * B_a:B_a ].

    The reciprocal couplings to the spin and tidal forcings are already carried
    by the spin kinetic energy through J(B) and by the orbital interaction
    through I_quad(B). Adding an extra B:E term here would double count a
    conservative coupling that is already present in the pair potential.
    """
    if not params.has_split_bulge:
        return 0.0
    eta = np.asarray(params.bulge_response_coeff, dtype=float)
    cjr = np.asarray(params.rot_inertia_shape_coeff, dtype=float)
    active = np.abs(eta) > 1.0e-14
    if not np.any(active):
        return 0.0
    coeff = np.zeros_like(eta, dtype=float)
    coeff[active] = cjr[active] / (4.0 * eta[active])
    self_energy = np.sum(coeff[:, None, None] * B_body * B_body)
    return float(self_energy)

def split_bulge_dissipation_rate(
    omega_body: np.ndarray,
    E_body: np.ndarray,
    B_body: np.ndarray,
    params: QuadrupoleParameters,
) -> float:
    """Non-negative dissipation rate associated with the split bulge relaxation."""
    if not params.has_split_bulge:
        return 0.0
    target, _, instant_mask = split_bulge_target_and_effective(omega_body, E_body, params, B_body)
    tauB = np.asarray(params.bulge_relax_time, dtype=float)
    eta = np.asarray(params.bulge_response_coeff, dtype=float)
    cjr = np.asarray(params.rot_inertia_shape_coeff, dtype=float)
    active = (~instant_mask) & (tauB > 1.0e-14) & (np.abs(eta) > 1.0e-14) & (np.abs(cjr) > 1.0e-14)
    if not np.any(active):
        return 0.0
    lag = stf(B_body - target)
    coeff = np.zeros_like(tauB, dtype=float)
    coeff[active] = cjr[active] / (2.0 * eta[active] * tauB[active])
    return float(np.sum(coeff[:, None, None] * lag * lag))


def spin_shape_coupling_energy(
    S_body: np.ndarray,
    omega_body: np.ndarray,
    params: QuadrupoleParameters,
) -> np.ndarray:
    """No separate stored energy.

    The rotational-flattening coupling is accounted for inside the spin kinetic
    energy through current_body_inertia(); adding an extra S:STF(omega omega)
    term here would double count it.
    """
    return np.zeros(S_body.shape[0], dtype=float)


def quadrupole_mode_angular_momentum(S_body: np.ndarray, W_body: np.ndarray) -> np.ndarray:
    """Leading body-frame angular momentum carried by the quadrupole motion.

    For symmetric trace-free S and W = D_t S, the commutator [W, S] is skew,
    and the associated modal angular momentum is
        Pi_S = 2 * axl([W, S]).
    """
    comm_WS = W_body @ S_body - S_body @ W_body
    return 2.0 * axl(comm_WS)


def quadrupole_mode_angular_momentum_rate(
    S_body: np.ndarray,
    W_body: np.ndarray,
    S_dot: np.ndarray,
    W_dot: np.ndarray,
) -> np.ndarray:
    """Time derivative of the leading quadrupole modal angular momentum."""
    comm_dot = (W_dot @ S_body - S_body @ W_dot) + (W_body @ S_dot - S_dot @ W_body)
    return 2.0 * axl(comm_dot)


def viscous_mode_dissipative_torque(
    S_body: np.ndarray,
    W_body: np.ndarray,
    params: QuadrupoleParameters,
) -> np.ndarray:
    """Reciprocal body torque from the Rayleigh term gamma * W:W."""
    pi_mode = quadrupole_mode_angular_momentum(S_body, W_body)
    return -2.0 * params.gamma[:, None] * pi_mode


def maxwell_branch_dissipative_torque(
    S_body: np.ndarray,
    Z_body: np.ndarray,
    params: QuadrupoleParameters,
) -> np.ndarray:
    """Reciprocal body torque from the branch Rayleigh terms.

    For R_Z = sum (c tau / 2) |D_t Z|^2 with D_t Z = (S-Z)/tau, the body-frame
    generalized torque is
        N_Z^diss = -2 sum c * axl([S-Z, Z]).
    """
    if params.n_relax == 0:
        return np.zeros((S_body.shape[0], 3), dtype=float)
    delta = S_body[:, None, :, :] - Z_body
    comm = delta @ Z_body - Z_body @ delta
    return -2.0 * np.sum(params.relax_strength[:, :, None] * axl(comm), axis=1)


def split_bulge_branch_dissipative_torque(
    B_body: np.ndarray,
    B_target: np.ndarray,
    params: QuadrupoleParameters,
) -> np.ndarray:
    """Objective-rate branch torque for the split rotational bulge.

    This captures the corotational part of the split-branch Rayleigh functional
    R_B = sum_a [cJ^(r)_a tau_B,a / (4 eta_a)] |D_t B_a|^2,
    giving
        N_B^diss = -(cJ^(r)/eta) axl([B_target - B, B]).
    Bodies with vanishing eta or disabled split response contribute zero.
    """
    if not params.has_split_bulge:
        return np.zeros((B_body.shape[0], 3), dtype=float)
    eta = np.asarray(params.bulge_response_coeff, dtype=float)
    cjr = np.asarray(params.rot_inertia_shape_coeff, dtype=float)
    coeff = np.zeros_like(eta, dtype=float)
    active = np.abs(eta) > 1.0e-14
    coeff[active] = cjr[active] / eta[active]
    lag = B_target - B_body
    comm = lag @ B_body - B_body @ lag
    return -coeff[:, None] * axl(comm)


def compute_fields_and_forces(
    x: np.ndarray,
    q: np.ndarray,
    omega: np.ndarray,
    S_body: np.ndarray,
    params: QuadrupoleParameters,
    B_body: np.ndarray | None = None,
):
    """Compute accelerations, body-frame tides, diagnostic torques, quadrupoles, and pair potential."""
    N = params.n_bodies
    G = params.G
    m = params.mass

    qn = quat_normalize(q)
    Q = quat_to_matrix(qn)

    S_body = stf(S_body)
    if B_body is None:
        B_body = np.zeros_like(S_body)
    else:
        B_body = stf(B_body)
    I_body = total_body_quadrupole(S_body, params, B_body)
    I_in = np.einsum("aik,akl,ajl->aij", Q, I_body, Q)

    r, d, n, inv_d3, inv_d5, inv_d7, mask = pairwise_geometry(x)

    eye = np.eye(3)
    nn = np.einsum("abi,abj->abij", n, n)
    E_terms = G * m[None, :, None, None] * inv_d3[:, :, None, None] * (eye[None, None, :, :] - 3.0 * nn)
    E_in = np.sum(E_terms, axis=1)
    E_in = stf(E_in)
    QT = np.swapaxes(Q, 1, 2)
    E_body = np.einsum("aik,akl,alj->aij", QT, E_in, Q)
    E_body = stf(E_body)

    F_mon = G * (m[:, None] * m[None, :])[:, :, None] * inv_d3[:, :, None] * r

    Ir_i = np.einsum("aij,abj->abi", I_in, r)
    Ir_j = np.einsum("bij,abj->abi", I_in, r)
    s_i = np.sum(r * Ir_i, axis=-1)
    s_j = np.sum(r * Ir_j, axis=-1)

    grad_i = 2.0 * Ir_i * inv_d5[:, :, None] - 5.0 * s_i[:, :, None] * r * inv_d7[:, :, None]
    grad_j = 2.0 * Ir_j * inv_d5[:, :, None] - 5.0 * s_j[:, :, None] * r * inv_d7[:, :, None]
    F_quad = -(1.5 * G) * (m[None, :, None] * grad_i + m[:, None, None] * grad_j)

    F_total = F_mon + F_quad
    accel = np.sum(F_total, axis=1) / m[:, None]

    In_i = np.einsum("aij,abj->abi", I_in, n)
    torque_in_pair = 3.0 * G * m[None, :, None] * inv_d3[:, :, None] * np.cross(In_i, n)
    torque_in = np.sum(torque_in_pair, axis=1)
    torque_body = np.einsum("aji,aj->ai", Q, torque_in)

    pair_mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d_safe = np.where(pair_mask, d, 1.0)
    nTIn_i = np.sum(n * np.einsum("aij,abj->abi", I_in, n), axis=-1)
    nTIn_j = np.sum(n * np.einsum("bij,abj->abi", I_in, n), axis=-1)
    U_pair = np.zeros((N, N))
    U_pair[pair_mask] = (
        -G * (m[:, None] * m[None, :])[pair_mask] / d_safe[pair_mask]
        -1.5 * G * (m[None, :] * nTIn_i + m[:, None] * nTIn_j)[pair_mask] * inv_d3[pair_mask]
    )
    pair_potential = np.sum(U_pair[pair_mask])

    return accel, E_body, torque_body, I_in, pair_potential


def rhs(t: float, y: np.ndarray, params: QuadrupoleParameters) -> np.ndarray:
    N = params.n_bodies
    x, v, q, omega, S, W, B, Z, D = unpack_state_full(y, N, params.n_relax, params.has_split_bulge)

    qn = quat_normalize(q)
    S = stf(S)
    W = stf(W)
    B = stf(B)
    if params.n_relax > 0:
        Z = stf(Z)

    if params.has_split_bulge:
        # First evaluate the field with the current split bulge state. The
        # energy-consistent target then uses the reciprocal spin and tidal
        # forcings, and the effective bulge is reinserted into the forces.
        accel, E_body, torque_body, _, _ = compute_fields_and_forces(x, qn, omega, S, params, B)
        B_target, B_eff, instant_B_mask = split_bulge_target_and_effective(omega, E_body, params, B)
        accel, E_body, torque_body, _, _ = compute_fields_and_forces(x, qn, omega, S, params, B_eff)
    else:
        B_target = np.zeros_like(B)
        B_eff = B
        instant_B_mask = np.zeros(N, dtype=bool)
        accel, E_body, torque_body, _, _ = compute_fields_and_forces(x, qn, omega, S, params, B_eff)

    Om_hat = hat(omega)
    comm_OM_S = Om_hat @ S - S @ Om_hat
    comm_OM_W = Om_hat @ W - W @ Om_hat

    # First-order form with W = D_t S = S_dot + [Omega_hat, S]
    S_dot = W - comm_OM_S

    if params.n_relax > 0:
        comm_OM_Z = Om_hat[:, None, :, :] @ Z - Z @ Om_hat[:, None, :, :]
        tau_relax = np.maximum(params.relax_time, 1.0e-14)[:, :, None, None]
        Z_dot = (S[:, None, :, :] - Z) / tau_relax - comm_OM_Z
        relax_force = np.sum(params.relax_strength[:, :, None, None] * (S[:, None, :, :] - Z), axis=1)
    else:
        Z_dot = np.zeros((N, 0, 3, 3), dtype=float)
        relax_force = np.zeros_like(S)

    x_dot = v
    v_dot = accel
    q_dot = quat_derivative_body_to_inertial(qn, omega)

    if params.has_split_bulge:
        tidal_force = 0.5 * params.tidal_quad_coeff[:, None, None] * E_body
        W_dot = (
            -comm_OM_W
            - 2.0 * params.gamma[:, None, None] * W
            - params.omega2[:, None, None] ** 2 * S
            - relax_force
            - tidal_force
        )

        comm_OM_B = Om_hat @ B - B @ Om_hat
        tauB = np.maximum(params.bulge_relax_time, 1e-14)
        B_dot = (B_target - B) / tauB[:, None, None] - comm_OM_B
        if np.any(instant_B_mask):
            B_dot = np.where(instant_B_mask[:, None, None], 0.0, B_dot)

        J_body = current_body_inertia(S, params, B_eff)
        J_dot = -params.rot_inertia_shape_coeff[:, None, None] * B_dot
    else:
        spin_force = 0.5 * effective_shape_spin_coeff(params)[:, None, None] * spin_flattening_tensor(omega)
        tidal_force = 0.5 * params.quad_coeff[:, None, None] * E_body
        W_dot = (
            -comm_OM_W
            - 2.0 * params.gamma[:, None, None] * W
            - params.omega2[:, None, None] ** 2 * S
            - relax_force
            - tidal_force
            - spin_force
        )
        B_dot = np.zeros_like(S)
        J_body = current_body_inertia(S, params)
        J_dot = -effective_shape_spin_coeff(params)[:, None, None] * S_dot

    L_spin = np.einsum("aij,aj->ai", J_body, omega)
    Pi_mode = quadrupole_mode_angular_momentum(S, W)
    Pi_mode_dot = quadrupole_mode_angular_momentum_rate(S, W, S_dot, W_dot)

    torque_diss_W = viscous_mode_dissipative_torque(S, W, params)
    torque_diss_Z = maxwell_branch_dissipative_torque(S, Z, params)
    torque_diss_B = split_bulge_branch_dissipative_torque(B_eff, B_target, params) if params.has_split_bulge else np.zeros_like(torque_body)

    spin_rhs = (
        torque_body
        + torque_diss_W
        + torque_diss_Z
        + torque_diss_B
        - np.einsum("aij,aj->ai", J_dot, omega)
        - Pi_mode_dot
        - np.cross(omega, L_spin + Pi_mode)
    )
    omega_dot = solve_body_inertia(J_body, spin_rhs, params.J0)

    D_dot = dissipation_rate_from_state(W, S, Z, params)
    if params.has_split_bulge:
        D_dot += split_bulge_dissipation_rate(omega, E_body, B_eff, params)

    return pack_state_full(x_dot, v_dot, q_dot, omega_dot, S_dot, W_dot, B_dot, Z_dot, D_dot, has_split_bulge=params.has_split_bulge)


# =========================
# Simulation
# =========================


def simulate(
    params: QuadrupoleParameters,
    y0: np.ndarray,
    t_span=(0.0, 90.0),
    n_samples: int = 700,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    method: str = "DOP853",
):
    t_eval = np.linspace(t_span[0], t_span[1], n_samples)

    chosen_method = method
    if method.upper() == "DOP853":
        span = max(float(t_span[1]) - float(t_span[0]), 1.0)
        sample_dt = span / max(int(n_samples) - 1, 1)
        stiff_scales = []
        if params.has_split_bulge and np.any(params.bulge_relax_time > 0.0):
            stiff_scales.append(float(np.min(params.bulge_relax_time[params.bulge_relax_time > 0.0])))
        if params.n_relax > 0 and np.any(params.relax_time > 0.0):
            stiff_scales.append(float(np.min(params.relax_time[params.relax_time > 0.0])))
        if stiff_scales and min(stiff_scales) < 0.1 * sample_dt:
            chosen_method = "Radau"

    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y, params),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        method=chosen_method,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol


def record_solution(sol, params: QuadrupoleParameters) -> Dict[str, np.ndarray]:
    N = params.n_bodies
    T = sol.t.size

    x = np.empty((T, N, 3))
    v = np.empty((T, N, 3))
    q = np.empty((T, N, 4))
    omega = np.empty((T, N, 3))
    S_body = np.empty((T, N, 3, 3))
    W_body = np.empty((T, N, 3, 3))
    B_body = np.empty((T, N, 3, 3))
    Z_body = np.empty((T, N, params.n_relax, 3, 3)) if params.n_relax > 0 else np.zeros((T, N, 0, 3, 3))
    S_in = np.empty((T, N, 3, 3))
    B_in = np.empty((T, N, 3, 3))
    I_in = np.empty((T, N, 3, 3))
    J_body = np.empty((T, N, 3, 3))
    energy_orb_kin = np.empty(T)
    energy_spin_kin = np.empty(T)
    energy_shape_kin = np.empty(T)
    energy_shape_pot = np.empty(T)
    energy_relax = np.empty(T)
    energy_grav_pot = np.empty(T)
    energy_spin_shape = np.empty(T)
    energy_bulge_free = np.empty(T)
    energy_mech = np.empty(T)

    for k in range(T):
        xk, vk, qk, omegak, Sk, Wk, Bk, Zk, Dk = unpack_state_full(sol.y[:, k], N, params.n_relax, params.has_split_bulge)
        qk = quat_normalize(qk)
        Sk = stf(Sk)
        Wk = stf(Wk)
        Bk = stf(Bk)
        if params.n_relax > 0:
            Zk = stf(Zk)

        Qk = quat_to_matrix(qk)
        S_ink = np.einsum("aik,akl,ajl->aij", Qk, Sk, Qk)
        B_ink = np.einsum("aik,akl,ajl->aij", Qk, Bk, Qk)
        _, E_body_k, _, _, _ = compute_fields_and_forces(xk, qk, omegak, Sk, params, Bk)
        B_target_k, B_eff_k, _ = split_bulge_target_and_effective(omegak, E_body_k, params, Bk) if params.has_split_bulge else (Bk, Bk, np.zeros(N, dtype=bool))
        _, E_body_k, _, I_ink, U_pair = compute_fields_and_forces(xk, qk, omegak, Sk, params, B_eff_k)
        Jk = current_body_inertia(Sk, params, B_eff_k)

        kinetic_orb = 0.5 * np.sum(params.mass[:, None] * vk * vk)
        kinetic_spin = 0.5 * np.sum(omegak * np.einsum("aij,aj->ai", Jk, omegak))
        shape_kin = 0.5 * np.sum(Wk * Wk)
        shape_pot = 0.5 * np.sum((params.omega2[:, None, None] ** 2) * Sk * Sk)
        spin_shape_energy = 0.0
        bulge_free_energy = rotational_bulge_free_energy(omegak, E_body_k, B_eff_k, params) if params.has_split_bulge else 0.0
        if params.n_relax > 0:
            branch_energy = 0.5 * np.sum(
                params.relax_strength[:, :, None, None] * (Sk[:, None, :, :] - Zk) ** 2
            )
        else:
            branch_energy = 0.0

        energy_orb_kin[k] = kinetic_orb
        energy_spin_kin[k] = kinetic_spin
        energy_shape_kin[k] = shape_kin
        energy_shape_pot[k] = shape_pot
        energy_relax[k] = branch_energy
        energy_grav_pot[k] = U_pair
        energy_spin_shape[k] = spin_shape_energy
        energy_bulge_free[k] = bulge_free_energy
        energy_mech[k] = kinetic_orb + kinetic_spin + shape_kin + shape_pot + branch_energy + bulge_free_energy + spin_shape_energy + U_pair

        x[k] = xk
        v[k] = vk
        q[k] = qk
        omega[k] = omegak
        # Backward-compatible public shape diagnostics expose the total visible
        # quadrupolar deformation, while the split pieces remain available under
        # B_body / B_in and the new S_tidal_* keys below.
        S_total_k = Sk + Bk if params.has_split_bulge else Sk
        S_total_in_k = S_ink + B_ink if params.has_split_bulge else S_ink

        S_body[k] = S_total_k
        W_body[k] = Wk
        B_body[k] = Bk
        if params.n_relax > 0:
            Z_body[k] = Zk
        S_in[k] = S_total_in_k
        B_in[k] = B_ink
        I_in[k] = I_ink
        J_body[k] = Jk

    dissipation_rate = np.empty(T)
    for k in range(T):
        S_tidal_k = S_body[k] - B_body[k] if params.has_split_bulge else S_body[k]
        dissipation_rate[k] = dissipation_rate_from_state(W_body[k], S_tidal_k, Z_body[k], params)
        if params.has_split_bulge:
            qk = q[k]
            _, E_body_k, _, _, _ = compute_fields_and_forces(x[k], qk, omega[k], S_tidal_k, params, B_body[k])
            dissipation_rate[k] += split_bulge_dissipation_rate(omega[k], E_body_k, B_body[k], params)
    cumulative_dissipation_state = np.array([
        unpack_state_full(sol.y[:, k], N, params.n_relax, params.has_split_bulge)[-1] for k in range(T)
    ])
    cumulative_dissipation_from_rate = np.zeros(T, dtype=float)
    if T > 1:
        dt = np.diff(sol.t)
        cumulative_dissipation_from_rate[1:] = np.cumsum(0.5 * (dissipation_rate[:-1] + dissipation_rate[1:]) * dt)

    # Use the ODE-carried dissipation counter as the primary quantity, but keep
    # the independently reconstructed integral from the sampled dissipation rate
    # for diagnostics and plotting cross-checks.
    cumulative_dissipation = cumulative_dissipation_state

    total_energy_with_dissipation = energy_mech + cumulative_dissipation
    total_energy_with_dissipation_from_rate = energy_mech + cumulative_dissipation_from_rate
    total_energy_drift = total_energy_with_dissipation - total_energy_with_dissipation[0]
    total_energy_drift_from_rate = total_energy_with_dissipation_from_rate - total_energy_with_dissipation_from_rate[0]

    return {
        "t": sol.t.copy(),
        "x": x,
        "v": v,
        "q": q,
        "omega": omega,
        "S_body": S_body,
        "S_tidal_body": S_body - B_body,
        "W_body": W_body,
        "B_body": B_body,
        "Z_body": Z_body,
        "S_in": S_in,
        "S_tidal_in": S_in - B_in,
        "B_in": B_in,
        "I_in": I_in,
        "J_body": J_body,
        "energy_orb_kin": energy_orb_kin,
        "energy_spin_kin": energy_spin_kin,
        "energy_shape_kin": energy_shape_kin,
        "energy_shape_pot": energy_shape_pot,
        "energy_relax": energy_relax,
        "energy_grav_pot": energy_grav_pot,
        "energy_spin_shape": energy_spin_shape,
        "energy_bulge_free": energy_bulge_free,
        "energy_mech": energy_mech,
        "energy_total_with_dissipation": total_energy_with_dissipation,
        "energy_total_with_dissipation_from_rate": total_energy_with_dissipation_from_rate,
        "energy_total_drift": total_energy_drift,
        "energy_total_drift_from_rate": total_energy_drift_from_rate,
        "dissipation_rate": dissipation_rate,
        "cumulative_dissipation": cumulative_dissipation,
        "cumulative_dissipation_state": cumulative_dissipation_state,
        "cumulative_dissipation_from_rate": cumulative_dissipation_from_rate,
    }


# =========================
# Demo setup with explicit initial conditions
# =========================


def random_unit_vectors(rng: np.random.Generator, n: int) -> np.ndarray:
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def random_quaternions(rng: np.random.Generator, n: int) -> np.ndarray:
    u1 = rng.random(n)
    u2 = rng.random(n)
    u3 = rng.random(n)
    q = np.stack([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
    ], axis=-1)
    q = q[:, [3, 0, 1, 2]]
    return quat_normalize(q)



def random_stf_matrices(rng: np.random.Generator, n: int, amplitude: float) -> np.ndarray:
    A = rng.normal(size=(n, 3, 3))
    A = stf(A)
    norms = np.sqrt(np.sum(A * A, axis=(1, 2), keepdims=True))
    norms = np.where(norms > 0.0, norms, 1.0)
    return amplitude * A / norms


def build_demo_problem(seed: int = 7):
    """
    ==============================================================
    INITIAL CONDITIONS FOR THE DEMO PROBLEM -- EDIT THIS SECTION
    ==============================================================
    The parameters below are the intended user-facing place to edit:
      - masses and radii
      - orbital separation and eccentricity
      - orbital plane orientation
      - random seed for spin axes / body orientation / vibration modes
      - initial spin rates
      - initial quadrupole deformation S0 and vibration rate W0
    """
    rng = np.random.default_rng(seed)

    G = 1.0
    mass = np.array([1.00, 0.86], dtype=float)
    radius = np.array([0.34, 0.30], dtype=float)

    # ---- Orbital initial conditions ----
    separation_scale = 2.65           # semi-major axis scale for the tight binary
    eccentricity = 0.6                    # slight but clearly visible eccentricity
    orbital_phase = np.pi                 # initial true anomaly [rad]

    # Randomize the orbital plane orientation in 3D so the motion is not confined to xy.
    orbital_rotation = random_rotation_matrix(rng)

    # ---- Rotational initial conditions ----
    # Random spin axes and deliberately non-synchronous spin rates.
    spin_axes = random_unit_vectors(rng, 2)
    spin_rates = np.array([1.18, -0.83], dtype=float)

    # ---- Shape / tidal-mode initial conditions ----
    # Random initial quadrupole deformation (S0) and quadrupole mode velocity (W0 = D_t S).
    S0_amplitude = 0  # 0.065
    W0_amplitude = 0  # 0.095

    # ---- Material / damping / rheology parameters ----
    extra_stiffness = np.array([0.52, 0.38], dtype=float)
    # gamma = np.array([0.030, 0.040], dtype=float)
    # One Maxwell-like relaxation branch per body. Add more columns for richer rheology.
    # relax_strength = np.array([[0.55], [0.42]], dtype=float)
    # relax_time = np.array([[2.8], [2.1]], dtype=float)
    # Rotational-flattening forcing coefficient in the quadrupole equation.
    # rot_flattening_coeff = np.array([0.22, 0.18], dtype=float)

    relax_strength = np.array([[0.05], [0.05]])
    relax_time = np.array([[15.0], [20.0]])
    gamma = np.array([0.003, 0.004])
    rot_flattening_coeff = np.array([0.003, 0.003])

    # ---- Construct the osculating two-body orbit from (a, e, f) ----
    total_mass = np.sum(mass)
    mu = G * total_mass
    a = separation_scale
    e = eccentricity
    f = orbital_phase
    p = a * (1.0 - e * e)
    r_mag = p / (1.0 + e * np.cos(f))

    r_pf = np.array([r_mag * np.cos(f), r_mag * np.sin(f), 0.0])
    v_pf = np.sqrt(mu / p) * np.array([-np.sin(f), e + np.cos(f), 0.0])

    r_rel = orbital_rotation @ r_pf
    v_rel = orbital_rotation @ v_pf

    # Barycentric positions / velocities.
    x = np.array([
        -(mass[1] / total_mass) * r_rel,
        +(mass[0] / total_mass) * r_rel,
    ])
    v = np.array([
        -(mass[1] / total_mass) * v_rel,
        +(mass[0] / total_mass) * v_rel,
    ])

    q = random_quaternions(rng, 2)
    omega = spin_rates[:, None] * spin_axes

    S0 = random_stf_matrices(rng, 2, amplitude=S0_amplitude)
    W0 = random_stf_matrices(rng, 2, amplitude=W0_amplitude)

    J0 = 0.4 * mass * radius ** 2
    omega_grav_sq = (4.0 / 5.0) * G * mass / radius ** 3
    omega2 = np.sqrt(omega_grav_sq + extra_stiffness ** 2)
    quad_coeff = 0.4 * mass * radius ** 2
    inertia_shape_coeff = quad_coeff.copy()

    rot_flattening_arr = np.asarray(rot_flattening_coeff, dtype=float)
    gamma_arr = np.asarray(gamma, dtype=float)
    rot_inertia_shape_coeff = inertia_shape_coeff + 2.0 * rot_flattening_arr
    bulge_response_coeff = 0.5 * rot_inertia_shape_coeff / np.maximum(omega2 ** 2, 1e-14)
    bulge_relax_time = 2.0 * gamma_arr / np.maximum(omega2 ** 2, 1e-14)

    params = QuadrupoleParameters(
        G=G,
        mass=mass,
        radius=radius,
        gamma=gamma_arr,
        omega2=omega2,
        J0=J0,
        quad_coeff=quad_coeff,
        inertia_shape_coeff=inertia_shape_coeff,
        rot_flattening_coeff=rot_flattening_coeff,
        relax_strength=relax_strength,
        relax_time=relax_time,
        tidal_quad_coeff=quad_coeff,
        rot_quad_coeff=quad_coeff,
        tidal_inertia_shape_coeff=np.zeros_like(inertia_shape_coeff),
        rot_inertia_shape_coeff=rot_inertia_shape_coeff,
        bulge_response_coeff=bulge_response_coeff,
        bulge_relax_time=bulge_relax_time,
        colors=("royalblue", "orangered"),
    )

    if params.has_split_bulge:
        _, E0_body, _, _, _ = compute_fields_and_forces(x, quat_normalize(q), omega, S0, params, np.zeros_like(S0))
        B0, _, _ = split_bulge_target_and_effective(omega, E0_body, params, np.zeros_like(S0))
    else:
        B0 = np.zeros_like(S0)
    # Start the rheology branch near equilibrium to avoid an artificial initial transient.
    Z0 = np.repeat(S0[:, None, :, :], params.n_relax, axis=1) if params.n_relax > 0 else np.zeros((params.n_bodies, 0, 3, 3))
    y0 = pack_state_full(x, v, q, omega, S0, W0, B0 if params.has_split_bulge else None, Z0, 0.0, has_split_bulge=params.has_split_bulge)
    return params, y0


def describe_initial_conditions(params: QuadrupoleParameters, y0: np.ndarray, seed: int = 7) -> str:
    x, v, q, omega, S, W, B, Z, D = unpack_state_full(y0, params.n_bodies, params.n_relax, params.has_split_bulge)
    Q = quat_to_matrix(quat_normalize(q))
    spin_in = np.einsum("aij,aj->ai", Q, omega)
    orbital_r = x[1] - x[0]
    orbital_v = v[1] - v[0]
    h = np.cross(orbital_r, orbital_v)
    hhat = h / max(np.linalg.norm(h), 1e-14)
    evec = np.cross(orbital_v, h) / (params.G * np.sum(params.mass)) - orbital_r / np.linalg.norm(orbital_r)
    e = np.linalg.norm(evec)

    lines = []
    lines.append("=" * 78)
    lines.append("INITIAL CONDITIONS")
    lines.append("=" * 78)
    lines.append(f"Random seed: {seed}")
    lines.append(f"Masses:  {np.array2string(params.mass, precision=4)}")
    lines.append(f"Radii:   {np.array2string(params.radius, precision=4)}")
    lines.append(f"Viscous gamma: {np.array2string(params.gamma, precision=4)}")
    lines.append(f"Mode frequencies omega2: {np.array2string(params.omega2, precision=4)}")
    lines.append(f"Shape-dependent inertia coeff: {np.array2string(params.inertia_shape_coeff, precision=4)}")
    lines.append(f"Legacy rotational-flattening coeff: {np.array2string(params.rot_flattening_coeff, precision=4)}")
    if params.has_split_bulge:
        lines.append(f"Split tidal quadrupole coeff: {np.array2string(params.tidal_quad_coeff, precision=4)}")
        lines.append(f"Split rotational quadrupole coeff: {np.array2string(params.rot_quad_coeff, precision=4)}")
        lines.append(f"Split tidal inertia coeff: {np.array2string(params.tidal_inertia_shape_coeff, precision=4)}")
        lines.append(f"Split rotational inertia coeff: {np.array2string(params.rot_inertia_shape_coeff, precision=4)}")
        lines.append(f"Rotational-bulge response coeff eta: {np.array2string(params.bulge_response_coeff, precision=4)}")
        lines.append(f"Rotational-bulge relax time tau_B: {np.array2string(params.bulge_relax_time, precision=4)}")
    if params.n_relax > 0:
        lines.append(f"Relax strengths: {np.array2string(params.relax_strength, precision=4)}")
        lines.append(f"Relax times:    {np.array2string(params.relax_time, precision=4)}")
    lines.append("")
    lines.append("Initial barycentric positions x0:")
    lines.append(np.array2string(x, precision=5, floatmode="fixed"))
    lines.append("Initial barycentric velocities v0:")
    lines.append(np.array2string(v, precision=5, floatmode="fixed"))
    lines.append(f"Initial relative eccentricity: {e:.6f}")
    lines.append(f"Initial orbit normal: {np.array2string(hhat, precision=5, floatmode='fixed')}")
    lines.append("")
    for i in range(params.n_bodies):
        evals, _ = np.linalg.eigh(stf(S[i]))
        lines.append(f"Planet {i + 1} quaternion q0: {np.array2string(q[i], precision=5, floatmode='fixed')}")
        lines.append(f"Planet {i + 1} body-frame omega0: {np.array2string(omega[i], precision=5, floatmode='fixed')}")
        lines.append(f"Planet {i + 1} inertial spin axis: {np.array2string(spin_in[i] / max(np.linalg.norm(spin_in[i]), 1e-14), precision=5, floatmode='fixed')}")
        lines.append(f"Planet {i + 1} initial STF mode eigenvalues: {np.array2string(evals, precision=5, floatmode='fixed')}")
        lines.append(f"Planet {i + 1} initial ||S||_F = {np.linalg.norm(S[i]):.6f},  ||W||_F = {np.linalg.norm(W[i]):.6f}")
        lines.append("")
    return "\n".join(lines)


# =========================
# Diagnostics / statistics
# =========================


def two_body_diagnostics(record: Dict[str, np.ndarray], params: QuadrupoleParameters) -> Dict[str, np.ndarray]:
    if params.n_bodies != 2:
        raise ValueError("Diagnostics are currently implemented for two bodies.")

    x = record["x"]
    v = record["v"]
    q = record["q"]
    omega_body = record["omega"]
    S_in = record["S_in"]
    t = record["t"]

    G = params.G
    mu = G * np.sum(params.mass)

    r = x[:, 1] - x[:, 0]
    vrel = v[:, 1] - v[:, 0]
    rnorm = np.linalg.norm(r, axis=1)
    h = np.cross(r, vrel)
    hnorm = np.linalg.norm(h, axis=1)
    orbit_normal = normalize_rows(h)

    evec = np.cross(vrel, h) / mu - r / rnorm[:, None]
    ecc = np.linalg.norm(evec, axis=1)

    speed2 = np.sum(vrel * vrel, axis=1)
    inv_a = 2.0 / rnorm - speed2 / mu
    a = np.where(np.abs(inv_a) > 1e-14, 1.0 / inv_a, np.nan)
    mean_motion = np.sqrt(np.maximum(mu / np.maximum(a, 1e-12) ** 3, 0.0))

    Q = quat_to_matrix(q.reshape(-1, 4)).reshape(q.shape[0], q.shape[1], 3, 3)
    spin_in = np.einsum("taij,taj->tai", Q, omega_body)
    spin_rate = np.linalg.norm(spin_in, axis=2)
    spin_axis = normalize_rows(spin_in.reshape(-1, 3)).reshape(spin_in.shape)

    spin_orbit_angle = safe_angle_deg_from_dot(np.einsum("tai,ti->ta", spin_axis, orbit_normal), axis_like=False)
    spin_ratio = spin_rate / mean_motion[:, None]

    # Dominant bulge axis = eigenvector of largest eigenvalue of S_in.
    evals, evecs = np.linalg.eigh(S_in.reshape(-1, 3, 3))
    evals = evals.reshape(S_in.shape[0], S_in.shape[1], 3)
    evecs = evecs.reshape(S_in.shape[0], S_in.shape[1], 3, 3)
    dominant_axis = evecs[:, :, :, 2]
    line_of_centers = normalize_rows(r)
    bulge_line_angle = safe_angle_deg_from_dot(
        np.einsum("tai,ti->ta", dominant_axis, line_of_centers), axis_like=True
    )

    mode_norm = np.linalg.norm(record["S_body"].reshape(record["S_body"].shape[0], record["S_body"].shape[1], -1), axis=2)
    mode_speed_norm = np.linalg.norm(record["W_body"].reshape(record["W_body"].shape[0], record["W_body"].shape[1], -1), axis=2)

    initial_orbit_normal = orbit_normal[0]
    orbit_plane_drift = safe_angle_deg_from_dot(orbit_normal @ initial_orbit_normal, axis_like=True)

    return {
        "t": t,
        "r": r,
        "vrel": vrel,
        "ecc": ecc,
        "a": a,
        "mean_motion": mean_motion,
        "orbit_normal": orbit_normal,
        "orbit_plane_drift_deg": orbit_plane_drift,
        "spin_axis": spin_axis,
        "spin_rate": spin_rate,
        "spin_ratio": spin_ratio,
        "spin_orbit_angle_deg": spin_orbit_angle,
        "bulge_line_angle_deg": bulge_line_angle,
        "mode_norm": mode_norm,
        "mode_speed_norm": mode_speed_norm,
        "evals": evals,
    }


def summarize_series(name: str, values: np.ndarray) -> str:
    return (
        f"{name}: initial={values[0]:.6f}, final={values[-1]:.6f}, "
        f"min={np.min(values):.6f}, max={np.max(values):.6f}, mean={np.mean(values):.6f}"
    )


def format_stats_report(record: Dict[str, np.ndarray], params: QuadrupoleParameters) -> str:
    diag = two_body_diagnostics(record, params)

    lines = []
    lines.append("=" * 78)
    lines.append("POST-SIMULATION TIDAL-EVOLUTION STATISTICS")
    lines.append("=" * 78)
    lines.append(summarize_series("Relative eccentricity", diag["ecc"]))
    lines.append(summarize_series("Orbit-plane drift angle [deg]", diag["orbit_plane_drift_deg"]))
    lines.append(summarize_series("Mechanical energy estimate", record["energy_mech"]))
    lines.append(summarize_series("Cumulative dissipated energy", record["cumulative_dissipation"]))
    lines.append(summarize_series("Mechanical + dissipated energy", record["energy_total_with_dissipation"]))
    lines.append(summarize_series("Conserved-energy drift", record["energy_total_drift"]))
    lines.append("")

    for i in range(params.n_bodies):
        lines.append(f"Planet {i + 1}")
        lines.append("-" * 78)
        lines.append(summarize_series("Spin rate / mean motion", diag["spin_ratio"][:, i]))
        lines.append(summarize_series("Spin axis vs orbit normal angle [deg]", diag["spin_orbit_angle_deg"][:, i]))
        lines.append(summarize_series("Bulge axis vs line-of-centers angle [deg]", diag["bulge_line_angle_deg"][:, i]))
        lines.append(summarize_series("Quadrupole mode norm ||S||_F", diag["mode_norm"][:, i]))
        lines.append(summarize_series("Quadrupole mode speed ||W||_F", diag["mode_speed_norm"][:, i]))
        lines.append(
            summarize_series("Largest quadrupole eigenvalue", diag["evals"][:, i, 2])
        )
        lines.append(
            summarize_series("Most negative quadrupole eigenvalue", diag["evals"][:, i, 0])
        )
        # Axial migration of the spin direction.
        spin_axis = diag["spin_axis"][:, i]
        spin_axis0 = spin_axis[0]
        spin_axis_drift = safe_angle_deg_from_dot(spin_axis @ spin_axis0, axis_like=True)
        lines.append(summarize_series("Spin-axis drift from initial [deg]", spin_axis_drift))
        lines.append("")

    return "\n".join(lines)


# =========================
# Matplotlib diagnostics plot
# =========================


def make_diagnostics_plot(record: Dict[str, np.ndarray], params: QuadrupoleParameters, filename: str = "quadrupole_tidal_diagnostics.png"):
    """Create a single matplotlib figure with the requested tidal-evolution diagnostics."""
    np.savez_compressed('tidal_cache.npz', record=record, params=params)

    diag = two_body_diagnostics(record, params)
    t = diag["t"]

    energy_dissipated = record.get("cumulative_dissipation_from_rate", record["cumulative_dissipation"])
    mech_energy = record["energy_mech"]
    total_energy = mech_energy + energy_dissipated
    eccentricity = diag["ecc"]
    semi_major_axis = diag["a"]
    spin_rate = diag["spin_rate"]
    mean_motion = diag["mean_motion"]

    W_norm_sq = np.sum(record["W_body"] ** 2, axis=(2, 3))
    S_for_lag = record["S_tidal_body"] if "S_tidal_body" in record else record["S_body"]
    if params.n_relax > 0:
        delta_SZ = S_for_lag[:, :, None, :, :] - record["Z_body"]
        SZ_lag_norm_sq = np.sum(delta_SZ ** 2, axis=(2, 3, 4))
    else:
        SZ_lag_norm_sq = np.zeros_like(W_norm_sq)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    ax = axes[0, 0]
    mech_line, = ax.plot(t, mech_energy, lw=2.0, color="tab:green", ls="--", label="Mechanical energy")
    total_line, = ax.plot(t, total_energy, lw=2.0, color="tab:red", label="Mechanical + dissipated")
    ax.set_title("Energy bookkeeping")
    ax.set_xlabel("time")
    ax.set_ylabel("mechanical / total energy")
    ax.grid(True, alpha=0.3)

    ax_d = ax.twinx()
    diss_line, = ax_d.plot(t, energy_dissipated, lw=2.0, color="black", label="Cumulative dissipated")
    ax_d.set_ylabel("cumulative dissipated")

    ax.legend(
        [mech_line, diss_line, total_line],
        [mech_line.get_label(), diss_line.get_label(), total_line.get_label()],
        loc="center right",
    )

    ax = axes[0, 1]
    for i, color in enumerate(params.colors[:params.n_bodies]):
        ax.plot(t, W_norm_sq[:, i], lw=2.0, color=color, label=fr"Planet {i + 1} $\|W\|_F^2$")
        ax.plot(t, SZ_lag_norm_sq[:, i], lw=2.0, color=color, ls="--", label=fr"Planet {i + 1} $\|S-Z\|_F^2$")
    ax.set_title(r"Internal quadrupole activity: $\|W\|_F^2$ and $\|S-Z\|_F^2$")
    ax.set_xlabel("time")
    ax.set_ylabel("squared Frobenius norm")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ecc_line, = ax.plot(t, eccentricity, lw=2.0, color="purple", label="eccentricity")
    ax.set_title("Orbital eccentricity")
    ax.set_xlabel("time")
    ax.set_ylabel("e")
    ax.grid(True, alpha=0.3)

    ax_a = ax.twinx()
    a_line, = ax_a.plot(t, semi_major_axis, lw=2.0, color="gray", ls="--", label="semi-major axis")
    ax_a.set_ylabel("a")
    ax.legend([ecc_line, a_line], [ecc_line.get_label(), a_line.get_label()], loc="upper right")

    ax = axes[1, 1]
    for i, color in enumerate(params.colors[:params.n_bodies]):
        ax.plot(t, spin_rate[:, i], lw=2.0, color=color, label=f"Planet {i + 1}")
    ax.plot(t, mean_motion, lw=2.0, color="black", ls="--", label="Orbital frequency")
    ax.set_title("Rotation angular frequency")
    ax.set_xlabel("time")
    ax.set_ylabel(r"$|\Omega|$")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle("Quadrupole tidal model diagnostics", fontsize=14)
    # fig.savefig(filename, dpi=300, bbox_inches='tight')
    # fig.clear()
    plt.show()
    return fig, filename


# =========================
# Visualization
# =========================


def sphere_directions(n_theta: int = 34, n_phi: int = 68):
    theta = np.linspace(0.0, np.pi, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi)
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    dirs = np.stack([
        np.sin(th) * np.cos(ph),
        np.sin(th) * np.sin(ph),
        np.cos(th),
    ], axis=-1)
    return th, ph, dirs


def deformed_surface(center: np.ndarray, radius: float, S_in: np.ndarray, dirs: np.ndarray, min_factor: float = 0.55):
    quad = np.einsum("...i,ij,...j->...", dirs, S_in, dirs)
    rr = radius * (1.0 + quad)
    rr = np.maximum(rr, min_factor * radius)
    xyz = center[None, None, :] + rr[..., None] * dirs
    return xyz[..., 0], xyz[..., 1], xyz[..., 2]


def constant_surface_trace(x, y, z, color: str, name: str):
    zero = np.zeros_like(x)
    return go.Surface(
        x=x,
        y=y,
        z=z,
        surfacecolor=zero,
        colorscale=[[0.0, color], [1.0, color]],
        cmin=0.0,
        cmax=1.0,
        showscale=False,
        opacity=1.0,
        name=name,
        hoverinfo="skip",
        lighting=dict(ambient=0.8, diffuse=0.7, specular=0.1, roughness=0.95, fresnel=0.02),
        lightposition=dict(x=500, y=350, z=700),
        contours={"x": {"show": False}, "y": {"show": False}, "z": {"show": False}},
    )


# =========================
# End-to-end run helpers
# =========================


def save_demo_outputs(
    stats_filename: str = "quadrupole_tidal_stats.txt",
    diagnostics_filename: str = "quadrupole_tidal_diagnostics.png",
    seed: int = 7
):
    params, y0 = build_demo_problem(seed=seed)
    initial_text = describe_initial_conditions(params, y0, seed=seed)
    print(initial_text)
    print("Running solve_ivp() ...")
    sol = simulate(params, y0, t_span=(0.0, 60 * 90.0), n_samples=700 * 30, method="DOP853")
    record = record_solution(sol, params)
    report = format_stats_report(record, params)
    print(report)

    with open(stats_filename, "w", encoding="utf-8") as f:
        f.write(initial_text)
        f.write("\n\n")
        f.write(report)
        f.write("\n")

    diag_fig, diagnostics_filename = make_diagnostics_plot(record, params, filename=diagnostics_filename)
    plt.close(diag_fig)

    return stats_filename, diagnostics_filename


def main():
    stats_name, diagnostics_name = save_demo_outputs(
        stats_filename="quadrupole_tidal_stats.txt",
        diagnostics_filename="quadrupole_tidal_diagnostics.png",
        seed=7,
    )
    print(f"Saved statistics report to {stats_name}")
    print(f"Saved matplotlib diagnostics plot to {diagnostics_name}")


def run_tests(output_dir="quadrupole_test_outputs", show_plots=False, quick=False):
    """
    Run a suite of synthetic validation tests for the quadrupole tidal model.

    Parameters
    ----------
    output_dir : str
        Directory where diagnostic PNG files will be saved.
    show_plots : bool
        If False, suppress interactive plt.show() while still saving figures.
    quick : bool
        If True, shorten integrations for a faster smoke-test version.

    Returns
    -------
    dict
        Summary dictionary with pass/fail results.
    """
    os.makedirs(output_dir, exist_ok=True)

    def _scaled(t_end, n_samples):
        if quick:
            return max(20.0, 0.45 * float(t_end)), max(240, int(0.45 * int(n_samples)))
        return float(t_end), int(n_samples)

    def _mean_motion(a, total_mass=2.0, G=1.0):
        return np.sqrt(G * total_mass / a ** 3)

    def _sph(theta_deg, phi_deg=0.0, mag=1.0):
        th = np.deg2rad(theta_deg)
        ph = np.deg2rad(phi_deg)
        return np.array([
            mag * np.sin(th) * np.cos(ph),
            mag * np.sin(th) * np.sin(ph),
            mag * np.cos(th),
        ], dtype=float)

    def _make_two_body_case(
        *,
        G=1.0,
        mass=(1.0, 1.0),
        radius=(0.25, 0.25),
        a=3.0,
        e=0.0,
        f=0.0,
        orbital_rotation=None,
        spin_vecs=((0.0, 0.0, 0.5), (0.0, 0.0, 0.5)),
        q=((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        S0=None,
        W0=None,
        extra_stiffness=(0.3, 0.3),
        gamma=(0.0, 0.0),
        relax_strength=None,
        relax_time=None,
        rot_flattening_coeff=(0.0, 0.0),
        colors=("royalblue", "orangered"),
    ):
        mass_arr = np.asarray(mass, dtype=float)
        radius_arr = np.asarray(radius, dtype=float)

        total_mass = float(np.sum(mass_arr))
        mu = G * total_mass
        p = a * (1.0 - e * e)
        r_mag = p / (1.0 + e * np.cos(f))

        r_pf = np.array([r_mag * np.cos(f), r_mag * np.sin(f), 0.0], dtype=float)
        v_pf = np.sqrt(mu / p) * np.array([-np.sin(f), e + np.cos(f), 0.0], dtype=float)

        if orbital_rotation is None:
            orbital_rotation = np.eye(3)
        orbital_rotation = np.asarray(orbital_rotation, dtype=float)

        r_rel = orbital_rotation @ r_pf
        v_rel = orbital_rotation @ v_pf

        x = np.array([
            -(mass_arr[1] / total_mass) * r_rel,
            +(mass_arr[0] / total_mass) * r_rel,
        ], dtype=float)
        v = np.array([
            -(mass_arr[1] / total_mass) * v_rel,
            +(mass_arr[0] / total_mass) * v_rel,
        ], dtype=float)

        q_arr = quat_normalize(np.asarray(q, dtype=float))
        omega = np.asarray(spin_vecs, dtype=float)

        if S0 is None:
            S0 = np.zeros((2, 3, 3), dtype=float)
        if W0 is None:
            W0 = np.zeros((2, 3, 3), dtype=float)
        S0 = stf(np.asarray(S0, dtype=float))
        W0 = stf(np.asarray(W0, dtype=float))

        J0 = 0.4 * mass_arr * radius_arr ** 2
        omega_grav_sq = (4.0 / 5.0) * G * mass_arr / radius_arr ** 3
        omega2 = np.sqrt(omega_grav_sq + np.asarray(extra_stiffness, dtype=float) ** 2)
        quad_coeff = 0.4 * mass_arr * radius_arr ** 2
        inertia_shape_coeff = quad_coeff.copy()

        if relax_strength is None:
            relax_strength_arr = np.zeros((2, 0), dtype=float)
            relax_time_arr = np.zeros((2, 0), dtype=float)
        else:
            relax_strength_arr = np.asarray(relax_strength, dtype=float)
            relax_time_arr = np.asarray(relax_time, dtype=float)

        gamma_arr = np.asarray(gamma, dtype=float)
        rot_flattening_arr = np.asarray(rot_flattening_coeff, dtype=float)
        rot_inertia_shape_coeff = inertia_shape_coeff + 2.0 * rot_flattening_arr
        bulge_response_coeff = 0.5 * (inertia_shape_coeff + 2.0 * rot_flattening_arr) / np.maximum(omega2 ** 2, 1e-14)
        # Match the low-frequency phase lag of the original second-order closure:
        #   D_t^2 S + 2 gamma D_t S + omega2^2 S = forcing
        # has small-forcing lag ~ nu * (2 gamma / omega2^2), so the first-order
        # rotational bulge is calibrated with tau_B = 2 gamma / omega2^2.
        bulge_relax_time = 2.0 * gamma_arr / np.maximum(omega2 ** 2, 1e-14)

        params = QuadrupoleParameters(
            G=G,
            mass=mass_arr,
            radius=radius_arr,
            gamma=gamma_arr,
            omega2=omega2,
            J0=J0,
            quad_coeff=quad_coeff,
            inertia_shape_coeff=inertia_shape_coeff,
            rot_flattening_coeff=rot_flattening_arr,
            relax_strength=relax_strength_arr,
            relax_time=relax_time_arr,
            tidal_quad_coeff=quad_coeff,
            rot_quad_coeff=np.where(rot_flattening_arr != 0.0, quad_coeff, 0.0),
            tidal_inertia_shape_coeff=np.zeros_like(inertia_shape_coeff),
            rot_inertia_shape_coeff=np.where(rot_flattening_arr != 0.0, rot_inertia_shape_coeff, 0.0),
            bulge_response_coeff=np.where(rot_flattening_arr != 0.0, bulge_response_coeff, 0.0),
            bulge_relax_time=bulge_relax_time,
            colors=colors,
        )

        if params.has_split_bulge:
            _, E0_body, _, _, _ = compute_fields_and_forces(x, quat_normalize(q_arr), omega, S0, params, np.zeros_like(S0))
            B0, _, _ = split_bulge_target_and_effective(omega, E0_body, params, np.zeros_like(S0))
        else:
            B0 = np.zeros_like(S0)
        if params.n_relax > 0:
            Z0 = np.repeat(S0[:, None, :, :], params.n_relax, axis=1)
        else:
            Z0 = np.zeros((params.n_bodies, 0, 3, 3), dtype=float)

        y0 = pack_state_full(x, v, q_arr, omega, S0, W0, B0 if params.has_split_bulge else None, Z0, 0.0, has_split_bulge=params.has_split_bulge)
        return params, y0

    def _equilibrium_S(x, q, omega, params):
        zero_S = np.zeros((params.n_bodies, 3, 3), dtype=float)
        _, E_body, _, _, _ = compute_fields_and_forces(x, quat_normalize(q), omega, zero_S, params, np.zeros_like(zero_S))
        tidal_term = 0.5 * params.tidal_quad_coeff[:, None, None] * E_body
        if params.has_split_bulge:
            return -tidal_term / (params.omega2[:, None, None] ** 2)
        spin_term = 0.5 * effective_shape_spin_coeff(params)[:, None, None] * spin_flattening_tensor(omega)
        return -(tidal_term + spin_term) / (params.omega2[:, None, None] ** 2)

    def _run_case(name, params, y0, *, t_end, n_samples, rtol=1e-8, atol=1e-10, make_plot=True):
        t_end, n_samples = _scaled(t_end, n_samples)

        print("\n" + "=" * 88)
        print(f"RUNNING TEST: {name}")
        print("=" * 88)

        sol = simulate(
            params,
            y0,
            t_span=(0.0, t_end),
            n_samples=n_samples,
            rtol=rtol,
            atol=atol,
            method="DOP853",
        )
        record = record_solution(sol, params)
        diag = two_body_diagnostics(record, params)

        plot_path = None
        if make_plot:
            plot_path = os.path.join(output_dir, f"{name}.jpg")

            old_show = plt.show
            try:
                if not show_plots:
                    plt.show = lambda *args, **kwargs: None
                fig, _ = make_diagnostics_plot(record, params, filename=plot_path)
            finally:
                plt.show = old_show

            try:
                fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            finally:
                if not show_plots:
                    plt.close(fig)

        return record, diag, plot_path

    def _report_case(name, checks, plot_path=None, extra_lines=None):
        passed = True
        print(f"[{name}] criteria")
        for label, ok, value in checks:
            passed = passed and bool(ok)
            status = "PASS" if ok else "FAIL"
            print(f"  {status:4s}  {label}: {value}")

        if extra_lines:
            for line in extra_lines:
                print(f"         {line}")

        if plot_path is not None:
            print(f"         plot: {plot_path}")

        print(f"[{name}] OVERALL: {'PASS' if passed else 'FAIL'}")
        return passed

    results = []

    params, y0 = _make_two_body_case(
        a=3.5,
        e=0.30,
        radius=(0.30, 0.30),
        spin_vecs=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        gamma=(0.0, 0.0),
    )
    params.quad_coeff[:] = 0.0
    params.inertia_shape_coeff[:] = 0.0
    params.rot_flattening_coeff[:] = 0.0

    record, diag, plot_path = _run_case(
        "01_null_kepler_limit",
        params,
        y0,
        t_end=120.0,
        n_samples=600,
        rtol=1e-10,
        atol=1e-12,
    )

    e_drift = np.max(np.abs(diag["ecc"] - diag["ecc"][0]))
    a_drift = np.max(np.abs(diag["a"] - diag["a"][0]))
    energy_drift = np.max(np.abs(record["energy_total_drift"]))

    passed = _report_case(
        "01_null_kepler_limit",
        [
            ("max |Δe| < 5e-6", e_drift < 5e-6, f"{e_drift:.3e}"),
            ("max |Δa| < 5e-6", a_drift < 5e-6, f"{a_drift:.3e}"),
            ("max |ΔE_total| < 5e-7", energy_drift < 5e-7, f"{energy_drift:.3e}"),
        ],
        plot_path,
    )
    results.append(("01_null_kepler_limit", passed))

    a = 4.0
    n = _mean_motion(a)

    params, y0 = _make_two_body_case(
        a=a,
        e=0.0,
        radius=(0.25, 0.25),
        spin_vecs=((0.0, 0.0, n), (0.0, 0.0, n)),
        gamma=(0.0, 0.0),
        rot_flattening_coeff=(0.0, 0.0),
    )

    x, v, q, omega, S, W, B, Z, D = unpack_state_full(y0, params.n_bodies, params.n_relax, params.has_split_bulge)
    S_eq = _equilibrium_S(x, q, omega, params)
    y0 = pack_state_full(x, v, q, omega, S_eq, np.zeros_like(S_eq), B if params.has_split_bulge else None, Z, 0.0, has_split_bulge=params.has_split_bulge)

    record, diag, plot_path = _run_case(
        "02_conservative_equilibrium",
        params,
        y0,
        t_end=180.0,
        n_samples=700,
        rtol=1e-9,
        atol=1e-11,
    )

    max_energy_drift = np.max(np.abs(record["energy_total_drift"]))
    max_ecc = np.max(diag["ecc"])
    mean_bulge_angle = float(np.mean(diag["bulge_line_angle_deg"]))

    passed = _report_case(
        "02_conservative_equilibrium",
        [
            ("max |ΔE_total| < 2e-5", max_energy_drift < 2e-5, f"{max_energy_drift:.3e}"),
            ("max eccentricity < 2e-3", max_ecc < 2e-3, f"{max_ecc:.3e}"),
            ("mean bulge-line angle < 2 deg", mean_bulge_angle < 2.0, f"{mean_bulge_angle:.3f} deg"),
        ],
        plot_path,
    )
    results.append(("02_conservative_equilibrium", passed))

    params, y0 = _make_two_body_case(
        a=8.0,
        e=0.0,
        radius=(0.25, 0.25),
        spin_vecs=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        gamma=(2.0, 2.0),
    )

    record, diag, plot_path = _run_case(
        "03_quasi_static_bulge",
        params,
        y0,
        t_end=120.0,
        n_samples=800,
        rtol=1e-9,
        atol=1e-11,
    )

    errs = []
    for k in range(record["t"].size):
        S_eq = _equilibrium_S(record["x"][k], record["q"][k], record["omega"][k], params)
        denom = max(np.linalg.norm(S_eq), 1e-30)
        errs.append(np.linalg.norm(record["S_body"][k] - S_eq) / denom)
    errs = np.asarray(errs)

    mask = record["t"] > 0.25 * record["t"][-1]
    mean_err = float(np.mean(errs[mask]))
    final_err = float(errs[-1])
    mean_bulge = float(np.mean(diag["bulge_line_angle_deg"]))

    passed = _report_case(
        "03_quasi_static_bulge",
        [
            ("mean relative shape error < 0.20", mean_err < 0.20, f"{mean_err:.3e}"),
            ("final relative shape error < 0.10", final_err < 0.10, f"{final_err:.3e}"),
            ("mean bulge-line angle < 5 deg", mean_bulge < 5.0, f"{mean_bulge:.3f} deg"),
        ],
        plot_path,
    )
    results.append(("03_quasi_static_bulge", passed))

    params, y0 = _make_two_body_case(
        a=100.0,
        e=0.0,
        mass=(1.0, 1e-6),
        radius=(0.50, 0.05),
        spin_vecs=((0.0, 0.0, 1.5), (0.0, 0.0, 0.0)),
        gamma=(1.0, 0.0),
        rot_flattening_coeff=(0.25, 0.0),
    )

    record, diag, plot_path = _run_case(
        "04_rotational_flattening",
        params,
        y0,
        t_end=30.0,
        n_samples=500,
        rtol=1e-9,
        atol=1e-11,
    )

    S1 = record["S_body"][-1, 0]
    omega1 = record["omega"][-1, 0][None, :]
    c_eff = effective_shape_spin_coeff(params)[0]
    S_rot = -(0.5 * c_eff * spin_flattening_tensor(omega1)[0]) / (params.omega2[0] ** 2)

    rel_err = np.linalg.norm(S1 - S_rot) / max(np.linalg.norm(S_rot), 1e-30)
    evals = np.linalg.eigvalsh(S1)
    oblate_signature = bool((evals[0] < 0.0) and (evals[2] > 0.0))

    passed = _report_case(
        "04_rotational_flattening",
        [
            ("relative error to isolated rotational equilibrium < 0.10", rel_err < 0.10, f"{rel_err:.3e}"),
            ("oblate eigenvalue signature present", oblate_signature, np.array2string(evals, precision=4)),
        ],
        plot_path,
    )
    results.append(("04_rotational_flattening", passed))

    a = 2.4
    n = _mean_motion(a)

    common_kwargs = dict(
        a=a,
        e=0.12,
        radius=(0.43, 0.35),
        spin_vecs=((0.0, 0.0, 3.5 * n), (0.0, 0.0, 1.0 * n)),
        rot_flattening_coeff=(0.16, 0.10),
    )

    params_A, y0_A = _make_two_body_case(
        **common_kwargs,
        gamma=(0.6, 0.0),
    )
    record_A, diag_A, plot_A = _run_case(
        "05A_rheology_viscous",
        params_A,
        y0_A,
        t_end=120.0,
        n_samples=700,
    )

    params_B, y0_B = _make_two_body_case(
        **common_kwargs,
        gamma=(0.0, 0.0),
        relax_strength=((0.8,), (0.0,)),
        relax_time=((2.0,), (1.0,)),
    )
    record_B, diag_B, plot_B = _run_case(
        "05B_rheology_maxwell",
        params_B,
        y0_B,
        t_end=120.0,
        n_samples=700,
    )

    lag_A = float(np.mean(diag_A["bulge_line_angle_deg"][:, 0]))
    lag_B = float(np.mean(diag_B["bulge_line_angle_deg"][:, 0]))
    diss_A = float(record_A["cumulative_dissipation"][-1])
    diss_B = float(record_B["cumulative_dissipation"][-1])

    lag_diff = abs(lag_A - lag_B)
    diss_rel_diff = abs(diss_A - diss_B) / max(abs(diss_A), abs(diss_B), 1e-12)

    passed = _report_case(
        "05_rheology_comparison",
        [
            ("|mean lag difference| > 0.5 deg", lag_diff > 0.5, f"{lag_diff:.3f} deg"),
            ("relative dissipation difference > 5%", diss_rel_diff > 0.05, f"{100.0 * diss_rel_diff:.2f} %"),
        ],
        extra_lines=[
            f"viscous plot: {plot_A}",
            f"maxwell plot: {plot_B}",
            f"viscous mean lag = {lag_A:.3f} deg, maxwell mean lag = {lag_B:.3f} deg",
            f"viscous final D = {diss_A:.6e}, maxwell final D = {diss_B:.6e}",
        ],
    )
    results.append(("05_rheology_comparison", passed))

    a = 2.6
    n = _mean_motion(a)

    params, y0 = _make_two_body_case(
        a=a,
        e=0.45,
        radius=(0.45, 0.40),
        spin_vecs=((0.0, 0.0, 1.8 * n), (0.0, 0.0, 0.6 * n)),
        gamma=(0.35, 0.45),
        relax_strength=((0.5,), (0.3,)),
        relax_time=((2.5,), (1.6,)),
        rot_flattening_coeff=(0.18, 0.14),
    )

    record, diag, plot_path = _run_case(
        "06_circularization",
        params,
        y0,
        t_end=180.0,
        n_samples=900,
    )

    e0, ef = float(diag["ecc"][0]), float(diag["ecc"][-1])
    a0, af = float(diag["a"][0]), float(diag["a"][-1])
    energy_rel_drift = np.max(np.abs(record["energy_total_drift"])) / max(abs(record["energy_total_with_dissipation"][0]), 1e-12)

    passed = _report_case(
        "06_circularization",
        [
            ("final eccentricity < initial eccentricity", ef < e0, f"e0={e0:.4f}, ef={ef:.4f}"),
            ("final semi-major axis < initial semi-major axis", af < a0, f"a0={a0:.4f}, af={af:.4f}"),
            ("relative total-energy drift < 2%", energy_rel_drift < 2.0e-2, f"{100.0 * energy_rel_drift:.3f} %"),
        ],
        plot_path,
    )
    results.append(("06_circularization", passed))

    a = 2.0
    n = _mean_motion(a)

    params, y0 = _make_two_body_case(
        a=a,
        e=0.0,
        radius=(0.45, 0.45),
        spin_vecs=((0.0, 0.0, 4.0 * n), (0.0, 0.0, 1.0 * n)),
        gamma=(0.3, 0.3),
        rot_flattening_coeff=(0.20, 0.20),
    )

    record, diag, plot_path = _run_case(
        "07_spin_synchronization",
        params,
        y0,
        t_end=180.0,
        n_samples=900,
    )

    sr0 = float(diag["spin_ratio"][0, 0])
    srf = float(diag["spin_ratio"][-1, 0])
    min_sr = float(np.min(diag["spin_ratio"][:, 0]))
    mean_lag = float(np.mean(diag["bulge_line_angle_deg"][:, 0]))

    passed = _report_case(
        "07_spin_synchronization",
        [
            ("spin ratio decreases", srf < sr0, f"initial={sr0:.4f}, final={srf:.4f}"),
            ("final spin ratio is closer to 1 than initial", abs(srf - 1.0) < abs(sr0 - 1.0),
             f"|final-1|={abs(srf - 1.0):.4f}, |initial-1|={abs(sr0 - 1.0):.4f}"),
            ("mean bulge lag > 0.2 deg", mean_lag > 0.2, f"{mean_lag:.3f} deg"),
            ("minimum spin ratio is below the initial one", min_sr < sr0, f"min={min_sr:.4f}"),
        ],
        plot_path,
    )
    results.append(("07_spin_synchronization", passed))

    representative_plot = None
    survey = []

    for tilt in (15.0, 45.0, 75.0):
        for gam in (0.10, 0.30, 0.60):
            params, y0 = _make_two_body_case(
                a=2.5,
                e=0.15,
                radius=(0.42, 0.38),
                spin_vecs=(
                    _sph(tilt, phi_deg=0.0, mag=1.8 * _mean_motion(2.5)),
                    _sph(0.5 * tilt, phi_deg=180.0, mag=0.9 * _mean_motion(2.5)),
                ),
                gamma=(gam, gam),
                rot_flattening_coeff=(0.18, 0.14),
            )

            make_plot = (tilt == 45.0 and abs(gam - 0.30) < 1e-12)

            record, diag, plot_path = _run_case(
                f"08_obliquity_tilt_{int(tilt):02d}_gamma_{int(round(100 * gam)):02d}",
                params,
                y0,
                t_end=120.0,
                n_samples=600,
                make_plot=make_plot,
            )

            dtheta1 = float(diag["spin_orbit_angle_deg"][-1, 0] - diag["spin_orbit_angle_deg"][0, 0])
            survey.append((tilt, gam, dtheta1))

            if make_plot:
                representative_plot = plot_path

    dtheta_vals = np.array([row[2] for row in survey], dtype=float)
    frac_negative = np.mean(dtheta_vals < 0.0)
    median_dtheta = float(np.median(dtheta_vals))

    passed = _report_case(
        "08_obliquity_survey",
        [
            ("at least half the survey runs damp obliquity", frac_negative >= 0.5, f"fraction={frac_negative:.3f}"),
            ("median Δ(theta_spin-orbit) < 0", median_dtheta < 0.0, f"median={median_dtheta:.4f} deg"),
        ],
        representative_plot,
        extra_lines=[
            "survey grid = tilts {15,45,75} deg × gamma {0.10,0.30,0.60}",
            "Δ(theta) < 0 means net obliquity damping over the run",
        ],
    )
    results.append(("08_obliquity_survey", passed))

    print("\n" + "#" * 88)
    print("TEST SUMMARY")
    print("#" * 88)
    n_pass = 0
    for name, passed in results:
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
        n_pass += int(bool(passed))
    print(f"Passed {n_pass} / {len(results)} tests.")
    print(f"Plots saved in: {output_dir}")

    return {
        "results": results,
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    # run_tests()

    # with np.load(r'tidal_cache.npz', allow_pickle=True) as dat:
    #     record = dat['record'].item()
    #     params = dat['params'].item()
    #
    # make_diagnostics_plot(record, params)
    # exit()

    main()
