from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
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

    @property
    def n_bodies(self) -> int:
        return int(self.mass.size)

    @property
    def n_relax(self) -> int:
        if self.relax_strength.ndim != 2:
            return 0
        return int(self.relax_strength.shape[1])


# =========================
# Packing / unpacking state
# =========================


def pack_state(x, v, q, omega, S, W, Z=None, D=None) -> np.ndarray:
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


def unpack_state(y: np.ndarray, n: int, n_relax: int = 0):
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


def current_body_inertia(S_body: np.ndarray, params: QuadrupoleParameters) -> np.ndarray:
    """Linearized body-frame spin inertia, including rotational flattening consistently."""
    eye = np.eye(3)
    S_body = stf(S_body)
    coeff = effective_shape_spin_coeff(params)
    J_body = params.J0[:, None, None] * eye[None, :, :] - coeff[:, None, None] * S_body
    return 0.5 * (J_body + np.swapaxes(J_body, -1, -2))



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



def compute_fields_and_forces(
    x: np.ndarray,
    q: np.ndarray,
    omega: np.ndarray,
    S_body: np.ndarray,
    params: QuadrupoleParameters,
):
    """Compute accelerations, body-frame tides, diagnostic torques, quadrupoles, and pair potential."""
    N = params.n_bodies
    G = params.G
    m = params.mass

    qn = quat_normalize(q)
    Q = quat_to_matrix(qn)

    S_body = stf(S_body)
    S_in = np.einsum("aik,akl,ajl->aij", Q, S_body, Q)
    I_in = params.quad_coeff[:, None, None] * S_in

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
    # torque_body is returned for diagnostics only.  The dynamical coupling of the tidal
    # potential to orientation is already represented by the commutator term in S_dot.

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
    x, v, q, omega, S, W, Z, D = unpack_state(y, N, params.n_relax)

    qn = quat_normalize(q)
    S = stf(S)
    W = stf(W)
    if params.n_relax > 0:
        Z = stf(Z)

    accel, E_body, _torque_body_diag, _, _ = compute_fields_and_forces(x, qn, omega, S, params)

    Om_hat = hat(omega)
    comm_OM_S = Om_hat @ S - S @ Om_hat
    comm_OM_W = Om_hat @ W - W @ Om_hat

    # First-order form with W = D_t S = S_dot + [Omega_hat, S]
    S_dot = W - comm_OM_S

    spin_force = 0.5 * effective_shape_spin_coeff(params)[:, None, None] * spin_flattening_tensor(omega)

    if params.n_relax > 0:
        comm_OM_Z = Om_hat[:, None, :, :] @ Z - Z @ Om_hat[:, None, :, :]
        Z_dot = (S[:, None, :, :] - Z) / params.relax_time[:, :, None, None] - comm_OM_Z
        relax_force = np.sum(params.relax_strength[:, :, None, None] * (S[:, None, :, :] - Z), axis=1)
    else:
        Z_dot = np.zeros((N, 0, 3, 3), dtype=float)
        relax_force = np.zeros_like(S)

    x_dot = v
    v_dot = accel
    q_dot = quat_derivative_body_to_inertial(qn, omega)

    # Energy-consistent tidal coupling:
    # the orbital force/potential uses I_in = quad_coeff * S_in, so the generalized
    # force on the internal quadrupole coordinate S must use the reciprocal coefficient
    # derived from the same interaction energy U_tide = 0.5 * I:E.  Using bare E_body here
    # breaks action-reaction in the energy budget and produces spurious orbital-frequency
    # oscillations in the supposed total energy diagnostic.
    tidal_force = 0.5 * params.quad_coeff[:, None, None] * E_body

    W_dot = (
        -comm_OM_W
        - 2.0 * params.gamma[:, None, None] * W
        - params.omega2[:, None, None] ** 2 * S
        - relax_force
        - tidal_force
        - spin_force
    )

    J_body = current_body_inertia(S, params)
    J_dot = -effective_shape_spin_coeff(params)[:, None, None] * S_dot
    L_spin = np.einsum("aij,aj->ai", J_body, omega)
    # Do not add the external tidal torque here.
    #
    # The quadrupole interaction U(I,E) is already coupled to the orientation through
    # the kinematics S_dot = W - [Omega_hat, S].  The commutator piece carries the same
    # conservative orientational coupling that torque_body would represent.  Keeping both
    # terms double counts the q/S rotational interaction and produces a secular drift in
    # mechanical energy + cumulative dissipation.
    spin_rhs = -np.einsum("aij,aj->ai", J_dot, omega) - np.cross(omega, L_spin)
    omega_dot = np.linalg.solve(J_body, spin_rhs[..., None])[..., 0]
    D_dot = dissipation_rate_from_state(W, S, Z, params)

    return pack_state(x_dot, v_dot, q_dot, omega_dot, S_dot, W_dot, Z_dot, D_dot)


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
    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y, params),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        method=method,
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
    Z_body = np.empty((T, N, params.n_relax, 3, 3)) if params.n_relax > 0 else np.zeros((T, N, 0, 3, 3))
    S_in = np.empty((T, N, 3, 3))
    I_in = np.empty((T, N, 3, 3))
    J_body = np.empty((T, N, 3, 3))
    energy_orb_kin = np.empty(T)
    energy_spin_kin = np.empty(T)
    energy_shape_kin = np.empty(T)
    energy_shape_pot = np.empty(T)
    energy_relax = np.empty(T)
    energy_grav_pot = np.empty(T)
    energy_spin_shape = np.empty(T)
    energy_mech = np.empty(T)

    for k in range(T):
        xk, vk, qk, omegak, Sk, Wk, Zk, Dk = unpack_state(sol.y[:, k], N, params.n_relax)
        qk = quat_normalize(qk)
        Sk = stf(Sk)
        Wk = stf(Wk)
        if params.n_relax > 0:
            Zk = stf(Zk)

        Qk = quat_to_matrix(qk)
        S_ink = np.einsum("aik,akl,ajl->aij", Qk, Sk, Qk)
        _, _, _, I_ink, U_pair = compute_fields_and_forces(xk, qk, omegak, Sk, params)
        Jk = current_body_inertia(Sk, params)

        kinetic_orb = 0.5 * np.sum(params.mass[:, None] * vk * vk)
        kinetic_spin = 0.5 * np.sum(omegak * np.einsum("aij,aj->ai", Jk, omegak))
        shape_kin = 0.5 * np.sum(Wk * Wk)
        shape_pot = 0.5 * np.sum((params.omega2[:, None, None] ** 2) * Sk * Sk)
        spin_shape_energy = 0.0
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
        energy_mech[k] = kinetic_orb + kinetic_spin + shape_kin + shape_pot + branch_energy + spin_shape_energy + U_pair

        x[k] = xk
        v[k] = vk
        q[k] = qk
        omega[k] = omegak
        S_body[k] = Sk
        W_body[k] = Wk
        if params.n_relax > 0:
            Z_body[k] = Zk
        S_in[k] = S_ink
        I_in[k] = I_ink
        J_body[k] = Jk

    dissipation_rate = np.array([
        dissipation_rate_from_state(W_body[k], S_body[k], Z_body[k], params) for k in range(T)
    ])
    cumulative_dissipation = np.array([
        unpack_state(sol.y[:, k], N, params.n_relax)[-1] for k in range(T)
    ])

    total_energy_with_dissipation = energy_mech + cumulative_dissipation
    total_energy_drift = total_energy_with_dissipation - total_energy_with_dissipation[0]

    return {
        "t": sol.t.copy(),
        "x": x,
        "v": v,
        "q": q,
        "omega": omega,
        "S_body": S_body,
        "W_body": W_body,
        "Z_body": Z_body,
        "S_in": S_in,
        "I_in": I_in,
        "J_body": J_body,
        "energy_orb_kin": energy_orb_kin,
        "energy_spin_kin": energy_spin_kin,
        "energy_shape_kin": energy_shape_kin,
        "energy_shape_pot": energy_shape_pot,
        "energy_relax": energy_relax,
        "energy_grav_pot": energy_grav_pot,
        "energy_spin_shape": energy_spin_shape,
        "energy_mech": energy_mech,
        "energy_total_with_dissipation": total_energy_with_dissipation,
        "energy_total_drift": total_energy_drift,
        "dissipation_rate": dissipation_rate,
        "cumulative_dissipation": cumulative_dissipation,
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
    S0_amplitude = 0.065
    W0_amplitude = 0.095

    # ---- Material / damping / rheology parameters ----
    extra_stiffness = np.array([0.52, 0.38], dtype=float)
    gamma = np.array([0.030, 0.040], dtype=float)
    # One Maxwell-like relaxation branch per body. Add more columns for richer rheology.
    relax_strength = np.array([[0.55], [0.42]], dtype=float)
    relax_time = np.array([[2.8], [2.1]], dtype=float)
    # Rotational-flattening forcing coefficient in the quadrupole equation.
    rot_flattening_coeff = np.array([0.22, 0.18], dtype=float)

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

    params = QuadrupoleParameters(
        G=G,
        mass=mass,
        radius=radius,
        gamma=gamma,
        omega2=omega2,
        J0=J0,
        quad_coeff=quad_coeff,
        inertia_shape_coeff=inertia_shape_coeff,
        rot_flattening_coeff=rot_flattening_coeff,
        relax_strength=relax_strength,
        relax_time=relax_time,
        colors=("royalblue", "orangered"),
    )

    # Start the rheology branch near equilibrium to avoid an artificial initial transient.
    Z0 = np.repeat(S0[:, None, :, :], params.n_relax, axis=1) if params.n_relax > 0 else np.zeros((params.n_bodies, 0, 3, 3))
    y0 = pack_state(x, v, q, omega, S0, W0, Z0, 0.0)
    return params, y0


def describe_initial_conditions(params: QuadrupoleParameters, y0: np.ndarray, seed: int = 7) -> str:
    x, v, q, omega, S, W, Z, D = unpack_state(y0, params.n_bodies, params.n_relax)
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
    lines.append(f"Extra rotational-flattening coeff (absorbed into J_eff): {np.array2string(params.rot_flattening_coeff, precision=4)}")
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
    # np.savez_compressed('tidal_cache.npz', record=record, params=params)

    diag = two_body_diagnostics(record, params)
    t = diag["t"]

    energy_dissipated = record["cumulative_dissipation"]
    orbit_normal = diag["orbit_normal"]
    spin_axis = diag["spin_axis"]
    axis_plane_projection = np.sqrt(np.clip(1.0 - np.einsum("tai,ti->ta", spin_axis, orbit_normal) ** 2, 0.0, 1.0))
    eccentricity = diag["ecc"]
    semi_major_axis = diag["a"]
    spin_rate = diag["spin_rate"]
    mean_motion = diag["mean_motion"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    ax = axes[0, 0]
    mech_energy = record["energy_mech"]
    total_energy = record["energy_total_with_dissipation"]

    diss_line, = ax.plot(t, energy_dissipated, lw=2.0, color="black", label="Cumulative dissipated")
    ax.set_title("Energy bookkeeping")
    ax.set_xlabel("time")
    ax.set_ylabel("energy")
    ax.grid(True, alpha=0.3)

    ax_energy = ax.twinx()
    mech_line, = ax_energy.plot(t, mech_energy, lw=2.0, color="tab:green", ls="--", label="Mechanical energy")
    total_line, = ax_energy.plot(t, total_energy, lw=2.0, color="tab:red", label="Mechanical + dissipated")
    ax_energy.set_ylabel("energy")
    # ax.legend(
    #     [diss_line, mech_line, total_line],
    #     [diss_line.get_label(), mech_line.get_label(), total_line.get_label()],
    #     loc="best",
    # )
    ax.legend(
        [diss_line, mech_line, total_line],
        [diss_line.get_label(), mech_line.get_label(), total_line.get_label()],
        loc="center right",
    )

    ax = axes[0, 1]
    for i, color in enumerate(params.colors[:params.n_bodies]):
        ax.plot(t, axis_plane_projection[:, i], lw=2.0, color=color, label=f"Planet {i + 1}")
    ax.set_title("Spin-axis projection into orbital plane")
    ax.set_xlabel("time")
    ax.set_ylabel(r"$|\hat s-(\hat s\cdot \hat h)\hat h|$")
    ax.set_ylim(0.0, 1.05)
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
    # fig.savefig(filename, dpi=300, bbox_inches="tight")
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


if __name__ == "__main__":
    # with np.load(r'c:\Programs\Shapley\pythonProject\tidal_cache.npz', allow_pickle=True) as dat:
    #     record = dat['record'].item()
    #     params = dat['params'].item()
    #
    # make_diagnostics_plot(record, params)
    # exit()

    main()
