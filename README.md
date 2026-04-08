# Tidal Deformation of Pseudo-Fluid Planets — Quadrupole Model

## Overview

This repository provides a **reduced Newtonian tidal model** that sits between two classical extremes: simple point-mass *N*-body gravity (no tides) and full viscoelastic interior simulations (expensive and problem-specific). The model is a structured "middle ground" that still captures the key mechanisms of tidal evolution:

- tidal bulge formation and gravitational backreaction,
- spin torques and spin–orbit coupling,
- orbital circularization from dissipative lag,
- spin synchronization,
- obliquity evolution.

Each planet's surface is expanded in spherical harmonics and truncated at the **quadrupole** (ℓ = 2) level. The shape is described by symmetric trace-free (STF) tensors that evolve as ordinary differential equations (ODEs), making the system straightforward to integrate numerically.

Compared with the simplest damped-quadrupole model, this implementation adds four improvements:

1. **Deformation-dependent spin inertia** — the moment of inertia tensor varies with the tidal shape.
2. **Rotational flattening** — centrifugal oblateness enters through the same shape-dependent coefficient as the spin kinetic energy, keeping the energy bookkeeping consistent.
3. **Frequency-dependent rheology** — one or more Maxwell-like relaxation branches are supported alongside the direct viscous closure, all in ODE form.
4. **Quadrupole modal angular momentum** — the leading angular momentum carried by the deformation mode itself is retained; this restores the explicit quadrupole torque in the spin balance without upsetting conservative bookkeeping, which is important for obliquity studies.

An optional **split-bulge extension** introduces separate tidal and rotational quadrupole tensors, providing additional orientational degrees of freedom for obliquity evolution studies.

---

![Curious Double Planet](media/Curious%20Double%20Planet.jpg)
*A binary planet system illustrating the kind of two-body configuration studied in this work. The two bodies are of comparable size — a configuration sometimes called a "double planet" — and both are subject to mutual tidal forcing. In the model each body develops a quadrupole bulge that lags slightly behind the line of centers, producing the torques responsible for spin synchronization, orbital circularization, and obliquity damping.*

![Orbital animation](media/tidal.gif)
*Animation of the two planets as they orbit each other, shape changes 30x magnified.*

---

## Repository Structure

```
.
├── multipoleGravity.py       # Full model implementation and test suite
├── tidal.tex                 # LaTeX source of the companion paper
├── media/
│   ├── Curious Double Planet.jpg
│   └── tidal.gif
└── README.md
```

---

## Physics Summary

### State variables (per body)

| Variable | Description |
|---|---|
| **x** | Center-of-mass position (inertial frame) |
| **v** | Center-of-mass velocity |
| **q** | Unit quaternion encoding body orientation |
| **Ω** | Body-frame angular velocity |
| **S** | Quadrupole STF shape tensor (body frame) |
| **W** | Time derivative of S (body frame) |
| **B** | *(split-bulge only)* Rotational bulge STF tensor |
| **Z** | Relaxation branch memory tensors |
| *D* | Cumulative dissipated energy (scalar diagnostic) |

### Key physical effects modeled

- **Tidal potential** — quadrupole gravitational interaction between shape tensors and the tidal field of each companion body.
- **Spin–shape coupling** — the body's moment of inertia is modified by the deformation, so spin and shape equations are coupled through the kinetic energy.
- **Rheology** — two closures are available and can be mixed:
  - *Viscous*: shape relaxation proportional to `W` (shape rate), producing a frequency-independent lag.
  - *Maxwell branch*: internal memory variable `Z` that relaxes toward `S` on a characteristic timescale `τ`, producing a frequency-dependent (realistic) lag.
- **Rotational flattening** — spin-driven oblateness is computed from the centrifugal STF tensor `STF(Ω ⊗ Ω)`.

### Numerical integration

The ODE system is integrated with `scipy.integrate.solve_ivp` using a high-order Runge–Kutta method (`RK45` by default) with tight tolerances (`rtol = 1e-10`, `atol = 1e-11`). Quaternion normalization is applied at each recorded step to prevent orientation drift.

---

## Installation

```bash
pip install numpy scipy plotly matplotlib
```

No other dependencies are required. The code is a single self-contained module.

---

## Usage

Running the default simulation:

```bash
python multipoleGravity.py
```

This calls `main()`, which sets up a representative two-body system and runs the integrator, producing diagnostic plots via Plotly and Matplotlib.

Running the full regression test suite:

```python
from multipoleGravity import run_tests
run_tests()
```

Test plots are saved to a timestamped output directory and a summary is printed to the console.

---

## Test Suite

Eight regression tests verify that the model behaves correctly in the limits it was designed for. All 8 currently pass.

| # | Test | What is checked |
|---|---|---|
| 01 | `null_kepler_limit` | Reduces to exact Kepler orbit when quadrupole couplings are off |
| 02 | `conservative_equilibrium` | Circular synchronized state stays near equilibrium with no dissipation |
| 03 | `quasi_static_bulge` | Strongly damped shape tracks the hydrostatic tide in the slow-forcing limit |
| 04 | `rotational_flattening` | Spin-driven oblateness matches the expected oblate equilibrium |
| 05 | `rheology_comparison` | Viscous and Maxwell closures produce observably different lags and dissipation |
| 06 | `circularization` | Eccentric orbit circularizes and inspirals secularly under active tides |
| 07 | `spin_synchronization` | Super-synchronous spin is torqued toward the orbital frequency |
| 08 | `obliquity_survey` | Net obliquity damping is found across a grid of tilted initial states |

Achieved pass margins are tight: the nondissipative tests show residuals at the level of 10⁻⁹ to 10⁻¹⁴.

---

## Key Classes and Functions

### `QuadrupoleParameters` (dataclass)
Holds all physical parameters for an *N*-body system: masses, radii, elastic coefficients, dissipation rates, relaxation branch strengths and timescales. Provides convenience properties `n_bodies`, `n_relax`, and `has_split_bulge`.

### Core linear algebra utilities
- `stf(a)` — symmetric trace-free projection for (…, 3, 3) arrays.
- `hat(ω)` / `axl(A)` — conversion between angular velocity vectors and skew-symmetric matrices.
- `quat_multiply`, `quat_to_matrix`, `quat_derivative_body_to_inertial` — quaternion algebra for orientation tracking.

### Physics functions
- `pairwise_geometry(x)` — computes all pairwise separations, unit vectors, and inverse-distance powers.
- `current_body_inertia(S, params)` — linearized moment of inertia tensor as a function of the current shape.
- `total_body_quadrupole(S, params)` — effective quadrupole mass tensor for the gravitational interaction.
- `dissipation_rate_from_state(W, S, Z, params)` — instantaneous dissipation power from all active branches.
- `quadrupole_mode_angular_momentum(S, W)` — leading modal angular momentum `2 axl([W, S])`.

### State packing / unpacking
- `pack_state_full` / `unpack_state_full` — flatten the full state (including optional split-bulge and relaxation tensors) into a 1-D array suitable for `solve_ivp`, and recover it afterward.

---

