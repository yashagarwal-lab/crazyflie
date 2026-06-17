"""
Control Barrier Function (CBF) safety filter for Crazyflie drones.

Enforces safety constraints by filtering velocity commands through a QP:
  - Cylindrical workspace boundary (radius R, height [z_min, z_max])
  - Inter-drone minimum separation distance
  - Velocity magnitude limits

The drone dynamics are single-integrator (ẋ = v) since we command
velocities via send_velocity_world_setpoint. This makes all CBF
constraints linear in v, yielding a simple QP solvable in <2ms.

Reference:
  Ames et al., "Control Barrier Functions: Theory and Applications", 2019
"""

import numpy as np
from scipy.optimize import minimize


class CBFSafetyFilter:
    """
    Cylindrical workspace CBF safety filter.

    The filter solves at each timestep:
        minimize    ||v - v_des||²
        subject to  ∇hₖ · v ≥ -αₖ · hₖ(x)   for each barrier hₖ

    Barrier functions:
        h₁ = R² - (x² + y²)           cylinder wall
        h₂ = z_max - z                 ceiling
        h₃ = z - z_min                 floor
        h₄ = ||pᵢ - pⱼ||² - d_min²    inter-drone separation (per pair)

    Parameters
    ----------
    radius : float
        Workspace XY radius in metres (default 1.25).
    z_max : float
        Maximum altitude in metres (default 1.8).
    z_min : float
        Minimum altitude in metres (default 0.05).
    d_min : float
        Minimum inter-drone distance in metres (default 0.3).
    alpha_boundary : float
        CBF gain for workspace boundaries. Higher = more aggressive
        approach allowed. Lower = drone slows earlier (default 1.0).
    alpha_separation : float
        CBF gain for inter-drone separation (default 0.8).
    max_speed : float
        Maximum velocity magnitude in m/s (default 0.3).
    """

    def __init__(self,
                 radius=1.25,
                 z_max=1.8,
                 z_min=0.05,
                 d_min=0.3,
                 alpha_boundary=1.0,
                 alpha_separation=0.8,
                 max_speed=0.3):
        self.R2 = radius ** 2
        self.radius = radius
        self.z_max = z_max
        self.z_min = z_min
        self.d_min2 = d_min ** 2
        self.d_min = d_min
        self.alpha_b = alpha_boundary
        self.alpha_s = alpha_separation
        self.max_speed = max_speed

    def _build_constraints(self, pos, other_positions):
        """
        Build CBF inequality constraints: ∇h · v ≥ -α · h(x)
        Rearranged to scipy form: c(v) ≥ 0

        Returns list of constraint dicts for scipy.optimize.minimize.
        """
        x, y, z = pos
        constraints = []

        # ── h₁: cylinder wall  h = R² - (x² + y²) ──────────────────────
        h1 = self.R2 - (x**2 + y**2)
        # ∇h₁ · v = -2x·vx - 2y·vy
        # Constraint: -2x·vx - 2y·vy ≥ -α·h₁
        # scipy form: -2x·vx - 2y·vy + α·h₁ ≥ 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, _x=x, _y=y, _h=h1: (
                -2*_x*v[0] - 2*_y*v[1] + self.alpha_b * _h
            )
        })

        # ── h₂: ceiling  h = z_max - z ──────────────────────────────────
        h2 = self.z_max - z
        # ∇h₂ · v = -vz
        # Constraint: -vz + α·h₂ ≥ 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, _h=h2: -v[2] + self.alpha_b * _h
        })

        # ── h₃: floor  h = z - z_min ────────────────────────────────────
        h3 = z - self.z_min
        # ∇h₃ · v = vz
        # Constraint: vz + α·h₃ ≥ 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, _h=h3: v[2] + self.alpha_b * _h
        })

        # ── h₄: inter-drone separation (one per other drone) ────────────
        for other in other_positions:
            ox, oy, oz = other
            dx, dy, dz = x - ox, y - oy, z - oz
            dist_sq = dx**2 + dy**2 + dz**2
            h4 = dist_sq - self.d_min2
            # ∇h₄ · v = 2·dx·vx + 2·dy·vy + 2·dz·vz
            # (only the moving drone's velocity; other drone treated as static)
            # Constraint: 2·dx·vx + 2·dy·vy + 2·dz·vz + α·h₄ ≥ 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, _dx=dx, _dy=dy, _dz=dz, _h=h4: (
                    2*_dx*v[0] + 2*_dy*v[1] + 2*_dz*v[2]
                    + self.alpha_s * _h
                )
            })

        return constraints

    def filter(self, pos, v_des, other_positions=None):
        """
        Filter a desired velocity through the CBF safety constraints.

        Parameters
        ----------
        pos : tuple (x, y, z)
            Current drone position in metres.
        v_des : tuple (vx, vy, vz)
            Desired velocity from PID controller.
        other_positions : list of (x, y, z) tuples, optional
            Positions of other drones for separation constraint.

        Returns
        -------
        tuple (vx, vy, vz)
            Safe velocity command.
        """
        if other_positions is None:
            other_positions = []

        v0 = np.array(v_des, dtype=float)

        # Build constraints
        constraints = self._build_constraints(pos, other_positions)

        # Velocity magnitude bound
        constraints.append({
            'type': 'ineq',
            'fun': lambda v: self.max_speed**2 - (v[0]**2 + v[1]**2 + v[2]**2)
        })

        # Objective: minimize ||v - v_des||²
        def objective(v):
            d = v - v0
            return d[0]**2 + d[1]**2 + d[2]**2

        def objective_jac(v):
            return 2.0 * (v - v0)

        result = minimize(
            objective,
            v0,
            jac=objective_jac,
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': 50, 'ftol': 1e-8},
        )

        if result.success:
            v_safe = result.x
            # Enforce speed limit (belt and suspenders)
            speed = np.linalg.norm(v_safe)
            if speed > self.max_speed:
                v_safe = v_safe * (self.max_speed / speed)
            return (float(v_safe[0]), float(v_safe[1]), float(v_safe[2]))
        else:
            # Solver failed — safest action is hover in place
            return (0.0, 0.0, 0.0)

    def is_safe(self, pos, other_positions=None):
        """
        Check if current position is within all safety constraints.

        Returns
        -------
        bool
            True if all barrier functions h(x) > 0.
        """
        if other_positions is None:
            other_positions = []

        x, y, z = pos

        # Cylinder wall
        if (x**2 + y**2) >= self.R2:
            return False
        # Ceiling
        if z >= self.z_max:
            return False
        # Floor
        if z <= self.z_min:
            return False
        # Drone separation
        for ox, oy, oz in other_positions:
            dist_sq = (x-ox)**2 + (y-oy)**2 + (z-oz)**2
            if dist_sq <= self.d_min2:
                return False

        return True

    def margin(self, pos, other_positions=None):
        """
        Return the minimum barrier value across all constraints.
        Positive = safe, zero = at boundary, negative = violated.

        Useful for diagnostics and logging.
        """
        if other_positions is None:
            other_positions = []

        x, y, z = pos
        margins = [
            self.R2 - (x**2 + y**2),       # cylinder wall
            self.z_max - z,                  # ceiling
            z - self.z_min,                  # floor
        ]
        for ox, oy, oz in other_positions:
            dist_sq = (x-ox)**2 + (y-oy)**2 + (z-oz)**2
            margins.append(dist_sq - self.d_min2)

        return min(margins)
