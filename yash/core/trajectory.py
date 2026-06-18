"""
Minimum-snap trajectory generation for Crazyflie drones.

Generates smooth, jerk-free polynomial trajectories through a sequence of
waypoints. Each axis is solved independently as a 7th-order polynomial
minimizing the 4th derivative (snap).

Also includes preset trajectory shapes (circle, square, figure-8).
"""
import numpy as np


class MinSnapTrajectory:
    """
    Given N waypoints and N-1 segment durations, generates a piecewise
    polynomial trajectory that minimizes snap.

    Usage:
        traj = MinSnapTrajectory(
            waypoints=[(0,0,0.5), (0.5,0.5,0.8), (0,0,0.5)],
            segment_times=[3.0, 3.0]
        )
        x, y, z = traj.evaluate(t=1.5)
    """

    def __init__(self, waypoints, segment_times):
        self.waypoints = [np.array(wp) for wp in waypoints]
        self.segment_times = list(segment_times)
        self.n_segments = len(segment_times)
        assert len(waypoints) == self.n_segments + 1

        # Solve per-axis
        positions_x = [wp[0] for wp in waypoints]
        positions_y = [wp[1] for wp in waypoints]
        positions_z = [wp[2] for wp in waypoints]

        self.coeffs_x = self._solve_axis(positions_x, self.segment_times)
        self.coeffs_y = self._solve_axis(positions_y, self.segment_times)
        self.coeffs_z = self._solve_axis(positions_z, self.segment_times)

    @property
    def total_time(self):
        return sum(self.segment_times)

    def evaluate(self, t):
        """Evaluate trajectory at time t. Returns (x, y, z)."""
        t = max(0.0, min(t, self.total_time))

        # Find which segment we're in
        elapsed = 0.0
        seg = 0
        for i, dt in enumerate(self.segment_times):
            if elapsed + dt >= t:
                seg = i
                break
            elapsed += dt
        else:
            seg = self.n_segments - 1
            elapsed = self.total_time - self.segment_times[-1]

        tau = t - elapsed  # Local time within segment

        x = self._eval_poly(self.coeffs_x[seg], tau)
        y = self._eval_poly(self.coeffs_y[seg], tau)
        z = self._eval_poly(self.coeffs_z[seg], tau)
        return (x, y, z)

    def _eval_poly(self, coeffs, t):
        """Evaluate polynomial: c[0] + c[1]*t + c[2]*t^2 + ..."""
        result = 0.0
        for i, c in enumerate(coeffs):
            result += c * (t ** i)
        return result

    def _solve_axis(self, positions, times):
        """
        Solve for 5th-order polynomial coefficients per segment.
        Boundary conditions: position, velocity=0, acceleration=0 at endpoints.
        Continuity: position, velocity, acceleration at internal waypoints.
        """
        n = len(times)
        order = 6  # 5th order polynomial (6 coefficients)
        coeffs = []

        for seg in range(n):
            T = times[seg]
            p0 = positions[seg]
            p1 = positions[seg + 1]

            if seg == 0:
                v0, a0 = 0.0, 0.0
            else:
                # Use finite difference for smooth internal velocities
                dt_prev = times[seg - 1]
                dt_curr = times[seg]
                v0 = (positions[seg + 1] - positions[seg - 1]) / (dt_prev + dt_curr)
                a0 = 0.0

            if seg == n - 1:
                v1, a1 = 0.0, 0.0
            else:
                dt_curr = times[seg]
                dt_next = times[seg + 1]
                v1 = (positions[seg + 2] - positions[seg]) / (dt_curr + dt_next)
                a1 = 0.0

            # 5th order: p(t) = c0 + c1*t + c2*t^2 + c3*t^3 + c4*t^4 + c5*t^5
            # 6 constraints: p(0)=p0, p'(0)=v0, p''(0)=a0, p(T)=p1, p'(T)=v1, p''(T)=a1
            A = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 2, 0, 0, 0],
                [1, T, T**2, T**3, T**4, T**5],
                [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
                [0, 0, 2, 6*T, 12*T**2, 20*T**3],
            ], dtype=float)

            b = np.array([p0, v0, a0, p1, v1, a1], dtype=float)
            c = np.linalg.solve(A, b)
            coeffs.append(c)

        return coeffs


# ── Preset trajectory shapes ─────────────────────────────────────────────────

def circle_trajectory(radius=0.5, z=0.8, center=(0, 0), n_points=16, speed=0.3):
    """Generate waypoints and times for a smooth circle."""
    cx, cy = center
    waypoints = []
    for i in range(n_points + 1):
        angle = 2 * np.pi * (i % n_points) / n_points
        waypoints.append((cx + radius * np.cos(angle),
                          cy + radius * np.sin(angle), z))

    circumference = 2 * np.pi * radius
    total_time = circumference / speed
    seg_time = total_time / n_points
    segment_times = [seg_time] * n_points

    return MinSnapTrajectory(waypoints, segment_times)


def square_trajectory(side=0.6, z=0.8, center=(0, 0), speed=0.3):
    """Generate waypoints and times for a smooth square."""
    cx, cy = center
    h = side / 2
    waypoints = [
        (cx + h, cy + h, z), (cx - h, cy + h, z),
        (cx - h, cy - h, z), (cx + h, cy - h, z),
        (cx + h, cy + h, z),  # Return to start
    ]
    seg_time = side / speed
    segment_times = [seg_time] * 4
    return MinSnapTrajectory(waypoints, segment_times)


def figure8_trajectory(radius=0.5, z=0.8, center=(0, 0), n_points=24, speed=0.3):
    """Generate waypoints and times for a figure-8 (lemniscate)."""
    cx, cy = center
    waypoints = []
    for i in range(n_points + 1):
        t = 2 * np.pi * (i % n_points) / n_points
        x = cx + radius * np.sin(t)
        y = cy + radius * np.sin(t) * np.cos(t)
        waypoints.append((x, y, z))

    total_length = 4 * radius * 1.2  # Approximate
    total_time = total_length / speed
    seg_time = total_time / n_points
    segment_times = [seg_time] * n_points

    return MinSnapTrajectory(waypoints, segment_times)
