"""
Position-domain collision avoidance for Crazyflie swarms.

Clamps target positions to maintain inter-drone separation and workspace bounds.
Works with any controller mode (PID, Mellinger, etc.) since it only modifies
the target coordinates before they reach the backend.
"""
import math

WORKSPACE_RADIUS = 1.5
WORKSPACE_Z_MAX  = 2.0
WORKSPACE_Z_MIN  = 0.05


class CollisionAvoider:
    def __init__(self, controller, d_min=0.3, radius=WORKSPACE_RADIUS,
                 z_min=WORKSPACE_Z_MIN, z_max=WORKSPACE_Z_MAX):
        self.controller = controller
        self.d_min = d_min
        self.radius = radius
        self.z_min = z_min
        self.z_max = z_max

    def safe_target(self, drone_number, x, y, z):
        """
        Clamp (x, y, z) to workspace boundary and push away from other drones.
        Returns (safe_x, safe_y, safe_z).
        """
        # 1. Workspace clamp
        z = max(self.z_min, min(z, self.z_max))
        r = math.sqrt(x**2 + y**2)
        if r > self.radius:
            x *= self.radius / r
            y *= self.radius / r

        # 2. Push away from each other drone's position and target
        for d in self.controller.drones:
            if d.number == drone_number:
                continue

            # Check against current position
            ox, oy, oz, _, _, _, _, _ = d.get_pose()
            x, y, z = self._push_apart(x, y, z, ox, oy, oz)

            # Check against other drone's current target
            with d.nav_lock:
                otx, oty, otz = d.target_x, d.target_y, d.target_z
            x, y, z = self._push_apart(x, y, z, otx, oty, otz)

        # 3. Re-clamp after pushes
        z = max(self.z_min, min(z, self.z_max))
        r = math.sqrt(x**2 + y**2)
        if r > self.radius:
            x *= self.radius / r
            y *= self.radius / r

        return x, y, z

    def _push_apart(self, x, y, z, ox, oy, oz):
        """Push (x,y,z) away from (ox,oy,oz) if closer than d_min."""
        dx, dy, dz = x - ox, y - oy, z - oz
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        if dist < self.d_min and dist > 0.001:
            push = (self.d_min - dist) / dist
            x += dx * push
            y += dy * push
            z += dz * push
        return x, y, z

    def safe_goto(self, drone_number, x, y, z):
        """Filter target through collision avoidance, then send to controller."""
        sx, sy, sz = self.safe_target(drone_number, x, y, z)
        return self.controller.goto_point(drone_number, sx, sy, sz)
