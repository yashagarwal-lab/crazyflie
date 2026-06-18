"""
Formation flying for Crazyflie swarms.

Computes per-drone target positions for geometric shapes and dispatches
them via controller.goto_point(). Controller-agnostic.
"""
import math


class FormationManager:
    def __init__(self, controller, avoider=None):
        self.controller = controller
        self.avoider = avoider  # Optional CollisionAvoider
        self.current_slots = {}  # drone_number -> (x, y, z)
        self.center = (0.0, 0.0, 0.5)

    def _send(self, drone_number, x, y, z):
        """Send target through collision filter if available."""
        if self.avoider:
            self.avoider.safe_goto(drone_number, x, y, z)
        else:
            self.controller.goto_point(drone_number, x, y, z)

    def _active_drones(self):
        """Return list of drone numbers that are currently flying."""
        return [d['number'] for d in self.controller.get_state()
                if d['state'] == 'FLYING']

    def apply(self, shape, **kwargs):
        """Compute formation and dispatch."""
        drones = self._active_drones()
        if not drones:
            return False

        if shape == 'line':
            slots = self.line(len(drones), **kwargs)
        elif shape == 'triangle':
            slots = self.triangle(len(drones), **kwargs)
        elif shape == 'circle':
            slots = self.circle(len(drones), **kwargs)
        else:
            return False

        self.current_slots = {}
        for drone_num, (x, y, z) in zip(drones, slots):
            self.current_slots[drone_num] = (x, y, z)
            self._send(drone_num, x, y, z)
        return True

    def line(self, n, spacing=0.4, z=None):
        """N drones in a line centered at self.center, along X axis."""
        cx, cy, cz = self.center
        z = z if z is not None else cz
        slots = []
        for i in range(n):
            offset = (i - (n - 1) / 2.0) * spacing
            slots.append((cx + offset, cy, z))
        return slots

    def triangle(self, n, radius=0.5, z=None):
        """N drones in an equilateral arrangement (works best with 3)."""
        return self.circle(n, radius=radius, z=z)

    def circle(self, n, radius=0.5, z=None):
        """N drones evenly spaced around a circle."""
        cx, cy, cz = self.center
        z = z if z is not None else cz
        slots = []
        for i in range(n):
            angle = 2 * math.pi * i / n
            slots.append((cx + radius * math.cos(angle),
                          cy + radius * math.sin(angle), z))
        return slots

    def translate(self, dx, dy, dz):
        """Shift the entire formation by an offset."""
        cx, cy, cz = self.center
        self.center = (cx + dx, cy + dy, cz + dz)

        new_slots = {}
        for drone_num, (x, y, z) in self.current_slots.items():
            nx, ny, nz = x + dx, y + dy, z + dz
            new_slots[drone_num] = (nx, ny, nz)
            self._send(drone_num, nx, ny, nz)
        self.current_slots = new_slots

    def rotate(self, angle_deg):
        """Rotate the formation around its centroid."""
        cx, cy, cz = self.center
        rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)

        new_slots = {}
        for drone_num, (x, y, z) in self.current_slots.items():
            dx, dy = x - cx, y - cy
            nx = cx + dx * cos_a - dy * sin_a
            ny = cy + dx * sin_a + dy * cos_a
            new_slots[drone_num] = (nx, ny, z)
            self._send(drone_num, nx, ny, z)
        self.current_slots = new_slots
