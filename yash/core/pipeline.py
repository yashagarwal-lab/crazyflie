import math

class PositionSetpoint:
    def __init__(self, x, y, z, yaw):
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw

class BaseController:
    def compute(self, state, target_pos, target_yaw, dt):
        """
        Computes and returns the next PositionSetpoint.
        """
        raise NotImplementedError

class MellingerController(BaseController):
    """
    Standard position interpolation controller.
    Gradually moves a setpoint towards the target at a maximum speed.
    """
    def __init__(self, max_speed=1.0):
        self.max_speed = max_speed
        self.setpoint_x = 0.0
        self.setpoint_y = 0.0
        self.setpoint_z = 0.0
        self.initialized = False

    def compute(self, state, target_pos, target_yaw, dt):
        tx, ty, tz = target_pos
        cx, cy, cz = state.get_position()
        
        if not self.initialized:
            self.setpoint_x, self.setpoint_y, self.setpoint_z = cx, cy, cz
            self.initialized = True
            
        # Anti-windup clamp (if setpoint gets too far from actual pose, snap it back)
        dd = math.sqrt((self.setpoint_x-cx)**2 + (self.setpoint_y-cy)**2 + (self.setpoint_z-cz)**2)
        if dd > 0.3:
            s = 0.3/dd
            self.setpoint_x = cx + (self.setpoint_x-cx)*s
            self.setpoint_y = cy + (self.setpoint_y-cy)*s
            self.setpoint_z = cz + (self.setpoint_z-cz)*s

        # Interpolate setpoint toward target
        ex, ey, ez = tx - self.setpoint_x, ty - self.setpoint_y, tz - self.setpoint_z
        dist = math.sqrt(ex**2 + ey**2 + ez**2)
        step = self.max_speed * dt
        
        if dist > step:
            self.setpoint_x += (ex/dist) * step
            self.setpoint_y += (ey/dist) * step
            self.setpoint_z += (ez/dist) * step
        else:
            self.setpoint_x, self.setpoint_y, self.setpoint_z = tx, ty, tz
            
        return PositionSetpoint(self.setpoint_x, self.setpoint_y, self.setpoint_z, target_yaw)

class CBFSafetyWrapper(BaseController):
    """
    Wraps another controller and applies a Control Barrier Function (CBF) filter 
    to prevent collisions with boundaries and other drones.
    """
    def __init__(self, base_controller, cbf_filter, all_drones_provider):
        self.base_controller = base_controller
        self.cbf = cbf_filter
        self.all_drones_provider = all_drones_provider # Function returning a list of all DroneState objects
        self.max_speed = getattr(base_controller, 'max_speed', 1.0)
        self.prev_setpoint_x = None
        self.prev_setpoint_y = None
        self.prev_setpoint_z = None
        
    @property
    def setpoint_x(self):
        return self.base_controller.setpoint_x

    @setpoint_x.setter
    def setpoint_x(self, value):
        self.base_controller.setpoint_x = value
        self.prev_setpoint_x = value

    @property
    def setpoint_y(self):
        return self.base_controller.setpoint_y

    @setpoint_y.setter
    def setpoint_y(self, value):
        self.base_controller.setpoint_y = value
        self.prev_setpoint_y = value

    @property
    def setpoint_z(self):
        return self.base_controller.setpoint_z

    @setpoint_z.setter
    def setpoint_z(self, value):
        self.base_controller.setpoint_z = value
        self.prev_setpoint_z = value

    def compute(self, state, target_pos, target_yaw, dt):
        cx, cy, cz = state.get_position()
        
        # 1. Ask base controller for its desired setpoint
        base_setpoint = self.base_controller.compute(state, target_pos, target_yaw, dt)
        
        if self.prev_setpoint_x is None:
            self.prev_setpoint_x = base_setpoint.x
            self.prev_setpoint_y = base_setpoint.y
            self.prev_setpoint_z = base_setpoint.z

        # 2. What velocity does the *setpoint trajectory* imply?
        # (Must differentiate the setpoint, NOT the error from actual position)
        vx_nom = (base_setpoint.x - self.prev_setpoint_x) / dt if dt > 0 else 0
        vy_nom = (base_setpoint.y - self.prev_setpoint_y) / dt if dt > 0 else 0
        vz_nom = (base_setpoint.z - self.prev_setpoint_z) / dt if dt > 0 else 0
        
        # Clamp implied velocity just in case
        v_mag = math.sqrt(vx_nom**2 + vy_nom**2 + vz_nom**2)
        if v_mag > self.max_speed:
            scale = self.max_speed / v_mag
            vx_nom *= scale; vy_nom *= scale; vz_nom *= scale

        # 3. Gather obstacle data
        other_poses = []
        for other_state in self.all_drones_provider():
            if other_state is not state and other_state.pose_valid:
                other_poses.append(other_state.get_position())
                
        # 4. Filter velocity through CBF using current physical position as the state
        vx_safe, vy_safe, vz_safe = self.cbf.filter(
            pos=(cx, cy, cz),
            v_des=(vx_nom, vy_nom, vz_nom),
            other_positions=other_poses
        )
        
        # 5. Generate safe position setpoint by integrating safe velocity
        safe_x = self.prev_setpoint_x + vx_safe * dt
        safe_y = self.prev_setpoint_y + vy_safe * dt
        safe_z = self.prev_setpoint_z + vz_safe * dt
        
        # Keep internal states synced
        self.prev_setpoint_x = safe_x
        self.prev_setpoint_y = safe_y
        self.prev_setpoint_z = safe_z
        self.base_controller.setpoint_x = safe_x
        self.base_controller.setpoint_y = safe_y
        self.base_controller.setpoint_z = safe_z
        
        return PositionSetpoint(safe_x, safe_y, safe_z, target_yaw)
