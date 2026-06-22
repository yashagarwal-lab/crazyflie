import time
import math
import threading

def quat_to_yaw(qx, qy, qz, qw):
    # Yaw from raw Optitrack quaternion (Y-up frame, rotation around Y)
    siny_cosp = 2.0 * (qw * qy + qz * qx)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))

class DroneState:
    """Thread-safe data class for tracking physical drone reality."""
    def __init__(self):
        self.lock = threading.Lock()
        
        # Position
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        
        # Orientation (Quaternion)
        self.qx = 0.0
        self.qy = 0.0
        self.qz = 0.0
        self.qw = 1.0
        
        # Metadata
        self.pose_valid = False
        self.last_update = 0.0
        self.battery = 4.2
        self.status = "INIT"
        
    def update_pose(self, position, rotation):
        opti_qx, opti_qy, opti_qz, opti_qw = rotation
        with self.lock:
            # Map Optitrack (X, Y_up, Z) -> Crazyflie (Z, X, Y_up)
            self.x, self.y, self.z = position[2], position[0], position[1]
            
            # Store raw Optitrack quaternion — the EKF handles the frame
            # (matches proven 3m_bounds.py behavior)
            self.qx = opti_qx
            self.qy = opti_qy
            self.qz = opti_qz
            self.qw = opti_qw
            self.pose_valid = True
            self.last_update = time.time()
            
    def get_pose(self):
        with self.lock:
            return self.x, self.y, self.z, self.qx, self.qy, self.qz, self.qw, self.pose_valid
            
    def get_position(self):
        with self.lock:
            return self.x, self.y, self.z

    def get_yaw(self):
        with self.lock:
            return quat_to_yaw(self.qx, self.qy, self.qz, self.qw)

    def get_pose_age(self):
        with self.lock:
            if not self.pose_valid:
                return 999.0
            return time.time() - self.last_update
