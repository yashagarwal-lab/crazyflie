"""PID controller and utility functions for Crazyflie flight control."""


class PID:
    def __init__(self, kp, ki, kd, integral_limit=1.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral_limit = integral_limit
        self._integral   = 0.0
        self._prev_error = 0.0
        self._prev_time  = None

    def update(self, error, now):
        dt = (now - self._prev_time) if self._prev_time else 0.0
        self._prev_time = now
        self._integral = max(-self.integral_limit,
                             min(self.integral_limit,
                                 self._integral + error * dt))
        deriv = (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error
        return self.kp * error + self.ki * self._integral + self.kd * deriv

    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0
        self._prev_time  = None


def clamp(v, lim):
    """Clamp value v to the range [-lim, +lim]."""
    return max(-lim, min(lim, v))
