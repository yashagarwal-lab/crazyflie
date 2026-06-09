# Crazyflie 2.x Flight Controller

Keyboard-based flight controller and motor test utilities for the Crazyflie 2.x quadcopter using [cflib](https://github.com/bitcraze/crazyflie-lib-python).

## Setup

```bash
# Install udev rules (Linux)
sudo cp 99-bitcraze.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules

# Install dependencies
uv sync
```

## Usage

```bash
# Test individual motors (remove propellers first!)
uv run python motor_test.py

# Fly with keyboard
uv run python fly.py
```

## Controls (fly.py)

| Key | Action |
|---|---|
| ↑ / ↓ | Thrust up / down |
| W / S | Pitch forward / back |
| A / D | Roll left / right |
| Q / E | Yaw left / right |
| Space | Kill motors |
| Esc | Land & quit |
