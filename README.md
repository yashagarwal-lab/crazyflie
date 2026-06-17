# Crazyflie 2.x Flight Controller

Keyboard-based flight controller and motor test utilities for the Crazyflie 2.x quadcopter using [cflib](https://github.com/bitcraze/crazyflie-lib-python).


## Collaboration & Git Workflow

Yash and Nathan work in their respective directories (`yash/` and `nathan/`) to avoid merge conflicts.

### Daily Git Cheat Sheet
1. **Get the latest updates:**
   ```bash
   git pull origin main
   ```
2. **Work on your code** e.g. - in `/nathan/` or `/yash/` folders.
3. **Save and share your updates:**
   ```bash
   git add nathan/
   git commit -m "nathan: <describe your changes>"
   git push origin main
   ```
*(Using `git add nathan/` or `git add yash/` ensures you only upload your own files.)*
