import time
import cflib.crtp
from cflib.crazyflie.swarm import CachedCfFactory, Swarm
from cflib.positioning.motion_commander import MotionCommander

# List of URIs for your crazyflies.
# IMPORTANT: Each drone must have a unique radio address! 
# You can change the address of a drone using the bitcraze python client.
URIS = [
    'radio://0/80/2M/E7E7E7E701',  # Drone 1
    'radio://0/80/2M/E7E7E7E702',  # Drone 2 (CHANGE THIS ADDRESS TO MATCH YOUR 2ND DRONE)
]

def run_sequence(scf):
    """
    This function will be executed in parallel for each drone.
    The scf (SyncCrazyflie) object is passed automatically by the Swarm.
    """
    uri = scf.cf.link_uri
    print(f"[{uri}] Starting sequence...")
    
    # MotionCommander takes off automatically to default_height when entering the context
    with MotionCommander(scf, default_height=0.3) as mc:
        print(f"[{uri}] Hovering for 3 seconds...")
        time.sleep(3)
        
        print(f"[{uri}] Moving up...")
        mc.up(0.2)
        time.sleep(2)
        
        print(f"[{uri}] Moving down...")
        mc.down(0.2)
        time.sleep(2)
        
        print(f"[{uri}] Landing...")
        # Landing is handled automatically when exiting the context manager
        
    print(f"[{uri}] Sequence completed.")

if __name__ == '__main__':
    # Initialize the low-level drivers
    cflib.crtp.init_drivers()

    # Factory handles the creation and caching of Crazyflie objects
    factory = CachedCfFactory(rw_cache='./cache')

    print("Connecting to swarm...")
    # The Swarm class connects to all URIs in parallel
    try:
        with Swarm(URIS, factory=factory) as swarm:
            print("Connected to all Crazyflies. Starting sequence.")
            # Execute the run_sequence function on all connected crazyflies in parallel
            swarm.parallel_safe(run_sequence)
            
        print("Swarm operation finished successfully.")
    except Exception as e:
        print(f"Error during swarm operation: {e}")
