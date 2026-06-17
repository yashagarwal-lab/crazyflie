from NatNetClient import NatNetClient
import time

def labeled_marker_callback(marker_id, pos):
    # Same remap as flight code
    cf_x = pos[2]   # OptiTrack Z → Crazyflie X
    cf_y = pos[0]   # OptiTrack X → Crazyflie Y
    cf_z = pos[1]   # OptiTrack Y → Crazyflie Z (height)
    print(f"  Marker ID: {marker_id}   x={cf_x:+.3f}  y={cf_y:+.3f}  z={cf_z:+.3f}  (height={cf_z:.3f}m)")

client = NatNetClient()
client.labeledMarkerListener = labeled_marker_callback
client.run()
print("Listening — move each drone to identify it. Ctrl+C to stop.\n")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
client.stop()