'''
This file is used to test the airsim simulator, and to make sure you can control it
'''


import airsim
import time

# Connect to AirSim simulator (default localhost)
client = airsim.MultirotorClient()
client.confirmConnection()

# Enable API control and arm the drone
client.enableApiControl(True)
client.armDisarm(True)

# Takeoff
print("Taking off...")
client.takeoffAsync().join()

# Move up a bit
client.moveToZAsync(-10, 3).join()  # fly to 10m above ground

# Move forward 20m at 5 m/s
client.moveByVelocityAsync(5, 0, 0, 4).join()

# Hover
client.hoverAsync().join()

# Land
print("Landing...")
client.landAsync().join()

# Disarm and release control
client.armDisarm(False)
client.enableApiControl(False)

print("Done.")
