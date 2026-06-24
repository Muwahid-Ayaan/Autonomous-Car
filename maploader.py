import carla
import time




client = carla.Client('localhost', 2000)
# Load a known CARLA map by name (avoid backslash escapes and relative paths)
client.load_world("Town04")












