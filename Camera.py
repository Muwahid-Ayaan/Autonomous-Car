import carla
import cv2
import numpy as np

class Camera:
    def __init__(self,world,vehicle,fps = "0.1",offsetx = 1.5, offsetz = 1.8,pitch=-5):
        self.world = world
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('sensor_tick', fps)
        camera_transform = carla.Transform(
        carla.Location(x=offsetx, z=offsetz),
        carla.Rotation(pitch=pitch)
        )
        self.current_frame = None
        self.camera = world.spawn_actor(
        camera_bp, 
        camera_transform, 
        attach_to=vehicle, 
        attachment_type=carla.AttachmentType.Rigid
        )


    def process(self, image):
        self.current_frame = image
    
    def start_camera(self):
        self.camera.listen(self.process)
    
    def stop_camera(self):
        self.camera.stop()

    def get_current_frame(self):
        return self.current_frame            
    
    def save_frame(self):
        self.current_frame.save_to_disk("Latest_frame")
        