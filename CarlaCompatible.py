
import cv2
import carla
import numpy as np



class LaneDetectorSlidingWindow:
    def __init__(self):
        
        self.right_base = 0
        self.left_base = 0
        self.prevlx = []
        self.prevrx = []
        
        self.width = 640
        self.height = 480
        
        self.Lower_H = 0 
        self.Lower_S = 0
        self.Lower_V = 175
        self.Upper_H = 248
        self.Upper_S = 228
        self.Upper_V = 255
        
        self.top_left = (270,240)
        self.top_right = (365,240)
        self.bottom_left = (0,472)
        self.bottom_right = (600,472)
    
        self.window_width = 100

        self.pt1 = np.float32([self.top_left,self.bottom_left,self.top_right,self.bottom_right])
        self.pt2 = np.float32([[0,0],[0,self.height],[self.width,0],[self.width,self.height]])

        self.latest_mask = None
        self.latest_sliding_window = None
        self.latest_result = None

       
       


    def add_mask(self, frame):
        hsv_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        lower = np.array([self.Lower_H,self.Lower_S,self.Lower_V])
        upper = np.array([self.Upper_H,self.Upper_S,self.Upper_V])
    
        mask = cv2.inRange(hsv_frame,lower,upper)
        result = cv2.bitwise_and(hsv_frame,hsv_frame,mask = mask)

        return mask


    def generate_histogram(self,frame):
        histogram = np.sum(frame[frame.shape[0]//2:,:], axis=0)
        midpoint = int(histogram.shape[0]/2)
        self.left_base = np.argmax(histogram[:midpoint])
        self.right_base = np.argmax(histogram[midpoint:]) + midpoint

    
    def get_perspective_transform(self, frame):
        
        matrix = cv2.getPerspectiveTransform(self.pt1,self.pt2)
        transformed_frame = cv2.warpPerspective(frame,matrix,(self.width,self.height))

        return transformed_frame

    def get_inversed_perspective_tranform(self,frame,transformed_frame,leftx_cords,rightx_cords):
        min_length = min(len(leftx_cords),len(rightx_cords))
        top_left = (leftx_cords[0],472)
        top_right = (rightx_cords[0],472)
        bottom_left = (leftx_cords[min_length-1],0)
        bottom_right = (rightx_cords[min_length-1],0)

        quad_points = np.array([top_left,bottom_left,bottom_right,top_right],dtype=np.int32)
        quad_points = quad_points.reshape((-1,1,2))

        overlay = transformed_frame.copy()

        cv2.fillPoly(overlay,[quad_points],(0,255,0))
        
        
        alpha = 0.2
        cv2.addWeighted(overlay,alpha,transformed_frame, 1 - alpha, 0 ,transformed_frame)

        inv_matrix = cv2.getPerspectiveTransform(self.pt2,self.pt1)
        inversed_orignal_frame = cv2.warpPerspective(transformed_frame, inv_matrix, (640,480))

        result = cv2.addWeighted(frame, 1, inversed_orignal_frame, 0.5, 0)
        return result

    def make_compatible(self,image):
        frame = np.frombuffer(image.raw_data, dtype=np.uint8)
        frame = frame.reshape((image.height, image.width, 4))
        frame = frame[:, :, :3]
        frame = cv2.resize(frame,(self.width ,self.height))
        
        return frame
    

    def sliding_window(self,mask):
        y = 472 
        leftx_cords = []
        rightx_cords = []
        mask_copy = mask.copy()

        while y>0:
            img = mask[y-40:y, self.left_base-self.window_width:self.left_base+self.window_width]
            contours, _ = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"    ]/M["m00"])
                    cy = int(M["m01"]/M["m00"])
                    leftx_cords.append(self.left_base-self.window_width + cx)
                    self.left_base = self.left_base-self.window_width + cx

            img = mask[y-40:y, self.right_base-self.window_width:self.right_base+self.window_width]
            contours, _ = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"]/M["m00"])
                    cy = int(M["m01"]/M["m00"])
                    rightx_cords.append(self.right_base-self.window_width + cx)
                    self.right_base = self.right_base-self.window_width + cx

            cv2.rectangle(mask_copy,(self.left_base-self.window_width,y),(self.left_base+self.window_width,y-40),(255,255,255),2)
            cv2.rectangle(mask_copy,(self.right_base-self.window_width,y),(self.right_base+self.window_width,y-40),(255,255,255),2)

            y-=40
        if len(leftx_cords):
            self.prevlx = leftx_cords
        else:
            leftx_cords = self.prevlx

        if len(rightx_cords):
            self.prevrx = rightx_cords
        else:
            rightx_cords = self.prevrx
        
        return mask_copy, leftx_cords, rightx_cords, y

    def calculate_offset(self, leftx_cords, rightx_cords, y):
        
        min_length = min(len(leftx_cords),len(rightx_cords))

        left_points = [(leftx_cords[i], y + i * 40) for i in range(min_length)]
        right_points = [(rightx_cords[i], y + i * 40) for i in range(min_length)]

        left_fit = np.polyfit([p[1] for p in left_points], [p[0] for p in left_points], 2)
        right_fit = np.polyfit([p[1] for p in right_points], [p[0] for p in right_points], 2)

        y_eval = self.height
        left_curvature = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.abs(2*left_fit[0])
        right_curvature = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.abs(2*right_fit[0])
        curvature = (left_curvature + right_curvature) / 2  

        lane_center = (self.left_base + self.right_base) / 2
        car_position = 320  
        lane_offset = (car_position - lane_center) * 3.7 / 640

        return lane_offset
    
    def process_frame(self,frame : carla.Image):
        frame = self.make_compatible(frame) 
        BEV_transformed_frame = self.get_perspective_transform(frame)
        

        masked_frame = self.add_mask(BEV_transformed_frame)
        self.generate_histogram(masked_frame)

        mask_copy,leftx_cords,rightx_cords,y = self.sliding_window(masked_frame)

        lane_offset = self.calculate_offset(leftx_cords, rightx_cords, y)

        result = self.get_inversed_perspective_tranform(frame,BEV_transformed_frame,leftx_cords,rightx_cords)
        self.latest_mask = mask_copy
        self.latest_result = result
        self.latest_sliding_window = mask_copy
        if cv2.waitKey(1) == 27:
            return 0

        return lane_offset
