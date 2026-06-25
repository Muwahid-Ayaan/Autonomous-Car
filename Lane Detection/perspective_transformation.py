import cv2
import numpy as np

image = cv2.imread("Latest_frame.png")



success = True
# vidcap = cv2.VideoCapture("LaneVideo.mp4")

# success, image = vidcap.read()

def nothing(x):
    ...

cv2.namedWindow("Trackbars")

cv2.createTrackbar("Lower-H","Trackbars",0,255,nothing)
cv2.createTrackbar("Lower-S","Trackbars",0,255,nothing)
cv2.createTrackbar("Lower-V","Trackbars",175,255,nothing)
cv2.createTrackbar("Upper-H","Trackbars",248,255,nothing)
cv2.createTrackbar("Upper-S","Trackbars",228,255,nothing)
cv2.createTrackbar("Upper-V","Trackbars",255,255,nothing)
     
prevlx = []
prevrx = []

while success:
    
    # success, image = vidcap.read()
    frame = cv2.resize(image,(640,480))
    
    top_left = (265,250)
    top_right = (365,250)
    bottom_left =(0,472)
    bottom_right = (600,472)
    # cv2.circle(frame,top_left,5,(0,0,255),-1)
    # cv2.circle(frame,top_right,5,(0,0,255),-1)
    # cv2.circle(frame,bottom_left,5,(0,0,255),-1)
    # cv2.circle(frame,bottom_right,5,(0,0,255),-1)

    pt1 = np.float32([top_left,bottom_left,top_right,bottom_right])
    pt2 = np.float32([[0,0],[0,480],[640,0],[640,480]])

    matrix = cv2.getPerspectiveTransform(pt1,pt2)
    transformed_frame = cv2.warpPerspective(frame,matrix,(640,480))

    hsv_frame = cv2.cvtColor(transformed_frame,cv2.COLOR_BGR2HSV)

    Lower_H = cv2.getTrackbarPos("Lower-H","Trackbars")
    Lower_S = cv2.getTrackbarPos("Lower-S","Trackbars")
    Lower_V = cv2.getTrackbarPos("Lower-V","Trackbars")
    Upper_H = cv2.getTrackbarPos("Upper-H","Trackbars")
    Upper_S = cv2.getTrackbarPos("Upper-S","Trackbars")
    Upper_V = cv2.getTrackbarPos("Upper-V","Trackbars")


    lower = np.array([Lower_H,Lower_S,Lower_V])
    upper = np.array([Upper_H,Upper_S,Upper_V])
    
    mask = cv2.inRange(hsv_frame,lower,upper)
    result = cv2.bitwise_and(hsv_frame,hsv_frame,mask = mask)
    
    # Historgram
    histogram = np.sum(mask[mask.shape[0]//2:,:], axis=0)
    # histogram = np.sum(mask[380:,:], axis=0)
    midpoint = int(histogram.shape[0]/2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint
    
    #sliding window
    y = 480 
    leftx_cords = []
    rightx_cords = []

    mask_copy = mask.copy()
    lx1 = max(0, left_base - 75)
    lx2 = min(mask.shape[1], left_base + 150)

    img = mask[y-40:y, lx1:lx2]


    print("y:", y)
    print("left_base:", left_base)
    print("img shape:", img.shape)
    print("nonzero:", cv2.countNonZero(img))

    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.imshow("window", img)
    print("contours:", len(contours))
    while y > 0:
        # LEFT
        lx1 = max(0, left_base - 75)
        lx2 = min(mask.shape[1], left_base + 100)
        img = mask[y-40:y, lx1:lx2]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                leftx_cords.append(lx1 + cx)   # use lx1, not left_base - 75
                left_base = lx1 + cx

        # RIGHT
        rx1 = max(0, right_base - 100)
        rx2 = min(mask.shape[1], right_base + 75)
        img = mask[y-40:y, rx1:rx2]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                rightx_cords.append(rx1 + cx)  # use rx1, not right_base - 100
                right_base = rx1 + cx

        cv2.rectangle(mask_copy, (lx1, y), (lx2, y-40), (255,255,255), 2)
        cv2.rectangle(mask_copy, (rx1, y), (rx2, y-40), (255,255,255), 2)

        y -= 40

    if len(leftx_cords):
       prevlx = leftx_cords
    else:
        leftx_cords = prevlx

    if len(rightx_cords):
       prevrx = rightx_cords
    else:
        rightx_cords = prevrx
   
    
    hist_vis = np.zeros((480,640), dtype=np.uint8)

    for x,val in enumerate(histogram):
        cv2.line(
            hist_vis,
            (x,479),
            (x,479-int(val/255)),
            255,
            1
        )

    cv2.imshow("Histogram", hist_vis)
    print(
    "left peak:",
    histogram[left_base],
    "right peak:",
    histogram[right_base]
        )
    print("left_base =", left_base)
    print("left_peak =", histogram[left_base])

    nonzero_x = np.where(histogram > 0)[0]

    print("first 20 nonzero columns:", nonzero_x[:20])
    print("last 20 nonzero columns:", nonzero_x[-20:])

    hist_debug = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)

    cv2.line(
        hist_debug,
        (left_base, 0),
        (left_base, 480),
        (0,255,0),
        3
    )

    cv2.imshow("Histogram Debug", hist_debug)
    min_length = min(len(leftx_cords),len(rightx_cords))


    left_points = [(leftx_cords[i], y + i * 40) for i in range(min_length)]
    right_points = [(rightx_cords[i], y + i * 40) for i in range(min_length)]

    left_fit = np.polyfit([p[1] for p in left_points], [p[0] for p in left_points], 2)
    right_fit = np.polyfit([p[1] for p in right_points], [p[0] for p in right_points], 2)

    y_eval = 480
    left_curvature = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.abs(2*left_fit[0])
    right_curvature = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.abs(2*right_fit[0])
    curvature = (left_curvature + right_curvature) / 2

    lane_center = (left_base + right_base) / 2
    car_position = 320  
    lane_offset = (car_position - lane_center) * 3.7 / 640 
    
 
 
    steering_angle = np.arctan(lane_offset / curvature) * 180 / np.pi

    line_length = 100  
    end_x = int(320 + line_length * np.sin(np.radians(steering_angle)))
    end_y = int(480 - line_length * np.cos(np.radians(steering_angle)))

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

    inv_matrix = cv2.getPerspectiveTransform(pt2,pt1)
    inversed_orignal_frame = cv2.warpPerspective(transformed_frame, inv_matrix, (640,480))
 
 
    result = cv2.addWeighted(frame, 1, inversed_orignal_frame, 0.5, 0)
    cv2.line(result, (320, 480), (end_x, end_y), (255, 0, 0), 2)
    cv2.putText(result, f'Curvature: {curvature:.2f} m', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, f'Offset: {lane_offset:.2f} m', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, f'Angle: {steering_angle:.2f} deg', (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    cv2.line(mask_copy, (left_base, 480), (left_base, 0), 255, 2)
    cv2.line(mask_copy, (right_base, 480), (right_base, 0), 255, 2)

    cv2.imshow("Final Result",result)
    cv2.imshow("Inversed transform",inversed_orignal_frame)
    
    cv2.imshow("Orignal",frame)
    cv2.imshow("BEV ",transformed_frame)
    # cv2.imshow("HSV", hsv_frame)
    cv2.imshow("mask", mask)
    cv2.imshow("Sliding window",mask_copy)
    # cv2.imshow("result",result)

    # break
    if cv2.waitKey(1) == 27:
        break