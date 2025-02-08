import cv2
import numpy as np
import torch

# โหลดโมเดล YOLO (ใช้เวอร์ชันที่รองรับ PyTorch เช่น yolov5)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# พารามิเตอร์ Calibration ของกล้อง
sensor_width_mm = 1.177  # ความกว้างของเซ็นเซอร์ (mm)
focal_length_px = 543.78  # ระยะโฟกัส (px)
image_width_px = 640  # ความกว้างของภาพ (px)
baseline_m = 9.7 / 100  # ระยะห่างระหว่างกล้อง (เมตร)

# โหลดค่าพารามิเตอร์จากไฟล์ calibration
fs = cv2.FileStorage('D:/EE Engineer/Year4/project/Code/stereo_params.yml', cv2.FILE_STORAGE_READ)
M1 = fs.getNode('M1').mat()
D1 = fs.getNode('D1').mat()
M2 = fs.getNode('M2').mat()
D2 = fs.getNode('D2').mat()
R1 = fs.getNode('R1').mat()
R2 = fs.getNode('R2').mat()
P1 = fs.getNode('P1').mat()
P2 = fs.getNode('P2').mat()
Q = fs.getNode('Q').mat()
fs.release()

# ค่าพารามิเตอร์ของ StereoSGBM
minDisparity = 0
numDisparities = 16 * 8
blockSize = 7
P1 = 8 * 3 * blockSize**2
P2 = 32 * 3 * blockSize**2
disp12MaxDiff = 1
preFilterCap = 63
uniquenessRatio = 10
speckleWindowSize = 50
speckleRange = 1
mode = cv2.STEREO_SGBM_MODE_SGBM

# สร้างตัวคำนวณ Stereo Matcher
stereo = cv2.StereoSGBM_create(
    minDisparity=minDisparity,
    numDisparities=numDisparities,
    blockSize=blockSize,
    P1=P1,
    P2=P2,
    disp12MaxDiff=disp12MaxDiff,
    preFilterCap=preFilterCap,
    uniquenessRatio=uniquenessRatio,
    speckleWindowSize=speckleWindowSize,
    speckleRange=speckleRange,
    mode=mode,
)

# เปิดวิดีโอแทนกล้องซ้ายและขวา
cap_left = cv2.VideoCapture('C:/Users/user/Downloads/camera1.avi')
cap_right = cv2.VideoCapture('C:/Users/user/Downloads/camera2.avi')

if not cap_left.isOpened() or not cap_right.isOpened():
    print("ไม่สามารถเปิดไฟล์วิดีโอได้")
    exit()

fps = int(cap_left.get(cv2.CAP_PROP_FPS))
print(f"FPS ของวิดีโอ: {fps} เฟรมต่อวินาที")

while True:
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()

    if not ret_left or not ret_right:
        print("สิ้นสุดวิดีโอ")
        break

    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    # คำนวณ disparity map
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
    
    # กำจัดค่า disparity ที่เป็นศูนย์หรือน้อยกว่า
    mask = (disparity <= 0).astype(np.uint8)
    disparity_inpainted = cv2.inpaint(disparity.astype(np.float32), mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # แปลง Disparity Map เป็น Depth Map
    depth_map = cv2.reprojectImageTo3D(disparity_inpainted, Q)

    # ตรวจจับบุคคลในภาพ
    results = model(frame_left)

    for det in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = det
        if int(cls) == 0:  # ตรวจจับเฉพาะคน (Class 0 ใน YOLO)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            depth_value = depth_map[center_y, center_x, 2]
            
            if depth_value > 0:
                text = f"{depth_value:.2f}m"
                cv2.rectangle(frame_left, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame_left, text, (int(x1) + 5, int(y1) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            cv2.rectangle(frame_left, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    cv2.imshow("Stereo Depth & Object Detection", frame_left)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
'''
import cv2
import numpy as np
import torch

# โหลดโมเดล YOLO (ใช้เวอร์ชันที่รองรับ PyTorch เช่น yolov5)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# พารามิเตอร์ Calibration ของกล้อง
sensor_width_mm = 1.177  # ความกว้างของเซ็นเซอร์ (mm)
focal_length_px = 543.78  # ระยะโฟกัส (px)
image_width_px = 640  # ความกว้างของภาพ (px)
baseline_m = 9.7 / 100  # ระยะห่างระหว่างกล้อง (เมตร)

# โหลดค่าพารามิเตอร์จากไฟล์ calibration
fs = cv2.FileStorage('D:/EE Engineer/Year4/project/Code/stereo_params.yml', cv2.FILE_STORAGE_READ)
M1 = fs.getNode('M1').mat()
D1 = fs.getNode('D1').mat()
M2 = fs.getNode('M2').mat()
D2 = fs.getNode('D2').mat()
R1 = fs.getNode('R1').mat()
R2 = fs.getNode('R2').mat()
P1 = fs.getNode('P1').mat()
P2 = fs.getNode('P2').mat()
Q = fs.getNode('Q').mat()
fs.release()

# ค่าพารามิเตอร์ของ StereoSGBM
minDisparity = 0
numDisparities = 16 * 8
blockSize = 7
P1 = 8 * 3 * blockSize**2
P2 = 32 * 3 * blockSize**2
disp12MaxDiff = 1
preFilterCap = 63
uniquenessRatio = 10
speckleWindowSize = 50
speckleRange = 1
mode = cv2.STEREO_SGBM_MODE_SGBM

# สร้างตัวคำนวณ Stereo Matcher
stereo = cv2.StereoSGBM_create(
    minDisparity=minDisparity,
    numDisparities=numDisparities,
    blockSize=blockSize,
    P1=P1,
    P2=P2,
    disp12MaxDiff=disp12MaxDiff,
    preFilterCap=preFilterCap,
    uniquenessRatio=uniquenessRatio,
    speckleWindowSize=speckleWindowSize,
    speckleRange=speckleRange,
    mode=mode,
)

# เปิดวิดีโอแทนกล้องซ้ายและขวา
cap_left = cv2.VideoCapture('C:/Users/user/Downloads/camera1.avi')
cap_right = cv2.VideoCapture('C:/Users/user/Downloads/camera2.avi')

if not cap_left.isOpened() or not cap_right.isOpened():
    print("ไม่สามารถเปิดไฟล์วิดีโอได้")
    exit()

fps = int(cap_left.get(cv2.CAP_PROP_FPS))
frame_width = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))

# กำหนด VideoWriter สำหรับบันทึกวิดีโอ
out = cv2.VideoWriter('D:/EE Engineer/Year4/project/Code/output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

while True:
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()

    if not ret_left or not ret_right:
        print("สิ้นสุดวิดีโอ")
        break

    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    # คำนวณ disparity map
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
    
    # กำจัดค่า disparity ที่เป็นศูนย์หรือน้อยกว่า
    mask = (disparity <= 0).astype(np.uint8)
    disparity_inpainted = cv2.inpaint(disparity.astype(np.float32), mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # แปลง Disparity Map เป็น Depth Map
    depth_map = cv2.reprojectImageTo3D(disparity_inpainted, Q)

    # ตรวจจับบุคคลในภาพ
    results = model(frame_left)

    for det in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = det
        if int(cls) == 0:  # ตรวจจับเฉพาะคน (Class 0 ใน YOLO)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            depth_value = depth_map[center_y, center_x, 2]
            
            if depth_value > 0:
                text = f"{depth_value:.2f}m"
                cv2.rectangle(frame_left, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame_left, text, (int(x1) + 5, int(y1) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    print("Saving frame to video...")
    out.write(frame_left)  # บันทึกเฟรมลงวิดีโอ

cap_left.release()
cap_right.release()
out.release()
cv2.destroyAllWindows()
'''