import cv2
import numpy as np

# โหลดค่าพารามิเตอร์จากไฟล์ calibration
fs = cv2.FileStorage('stereo_params.yml', cv2.FILE_STORAGE_READ)
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

# ตั้งค่าพารามิเตอร์ของ StereoSGBM
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

# เปิดกล้องซ้ายและขวา (เปลี่ยนหมายเลขกล้องถ้าจำเป็น)
cap_left = cv2.VideoCapture(0)
cap_right = cv2.VideoCapture(1)

if not cap_left.isOpened() or not cap_right.isOpened():
    print("ไม่สามารถเปิดกล้องได้")
    exit()

while True:
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()

    if not ret_left or not ret_right:
        print("ไม่สามารถอ่านภาพจากกล้อง")
        break

    # แปลงเป็น Grayscale
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    # คำนวณ Disparity Map
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

    # กำจัดค่า disparity ที่เป็นศูนย์หรือน้อยกว่า
    mask = (disparity <= 0).astype(np.uint8)
    disparity_inpainted = cv2.inpaint(disparity, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # แปลง Disparity Map เป็น Depth Map
    depth_map = cv2.reprojectImageTo3D(disparity_inpainted, Q)

    # Normalize Depth Map เพื่อให้มองเห็นได้ง่ายขึ้น
    disp_vis = cv2.normalize(disparity_inpainted, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)

    # แสดงผล
    cv2.imshow("Left Camera", frame_left)
    cv2.imshow("Right Camera", frame_right)
    cv2.imshow("Disparity Map", disp_vis)

    # กด 'q' เพื่อออกจากลูป
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้อง
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
