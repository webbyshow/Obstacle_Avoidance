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

# ตั้งค่ากล้อง Stereo
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

# เปิดกล้องซ้ายและขวา (เปลี่ยน index ถ้ากล้องเป็นตัวอื่น)
cap_left = cv2.VideoCapture(0)
cap_right = cv2.VideoCapture(1)

if not cap_left.isOpened() or not cap_right.isOpened():
    print("ไม่สามารถเปิดกล้องได้")
    exit()

# อ่านเฟรมแรก
ret1, prev_left = cap_left.read()
ret2, prev_right = cap_right.read()
if not ret1 or not ret2:
    print("ไม่สามารถอ่านภาพจากกล้องได้")
    cap_left.release()
    cap_right.release()
    exit()

prev_gray = cv2.cvtColor(prev_left, cv2.COLOR_BGR2GRAY)
prev_right_gray = cv2.cvtColor(prev_right, cv2.COLOR_BGR2GRAY)

while True:
    # อ่านเฟรมใหม่
    ret1, frame_left = cap_left.read()
    ret2, frame_right = cap_right.read()
    if not ret1 or not ret2:
        break

    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    # คำนวณ Disparity Map
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

    # กำจัดค่า disparity ที่เป็นศูนย์หรือน้อยกว่า
    mask = (disparity <= 0).astype(np.uint8)
    disparity_inpainted = cv2.inpaint(disparity, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # Normalize Disparity Map ให้เห็นได้ชัด
    disp_vis = cv2.normalize(disparity_inpainted, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # === Optical Flow Calculation ===
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray_left, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # === แยก Background และ Foreground โดยใช้ Depth ===
    depth_threshold = np.percentile(disparity_inpainted, 70)  # เอา 70% เป็น Background
    background_mask = disparity_inpainted > depth_threshold  # Pixels ที่ลึกกว่าค่า threshold ถือเป็น Background

    # คำนวณ Camera Motion เฉพาะ Background Pixels
    if np.any(background_mask):
        avg_flow = np.mean(flow[background_mask], axis=0)  
    else:
        avg_flow = np.array([0, 0])  # ถ้าไม่มีจุด Background ให้ถือว่ากล้องไม่เคลื่อนที่

    # ลบ Camera Motion ออกจาก Optical Flow
    flow_corrected = flow - avg_flow

    # === วาด Optical Flow ===
    def draw_flow(img, flow, step=15, color=(0, 255, 0)):
        h, w = flow.shape[:2]
        for y in range(0, h, step):
            for x in range(0, w, step):
                fx, fy = flow[y, x]
                cv2.arrowedLine(img, (x, y), (int(x + fx), int(y + fy)), color, 1, tipLength=0.3)
        return img

    # แสดง Optical Flow ดิบ และ Optical Flow ที่ถูกลบ Camera Motion
    raw_visualization = draw_flow(frame_left.copy(), flow)
    corrected_visualization = draw_flow(frame_left.copy(), flow_corrected, color=(255, 0, 0))  # สีฟ้า

    # แสดงผล
    cv2.imshow("Left Camera", frame_left)
    cv2.imshow("Right Camera", frame_right)
    cv2.imshow("Disparity Map", disp_vis)
    cv2.imshow("Raw Optical Flow", raw_visualization)
    cv2.imshow("Corrected Optical Flow (Camera Motion Removed)", corrected_visualization)

    # อัปเดตเฟรมก่อนหน้า
    prev_gray = gray_left.copy()

    # กด 'q' เพื่อออกจากลูป
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้อง
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
