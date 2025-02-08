'''import cv2
import numpy as np

# เปิดกล้อง
cap = cv2.VideoCapture(1)

# ตรวจสอบว่ากล้องเปิดได้หรือไม่
if not cap.isOpened():
    print("ไม่สามารถเปิดกล้องได้")
    exit()

# อ่านเฟรมแรกและแปลงเป็น grayscale
ret, prev_frame = cap.read()
if not ret:
    print("ไม่สามารถอ่านภาพจากกล้องได้")
    cap.release()
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
orb = cv2.ORB_create()  # ใช้ ORB สำหรับ Feature Matching

while True:
    # อ่านเฟรมใหม่
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # คำนวณ Optical Flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # === หาการเคลื่อนที่ของกล้องโดยใช้ ORB + RANSAC ===
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(gray, None)

    if des1 is not None and des2 is not None:
        # ใช้ BFMatcher ค้นหาคู่ Feature ที่ตรงกัน
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        if len(matches) > 10:  # ต้องมีอย่างน้อย 10 คู่ที่ match
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # คำนวณ Homography เพื่อลบ Camera Motion
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is not None:
                # แปลง Optical Flow ตาม Camera Motion
                h, w = flow.shape[:2]
                y, x = np.mgrid[0:h, 0:w]
                flow_corrected = cv2.perspectiveTransform(np.stack([x, y], axis=-1).astype(np.float32).reshape(-1, 1, 2), H)
                flow_corrected = flow_corrected.reshape(h, w, 2)

                # ลบ Camera Motion ออกจาก Optical Flow
                flow = flow - (flow_corrected - np.stack([x, y], axis=-1))

    # === แสดงผล Optical Flow ที่ถูกลบ Camera Motion ออกแล้ว ===
    flow_visualization = frame.copy()
    step = 15  # ระยะห่างของลูกศร
    h, w = flow.shape[:2]
    for y in range(0, h, step):
        for x in range(0, w, step):
            fx, fy = flow[y, x]
            cv2.arrowedLine(flow_visualization, (x, y), (int(x + fx), int(y + fy)), (0, 255, 0), 1, tipLength=0.3)

    # แสดงผล
    cv2.imshow("Optical Flow (Camera Motion Removed)", flow_visualization)

    # อัปเดตเฟรมก่อนหน้า
    prev_gray = gray.copy()

    # กด 'q' เพื่อออกจากลูป
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้อง
cap.release()
cv2.destroyAllWindows()
'''

import cv2
import numpy as np

# เปิดกล้อง
cap = cv2.VideoCapture(1)

# ตรวจสอบว่ากล้องเปิดได้หรือไม่
if not cap.isOpened():
    print("ไม่สามารถเปิดกล้องได้")
    exit()

# อ่านเฟรมแรกและแปลงเป็น grayscale
ret, prev_frame = cap.read()
if not ret:
    print("ไม่สามารถอ่านภาพจากกล้องได้")
    cap.release()
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    # อ่านเฟรมใหม่
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # คำนวณ Optical Flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # === หาค่าการเคลื่อนที่ของฉากโดยใช้ Optical Flow-Based Camera Motion Estimation ===
    magnitude = np.linalg.norm(flow, axis=2)  # คำนวณขนาดของ Optical Flow ที่แต่ละจุด
    motion_threshold = np.percentile(magnitude, 80)  # หาค่าที่ 80th percentile (กรองจุดที่มีการเคลื่อนที่น้อยสุด)
    fixed_points = magnitude < motion_threshold  # สร้าง Mask สำหรับพื้นหลัง (Static Background)

    # หาค่าเฉลี่ย Optical Flow เฉพาะพื้นที่ที่ไม่เคลื่อนที่ (Background)
    if np.any(fixed_points):  # ตรวจสอบว่ามีจุด Background หรือไม่
        avg_flow = np.mean(flow[fixed_points], axis=0)  
    else:
        avg_flow = np.array([0, 0])  # ถ้าไม่มีจุดที่นิ่ง ให้ถือว่ากล้องไม่เคลื่อนที่

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
    raw_visualization = draw_flow(frame.copy(), flow)
    corrected_visualization = draw_flow(frame.copy(), flow_corrected, color=(255, 0, 0))  # สีฟ้า

    #cv2.imshow("Raw Optical Flow", raw_visualization)
    cv2.imshow("Corrected Optical Flow (Camera Motion Removed)", corrected_visualization)

    # อัปเดตเฟรมก่อนหน้า
    prev_gray = gray.copy()

    # กด 'q' เพื่อออกจากลูป
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้อง
cap.release()
cv2.destroyAllWindows()
