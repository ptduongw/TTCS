import cv2
import numpy as np
from insightface.app import FaceAnalysis

# 1. Khởi tạo InsightFace (Lần đầu chạy sẽ hơi lâu vì nó phải tải model về máy)
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # 2. AI bắt đầu quét khuôn mặt
    faces = app.get(frame)

    for face in faces:
        # Tự động lấy tọa độ khuôn mặt mà AI tìm thấy
        bbox = face.bbox.astype(int)
        
        # Vẽ khung xanh ôm sát mặt (thay cho khung tím cố định)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        # Hiện thông tin tuổi và giới tính mà AI đoán được
        label = f"{face.gender} | {int(face.age)}t"
        cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("He thong nhan dien AI", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
