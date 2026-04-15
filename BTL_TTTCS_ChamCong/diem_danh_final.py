import cv2
import numpy as np
from insightface.app import FaceAnalysis
import os
from datetime import datetime

# 1. Khởi tạo AI (Bộ não)
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# 2. Nạp ảnh mẫu của Duong để máy "học" mặt
img_path = "data_faces/duong.jpg"
if not os.path.exists(img_path):
    print(f"Lỗi: Không tìm thấy ảnh {img_path} rồi ông ơi!")
    exit()

img_sample = cv2.imread(img_path)
faces_sample = app.get(img_sample)

if len(faces_sample) == 0:
    print("AI không tìm thấy mặt trong ảnh mẫu. Ông chụp lại ảnh khác nhé!")
    exit()

# Lấy mã định danh (Embedding) của Duong
embedding_duong = faces_sample[0].normed_embedding

# 3. Mở Camera để bắt đầu điểm danh
cap = cv2.VideoCapture(0)

print("Hệ thống đang chạy... Nhìn vào Camera đi ông!")

last_recorded_times = {} 
COOLDOWN_MINUTES = 1 # Số phút giãn cách

while True:
    ret, frame = cap.read()
    if not ret: break

    # Quét khuôn mặt đang đứng trước cam
    faces = app.get(frame)

    for face in faces:
        bbox = face.bbox.astype(int)
        embedding_unknown = face.normed_embedding # Mã của người đang đứng trước cam
        
        # So sánh độ giống nhau giữa Cam và Ảnh mẫu (Cosine Similarity)
        # Điểm số càng gần 1.0 thì càng giống nhau
        score = np.dot(embedding_duong, embedding_unknown)
        
        if score > 0.45:
            name = "Duong"
            now = datetime.now()
            
            # Kiểm tra xem người này đã điểm danh trước đó chưa
            if name in last_recorded_times:
                last_time = last_recorded_times[name]
                # Tính khoảng cách thời gian (tính bằng giây)
                diff_seconds = (now - last_time).total_seconds()
                diff_minutes = diff_seconds / 60
            else:
                # Nếu là lần đầu tiên trong phiên chạy này
                diff_minutes = COOLDOWN_MINUTES + 1 

            # CHỈ GHI FILE VÀ HIỆN TÊN NẾU ĐÃ QUA 5 PHÚT
            if diff_minutes >= COOLDOWN_MINUTES:
                color = (255, 0, 255) # Màu tím khi ghi nhận thành công
                
                # Cập nhật lại thời gian ghi nhận mới nhất
                last_recorded_times[name] = now
                
                # Ghi vào file CSV
                time_str = now.strftime("%H:%M:%S")
                date_str = now.strftime("%Y-%m-%d")
                with open("attendance.csv", "a") as f:
                    f.write(f"{name},{date_str},{time_str}\n")
                
                print(f">>> Đã ghi nhận điểm danh cho {name} lúc {time_str}")
            else:
                # Nếu chưa đủ 5 phút, mình đổi màu khung để báo hiệu đang đợi
                name = f"Waiting ({int(COOLDOWN_MINUTES - diff_minutes)}m left)"
                color = (0, 255, 255) # Màu vàng báo hiệu đang chờ
        else:
            name = "Unknown"
            color = (0, 0, 255)

        # Vẽ khung và hiện tên lên màn hình
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(frame, f"{name} ({score:.2f})", (bbox[0], bbox[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("He thong Diem danh AI - PTIT", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
