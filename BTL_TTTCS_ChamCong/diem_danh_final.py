import cv2
import numpy as np
from insightface.app import FaceAnalysis
import os
from datetime import datetime

# ==========================================================
# 1. THIẾT LẬP HỆ THỐNG (SETUP)
# ==========================================================
# Khởi tạo AI (Sử dụng CPU, nếu có GPU thì đổi thành CUDAExecutionProvider)
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

known_embeddings = []
known_names = []
known_ids = []
known_roles = [] # Lưu: 'Admin' hoặc 'Staff'

# Tự động nạp dữ liệu từ thư mục ảnh mẫu
# Yêu cầu tên file: ChucVu_MaSV_Ten.jpg (VD: Admin_B23DCCN224_Duong.jpg)
folder_path = "data_faces"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"[*] Da tao thu muc {folder_path}. Hay bo anh vao day roi chay lai!")

for filename in os.listdir(folder_path):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        try:
            parts = filename.split("_")
            role = parts[0]
            user_id = parts[1]
            user_name = parts[2].split(".")[0]
            
            img = cv2.imread(os.path.join(folder_path, filename))
            res = app.get(img)
            if res:
                known_embeddings.append(res[0].normed_embedding)
                known_roles.append(role)
                known_ids.append(user_id)
                known_names.append(user_name)
                print(f"[+] Da nap: {role} - {user_name}")
        except Exception as e:
            print(f"[!] Loi dinh dang file {filename}: {e}")

# Biến quản lý dữ liệu trong phiên làm việc
# Cấu trúc: { "ID": {"name": "...", "role": "...", "in": datetime, "out": datetime} }
attendance_data = {}
current_admin_online = None

# ==========================================================
# 2. VÒNG LẶP NHẬN DIỆN VÀ CHẤM CÔNG
# ==========================================================
cap = cv2.VideoCapture(0)
print("\n>>> He thong dang chay. Nhan 'Q' de thoat.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    faces = app.get(frame)
    now = datetime.now()
    current_admin_online = None # Reset mỗi khung hình

    for face in faces:
        emb = face.normed_embedding
        # Thuật toán so khớp Cosine Similarity
        scores = np.dot(known_embeddings, emb)
        best_idx = np.argmax(scores)
        
        # Ngưỡng nhận diện (Threshold)
        if scores[best_idx] > 0.45:
            user_id = known_ids[best_idx]
            user_name = known_names[best_idx]
            user_role = known_roles[best_idx]
            
            # Nếu là Admin đang đứng trước máy
            if user_role == "Admin":
                current_admin_online = user_name

            # Logic Chấm công: Ghi nhận lần đầu (In) và cập nhật lần cuối (Out)
            if user_id not in attendance_data:
                attendance_data[user_id] = {
                    "name": user_name,
                    "role": user_role,
                    "in": now,
                    "out": now
                }
                print(f"[*] {user_name} ({user_role}) vao ca luc: {now.strftime('%H:%M:%S')}")
            else:
                attendance_data[user_id]["out"] = now

            # Hiển thị UI trên khung hình
            color = (0, 255, 0) if user_role == "Admin" else (255, 255, 0)
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, f"[{user_role}] {user_name}", (bbox[0], bbox[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            # Người lạ (Unknown)
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown", (bbox[0], bbox[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # ==========================================================
    # 3. QUẢN LÝ PHÂN QUYỀN VÀ XUẤT BÁO CÁO
    # ==========================================================
    key = cv2.waitKey(1) & 0xFF
    
    # Chỉ Admin mới thấy dòng hướng dẫn xuất file
    if current_admin_online:
        cv2.putText(frame, f"Admin {current_admin_online} - Bam 'S' de xuat bao cao", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if key == ord('s') or key == ord('S'):
            filename = f"Bao_cao_{now.strftime('%Y%m%d_%H%M%S')}.csv"
            with open(filename, "w", encoding="utf-8-sig") as f:
                f.write("Ma SV,Ho Ten,Chuc Vu,Gio Vao,Gio Ra,Tong Phut\n")
                for uid, data in attendance_data.items():
                    duration = (data['out'] - data['in']).total_seconds() / 60
                    f.write(f"{uid},{data['name']},{data['role']},{data['in'].strftime('%H:%M:%S')},"
                            f"{data['out'].strftime('%H:%M:%S')},{duration:.1f}\n")
            print(f"\n[OK] Da xuat bao cao bao mat: {filename}")
    else:
        cv2.putText(frame, "Che do Nhan vien: Dang quet...", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if key == ord('s'):
            print("\n[!] Canh bao: Ban khong co quyen xuat du lieu!")

    cv2.imshow("He thong Diem danh Phan quyen AI - PTIT", frame)
    if key == ord('q') or key == ord('Q'): break

cap.release()
cv2.destroyAllWindows()