
import cv2

# Kết nối với Camera (số 0 là camera mặc định của laptop)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không tìm thấy Camera rồi ông ơi!")
    exit()

print("Đang mở Camera... Nhấn phím 'q' để thoát nhé.")

while True:
    # Đọc hình ảnh từ camera
    ret, frame = cap.read()
    if not ret:
        break

    # Lấy kích thước màn hình để vẽ khung cho đẹp
    h, w, _ = frame.shape
    
    # Vẽ một cái khung màu tím (Purple) ở giữa màn hình để "giả vờ" chỗ đặt mặt
    # Màu trong OpenCV là BGR nên Tím là (255, 0, 255)
    cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (255, 0, 255), 2)
    cv2.putText(frame, "De mat vao day de diem danh", (w//4, h//4 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    # Hiển thị cửa sổ
    cv2.imshow("Test Camera BTL", frame)

    # Nhấn 'q' để đóng
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()