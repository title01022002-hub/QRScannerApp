import cv2
import socket
import time
import numpy as np
import threading

# Khởi tạo biến toàn cục
current_item_code = "1"
trigger_signal = None
qr_history = []
trigger_signal = False
trigger_lock = threading.Lock()

# Địa chỉ IP và cổng của PLC
PLC_IP = "192.168.0.10"
PLC_PORT = 8501

def process_image1(image, blur_strength, sharpness_strength, alpha, beta):
    """Xử lý ảnh cho mã hàng 1."""
    try:
        if image is None or image.size == 0:
            print("Ảnh đầu vào không hợp lệ.")
            return None, ""


        # Cân bằng sáng và điều chỉnh độ tương phản
        adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        # Xác định vùng ROI
        height, width = image.shape[:2]
        roi_x_start = int(1.5 * width / 6)
        roi_x_end = int(3.5 * width / 6)
        roi_y_start = int(1.5 * height / 6)
        roi_y_end = int(4 * height / 6)

        # Cắt vùng ROI
        roi = adjusted_image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

        # Chuyển đổi ảnh ROI sang màu xám
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Áp dụng bộ lọc Gaussian để làm mịn ảnh
        blurred = cv2.GaussianBlur(gray, (7, 7), blur_strength)

        # Tăng cường độ nét ảnh
        kernel = np.array([[-1, -1, -1], [-1, 9 + sharpness_strength, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(blurred, -1, kernel)

        _, thresh = cv2.threshold(sharpened, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Phát hiện và giải mã mã QR
        qr_detector = cv2.QRCodeDetector()
        decoded_text, points, _ = qr_detector.detectAndDecode(thresh)



        colored_image = image.copy()

        if points is not None:
            points = np.int32(points).reshape(-1, 2)
            points += np.array([roi_x_start, roi_y_start])
            print(f"Tọa độ: {points.tolist()}")
            for i in range(len(points)):
                cv2.line(colored_image, tuple(points[i]), tuple(points[(i + 1) % len(points)]), (0, 255, 0), 2)
            cv2.putText(colored_image, decoded_text, (points[0][0], points[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

        cv2.rectangle(colored_image, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (255, 0, 0), 2)
        cv2.imshow("Hình ảnh sau khi xử lý", thresh)
        cv2.imshow("Vùng ROI", roi)

        return colored_image, decoded_text

    except cv2.error as e:
        print(f"Lỗi OpenCV: {e}")
        return None, ""
    except Exception as e:
        print(f"Lỗi không xác định: {e}")
        return None, ""

def process_image2(image, blur_strength, sharpness_strength, alpha, beta):
    """Xử lý ảnh để phát hiện mã QR với các tham số khác nhau."""
    try:
        if image is None or image.size == 0:
            print("Ảnh đầu vào không hợp lệ.")
            return None, ""

        # Cân bằng sáng và điều chỉnh độ tương phản
        adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        # Xác định vùng ROI
        height, width = image.shape[:2]
        roi_x_start = int(1.5 * width / 6)
        roi_x_end = int(3.5 * width / 6)
        roi_y_start = int(1.5 * height / 6)
        roi_y_end = int(3.5 * height / 6)

        # Cắt vùng ROI
        roi = adjusted_image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

        # Chuyển đổi ảnh ROI sang màu xám
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        inverted_roi = cv2.bitwise_not(gray)
        # Áp dụng bộ lọc Gaussian để làm mịn ảnh
        blurred = cv2.GaussianBlur(inverted_roi, (7, 7), blur_strength)

        # Tăng cường độ nét ảnh
        kernel = np.array([[-1, -1, -1], [-1, 9 + sharpness_strength, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(blurred, -1, kernel)

        _, thresh = cv2.threshold(sharpened, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Phát hiện và giải mã mã QR
        qr_detector = cv2.QRCodeDetector()
        decoded_text, points, _ = qr_detector.detectAndDecode(thresh)

        colored_image = image.copy()

        if points is not None:
            points = np.int32(points).reshape(-1, 2)
            points += np.array([roi_x_start, roi_y_start])
            print(f"Tọa độ: {points.tolist()}")
            for i in range(len(points)):
                cv2.line(colored_image, tuple(points[i]), tuple(points[(i + 1) % len(points)]), (0, 255, 0), 2)
            cv2.putText(colored_image, decoded_text, (points[0][0], points[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

        cv2.rectangle(colored_image, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (255, 0, 0), 2)
        cv2.imshow("Hình ảnh sau khi xử lý", thresh)
        cv2.imshow("Vùng ROI", roi)

        return colored_image, decoded_text

    except cv2.error as e:
        print(f"Lỗi OpenCV: {e}")
        return None, ""
    except Exception as e:
        print(f"Lỗi không xác định: {e}")
        return None, ""

def find_optimal_settings(image,process_function):
    if image is None or image.size == 0:
        return None,"",{}
    best_image = None
    best_text = ""
    best_params ={"blur": 0, "sharpness":0, "alpha":1, "beta":0}
    blur_strengths = [0, 1, 2]
    sharpness_strengths =[0, 1, 2]
    beta_values = range(-50,50,5)
    alpha_values = np.linspace(0.5, 5.0, 10)

    for blur in blur_strengths:
        for sharpness in sharpness_strengths:
            for alpha in alpha_values:
                for beta in beta_values:
                    process_image, decoded_text = process_function(image, blur, sharpness, alpha, beta)
                    if decoded_text:
                        best_image = process_image
                        best_text = decoded_text
                        best_params ={"blur": blur, "sharpness":sharpness, "alpha":alpha, "beta":beta}
                        print(f"Thong so tot nhat Blur: {blur}, sharpness: {sharpness}, alpha: {alpha}, beta: {beta}")
                        return process_image, decoded_text,{"blur": blur, "sharpness":sharpness, "alpha":alpha, "beta":beta}
    return best_image, best_text, best_params

# Hàm tạo kết nối TCP đến PLC
def create_connection():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(15)
        sock.connect((PLC_IP, PLC_PORT))
        print("✅ Kết nối thành công đến PLC")
        return sock
    except Exception as e:
        print(f"⚠ Lỗi kết nối: {e}")
        return None

# Hàm đọc dữ liệu từ DM10000
def read_item_code(conn):
    try:
        command = "RDS DM10000 1\r\n"
        conn.send(command.encode())
        #time.sleep(0.05)
        response = conn.recv(1024).decode().strip()
        if response and response.isdigit():
            return str(int(response))
        else:
            print("⚠ Không nhận được dữ liệu hợp lệ từ DM10000")
            return None
    except socket.timeout:
        print("⚠ Timeout khi đọc DM10000")
        return None
    except Exception as e:
        print(f"⚠ Lỗi đọc DM10000: {e}")
        return None

# Hàm đọc trạng thái R300
def read_trigger(conn):
    try:
        command = "RD R300\r\n"
        conn.send(command.encode())
        #time.sleep(0.05)
        response = conn.recv(1024).decode().strip()
        if response in ["0", "1"]:
            return response
        else:
            print("⚠ Phản hồi từ R300 không hợp lệ")
            return None
    except socket.timeout:
        print("⚠ Timeout khi đọc R300")
        return None
    except Exception as e:
        print(f"⚠ Lỗi đọc R300: {e}")
        return None

# Hàm ghi dữ liệu vào thanh ghi
def write_data(conn, address, value):
    try:
        if "DM" in address:
            command = f"WRS {address} 1 {value}\r\n"
        else:
            command = f"WR {address} {value}\r\n"
        conn.send(command.encode())
        #time.sleep(0.05)
        response = conn.recv(1024).decode().strip()
        return response
    except socket.timeout:
        print("⚠ Timeout khi ghi dữ liệu")
        return None
    except Exception as e:
        print(f"⚠ Lỗi ghi dữ liệu: {e}")
        return None

# Hàm gửi kết quả QR hoặc trạng thái NG về PLC
def save_qr_result(conn, qr_text):
    try:
        base_address = 20000
        if qr_text and qr_text != "NG":
            # Cắt chuỗi tối đa 23 ký tự để tránh vượt quá bộ nhớ
            qr_text = qr_text[:23]
            # Lặp qua từng cặp ký tự
            for i in range(0, len(qr_text), 2):
                # Lấy cặp ký tự
                char_pair = qr_text[i:i+2]
                # Nếu chỉ có 1 ký tự trong cặp cuối (độ dài lẻ), thêm ký tự '0'
                if len(char_pair) == 1:
                    char_pair += ' '
                # Tính giá trị 16-bit: ký tự đầu (high byte) << 8 + ký tự sau (low byte)
                value = (ord(char_pair[0]) << 8) + ord(char_pair[1])
                address = f"DM{base_address + i // 2}"
                write_data(conn, address, value)
            #print(f"✅ Đã gửi kết quả QR: {qr_text}")
        else:
            # Gửi "NG" khi không có mã QR hợp lệ
            write_data(conn, "DM20000", (ord('N') << 8) + ord('G'))
            write_data(conn, "DM20001", 0)  # Thanh ghi tiếp theo là 0
            print("❌ Đã gửi kết quả NG")
    except Exception as e:
        print(f"⚠ Lỗi gửi dữ liệu: {e}")

# Trong plc_monitor()
def plc_monitor():
    global current_item_code, trigger_signal
    while True:
        conn = create_connection()
        if not conn:
            time.sleep(1)
            continue

        last_trigger = "0"
        while True:
            try:
                trigger = read_trigger(conn)
                if trigger is None:
                    break

                if trigger != last_trigger:
                    print(f"📡 Phản hồi từ R300: '{trigger}'")

                with trigger_lock:
                    if trigger == "1" and last_trigger == "0":
                        trigger_signal = True
                        item_code = read_item_code(conn)
                        if item_code:
                            print(f"📡 Phản hồi từ DM10000: '{item_code}'")
                            current_item_code = item_code
                            print(f"🔄 Trigger ON - Mã hàng: {current_item_code}")
                    elif trigger == "0" and last_trigger == "1":
                        trigger_signal = False
                        #print("🚦 Trigger OFF")

                last_trigger = trigger
                #time.sleep(0.05)
            except Exception as e:
                print(f"⚠ Lỗi PLC: {e}")
                break

        conn.close()
        time.sleep(1)


def qr_draw(qr_history, width, height):
    history_frame = np.ones((height, width // 2, 3), dtype=np.uint8) * 255
    font_scale, line_height = 0.5, 30
    for i, text in enumerate(qr_history[-30:]):
        y_position = 30 + i * line_height
        if y_position < height:
            cv2.putText(history_frame, text, (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)
    return history_frame

# Khởi tạo camera
#cap = None
#for i in range(4):
    #cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    #if cap.isOpened():
        #print(f"✅ Camera mở thành công với chỉ số {i}")
        #break
    #cap.release()
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Khong mo duoc Camera")
    exit()
#if current_item_code is None :
    #print("Ko doc duoc ma hang")
    #exit()
#if not cap or not cap.isOpened():
    #print("❌ Không mở được Camera")
    #exit()

# Khởi động luồng giám sát PLC
plc_thread = threading.Thread(target=plc_monitor, daemon=True)
plc_thread.start()

# Vòng lặp chính
while True:
    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        print("⚠ Không thể đọc khung hình từ camera")
        cv2.imshow('Camera', np.ones((480, 640, 3), dtype=np.uint8) * 0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    height, width = frame.shape[:2]
    roi_x_start = int(1.5 * width / 6)
    roi_x_end = int(3.5 * width / 6)
    roi_y_start = int(1.5 * height / 6)
    roi_y_end = int(3.5 * height / 6)
    cv2.rectangle(frame, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (255, 0, 0), 2)
    history_frame = qr_draw(qr_history, width, height)
    combined_frame = np.hstack((frame, history_frame))

    if len(qr_history) >= 10:
        qr_history = qr_history[-10:]

    with trigger_lock:
        if trigger_signal:
            decoded_text = ""
            start_time = time.time()
            if current_item_code == "1":
                _, _, optimal_params = find_optimal_settings(frame, process_image1)
                processed_frame, decoded_text = process_image1(
                    frame,
                    blur_strength=optimal_params["blur"],
                    sharpness_strength=optimal_params["sharpness"],
                    alpha=optimal_params["alpha"],
                    beta=optimal_params["beta"]
                )
            elif current_item_code == "2":
                _, _, optimal_params = find_optimal_settings(frame, process_image2)
                processed_frame, decoded_text = process_image2(
                    frame,
                    blur_strength=optimal_params["blur"],
                    sharpness_strength=optimal_params["sharpness"],
                    alpha=optimal_params["alpha"],
                    beta=optimal_params["beta"]
                )

            conn = create_connection()
            if conn:
                if decoded_text:
                    print(f"📸 Mã QR: {decoded_text}")
                    qr_history.append(decoded_text)
                    save_qr_result(conn, decoded_text)
                    history_frame = qr_draw(qr_history, width, height)
                    combined_frame = np.hstack((processed_frame, history_frame))
                else:
                    print("❌ NG: Không đọc được mã QR")
                    save_qr_result(conn, "NG")
                conn.close()

            trigger_signal = False  # Reset sau khi xử lý
            end_time = time.time()
            print(f"⏱ Tổng thời gian đọc mã QR: {end_time - start_time:.2f}s")

    cv2.imshow('Camera', combined_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()