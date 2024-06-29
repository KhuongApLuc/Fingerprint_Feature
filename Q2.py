from OperationCpv import *
#đánh dấu trên ảnh các điểm đặc trưng vân tay

def mark_minutiae(image):
    # Tìm contours trên ảnh đã xử lý
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Tạo một bản sao của ảnh để vẽ các điểm đặc trưng
    marked_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Duyệt qua tất cả các contours
    for contour in contours:
        for point in contour:
            # Lấy tọa độ của điểm
            x, y = point[0]

            # Vẽ một hình tròn tại điểm đặc trưng
            cv2.circle(marked_img, (x, y), 3, (0, 0, 255), -1)  # Màu đỏ

    return marked_img

def main():
    # Đọc ảnh từ phần cuối của đoạn mã bạn đã cung cấp
    closing = cv2.imread('final_result.jpg', cv2.IMREAD_GRAYSCALE)

    # Đánh dấu các điểm đặc trưng trên ảnh đã xử lý
    marked_image = mark_minutiae(closing)

    # Hiển thị ảnh đã đánh dấu các điểm đặc trưng
    display_image(marked_image, 'Minutiae Marked Image')

if __name__ == "__main__":
    main()