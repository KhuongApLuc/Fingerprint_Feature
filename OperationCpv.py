import cv2
import numpy as np

# Đọc ảnh fingerprint.tif
img = cv2.imread('fingerprint.tif', cv2.IMREAD_GRAYSCALE)

# Hiển thị ảnh gốc
cv2.imshow('Original Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

def display_image(image, window_name='Image'):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def main():
    # Đọc ảnh fingerprint
    img = cv2.imread('fingerprint.tif', cv2.IMREAD_GRAYSCALE)

    # Thực hiện phép toán co
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=1)
    display_image(erosion, 'Erosion')
    # Thực hiện phép toán giãn
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    display_image(dilation, 'Dilation')
    # Thực hiện phép toán mở
    opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel)
    display_image(opening, 'Opening')
    # Thực hiện phép toán đóng
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    display_image(closing, 'Closing')
    # Hiển thị kết quả cuối cùng
    display_image(closing, 'Final Result')

    cv2.imwrite('final_result.jpg', closing)

if __name__ == "__main__":
    main()