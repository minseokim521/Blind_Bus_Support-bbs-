# 이미지를 전처리하는 기능을 제공
import cv2

class ImageProcessor:
    @staticmethod
    def preprocess_image(image):
        """이미지 전처리 함수."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
        gray = clahe.apply(gray)
        blurred = cv2.GaussianBlur(cv2.medianBlur(gray, 7), (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, morph_kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, morph_kernel)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 70:
                binary[labels == i] = 0
        
        return binary
