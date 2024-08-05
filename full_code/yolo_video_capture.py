# YOLO 모델을 초기화하고 비디오에서 프레임을 읽는 기능을 제공
# 비디오 파일에서 프레임을 일정한 간격으로 추출하여 처리할 수 있게 함

import cv2
from ultralytics import YOLO

class YOLOVideoCapture:
    def __init__(self, model_path, video_path):
        # YOLO 모델 초기화
        self.model = YOLO(model_path)
        self.model.overrides['verbose'] = False

        # 비디오 캡처 초기화
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_interval = int(self.fps / 5)
    
    def read_frames(self):
        frames = []
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            if int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) % self.frame_interval == 0:
                frames.append(frame)
                if len(frames) == 15:
                    yield frames
                    frames = []
        
        if frames:
            yield frames
    
    def release(self):
        self.cap.release()
