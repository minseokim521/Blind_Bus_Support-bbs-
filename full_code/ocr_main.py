# 전체 프로그램의 실행을 관리하는 클래스인 `OCRMain`을 정의
# YOLO 모델과 비디오 캡처, 데이터베이스 연결, 그리고 프레임 처리 기능을 초기화하고 실행

import threading
from queue import Queue
import easyocr
from yolo_video_capture import YOLOVideoCapture
from database_connection import DatabaseConnection
from frame_processor import FrameProcessor

class OCRMain:
    def __init__(self, video_path, model_path, dbname, user, password, host, port):
        self.yolo_video_capture = YOLOVideoCapture(model_path, video_path)
        self.db = DatabaseConnection(dbname, user, password, host, port)
        self.reader = easyocr.Reader(['en'], gpu=True)
        plate_class_names = ['front_num', 'side_num', 'back_num']
        self.plate_class_indices = [idx for idx, name in self.yolo_video_capture.model.names.items() if name in plate_class_names]
        self.padding = 5
        self.min_confidence = 0.9

    def run(self):
        frame_queue = Queue()
        frame_processor = FrameProcessor(
            self.yolo_video_capture.model, self.plate_class_indices, self.reader,
            self.yolo_video_capture.width, self.yolo_video_capture.height, self.padding, self.db.cur, self.db.conn, self.min_confidence
        )
        frame_thread = threading.Thread(target=frame_processor.process_frame, args=(frame_queue,))
        frame_thread.start()

        for frames in self.yolo_video_capture.read_frames():
            frame_queue.put(frames)
        
        frame_queue.put(None)
        frame_thread.join()
        self.yolo_video_capture.release()
        self.db.close()

if __name__ == "__main__":
    main = OCRMain(
        "C:/Users/minseokim521/Desktop/YOLO/busbusbus.mp4",
        "C:/Users/minseokim521/Desktop/best.pt",
        "test", "postgres", "0521", "localhost", "5432"
    )
    main.run()
