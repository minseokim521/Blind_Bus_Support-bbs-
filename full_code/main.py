from func_utils import API
import psycopg2
import requests
import xml.etree.ElementTree as ET
import threading
from queue import Queue
import easyocr
from func_utils import YOLOVideoCapture, DatabaseConnection, FrameProcessor


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

temp = API("5618", "영등포역")

Bus_num = temp.Bus_num
Station_name = temp.Station_name

#데이터 베이스 상에서 버스 번호와 정류소 이름에 해당하는 id값 가져오기
bus_result, station_result = temp.database_query(Bus_num, Station_name)

# api 상에서 station id에 해당하는 정류소에 운행하는 버스정보 가져오기
response1 = temp.station_bus_list(station_result[0])

#xml 값을 가져옴
root1 = ET.fromstring(response1)

# 정류소에서 운행하는 버스 이름 정보 리스트
bus_list = temp.find_xml_val(root1, "busRouteAbrv")

# 버스 리스트에서 인식한 버스 정보가 있는지 찾음
index, result = temp.find_api_val(bus_list, Bus_num)

isArrive1 = []
arrmsg1_list = []
arrmsg2_list = []
if result:
    isArrive1 = temp.find_xml_val(root1, "isArrive1")
    arrmsg1_list = temp.find_xml_val(root1, "arrmsg1")
    arrmsg2_list = temp.find_xml_val(root1, "arrmsg2")


# 변수 출력
if not(isArrive1[index]):
    print("첫번째 버스가 도착했습니다.")
else:
    print("첫번째 버스가 도착하지 않았습니다.")
print("첫번째 버스 도착 예정시간 :", arrmsg1_list[index])
print("두번째 버스 도착 예정시간 :", arrmsg2_list[index])

print("end of the code")
