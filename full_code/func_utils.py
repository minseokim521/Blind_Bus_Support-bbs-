import psycopg2
import requests
import xml.etree.ElementTree as ET
import cv2
from collections import Counter
import cv2
from ultralytics import YOLO


# YOLO 모델을 초기화하고 비디오에서 프레임을 읽는 기능을 제공
# 비디오 파일에서 프레임을 일정한 간격으로 추출하여 처리할 수 있게 함
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

# 데이터베이스와의 연결을 관리하고, 특정 번호판 번호를 조회하는 기능을 제공
class DatabaseConnection:
    def __init__(self, dbname, user, password, host, port):
        self.conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
        self.cur = self.conn.cursor()
    
    def query_number(self, number):
        query = "SELECT num FROM bus_number WHERE num = %s;"
        self.cur.execute(query, (number,))
        return self.cur.fetchone()
    
    def close(self):
        self.cur.close()
        self.conn.close()


class API():
    def __init__(self, Bus_num, Station_name):
        self.Bus_num = Bus_num
        self.Station_name = Station_name
            
    def database_query(self, Bus_num, Station_name):
        #데이터베이스에 연결시도
        try:
            # 데이터베이스 연결 & 커넥트 객체 생성
            conn = psycopg2.connect(host="127.0.0.1", dbname="postgres", user="postgres", password="postgres")
        except:
            print("Not Connected!.")

        # 쿼리를 수행하는 cursor객체 생성
        cursor = conn.cursor()

        # 쿼리문
        sql_bus = "SELECT routeid FROM bus WHERE bus_id = %s;"
        sql_station = "SELECT node_id FROM station WHERE station_name = %s;"
        # 쿼리실행
        cursor.execute(sql_bus, (Bus_num,))
        #fetchone은 쿼리에 해당하는 열을 튜플형태로 반환
        bus_result = cursor.fetchone()

        # 예외 처리
        if bus_result:
            print("Bus Route ID :", bus_result[0])
        else:
            print("No bus found with the given bus_id")


        cursor.execute(sql_station, (Station_name,))

        station_result = cursor.fetchone()

        if station_result:
            print("Staion ID :", station_result[0])
        else:
            print("No Station found with the given Station_name")

        # 데이터베이스 연결 끊기
        conn.close()
        return bus_result, station_result
    
    # 특정 정류장의 정보(그 정류장에서 운행하는 버스들, 도착정보)
    def station_bus_list(self, station_result):
        url = 'http://ws.bus.go.kr/api/rest/arrive/getLowArrInfoByStId'
        service_key = "lnGvRUsSrOgezp/xjHmRf1XJipLQd9ANFdkUk5w2kB1FaTDTAcS88zmKBViC6HYFRcWfhGjkuNQD85aNrvoTTw=="

        params ={'serviceKey' : service_key,
                'stId' : str(station_result) }

        response = requests.get(url, params=params)
        return response.content


    #  특정 버스 노선이 경유하는 버스 정류소의 정보
    def bus_station_list(self, bus_result):
        url = 'http://ws.bus.go.kr/api/rest/busRouteInfo/getStaionByRoute'
        service_key = "lnGvRUsSrOgezp/xjHmRf1XJipLQd9ANFdkUk5w2kB1FaTDTAcS88zmKBViC6HYFRcWfhGjkuNQD85aNrvoTTw=="

        params ={'serviceKey' : service_key,
                'busRouteId' : str(bus_result) }

        response = requests.get(url, params=params)
        return response.content


    # 정류소 노선별 교통약자 도착예정정보
    def station_arrival_info(self, station_result, bus_result, ord):
        url = 'http://ws.bus.go.kr/api/rest/arrive/getLowArrInfoByRoute'
        service_key = "lnGvRUsSrOgezp/xjHmRf1XJipLQd9ANFdkUk5w2kB1FaTDTAcS88zmKBViC6HYFRcWfhGjkuNQD85aNrvoTTw=="


        params ={'serviceKey' : service_key,
                'stId' : str(station_result),
                'busRouteId' : str(bus_result),
                'ord' : str(ord) }

        response = requests.get(url, params=params)
        return response.content

    # xml 값에서 특정 val라는 tag안에 있는 item을 가져오는 함수
    def find_xml_val(self, root, val):
        item_list = []
        for item in root.findall(".//itemList"):
            item1 = item.find(str(val)).text  # val에 해당하는 태그의 텍스트 내용을 가져옵니다.
            item_list.append(item1)
        return item_list

    # 리스트에서 일치하는 값을 찾아서 True값과 인덱스를 반환해준다.
    def find_api_val(self, list, value_to_find):
        for i, val in enumerate(list):
            if str(val)==str(value_to_find):
                print(f"value {value_to_find} found at index {i}")
                return i, True
        print("could not found the value in the list")
        return None, False
    


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



# 프레임을 처리하여 번호판을 인식하고, 인식된 번호를 데이터베이스에서 조회하는 기능을 제공
# YOLO 모델을 사용하여 번호판을 감지하고, EasyOCR을 사용하여 번호판의 텍스트를 인식
class FrameProcessor:
    def __init__(self, model, plate_class_indices, reader, width, height, padding, cur, conn, min_confidence):
        self.model = model
        self.plate_class_indices = plate_class_indices
        self.reader = reader
        self.width = width
        self.height = height
        self.padding = padding
        self.cur = cur
        self.conn = conn
        self.min_confidence = min_confidence
        self.processed_numbers = set()
    
    def process_frame(self, frame_queue):
        while True:
            frames = frame_queue.get()
            if frames is None:
                break

            ocr_results = []

            for frame in frames:
                results = self.model(frame)
                boxes = results[0].boxes if len(results) > 0 else []

                for box in boxes:
                    cls = int(box.cls)
                    if cls in self.plate_class_indices:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        x1, y1 = max(x1 - self.padding, 0), max(y1 - self.padding, 0)
                        x2, y2 = min(x2 + self.padding, self.width), min(y2 + self.padding, self.height)

                        plate_image = frame[y1:y2, x1:x2]
                        if plate_image.size == 0:
                            continue

                        preprocessed_img = ImageProcessor.preprocess_image(plate_image)
                        ocr_result = self.reader.readtext(preprocessed_img, detail=1)

                        for res in ocr_result:
                            text, confidence = res[1], res[2]
                            if confidence >= self.min_confidence:
                                text = ''.join(filter(str.isdigit, text))
                                if 2 <= len(text) <= 4:
                                    ocr_results.append(text)

            if ocr_results:
                most_common_text = Counter(ocr_results).most_common(1)[0][0]
                if most_common_text not in self.processed_numbers:
                    print(f"Detected text: {most_common_text}")

                    row = self.cur.query_number(most_common_text)

                    if row:
                        print(f"Matching number in DB: {row[0]}")
                        self.processed_numbers.add(most_common_text)
