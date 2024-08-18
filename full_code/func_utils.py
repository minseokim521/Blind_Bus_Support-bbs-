import psycopg2
import requests
import xml.etree.ElementTree as ET
import time
from collections import Counter
import cv2
import threading
from ultralytics import YOLO
import pyaudio
from google.cloud import speech
from google.cloud import texttospeech
import rospy
import os
import sys
from sensor_msgs.msg import NavSatFix
import numpy as np
import re

# YOLO 모델을 초기화하고 비디오에서 프레임을 읽는 기능을 제공
# 비디오 파일에서 프레임을 일정한 간격으로 추출하여 처리할 수 있게 함
class YOLOVideoCapture:
    def __init__(self, model_path):
        try:
            # YOLO 모델 초기화
            self.model = YOLO(model_path)
            self.model.overrides['verbose'] = False
            
            # 비디오 캡처 초기화
            video_device = "/dev/video3"
            self.cap = cv2.VideoCapture(video_device)

            # 해상도를 명시적으로 1920x1080으로 설정
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            # 자동 초점 설정 (자동 초점을 지원하는 경우)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # 1로 설정하여 자동 초점 활성화

            # 웹캠 연결 확인
            if not self.cap.isOpened():
                print("웹캠을 열 수 없습니다. 웹캠이 제대로 연결되었는지 확인하세요.")
                self.cap = None  # 웹캠이 연결되지 않았음을 나타내기 위해 None으로 설정
            else:
                print("웹캠이 성공적으로 연결되었습니다.")
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"해상도: {self.width}x{self.height}")
                
                # 2초 대기 추가
                print("웹캠 초점 조정 시간을 위해 2초 대기합니다...")
                time.sleep(2)
                print("대기 완료, 비디오 캡처를 시작합니다.")

        except Exception as e:
            # 모델 로드 실패 시 예외 처리 및 로그 출력
            print(f"모델을 로드하는 중 오류가 발생했습니다: {str(e)}")
            self.model = None  # 모델 로드 실패 시 None으로 설정


    def read_frames(self):
        frames = []
        start_time = time.time()  # 시작 시간 설정

        if self.cap is None:
            print("웹캠이 열리지 않았기 때문에 프레임을 읽을 수 없습니다.")
            return

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("프레임을 읽지 못했습니다.")
                break  # 프레임을 읽지 못하면 종료

            current_time = time.time()
            if current_time - start_time <= 1.0:  # 1초 이내에만 프레임 캡처
                frames.append(frame)
            else:
                break  # 1초가 지나면 루프를 종료

        if frames:
            yield frames

        self.release()  # 비디오 캡처를 해제

    
    def release(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


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



class API():
    def __init__(self):
        pass
            
    def database_query(self, table, key1, key2, val):
        """
        지정된 테이블에서 키와 값을 기준으로 데이터를 조회합니다.

        Parameters:
            table (str): 조회할 table의 이름. EX) bus, station.
            key1 (str): 해당 테이블에 리턴받고자하는 column의 key값.
            key2 (str): 해당 테이블 에서 검색할 value가 해당하는 column.
            val (str): 조회 기준이 될 값.

        Returns:
            list: 조회 결과를 리스트 형태로 반환.

        Raises:
            DatabaseError: 데이터베이스 조회 중 오류가 발생한 경우.
        """
        #데이터베이스에 연결시도
        try:
            # 데이터베이스 연결 & 커넥트 객체 생성
            conn = psycopg2.connect(host="122.44.85.37", dbname="postgres", user="postgres", password="postgres", port='5432')
        except:
            print("Not Connected!.")

        # 쿼리를 수행하는 cursor객체 생성
        cursor = conn.cursor()

        # 쿼리문
        sql_query = f"SELECT {key1} FROM {table} WHERE {key2} = %s;"

        # 쿼리실행
        cursor.execute(sql_query, (val,))
        #fetchone은 쿼리에 해당하는 열을 튜플형태로 반환, 없다면 None
        query_result = cursor.fetchone()

        # 예외 처리
        if query_result:
            print("queryed value : ", query_result[0])
        else:
            print("No value found with the given value")


        # 데이터베이스 연결 끊기
        conn.close()
        return query_result
    
    def database_query_specific_column(self, table, column_name):
        """
        지정된 테이블에서 특정 열의 모든 데이터를 조회합니다.

        Parameters:
            table (str): 조회할 table의 이름.
            column_name (str): 데이터를 가져올 column의 이름.

        Returns:
            list: 조회된 데이터의 리스트.

        Raises:
            Exception: 데이터베이스 연결 실패 또는 쿼리 실행 중 오류 발생 시 예외 발생.
        """
        try:
            # 데이터베이스 연결
            conn = psycopg2.connect(host="122.44.85.37", dbname="postgres", user="postgres", password="postgres",port='5432')
            cursor = conn.cursor()

            # SQL 쿼리 실행 (주의: column_name을 쿼리에 직접 포함시킬 때 SQL 인젝션 위험이 있으므로 신뢰할 수 있는 입력만 처리해야 함)
            sql_query = f"SELECT {column_name} FROM {table};"
            cursor.execute(sql_query)

            # 결과 모두 가져오기
            results = cursor.fetchall()

            # 연결 종료
            conn.close()

            if results:
                print(f"successfully queried a column named {column_name}")
                return results
            else:
                print("No data found for the specified column")
                return []

        except Exception as e:
            print(f"Database error: {str(e)}")
            raise
    def find_nearest_index(self, x_coord, y_coord, x_list, y_list):
        # x_list와 y_list를 NumPy 배열로 변환
        x_array = np.array(x_list)
        y_array = np.array(y_list)

        # 주어진 x_coord, y_coord와 각 좌표 간의 거리 계산
        distances = np.sqrt((x_array - x_coord)**2 + (y_array - y_coord)**2)

        # 가장 작은 거리의 인덱스 반환
        nearest_index = np.argmin(distances)
        return nearest_index

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
        service_key = 'lnGvRUsSrOgezp/xjHmRf1XJipLQd9ANFdkUk5w2kB1FaTDTAcS88zmKBViC6HYFRcWfhGjkuNQD85aNrvoTTw=='

        params ={'serviceKey' : service_key,
                'busRouteId' : str(bus_result) }

        response = requests.get(url, params=params)
        return response.content


    # 정류소 노선별 교통약자 도착예정정보
    def station_arrival_info(self, station_result, bus_result, ord):
        url = 'http://ws.bus.go.kr/api/rest/arrive/getLowArrInfoByRoute'
        service_key = 'lnGvRUsSrOgezp/xjHmRf1XJipLQd9ANFdkUk5w2kB1FaTDTAcS88zmKBViC6HYFRcWfhGjkuNQD85aNrvoTTw=='



        params ={'serviceKey' : service_key,
                'stId' : str(station_result),
                'busRouteId' : str(bus_result),
                'ord' : str(ord) }

        response = requests.get(url, params=params)
        return response.content
    
        #  좌표기반 버스정류장 위치 조회
    def station_pose(self, X_location, Y_location, radius):
        url = 'http://ws.bus.go.kr/api/rest/busRouteInfo/getStaionByRoute'
        service_key = 'lnGvRUsSrOgezp/xjHmRf1XJipLQd9ANFdkUk5w2kB1FaTDTAcS88zmKBViC6HYFRcWfhGjkuNQD85aNrvoTTw=='

        params ={'serviceKey' : service_key,
                'tmX' : str(X_location),
                'tmY' : str(Y_location),
                'radius' : str(radius)
                }

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
    

# 프레임을 처리하여 번호판을 인식하고, 인식된 번호를 데이터베이스에서 조회하는 기능을 제공
# YOLO 모델을 사용하여 번호판을 감지하고, EasyOCR을 사용하여 번호판의 텍스트를 인식
class FrameProcessor:
    def __init__(self, model, plate_class_indices, reader, width, height, padding, min_confidence):
        self.model = model
        self.plate_class_indices = plate_class_indices
        self.reader = reader
        self.width = width
        self.height = height
        self.padding = padding
        self.min_confidence = min_confidence
        self.processed_numbers = set()

        # 경로 설정
        self.plate_img_dir = "/home/minseokim521/catkin_ws/src/bus/Blind_Bus_Support-bbs-/plate_img"
        self.preprocessed_img_dir = "/home/minseokim521/catkin_ws/src/bus/Blind_Bus_Support-bbs-/preprocessed_img"

        # 폴더 내 파일 삭제
        self.clear_directory(self.plate_img_dir)
        self.clear_directory(self.preprocessed_img_dir)

    def clear_directory(self, directory):
        """디렉터리 내의 모든 파일을 삭제"""
        if os.path.exists(directory):
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
        else:
            os.makedirs(directory, exist_ok=True)

    def process_frame(self, frames):
        ocr_results = []

        for frame_idx, frame in enumerate(frames):
            results = self.model(frame)
            boxes = results[0].boxes if len(results) > 0 else []

            if not boxes:
                print("No boxes detected by YOLO model.")
                continue

            for box_idx, box in enumerate(boxes):
                cls = int(box.cls)
                if cls in self.plate_class_indices:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1, y1 = max(x1 - self.padding, 0), max(y1 - self.padding, 0)
                    x2, y2 = min(x2 + self.padding, self.width), min(y2 + self.padding, self.height)

                    plate_image = frame[y1:y2, x1:x2]
                    if plate_image.size == 0:
                        print(f"Skipped empty plate image at frame {frame_idx}, box {box_idx}")
                        continue

                    # Save the cropped plate image for debugging
                    # plate_image_path = os.path.join(self.plate_img_dir, f"plate_frame{frame_idx}_box{box_idx}.png")
                    # cv2.imwrite(plate_image_path, plate_image)
                    # print(f"Saved plate image to {plate_image_path}")

                    preprocessed_img = ImageProcessor.preprocess_image(plate_image)
                    
                    # Save the preprocessed image for further inspection
                    # preprocessed_image_path = os.path.join(self.preprocessed_img_dir, f"preprocessed_plate_frame{frame_idx}_box{box_idx}.png")
                    # cv2.imwrite(preprocessed_image_path, preprocessed_img)
                    # print(f"Saved preprocessed plate image to {preprocessed_image_path}")

                    ocr_result = self.reader.readtext(preprocessed_img, detail=1)

                    if not ocr_result:
                        print(f"No OCR results for frame {frame_idx}, box {box_idx}")
                    else:
                        for res in ocr_result:
                            text, confidence = res[1], res[2]
                            # print(f"OCR detected text: {text}, Confidence: {confidence}")
                            if confidence >= self.min_confidence:
                                text = ''.join(filter(str.isdigit, text))
                                if 2 <= len(text) <= 4:
                                    ocr_results.append(text)

        if ocr_results:
            most_common_text = Counter(ocr_results).most_common(1)[0][0]
            if most_common_text not in self.processed_numbers:
                print(f"Detected text: {most_common_text}")
                return most_common_text

        return None
    
#=========================== STT =============================

def recognize_speech_from_audio(filename):
    stt_client = speech.SpeechClient()

    with open(filename, 'rb') as audio_file:
        audio_content = audio_file.read()
    
    audio = speech.RecognitionAudio(content=audio_content)
    
    config = speech.RecognitionConfig(
        language_code="ko-KR",
        speech_contexts=[speech.SpeechContext(phrases=["버스", "몇분 남았어", "언제 와", "언제 도착해", "얼마나", "남았어"])],
    )
    
    response = stt_client.recognize(config=config, audio=audio)
    
    for result in response.results:
        return result.alternatives[0].transcript
    return ""


def extract_bus_number(text):
    matches = re.findall(r'\d{3,}', text)
    return matches[0] if matches else None


def determine_intent(text):
    if any(phrase in text for phrase in ["몇분 남았어", "언제 와", "언제 도착해", "얼마나", "남았어"]):
        return "arrival_time"
    elif "탈건데" in text or "탈거야" in text:
        return "request_bus"
    else:
        return "unknown"


#=========================== TTS =============================
def text_to_speech_ssml(ssml_text, output_file):
    client = texttospeech.TextToSpeechClient()

    # SSML 입력 설정
    synthesis_input = texttospeech.SynthesisInput(ssml=ssml_text)

    # 음성 설정
    voice = texttospeech.VoiceSelectionParams(
        language_code="ko-KR",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
    )

    # 오디오 설정
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    # 음성 합성 요청
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # 음성 파일 저장
    with open(output_file, "wb") as out:
        out.write(response.audio_content)
        print(f'Audio content written to file "{output_file}"')

#=============================== GPS ===========================
# ROS 노드 초기화 및 GPS 데이터 수신
# 전역 변수 및 스레드 안전성 확보를 위한 락
latitude = None
longitude = None
gps_lock = threading.Lock()

def gps_callback(msg):
    global latitude, longitude
    with gps_lock:
        latitude = float(format(msg.latitude, f'.{sys.float_info.dig}f'))
        longitude = float(format(msg.longitude, f'.{sys.float_info.dig}f'))
        print(f"Latitude: {latitude}, Longitude: {longitude}")
        rospy.signal_shutdown('GPS data received.')
        

def gps_sub(timeout=10):
    latitude, longitude = 0,0
    rospy.init_node('gps_receive_node', anonymous=True)
    rospy.Subscriber("ublox_gps/fix", NavSatFix, gps_callback)
    return latitude, longitude
