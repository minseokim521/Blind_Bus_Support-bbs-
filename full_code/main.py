from func_utils import API, YOLOVideoCapture, FrameProcessor, text_to_speech_ssml
import psycopg2
import requests
import xml.etree.ElementTree as ET
import threading
from queue import Queue
import easyocr
import os
import pygame


# 환경 변수 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/home/LOE/workspace/yolo/Archive/zippy-brand-429513-k7-6ef67897540d.json' 



# 비디오 경로와 모델 경로 설정
video_path = '/home/LOE/workspace/yolo/Archive/vid/KakaoTalk_20240812_133651375.mp4'
model_path = '/home/LOE/workspace/yolo/Archive/models/best_3000_s.pt'

# YOLO 비디오 캡처와 EasyOCR 초기화
video_capture = YOLOVideoCapture(model_path, video_path)
easy_ocr = easyocr.Reader(['en'], gpu=True)

# 번호판 인식 클래스 이름 설정 및 인덱스 확인
plate_class_names = ['front_num', 'side_num', 'back_num']
plate_class_indices = [idx for idx, name in video_capture.model.names.items() if name in plate_class_names]

# YOLO 모델로 프레임 처리
padding = 5
min_confidence = 0.9
frame_processor = FrameProcessor(
    video_capture.model, plate_class_indices, easy_ocr,
    video_capture.width, video_capture.height, padding, min_confidence
)
ocr_number = []
# 비디오에서 프레임을 읽어와 처리
for i, frames in enumerate(video_capture.read_frames()):
    ocr_number.append(frame_processor.process_frame(frames))
    if ocr_number[i]:
        print(f"OCR 결과로 추출된 번호판: {ocr_number[i]}")
        
print(f"ocr_number : {ocr_number}")


# ocr결과에서 None을 제거
filtered_ocr_numbers = [num for num in ocr_number if num is not None]

print(f"filtered_ocr_number : {filtered_ocr_numbers}")


# 비디오 캡처 해제
video_capture.release()


temp = API(str(filtered_ocr_numbers[1]), "영등포역")

Bus_num = temp.Bus_num
Station_name = temp.Station_name

#데이터 베이스 상에서 버스 번호와 정류소 이름에 해당하는 id값 가져오기
bus_result, station_result = temp.database_query(Bus_num, Station_name)

if bus_result == None:
    print("OCR상의 버스 번호가 DB와 일치하지 않습니다.")
    exit()
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

msg1, msg2, msg3 = 0,0,0

# 변수 출력
if not(isArrive1[index]):
    msg1 = f"{filtered_ocr_numbers[1]}번 버스가 도착했습니다."
    print(msg1)
    
else:
    msg1 = f"{filtered_ocr_numbers[1]}번 버스가 도착하지 않았습니다."
    print(msg1)
    
msg2 = "첫번째 버스 도착 예정시간 :" + arrmsg1_list[index] 
msg3 = "두번째 버스 도착 예정시간 :" + arrmsg2_list[index] 
print("첫번째 버스 도착 예정시간 :", arrmsg1_list[index])
print("두번째 버스 도착 예정시간 :", arrmsg2_list[index])

print("end of the code")


# 텍스트를 mp3파일로 저장, 이미 있는경우 덮어씀
text_to_speech_ssml(msg1 + msg2 + msg3, "ocr.mp3")


# 소리재생

print('sound playing')

pygame.mixer.init()
pygame.mixer.music.load('ocr.mp3')
pygame.mixer.music.play()
while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10)

print('sound_ends')