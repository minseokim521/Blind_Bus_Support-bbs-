from func_utils import API, text_to_speech_ssml, gps_sub, recognize_speech_from_audio, extract_bus_number, determine_intent, record_audio
import os
import pyaudio
import simpleaudio as sa
from pydub import AudioSegment
from google.cloud import speech
from google.cloud import texttospeech
import requests 
import xml.etree.ElementTree as ET


# 환경 변수 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/이유진/Documents/2024/IDP_LAB/google cloud platform/zippy-brand-429513-k7-6ef67897540d.json"

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------
                                                        STT section
------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# 음성을 녹음하여 WAV 파일로 저장
audio_filename = "user_input.wav"
record_audio(5, audio_filename)

# 녹음된 오디오 파일에서 STT로 텍스트 인식
recognized_text = recognize_speech_from_audio(audio_filename)

# 인식된 텍스트 확인
print(f"인식된 텍스트: {recognized_text}")

# 버스 번호와 의도 추출
bus_number = extract_bus_number(recognized_text)
intent = determine_intent(recognized_text)

# 추출된 결과 확인
print(f"추출된 버스 번호: {bus_number}")
print(f"추출된 의도: {intent}")

if bus_number is None:
    msg = "버스 번호를 인식할 수 없습니다. 다시 시도해주세요."
    print(msg)
    text_to_speech_ssml(msg, "stt_error.mp3")
    
    # 음성 출력
    sound = AudioSegment.from_mp3("stt_error.mp3")
    sound.export("stt_error.wav", format="wav")
    wave_obj = sa.WaveObject.from_wave_file("stt_error.wav")
    play_obj = wave_obj.play()
    play_obj.wait_done()

    exit()


# GPS 데이터를 받아오고 그 값을 변수에 저장
latitude, longitude = gps_sub()

bus_api = API()



# 데이터베이스에서 버스 번호와 정류소 이름에 해당하는 id값 가져오기
bus_result = bus_api.database_query('bus', 'routeid', 'bus_id', bus_number)
station_list = bus_api.database_query_specific_column("station", 'node_id')
station_name_list = bus_api.database_query_specific_column("station", 'station_name')
X_locations = bus_api.database_query_specific_column("station", 'X_location')
Y_locations = bus_api.database_query_specific_column("station", 'Y_location')

# 리스트 평탄화
X_locations = [x[0] for x in X_locations]
Y_locations = [y[0] for y in Y_locations]

# 가장 가까운 정류소 인덱스 찾기
index = bus_api.find_nearest_index(longitude, latitude, X_locations, Y_locations)
station_name = station_name_list[index]
station_id = station_list[index]
print(f"찾아낸 정류소의 이름 :{station_name}, 찾아낸 정류소의 id :{station_id}")

if bus_result == None:
    print("STT로 인식된 버스 번호가 DB와 일치하지 않습니다.")
    exit()

# 정류소에 운행하는 버스정보 가져오기
response2 = bus_api.station_bus_list(station_id[0])

# xml 값을 가져옴
root2 = ET.fromstring(response2)

# 정류소에서 운행하는 버스 이름 정보 리스트
bus_list = bus_api.find_xml_val(root2, "busRouteAbrv")

# 버스 리스트에서 인식한 버스 정보가 있는지 찾음
index, result = bus_api.find_api_val(bus_list, bus_number)

isArrive1 = []
arrmsg1_list = []
arrmsg2_list = []

if result:
    isArrive1 = bus_api.find_xml_val(root2, "isArrive1")
    arrmsg1_list = bus_api.find_xml_val(root2, "arrmsg1")
    arrmsg2_list = bus_api.find_xml_val(root2, "arrmsg2")
else:
    print("인식된 버스가 해당 정류소에서 운행되지 않습니다.")
    exit()

stt_msg1, stt_msg2, stt_msg3 = 0, 0, 0

# 변수 출력
if not(isArrive1[index]):
    stt_msg1 = f"{bus_number}번 버스가 도착했습니다."
    print(stt_msg1)
else:
    stt_msg1 = f"{bus_number}번 버스가 도착하지 않았습니다."
    print(stt_msg1)
    
stt_msg2 = "첫번째 버스 도착 예정시간 :" + arrmsg1_list[index] 
stt_msg3 = "두번째 버스 도착 예정시간 :" + arrmsg2_list[index] 
print(stt_msg2)
print(stt_msg3)

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------
                                                        TTS section
------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# 텍스트를 mp3파일로 저장, 이미 있는경우 덮어씀
text_to_speech_ssml(stt_msg1 + stt_msg2 + stt_msg3, "bus_info.mp3")

# 소리재생
print('sound playing')

# MP3 파일 로드
sound = AudioSegment.from_mp3("bus_info.mp3")
sound.export("bus_info.wav", format="wav")
wave_obj = sa.WaveObject.from_wave_file("bus_info.wav")
play_obj = wave_obj.play()
play_obj.wait_done()  # 재생이 끝날 때까지 기다림

print('sound_ends')


