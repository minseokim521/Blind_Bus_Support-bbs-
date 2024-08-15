import re
import os
import pyaudio
import wave
import requests
import xml.etree.ElementTree as ET
from google.cloud import speech
from google.cloud import texttospeech


# 환경 변수 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/이유진/Documents/2024/IDP_LAB/google cloud platform/zippy-brand-429513-k7-6ef67897540d.json"

# Google Cloud STT, TTS client 설정
stt_client = speech.SpeechClient()
tts_client = texttospeech.TextToSpeechClient()

# 실시간 버스 정보 API 키 및 URL 설정
service_key = 'lnGvRUsSrOgezp/xjHmRf1XJipLQd9ANFdkUk5w2kB1FaTDTAcS88zmKBViC6HYFRcWfhGjkuNQD85aNrvoTTw=='
arrive_url = 'http://ws.bus.go.kr/api/rest/arrive/getLowArrInfoByStId'

# 영등포역 station_id (임의로 설정)
station_id = "118000005"

def record_audio(seconds, filename="request.wav"):
    #음성 녹음
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    rate = 44100

    p = pyaudio.PyAudio()

    stream = p.open(format=format,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)

    print("Recording...")
    frames = []

    for _ in range(0, int(rate / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

def recognize_speech_from_audio(filename):
    #녹음된 오디오 파일에서 텍스트 인식
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
    #텍스트에서 버스 번호 추출
    matches = re.findall(r'\d{3,}', text)
    return matches[0] if matches else None

def determine_intent(text):
    #텍스트에서 사용자 의도 파악
    if any(phrase in text for phrase in ["몇분 남았어", "언제 와", "언제 도착해", "얼마나", "남았어"]):
        return "arrival_time"
    elif "탈건데" in text or "타고 싶어요" in text:
        return "request_bus"
    else:
        return "unknown"

def find_xml_val(root, tag):
    #XML에서 특정 태그 값 리스트로 추출
    return [item.text for item in root.findall(f".//{tag}")]

def find_api_val(list_items, target):
    #리스트에서 특정 값을 찾고, 그 인덱스와 결과 반환
    for index, item in enumerate(list_items):
        if target in item:
            return index, True
    return None, False

def get_bus_arrival_info(station_id, bus_num):
    #실시간 버스 도착 정보를 API 통해 가져옴
    params = {
        'serviceKey': service_key,
        'stId': station_id,
    }
    response = requests.get(arrive_url, params=params)
    root = ET.fromstring(response.content)

    # 정류소에서 운행하는 버스 이름 정보 리스트
    bus_list = find_xml_val(root, "busRouteAbrv")

    # 버스 리스트에서 인식한 버스 정보가 있는지 찾음
    index, result = find_api_val(bus_list, bus_num)

    if result:
        isArrive1 = find_xml_val(root, "isArrive1")
        arrmsg1_list = find_xml_val(root, "arrmsg1")
        arrmsg2_list = find_xml_val(root, "arrmsg2")

        if not isArrive1[index]:
            return f"{bus_num}번 버스가 도착했습니다.", arrmsg1_list[index], arrmsg2_list[index]
        else:
            return f"{bus_num}번 버스가 도착하지 않았습니다.", arrmsg1_list[index], arrmsg2_list[index]
    else:
        return None, None, None

def speak_text(text, filename="response.mp3"):
    #텍스트를 음성으로 변환하여 재생
    synthesis_input = texttospeech.SynthesisInput(text=text)
    
    voice = texttospeech.VoiceSelectionParams(
        language_code="ko-KR",
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )
    
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    
    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    
    with open(filename, "wb") as out:
        out.write(response.audio_content)
        print(f"Audio content written to file '{filename}'")
    
    os.system(f"start {filename}")  # Windows에서 기본 오디오 플레이어로 재생

def notify_user(bus_number, station_id):
    #실시간 버스 도착 정보 음성으로 안내
    msg1, arrmsg1, arrmsg2 = get_bus_arrival_info(station_id, bus_number)
    
    if msg1:
        speak_text(msg1, "response_1.mp3")
        speak_text(f"첫번째 버스 도착 예정시간: {arrmsg1}", "response_2.mp3")
        speak_text(f"두번째 버스 도착 예정시간: {arrmsg2}", "response_3.mp3")
    else:
        speak_text(f"{bus_number}번 버스의 도착 정보를 가져올 수 없습니다.", "response_error.mp3")

# 실시간으로 음성을 녹음하여 STT 수행
record_audio(7, "request.wav")  # 7초간 음성 녹음
recognized_text = recognize_speech_from_audio("request.wav")

# 인식된 텍스트 확인
print(f"인식된 텍스트: {recognized_text}")

# 버스 번호와 의도 추출
bus_number = extract_bus_number(recognized_text)
intent = determine_intent(recognized_text)

# 추출된 결과 확인
print(f"추출된 버스 번호: {bus_number}")
print(f"추출된 의도: {intent}")

# 처리 결과에 따라 응답 생성 및 음성 안내
if intent == "arrival_time":
    notify_user(bus_number, station_id)
elif intent == "request_bus":
    notify_user(bus_number, station_id)
else:
    speak_text("죄송합니다. 요청을 이해하지 못했습니다.", "response_error.mp3")
