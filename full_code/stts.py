import os
import pyaudio
from google.cloud import speech
from google.cloud import texttospeech
import requests 


# 환경 변수 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/이유진/Documents/2024/IDP_LAB/google cloud platform/zippy-brand-429513-k7-6ef67897540d.json"

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


#=============== 실시간 bus_number, bus_stop 확인 ================

def check_bus_number(transcript, expected_bus_numbers):
    # 텍스트에서 숫자 부분만 추출 (예: "버스 번호는 2024입니다"에서 "2024" 추출)
    for bus_number in expected_bus_numbers:
        if bus_number in transcript:
            return bus_number
    return None

def get_real_time_bus_info(api_url):
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            # 실제 API 데이터 형식에 따라 처리 필요
            bus_numbers = data.get("bus_numbers", [])
            bus_stop = data.get("bus_stop", "알 수 없는 정류장")
            return bus_numbers, bus_stop
        else:
            print(f"Error fetching bus data: {response.status_code}")
            return [], "알 수 없는 정류장"
    except Exception as e:
        print(f"Error occurred: {e}")
        return [], "알 수 없는 정류장"

#=========================== STT =============================
def stream_audio(api_url):
    expected_bus_numbers, bus_stop = get_real_time_bus_info(api_url)

    # Google Cloud Speech-to-Text 클라이언트 생성
    client = speech.SpeechClient()

    # 마이크 설정
    RATE = 16000
    CHUNK = int(RATE / 10)  # 100ms

    # 오디오 스트림 생성
    audio_interface = pyaudio.PyAudio()
    audio_stream = audio_interface.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    # 오디오 데이터 생성기
    def generate_audio_data():
        while True:
            try:
                data = audio_stream.read(CHUNK, exception_on_overflow=False)
                yield speech.StreamingRecognizeRequest(audio_content=data)
            except Exception as e:
                print(f"Error capturing audio data: {e}")
                break

    # 자주 사용하는 숫자 목록 (예시)
    common_numbers = [str(i) for i in range(5000)]  # 0부터 4999까지의 숫자

    # 음성 인식 설정
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="ko-KR",
        use_enhanced=True,  # 향상된 모델 사용
        model="default",  # default 모델 사용, 필요에 따라 video, phone_call 등 설정 가능
        speech_contexts=[speech.SpeechContext(
            phrases=common_numbers  # 인식 정확도를 높이기 위한 프레이즈 힌트
        )]
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,  # 모든 결과를 받기 위해 interim_results를 True로 설정
    )

    # 스트리밍 음성 인식 요청
    requests = generate_audio_data()
    try:
        responses = client.streaming_recognize(config=streaming_config, requests=requests)
    except Exception as e:
        print(f"Error in streaming_recognize: {e}")
        return

    try:
        # 인식된 텍스트 출력 및 TTS 변환
        for response in responses:
            for result in response.results:
                if result.is_final:  # 최종 결과만 출력
                    transcript = result.alternatives[0].transcript
                    print(f"Final Transcript: {transcript}")
                    bus_number = check_bus_number(transcript, expected_bus_numbers)
                    if bus_number:
                        # TTS 변환
                        ssml_text = f"""
                        <speak>
                            <p>
                                <s><prosody rate="90%">
                                    현재 {bus_stop} 정류장에 들어오는 버스는
                                    <break time="200ms"/><emphasis level="moderate"><say-as interpret-as="characters">{bus_number}</say-as></emphasis>
                                    <break time="200ms"/><emphasis level="moderate"><say-as interpret-as="characters">{bus_number}</say-as> 번입니다.</emphasis>
                                    </prosody>
                                </s>
                            </p>
                        </speak>
                        """
                        text_to_speech_ssml(ssml_text, "output.mp3")
                    else:
                        # 버스 번호가 일치하지 않을 때 TTS 변환
                        error_message = "일치하는 버스 번호가 없습니다."
                        ssml_text = f"""
                        <speak>
                            <s>{error_message}</s>
                        </speak>
                        """
                        text_to_speech_ssml(ssml_text, "error_message.mp3")
    except Exception as e:
        print(f"Error occurred during response handling: {e}")
    except KeyboardInterrupt:
        print("Streaming stopped by user.")
    finally:
        audio_stream.stop_stream()
        audio_stream.close()
        audio_interface.terminate()

if __name__ == "__main__":
    api_url = "http://example.com/api/businfo"  # 실시간 버스 정보를 제공하는 API URL

    # 현재 작업 디렉토리 출력
    print("Current working directory:", os.getcwd())

    stream_audio(api_url)