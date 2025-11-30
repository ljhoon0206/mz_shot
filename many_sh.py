import streamlit as st
import cv2
import mediapipe as mp
import time
import numpy as np
import av
import queue
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import streamlit.components.v1 as components
import base64

# ---------------- Mediapipe 초기화 ----------------
mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# 전역 객체를 함수 밖에서 초기화 (성능 최적화)
FACE_DETECTOR = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.4)
HAND_DETECTOR = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

# ---------------- 웹 앱 전역 변수 설정 ----------------
TARGET_A_MIN, TARGET_A_MAX = 43, 47  # 눈-입 비율 (%)
TARGET_B_MIN, TARGET_B_MAX = 12, 15  # 코-입 비율 (%)
COUNTDOWN_TIME = 3.0  # 카운트다운 시간 설정

# STUN 서버 (검은 화면 방지)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# ---------------- Victory 제스처 판단 ----------------
def is_victory(lms):
    """검지+중지 펴짐, 약지+새끼 접힘이면 V 사인 True"""
    try:
        # 손가락 끝(tip)의 Y 좌표가 마디(knuckle)의 Y 좌표보다 작으면 (위쪽에 있으면) 펴진 상태 (Mediapipe 기준)
        return (
                lms.landmark[8].y < lms.landmark[5].y and  # 검지
                lms.landmark[12].y < lms.landmark[9].y and  # 중지
                lms.landmark[16].y > lms.landmark[13].y and  # 약지 (접힘)
                lms.landmark[20].y > lms.landmark[17].y  # 새끼 (접힘)
        )
    except Exception:
        return False


# ---------------- 비율 계산 함수 ----------------
def get_face_distances(detection, img_h):
    """얼굴 인식 결과를 바탕으로 눈-입, 코-입 비율을 계산."""
    keypoints = detection.location_data.relative_keypoints
    bbox_h = detection.location_data.relative_bounding_box.height

    if bbox_h == 0:
        return {'eye_mouth_ratio': 0.0, 'nose_mouth_ratio': 0.0}

    # 주요 키포인트 Y 좌표 (상대 좌표)
    y_eye_r = keypoints[1].y
    y_eye_l = keypoints[0].y
    y_eye_center = (y_eye_r + y_eye_l) / 2
    y_mouth = keypoints[3].y
    y_nose = keypoints[2].y

    # 눈 중앙 - 입 (Normalized Distance)
    distance_eye_mouth_norm = abs(y_mouth - y_eye_center)
    eye_mouth_ratio = distance_eye_mouth_norm / bbox_h

    # 코 - 입 (Normalized Distance)
    distance_nose_mouth_norm = abs(y_mouth - y_nose)
    nose_mouth_ratio = distance_nose_mouth_norm / bbox_h

    return {
        'eye_mouth_ratio': eye_mouth_ratio,
        'nose_mouth_ratio': nose_mouth_ratio
    }


# ---------------- 게이지 그리기 함수 ----------------
def draw_gauge(img, ratio_percent, x_offset, target_min, target_max, label):
    """화면 왼쪽에 수직 게이지를 그립니다."""
    gauge_x, gauge_y = 50 + x_offset, 80
    gauge_w, gauge_h = 20, 200

    # 0~100% 사이로 값 제한
    ratio_percent_clamped = max(0, min(100, ratio_percent))
    is_target = target_min <= ratio_percent_clamped <= target_max

    target_color = (0, 255, 0)
    base_color = (255, 255, 255)
    fill_color = target_color if is_target else (0, 0, 255)

    cv2.rectangle(img, (gauge_x, gauge_y), (gauge_x + gauge_w, gauge_y + gauge_h), base_color, 2)

    # 게이지 채우기
    fill_height = int(gauge_h * (ratio_percent_clamped / 100))
    fill_y_start = gauge_y + gauge_h - fill_height
    cv2.rectangle(img, (gauge_x, fill_y_start), (gauge_x + gauge_w, gauge_y + gauge_h), fill_color, cv2.FILLED)

    # 타겟 영역 표시 (노란색)
    y_max = gauge_y + gauge_h - int(gauge_h * (target_min / 100))
    y_min = gauge_y + gauge_h - int(gauge_h * (target_max / 100))
    cv2.rectangle(img, (gauge_x - 5, y_min), (gauge_x + gauge_w + 5, y_max), (0, 255, 255), 1)

    cv2.putText(img, label, (gauge_x - 10, gauge_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, base_color, 1)
    cv2.putText(img, f"{ratio_percent_clamped}%", (gauge_x - 10, gauge_y + gauge_h + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, base_color, 2)

    return is_target, ratio_percent_clamped


# ---------------- 셔터 소리용 HTML 생성 함수 ----------------
def load_shutter_html():
    """
    shutter.wav 파일을 base64로 읽어서 <audio> 자동 재생 HTML을 만들어 줌.
    shutter.wav는 이 파이썬 파일과 같은 폴더에 있어야 함.
    """
    try:
        # NOTE: 이 파일은 실제 환경에 있어야 소리가 납니다.
        with open("shutter.wav", "rb") as f:
            audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        html = f"""
        <audio autoplay>
            <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav" />
            브라우저가 audio 태그를 지원하지 않습니다.
        </audio>
        """
        return html
    except FileNotFoundError:
        # 파일이 없을 경우 경고 대신 빈 문자열 반환
        return ""


# ---------------- VideoProcessor 클래스 (핵심) ----------------
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        # 캡처된 RGB 이미지를 메인 스레드로 전송
        self.result_queue = queue.Queue(maxsize=1)
        # 셔터 소리 신호를 메인 스레드로 전송
        self.shutter_queue = queue.Queue(maxsize=1)

        self.captured = False
        self.last_capture_time = 0
        self.countdown_active = False
        self.countdown_start_time = 0
        self.face_detector = FACE_DETECTOR
        self.hand_detector = HAND_DETECTOR

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img_h, img_w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 캡처 직후는 홀드 상태
        if self.captured:
            cv2.putText(img, "CAPTURED! (Hold)", (img_w // 2 - 150, img_h // 2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # ---------------- 1. 얼굴/손 인식 및 비율 확인 ----------------
        face_detected = False
        ratio_ok_A, ratio_ok_B = False, False
        ratio_A_percent, ratio_B_percent = 0, 0
        victory_detected = False

        img_out = img.copy()

        face_res = self.face_detector.process(rgb)
        if face_res.detections:
            face_detected = True
            d = face_res.detections[0]
            current_ratios = get_face_distances(d, img_h)
            ratio_A_percent = int(current_ratios['eye_mouth_ratio'] * 100)
            ratio_B_percent = int(current_ratios['nose_mouth_ratio'] * 100)
            ratio_ok_A = TARGET_A_MIN <= ratio_A_percent <= TARGET_A_MAX
            ratio_ok_B = TARGET_B_MIN <= ratio_B_percent <= TARGET_B_MAX
            mp_draw.draw_detection(img_out, d)

        hand_res = self.hand_detector.process(rgb)
        if hand_res.multi_hand_landmarks:
            for handLms in hand_res.multi_hand_landmarks:
                if is_victory(handLms):
                    victory_detected = True
                mp_draw.draw_landmarks(img_out, handLms, mp_hands.HAND_CONNECTIONS)

        # ---------------- 2. 게이지 표시 ----------------
        draw_gauge(img_out, ratio_A_percent, 0, TARGET_A_MIN, TARGET_A_MAX, "E-M Ratio")
        draw_gauge(img_out, ratio_B_percent, 70, TARGET_B_MIN, TARGET_B_MAX, "N-M Ratio")

        total_ratio_ok = ratio_ok_A and ratio_ok_B
        all_conditions_met = face_detected and victory_detected and total_ratio_ok

        # ---------------- 3. 카운트다운 및 캡처 로직 ----------------
        if all_conditions_met:
            if not self.countdown_active:
                self.countdown_active = True
                self.countdown_start_time = time.time()
                # Streamlit Session State 메시지 업데이트
                st.session_state.capture_message = f"카운트다운 시작! {COUNTDOWN_TIME}초 유지하세요."

            elapsed = time.time() - self.countdown_start_time
            countdown_value = COUNTDOWN_TIME - elapsed

            if countdown_value <= 0:
                self.countdown_active = False
                self.captured = True
                self.last_capture_time = time.time()

                # ⭐️ 원본 RGB 이미지를 큐에 전송 (메인 스레드에서 처리)
                try:
                    self.result_queue.put(rgb.copy(), block=False)
                except queue.Full:
                    pass

                # ⭐️ 셔터 소리 신호 전송
                try:
                    self.shutter_queue.put(True, block=False)
                except queue.Full:
                    pass

            # 카운트다운 표시
            countdown_display = max(1, int(COUNTDOWN_TIME - (time.time() - self.countdown_start_time)) + 1)
            cv2.putText(img_out, f"Capturing in: {countdown_display}", (img_w // 2 - 150, img_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)

        else:
            if self.countdown_active:
                self.countdown_active = False
                st.session_state.capture_message = "⏳ 조건 미달로 카운트다운 중단."

            if not self.captured and not self.countdown_active:
                # 얼굴이나 V 사인이 감지되면 자세 안내 메시지를 좀 더 명확히 함
                if face_detected or victory_detected:
                    st.session_state.capture_message = "조건을 충족시켜주세요. (비율/V사인)"
                else:
                    st.session_state.capture_message = "카메라를 켜고 자세를 잡아주세요."

        # ---------------- 4. 디버그/상태 표시 갱신 ----------------
        status_text = (
            f"Face: {face_detected} | V: {victory_detected} | "
            f"Ratio A: {ratio_ok_A} | Ratio B: {ratio_ok_B}"
        )
        cv2.putText(img_out, status_text,
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img_out, format="bgr24")


# ---------------- Streamlit 메인 함수 ----------------
def main():
    st.set_page_config(page_title="비율 & V 사인 검출기", layout="wide")

    st.title("📸 비율 최적화 V-사인 자동 캡처 웹 앱")
    st.markdown("""
    모든 조건 (**얼굴 감지, V-사인, 최적 비율**)이 충족되면 **3초 카운트다운** 후 자동으로 캡처되며, 셔터 소리가 납니다.
    ---
    """)

    SHUTTER_HTML = load_shutter_html()
    if not SHUTTER_HTML:
        st.warning("⚠️ **경고**: `shutter.wav` 파일을 찾을 수 없어 셔터 소리 재생이 불가능합니다.")

    # Session State 초기화
    if 'capture_ready' not in st.session_state:
        st.session_state.capture_ready = False
        st.session_state.captured_image_rgb = None
        st.session_state.capture_message = "카메라를 켜고 자세를 잡아주세요."

    current_message = st.session_state.get('capture_message', "카메라를 켜고 자세를 잡아주세요.")

    col1, col2 = st.columns([2, 1])

    # ---------------- I. 웹캠 스트림 (col1) ----------------
    with col1:
        st.subheader("실시간 웹캠 스트림 (비전 처리)")

        webrtc_ctx = webrtc_streamer(
            key="media-pipe-detector",
            video_processor_factory=VideoProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if webrtc_ctx.state.playing:
            if "촬영 성공" in current_message or st.session_state.get('capture_ready'):
                st.success(f"현재 상태: **{current_message}**")
            elif "카운트다운 시작" in current_message:
                st.warning(f"현재 상태: **{current_message}**")
            else:
                st.info(f"현재 상태: **{current_message}**")
        else:
            st.warning("카메라를 불러오는 중입니다...")

    # ⭐ II. 캡처 및 셔터 소리 감지 루프 (메인 스레드)
    if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
        processor = webrtc_ctx.video_processor

        # 캡처 및 신호 감지 루프
        while webrtc_ctx.state.playing:

            # 1. 캡처 이미지 신호 확인
            result_img_rgb = None
            try:
                # 큐에서 캡처된 RGB 이미지를 가져옴
                result_img_rgb = processor.result_queue.get(timeout=0.01)
            except queue.Empty:
                pass
            except Exception:
                break

            # 2. 셔터 소리 신호 확인 및 HTML 삽입
            try:
                if processor.shutter_queue.get(timeout=0.01):
                    if SHUTTER_HTML:
                        components.html(SHUTTER_HTML, height=0)  # base64 audio 자동 재생
            except queue.Empty:
                pass
            except Exception:
                pass

            # 3. 캡처 이미지가 있으면 Session State 업데이트 후 UI 갱신
            if result_img_rgb is not None:
                st.session_state.captured_image_rgb = result_img_rgb
                st.session_state.capture_ready = True
                st.session_state.capture_message = "✅ 촬영 성공! 아래에서 결과를 확인하세요."

                # 프로세서의 홀드 상태 해제 (다음 프레임부터 정상 작동)
                processor.captured = False

                # UI 즉시 갱신
                st.rerun()
                # st.rerun()이 호출되면 이 루프는 종료됨

            time.sleep(0.01)  # CPU 부하 줄이기

    # ---------------- III. 결과 표시 (col2) ----------------
    with col2:
        st.subheader("✅ 캡처 조건 및 결과")
        st.markdown(
            f"""
            **✅ 최적 비율 목표:**
            * **눈-입 비율 (A):** ${TARGET_A_MIN}\\% \sim {TARGET_A_MAX}\\%$
            * **코-입 비율 (B):** ${TARGET_B_MIN}\\% \sim {TARGET_B_MAX}\\%$
            * **추가 조건:** **얼굴 감지** 및 **V-사인 감지**
            """
        )
        st.markdown("---")

        if st.session_state.get('capture_ready') and st.session_state.get('captured_image_rgb') is not None:
            st.success("🎉 **캡처 완료!**")

            captured_img_rgb = st.session_state.captured_image_rgb

            # 캡처 이미지 표시 (RGB)
            st.image(captured_img_rgb, caption="최근 캡처 이미지", use_container_width=True)

            # 다운로드를 위해 RGB를 BGR로 변환 후 PNG 인코딩
            img_bgr = cv2.cvtColor(captured_img_rgb, cv2.COLOR_RGB2BGR)
            ret, buffer = cv2.imencode(".png", img_bgr)

            if ret:
                st.download_button(
                    label="🖼️ 캡처 이미지 다운로드",
                    data=buffer.tobytes(),
                    file_name=f"capture_optimal_{time.strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )

            st.markdown("---")

            if st.button("🔄 다음 캡처 준비"):
                # 상태 초기화 후 재실행하여 스트림 재개
                st.session_state.capture_ready = False
                st.session_state.captured_image_rgb = None
                st.session_state.capture_message = "카메라를 켜고 자세를 잡아주세요."
                st.rerun()

        else:
            st.warning("아직 캡처된 이미지가 없습니다. 조건을 충족시켜보세요!")


if __name__ == "__main__":
    main()