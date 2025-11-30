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

FACE_DETECTOR = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.4)
HAND_DETECTOR = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

# ---------------- 웹 앱 전역 변수 설정 ----------------
TARGET_A_MIN, TARGET_A_MAX = 43, 47
TARGET_B_MIN, TARGET_B_MAX = 12, 15
COUNTDOWN_TIME = 3.0  # 카운트다운 시간 설정

# STUN 서버 (검은 화면 방지)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# ---------------- Victory 제스처 판단 ----------------
def is_victory(lms):
    """검지+중지 펴짐, 약지+새끼 접힘이면 V 사인 True"""
    try:
        # 손가락 끝(tip)이 마디(knuckle)보다 위(y좌표 작음)에 있으면 펴짐
        return (
                lms.landmark[8].y < lms.landmark[5].y and
                lms.landmark[12].y < lms.landmark[9].y and
                lms.landmark[16].y > lms.landmark[13].y and
                lms.landmark[20].y > lms.landmark[17].y
        )
    except Exception:
        return False


# ---------------- 비율 계산 함수 ----------------
def get_face_distances(detection, img_h):
    keypoints = detection.location_data.relative_keypoints
    bbox_h = detection.location_data.relative_bounding_box.height

    if bbox_h == 0:
        return {'eye_mouth_ratio': 0.0, 'nose_mouth_ratio': 0.0}

    y_eye_r = keypoints[1].y
    y_eye_l = keypoints[0].y
    y_eye_center = (y_eye_r + y_eye_l) / 2
    y_mouth = keypoints[3].y
    y_nose = keypoints[2].y

    distance_eye_mouth_norm = abs(y_mouth - y_eye_center)
    eye_mouth_ratio = distance_eye_mouth_norm / bbox_h

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

    ratio_percent_clamped = max(0, min(100, ratio_percent))
    is_target = target_min <= ratio_percent_clamped <= target_max

    target_color = (0, 255, 0)
    base_color = (255, 255, 255)
    fill_color = target_color if is_target else (0, 0, 255)

    cv2.rectangle(img, (gauge_x, gauge_y), (gauge_x + gauge_w, gauge_y + gauge_h), base_color, 2)

    fill_height = int(gauge_h * (ratio_percent_clamped / 100))
    fill_y_start = gauge_y + gauge_h - fill_height
    cv2.rectangle(img, (gauge_x, fill_y_start), (gauge_x + gauge_w, gauge_y + gauge_h), fill_color, cv2.FILLED)

    y_max = gauge_y + gauge_h - int(gauge_h * (target_min / 100))
    y_min = gauge_y + gauge_h - int(gauge_h * (target_max / 100))

    cv2.rectangle(img, (gauge_x, y_min), (gauge_x + gauge_w, y_max), (0, 255, 255), 1)

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
        st.error("shutter.wav 파일을 찾을 수 없습니다. 셔터 소리 재생이 불가능합니다.")
        return ""


# ---------------- VideoProcessor 클래스 (핵심) ----------------
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        # ✅ 이미지 캡처 결과를 메인 스레드로 전달하는 큐
        self.result_queue = queue.Queue()
        # ✅ 셔터 소리 재생 신호를 메인 스레드로 전달하는 큐
        self.shutter_queue = queue.Queue()

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

        if self.captured:
            cv2.putText(img, "CAPTURED! (Hold)", (img_w // 2 - 150, img_h // 2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # ---------------- 1. 얼굴/손 인식 및 비율 확인 ----------------
        face_res = self.face_detector.process(rgb)
        face_detected = False
        ratio_ok_A, ratio_ok_B = False, False
        ratio_A_percent, ratio_B_percent = 0, 0
        victory_detected = False

        img_out = img.copy()

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
                # 캡처 메시지를 Session State에 직접 저장하면 Multi-threading 환경에서 문제가 될 수 있어,
                # 이 부분은 메인 스레드에서만 처리하거나, 화면에 표시할 최종 프레임에만 넣도록 합니다.
                # 여기서는 UI에 영향을 주는 메시지 업데이트는 메인 스레드에 맡기기 위해 생략합니다.

            elapsed = time.time() - self.countdown_start_time
            countdown_value = COUNTDOWN_TIME - elapsed

            if countdown_value <= 0:
                self.countdown_active = False
                self.captured = True
                self.last_capture_time = time.time()

                # ✅ 캡처된 원본 RGB 이미지를 result_queue에 전송
                self.result_queue.put(rgb)

                # ✅ 셔터 소리 신호를 shutter_queue에 전송
                try:
                    self.shutter_queue.put(True, block=False)
                except queue.Full:
                    pass

        else:
            if self.countdown_active:
                self.countdown_active = False
                # 상태 메시지 업데이트는 메인 스레드에 맡김

        # ---------------- 4. 화면 표시 갱신 ----------------
        if self.countdown_active:
            # 카운트다운 숫자는 비디오 프레임에만 표시
            countdown_display = max(1, int(COUNTDOWN_TIME - (time.time() - self.countdown_start_time)) + 1)
            cv2.putText(img_out, f"Capturing in: {countdown_display}", (img_w // 2 - 150, img_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)

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
    모든 조건(얼굴, V-사인, 두 가지 비율)이 충족되면 **3초 카운트다운** 후 자동으로 캡처됩니다.
    ---
    """)

    SHUTTER_HTML = load_shutter_html()

    # Session State 초기화
    if 'capture_ready' not in st.session_state:
        st.session_state.capture_ready = False
        st.session_state.captured_image_rgb = None
        st.session_state.capture_message = "카메라를 켜고 자세를 잡아주세요."
        # ✅ 카운트다운 상태를 UI에 표시하기 위한 상태
        st.session_state.countdown_ui_active = False

    current_message = st.session_state.get('capture_message', "카메라를 켜고 자세를 잡아주세요.")

    col1, col2 = st.columns([2, 1])

    # ---------------- I. 웹캠 스트림 (col1) ----------------
    with col1:
        st.subheader("실시간 웹캠 스트림 (비전 처리)")
        status_placeholder = st.empty()  # 상태 메시지 출력을 위한 placeholder

        webrtc_ctx = webrtc_streamer(
            key="media-pipe-detector",
            video_processor_factory=VideoProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        # 웹캠 상태 메시지는 아래 루프에서 업데이트

    # ---------------- II. 캡처 및 셔터 소리 감지 루프 (메인 스레드) ----------------
    if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
        processor = webrtc_ctx.video_processor

        # 무한 루프 대신, Streamlit rerun() 후에는 루프를 종료하도록 처리
        while True:
            # 1. 이미지 캡처 결과 큐 확인
            try:
                # 큐 타임아웃을 짧게 설정하여 UI 업데이트를 방해하지 않도록 함
                result_img_rgb = processor.result_queue.get(timeout=0.01)
            except queue.Empty:
                result_img_rgb = None
            except Exception:
                break

            # 2. 셔터 소리 신호 큐 확인 및 HTML 삽입
            try:
                # 셔터 소리 신호는 이미지 큐와 독립적으로 감지
                shutter_signal = processor.shutter_queue.get(timeout=0.01)
            except queue.Empty:
                shutter_signal = None
            except Exception:
                break

            # 3. 캡처 성공 시 상태 업데이트 및 UI 재실행 (가장 중요)
            if result_img_rgb is not None:
                st.session_state.captured_image_rgb = result_img_rgb
                st.session_state.capture_ready = True
                st.session_state.capture_message = "✅ 촬영 성공! 아래에서 결과를 확인하세요."
                st.session_state.countdown_ui_active = False  # 카운트다운 비활성화

                # 비디오 프로세서의 상태를 재설정 (다음 캡처 준비)
                processor.captured = False

                # 셔터 소리 신호가 있으면 재생 (큐에서 꺼낸 후)
                if shutter_signal:
                    components.html(SHUTTER_HTML, height=0)

                # Streamlit을 재실행하여 UI 업데이트
                st.rerun()
                break  # 루프 종료

            # 4. 카운트다운 상태 감지 (단순 UI 메시지 업데이트용)
            # 카운트다운이 활성화되어 있고, 캡처된 상태가 아니며, 이미지 큐에 신호가 없는 경우
            if processor.countdown_active and not st.session_state.capture_ready:
                if not st.session_state.countdown_ui_active:
                    st.session_state.countdown_ui_active = True
                    st.session_state.capture_message = f"카운트다운 시작! {COUNTDOWN_TIME}초 유지하세요."
                    # 카운트다운 시작 메시지를 바로 띄우기 위해 한 번만 rerun (성능 최적화를 위해 지양되기도 하나, 즉각적인 피드백을 위해 사용)
                    # st.rerun() # 주석 처리: 매번 rerun하지 않고, 아래 status_placeholder를 사용

            elif not processor.countdown_active and st.session_state.countdown_ui_active:
                # 카운트다운이 중단됨
                st.session_state.countdown_ui_active = False
                st.session_state.capture_message = "⏳ 조건 미달로 카운트다운 중단."
                # st.rerun() # 주석 처리: 매번 rerun하지 않고, 아래 status_placeholder를 사용

            # 큐를 확인하는 동안 CPU 부하를 줄임
            time.sleep(0.01)

            # 실시간 상태 메시지 업데이트 (rerun 없이)
            if webrtc_ctx.video_processor:
                vp = webrtc_ctx.video_processor

                if vp.countdown_active:
                    countdown_time = max(1, int(COUNTDOWN_TIME - (time.time() - vp.countdown_start_time)) + 1)
                    status_placeholder.warning(f"현재 상태: **카운트다운 중! {countdown_time}초** 남았습니다.")
                elif st.session_state.capture_ready:
                    status_placeholder.success(f"현재 상태: **{st.session_state.capture_message}**")
                else:
                    status_text = "조건을 충족시켜주세요. (얼굴, V-사인, 비율)"
                    if not vp.face_detector.process(cv2.cvtColor(webrtc_ctx.video_processor.to_ndarray(format="bgr24"),
                                                                 cv2.COLOR_BGR2RGB)).detections:
                        status_text = "얼굴을 인식하지 못했습니다."

                    status_placeholder.info(f"현재 상태: **{status_text}**")

    # ---------------- III. 결과 표시 (col2) ----------------
    with col2:
        st.subheader("✅ 캡처 조건 및 결과")
        st.markdown(
            f"""
            **✅ 캡처 조건 (모두 충족해야 함):**
            * **얼굴 감지**
            * **V-사인 감지**
            * **눈-입 비율 (A):** ${TARGET_A_MIN}\\% \sim {TARGET_A_MAX}\\%$
            * **코-입 비율 (B):** ${TARGET_B_MIN}\\% \sim {TARGET_B_MAX}\\%$
            """
        )
        st.markdown("---")

        if st.session_state.get('capture_ready') and st.session_state.get('captured_image_rgb') is not None:
            st.success("🎉 **캡처 완료!**")

            captured_img = st.session_state.captured_image_rgb

            st.image(captured_img, caption="최근 캡처 이미지", use_container_width=True)

            img_bgr = cv2.cvtColor(captured_img, cv2.COLOR_RGB2BGR)
            ret, buffer = cv2.imencode(".png", img_bgr)

            if ret:
                st.download_button(
                    label="🖼️ 캡처 이미지 다운로드",
                    data=buffer.tobytes(),
                    file_name=f"capture_optimal_{int(time.time())}.png",
                    mime="image/png"
                )

            if st.button("🔄 다음 캡처 준비"):
                st.session_state.capture_ready = False
                st.session_state.captured_image_rgb = None
                st.session_state.capture_message = "카메라를 켜고 자세를 잡아주세요."
                st.rerun()

        else:
            st.warning("아직 캡처된 이미지가 없습니다. 조건을 충족시켜보세요!")


if __name__ == "__main__":
    main()