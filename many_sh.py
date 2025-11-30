import time
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
import av
import queue
import base64
import streamlit.components.v1 as components

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration,
    WebRtcMode,
)

mp_face = mp.solutions.face_detection


# ---------------- 셔터 소리용 HTML 생성 함수 ----------------
def load_shutter_html():
    """
    shutter.wav 파일을 base64로 읽어서 <audio> 자동 재생 HTML을 만들어 줌.
    shutter.wav는 이 파이썬 파일과 같은 폴더에 있어야 함.
    """
    try:
        # Note: 'shutter.wav' file must be present in the same directory.
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


def get_face_roll_angle(img_bgr):
    """BGR 이미지에서 첫 번째 얼굴의 roll angle(기울기) 계산."""
    h, w, _ = img_bgr.shape

    with mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.6
    ) as face_detector:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = face_detector.process(img_rgb)

        if not results.detections:
            return None, None

        detection = results.detections[0]
        keypoints = detection.location_data.relative_keypoints

        right_eye = keypoints[0]
        left_eye = keypoints[1]

        x1, y1 = right_eye.x * w, right_eye.y * h
        x2, y2 = left_eye.x * w, left_eye.y * h

        dx = x2 - x1
        dy = y2 - y1

        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)

        right_eye_pt = (int(x1), int(y1))
        left_eye_pt = (int(x2), int(y2))

        return angle_deg, (right_eye_pt, left_eye_pt)


def draw_angle_overlay(img_bgr, angle_deg, eye_pts, label=""):
    """타겟 이미지 위에 눈 선 + 각도 텍스트 표시."""
    img = img_bgr.copy()
    if angle_deg is not None and eye_pts is not None:
        (re, le) = eye_pts
        cv2.line(img, re, le, (0, 255, 0), 2)
        text = f"{label} roll: {angle_deg:.1f} deg"
    else:
        text = f"{label} No face"

    cv2.putText(
        img,
        text,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return img


class PoseMatchProcessor(VideoProcessorBase):
    """
    웹캠 프레임을 받아서:
      - 얼굴 roll angle 계산 및 비교
      - 조건 만족 시 자동 캡처 (단일)
      - 캡처 순간 셔터 소리 신호(shutter_queue) 보냄
    """

    def __init__(self):
        self._frame_format = "bgr24"

        self.ref_angle = None
        self.tolerance = 5.0
        self.capture_threshold = 90.0  # 유사도 캡처 기준 (90% 이상)

        # 단일 캡처 상태 관리 변수
        self.is_capturing_enabled = True  # 캡처가 활성화되었는지 (초기화 버튼으로 제어)
        self.captured_image_rgb = None  # 캡처된 단일 이미지 (RGB)

        # Streamlit session_state를 직접 조작하지 않도록 변경
        # 메인 스레드에서 주기적으로 값을 가져가도록 하기 위한 Queue
        self.capture_state_queue = queue.Queue(maxsize=1)
        self.shutter_queue = queue.Queue(maxsize=1)

        self.face_detector = mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.6
        )

    def recv(self, frame):
        img_bgr = frame.to_ndarray(format="bgr24")
        raw_img_rgb = cv2.cvtColor(img_bgr.copy(), cv2.COLOR_BGR2RGB)  # 캡처용 RGB 이미지

        h, w, _ = img_bgr.shape
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(img_rgb)

        # ------------------- 얼굴 감지 및 유사도 계산 -------------------
        faces_for_capture = []
        max_sim = 0.0

        if results and results.detections:
            for det in results.detections:
                keypoints = det.location_data.relative_keypoints
                right_eye = keypoints[0]
                left_eye = keypoints[1]

                x1, y1 = right_eye.x * w, right_eye.y * h
                x2, y2 = left_eye.x * w, left_eye.y * h
                dx, dy = x2 - x1, y2 - y1

                angle_deg = np.degrees(np.arctan2(dy, dx))

                sim = None
                if self.ref_angle is not None:
                    diff = abs(angle_deg - self.ref_angle)
                    if diff < self.tolerance:
                        sim = max(0.0, 100.0 * (1.0 - diff / self.tolerance))

                if sim is not None:
                    faces_for_capture.append(sim)
                    max_sim = max(max_sim, sim)

            # ------------------- 캡처 로직 (단일) -------------------

            # 조건: 캡처 활성화 & 타겟 설정됨 & 유사도 기준 충족
            if self.is_capturing_enabled and self.ref_angle is not None and max_sim >= self.capture_threshold:

                self.is_capturing_enabled = False  # 캡처 비활성화
                self.captured_image_rgb = raw_img_rgb  # 이미지 저장

                # 메인 스레드에 캡처 완료 신호 전송
                try:
                    # 캡처된 RGB 이미지와 셔터 신호를 큐에 전송
                    if self.capture_state_queue.empty():
                        self.capture_state_queue.put(self.captured_image_rgb.copy(), block=False)
                    if self.shutter_queue.empty():
                        self.shutter_queue.put(True, block=False)
                except queue.Full:
                    pass

                # 화면에만 CAPTURED! 텍스트 표시
                cv2.putText(
                    img_bgr,
                    "CAPTURED!",
                    (30, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )

            # ------------------- 시각적 피드백 (생략되지 않음) -------------------
            # 현재 얼굴 정보 표시 및 오버레이 그리기 (예: 가장 높은 유사도의 얼굴만)
            top_text = f"Faces: {len(results.detections)}"
            if self.ref_angle is not None:
                top_text += f" | target:{self.ref_angle:.1f} deg"
                if max_sim > 0.0:
                    top_text += f" | Max Sim: {max_sim:.0f}%"

        else:
            top_text = "No face"
            max_sim = 0.0

        # 상단 공통 텍스트
        cv2.putText(
            img_bgr,
            top_text,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")


def main():
    st.title("타겟 포즈 유사도 기반 단일 자동 촬영")

    st.markdown(
        """
        **기능:** 얼굴의 롤 각도(기울기)를 타겟 사진과 비교하여, 유사도가 **90% 이상**일 때 자동으로 단일 캡처를 수행합니다.
        ---
        """
    )

    # 🔔 셔터 소리 HTML 미리 준비
    SHUTTER_HTML = load_shutter_html()

    # Session State 초기화 및 캡처 상태 관리
    if "capture_ready" not in st.session_state:
        st.session_state["capture_ready"] = False
        st.session_state["captured_image_rgb"] = None
        st.session_state["ref_angle"] = None
        st.session_state["tolerance"] = 8.0

    col1, col2 = st.columns([2, 1])

    with col2:
        st.header("설정")
        st.subheader("① 타겟 사진 업로드")
        ref_file = st.file_uploader("타겟 포즈 사진 (jpg, png)", type=["jpg", "jpeg", "png"], key="ref_upload")

        ref_angle = None
        ref_disp = None

        if ref_file is not None:
            data = ref_file.read()
            arr = np.frombuffer(data, np.uint8)
            ref_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if ref_img is None:
                st.error("타겟 사진을 읽지 못했습니다.")
            else:
                ref_angle_new, eye_pts = get_face_roll_angle(ref_img)
                if ref_angle_new is None:
                    st.error("타겟 사진에서 얼굴을 찾지 못했습니다.")
                else:
                    ref_angle = ref_angle_new
                    st.success(f"타겟 얼굴 각도: {ref_angle:.1f}°")
                    ref_disp = draw_angle_overlay(ref_img, ref_angle, eye_pts, label="target")

        st.session_state["ref_angle"] = ref_angle

        st.subheader("② 촬영 조건")
        tolerance = st.slider("허용 각도 차 (deg)", min_value=2.0, max_value=30.0, value=st.session_state["tolerance"],
                              step=1.0)
        st.session_state["tolerance"] = tolerance

        st.markdown("---")

        if ref_disp is not None:
            st.subheader("타겟 포즈")
            st.image(ref_disp, channels="BGR", use_container_width=True)

    with col1:
        st.subheader("웹캠 스트림")
        if st.session_state["capture_ready"]:
            st.info("✅ 캡처 완료! 초기화 버튼을 눌러 다시 촬영하세요.")

        rtc_config = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        cam_mode = st.radio("카메라 선택", ["전면", "후면"], horizontal=True, key="cam_mode",
                            disabled=st.session_state["capture_ready"])

        base_constraints = {
            "width": {"ideal": 480},
            "height": {"ideal": 360},
            "frameRate": {"ideal": 15},
        }

        video_constraints = {
            **base_constraints,
            "facingMode": {"ideal": "user"} if cam_mode == "전면" else {"ideal": "environment"},
        }

        # 캡처 완료 상태이면 웹캠을 비활성화 (None)
        if not st.session_state["capture_ready"]:
            webrtc_ctx = webrtc_streamer(
                key="pose-match-capture-single",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=rtc_config,
                media_stream_constraints={"video": video_constraints, "audio": False},
                video_processor_factory=PoseMatchProcessor,
                async_processing=True,
            )
        else:
            # 캡처 완료 시 웹캠 대신 빈 컨테이너 표시
            st.empty()
            webrtc_ctx = None

    # ------------------ 메인 스레드 캡처 상태 업데이트 루프 (WebRTC 컨텍스트 이용) ------------------
    if webrtc_ctx and webrtc_ctx.video_processor:
        vp: PoseMatchProcessor = webrtc_ctx.video_processor

        # 타겟 각도/설정 전달
        vp.ref_angle = st.session_state["ref_angle"]
        vp.tolerance = st.session_state["tolerance"]

        # 프로세서 상태 확인
        try:
            # ⭐ 캡처 완료 신호 확인 (논블로킹)
            captured_img = vp.capture_state_queue.get(timeout=0.01)

            if captured_img is not None:
                # 캡처된 이미지와 상태를 Session State에 저장
                st.session_state["captured_image_rgb"] = captured_img
                st.session_state["capture_ready"] = True

                # 셔터 소리 재생
                components.html(SHUTTER_HTML, height=0)

                # UI 업데이트를 위해 Streamlit 재실행
                st.rerun()
        except queue.Empty:
            pass  # 신호 없으면 계속 진행
        except Exception:
            pass

    # ------------------ 캡처 완료 후 UI 로직 (Streamlit Standard UI) ------------------
    with col1:
        st.markdown("---")
        st.subheader("✨ 최종 캡처 결과")

        if st.session_state["capture_ready"] and st.session_state["captured_image_rgb"] is not None:
            st.success("🎉 **단일 캡처 완료!**")

            captured_img_rgb = st.session_state["captured_image_rgb"]

            st.image(captured_img_rgb, caption="최종 캡처 이미지", use_container_width=True)

            # 다운로드를 위해 RGB를 BGR로 변환 후 PNG 인코딩
            img_bgr = cv2.cvtColor(captured_img_rgb, cv2.COLOR_RGB2BGR)
            ret, buffer = cv2.imencode(".png", img_bgr)

            if ret:
                st.download_button(
                    label="🖼️ 캡처 이미지 다운로드",
                    data=buffer.tobytes(),
                    file_name=f"pose_capture_{int(time.time())}.png",
                    mime="image/png"
                )

            if st.button("🔄 다음 캡처 준비"):
                st.session_state["capture_ready"] = False
                st.session_state["captured_image_rgb"] = None

                # VideoProcessor의 상태 초기화 요청 (메인 스레드가 다음 실행 시 처리)
                if webrtc_ctx and webrtc_ctx.video_processor:
                    vp: PoseMatchProcessor = webrtc_ctx.video_processor
                    vp.is_capturing_enabled = True
                    vp.captured_image_rgb = None
                    # 큐 비우기 (불필요한 신호 방지)
                    while not vp.capture_state_queue.empty():
                        vp.capture_state_queue.get_nowait()
                    while not vp.shutter_queue.empty():
                        vp.shutter_queue.get_nowait()

                st.rerun()

        elif st.session_state["ref_angle"] is None:
            st.warning("먼저 타겟 사진을 업로드하여 타겟 각도를 설정해야 합니다.")
        else:
            st.info("웹캠을 켜고, 얼굴 각도를 타겟과 유사하게 맞춰보세요. (유사도 90% 이상)")


if __name__ == "__main__":
    main()