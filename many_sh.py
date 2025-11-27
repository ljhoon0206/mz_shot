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
      - 조건 만족 시 자동 캡처
      - 캡처 순간 셔터 소리 신호(shutter_queue) 보냄
    """

    def __init__(self):
        self._frame_format = "bgr24"

        self.ref_angle = None
        self.tolerance = 5.0
        self.cooldown_sec = 3.0
        self.last_capture_time = 0.0

        self.person_infos = []
        self.captured_images = []

        self.face_detector = mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.6
        )

        # ⭐ 셔터 소리 트리거용 큐 (추가됨)
        self.shutter_queue = queue.Queue()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        raw_img = img.copy()

        h, w, _ = img.shape

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(img_rgb)

        self.person_infos = []
        faces_for_capture = []

        if results and results.detections:
            # ... (유사도 계산 및 그리기 로직은 이전 코드와 동일하게 진행) ...
            temp_list = []
            for det in results.detections:
                keypoints = det.location_data.relative_keypoints
                right_eye = keypoints[0]
                left_eye = keypoints[1]

                x1, y1 = right_eye.x * w, right_eye.y * h
                x2, y2 = left_eye.x * w, left_eye.y * h
                dx, dy = x2 - x1, y2 - y1

                angle_rad = np.arctan2(dy, dx)
                angle_deg = np.degrees(angle_rad)

                sim = None
                if self.ref_angle is not None:
                    diff = abs(angle_deg - self.ref_angle)
                    if diff >= self.tolerance:
                        sim = 0.0
                    else:
                        sim = max(0.0, 100.0 * (1.0 - diff / self.tolerance))

                center_x = (x1 + x2) / 2.0
                temp_list.append(
                    {
                        "det": det,
                        "angle": angle_deg,
                        "sim": sim,
                        "center_x": center_x,
                    }
                )

            temp_list.sort(key=lambda d: d["center_x"])

            for idx, info in enumerate(temp_list, start=1):
                det = info["det"]
                angle_deg = info["angle"]
                sim = info["sim"]

                self.person_infos.append(
                    {
                        "id": idx,
                        "angle": angle_deg,
                        "sim": sim,
                    }
                )

                # 시각적 피드백 (눈 선, 박스, 텍스트, 유사도 바) 그리기 로직 (생략)

                if sim is not None:
                    faces_for_capture.append(sim)

            # 4) 여러 명 중 하나라도 유사도 기준 넘으면 자동 캡처
            if self.ref_angle is not None and faces_for_capture:
                max_sim = max(faces_for_capture)
                now = time.time()
                if max_sim >= 90.0 and now - self.last_capture_time > self.cooldown_sec:
                    self.last_capture_time = now

                    # ✅ UI 없는 원본(raw_img)을 저장
                    self.captured_images.append(raw_img.copy())
                    if len(self.captured_images) > 10:
                        self.captured_images.pop(0)

                    # ⭐ 셔터 소리 신호 보내기 (큐 사용)
                    try:
                        self.shutter_queue.put(True, block=False)
                    except queue.Full:
                        pass

                    # 화면에만 CAPTURED! 텍스트 표시
                    cv2.putText(
                        img,
                        "CAPTURED!",
                        (30, h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 255),
                        3,
                        cv2.LINE_AA,
                    )

            top_text = f"Faces: {len(temp_list)}"
            if self.ref_angle is not None:
                top_text += f" | target:{self.ref_angle:.1f} deg"
        else:
            top_text = "No face"
            self.person_infos = []

        # 상단 공통 텍스트
        cv2.putText(
            img,
            top_text,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    st.title("타겟 포즈 유사도 기반 자동 촬영 (여러 명 + 전면/후면 지원)")

    st.markdown(
        """
        **문제 해결:** 셔터 소리 지연 문제 해결을 위해, 캡처 시 신호를 **큐**에 넣어 메인 스레드에서 **지속적으로 감지**하도록 수정했습니다.
        ---
        """
    )

    # 🔔 셔터 소리 HTML 미리 준비
    SHUTTER_HTML = load_shutter_html()

    # 캡처된 이미지 목록은 Session State에 저장하여 UI에 반영
    if "captured_images_main" not in st.session_state:
        st.session_state["captured_images_main"] = []

    # --- (생략: 타겟 사진 업로드, 촬영 조건 설정 코드는 이전과 동일) ---
    st.sidebar.header("① 타겟 사진 업로드")
    ref_file = st.sidebar.file_uploader("타겟 포즈 사진 (jpg, png)", type=["jpg", "jpeg", "png"], key="ref_upload")

    ref_angle = None
    ref_disp = None

    if ref_file is not None:
        data = ref_file.read()
        arr = np.frombuffer(data, np.uint8)
        ref_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if ref_img is None:
            st.sidebar.error("타겟 사진을 읽지 못했습니다.")
        else:
            ref_angle, eye_pts = get_face_roll_angle(ref_img)
            if ref_angle is None:
                st.sidebar.error("타겟 사진에서 얼굴을 찾지 못했습니다.")
            else:
                st.sidebar.success(f"타겟 얼굴 각도: {ref_angle:.1f}°")
                ref_disp = draw_angle_overlay(ref_img, ref_angle, eye_pts, label="target")

    st.sidebar.header("② 촬영 조건")
    tolerance = st.sidebar.slider("허용 각도 차 (deg)", min_value=2.0, max_value=30.0, value=8.0, step=1.0)
    cooldown_sec = st.sidebar.slider("촬영 간 최소 간격 (초)", min_value=0.0, max_value=10.0, value=3.0, step=1.0)

    if ref_disp is not None:
        st.subheader("타겟 사진 (각도 표시)")
        st.image(ref_disp, channels="BGR")

    rtc_config = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    st.subheader("웹캠")

    cam_mode = st.radio("카메라 선택", ["전면", "후면"], horizontal=True)

    base_constraints = {
        "width": {"ideal": 480},
        "height": {"ideal": 360},
        "frameRate": {"ideal": 15},
    }

    video_constraints = {
        **base_constraints,
        "facingMode": {"ideal": "user"} if cam_mode == "전면" else {"ideal": "environment"},
    }

    webrtc_ctx = webrtc_streamer(
        key="pose-match-capture-multi",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": video_constraints, "audio": False},
        video_processor_factory=PoseMatchProcessor,
        async_processing=True,
    )

    # ⭐ 셔터 소리 감지 및 목록 업데이트 루프 (메인 스레드)
    if webrtc_ctx.video_processor:
        vp: PoseMatchProcessor = webrtc_ctx.video_processor

        # 타겟 각도/설정 전달
        vp.ref_angle = ref_angle
        vp.tolerance = tolerance
        vp.cooldown_sec = cooldown_sec

        # 캡처 신호 감지 루프
        while webrtc_ctx.state.playing:
            try:
                # ⭐ 셔터 소리 신호 확인 (논블로킹)
                if vp.shutter_queue.get(timeout=0.1):
                    components.html(SHUTTER_HTML, height=0)  # 소리 재생

                    # 캡처된 이미지 목록을 메인 스레드 상태로 복사
                    # 캡처된 이미지의 사본을 저장하여, 이후 UI 업데이트 시 반영
                    st.session_state["captured_images_main"] = vp.captured_images[:]

                    # ⭐ 상태가 변경되었으므로 Streamlit을 재실행하여 UI를 업데이트
                    # (이때 소리가 나지 않도록 이미 큐에서 신호를 꺼냈음)
                    st.rerun()
                    break

            except queue.Empty:
                pass  # 신호 없으면 계속 대기
            except Exception:
                break

            time.sleep(0.01)  # CPU 부하 줄이기

        st.subheader("현재 상태 (왼쪽부터 P1, P2, ...)")

        if ref_angle is None:
            st.warning("타겟 사진을 먼저 업로드해야 유사도 계산이 가능합니다.")
        else:
            if not vp.person_infos:
                st.write("현재 얼굴을 찾는 중입니다.")
            else:
                for info in vp.person_infos:
                    pid = info["id"]
                    angle = info["angle"]
                    sim = info["sim"]
                    if sim is None:
                        st.write(f"사람 P{pid}: 각도 **{angle:.1f}°** (타겟 없음)")
                    else:
                        st.write(
                            f"사람 P{pid}: 각도 **{angle:.1f}°**, "
                            f"유사도(각도 기준): **{sim:.0f}%**"
                        )

        st.subheader("자동 촬영된 사진들")

        # 캡처 목록은 셔터 소리 감지 루프에서 업데이트된 session_state를 사용
        if st.session_state["captured_images_main"]:
            st.button("캡처 목록 다시 그리기", key="refresh_ui")  # UI 갱신용

            for idx, img in enumerate(reversed(st.session_state["captured_images_main"]), start=1):
                st.image(img, channels="BGR", caption=f"캡처 #{idx}")

                success, buf = cv2.imencode(".jpg", img)
                if success:
                    st.download_button(
                        label=f"이 사진 다운로드 #{idx}",
                        data=buf.tobytes(),
                        file_name=f"capture_{idx}.jpg",
                        mime="image/jpeg",
                        key=f"download_{idx}",
                    )
        else:
            st.write("아직 캡처된 사진이 없습니다. 타겟 각도와 비슷하게 맞춰보세요.")


if __name__ == "__main__":
    main()