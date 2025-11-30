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

# Mediapipe Face Detection 초기화
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
        # 이 에러 메시지는 Streamlit UI에 표시될 수 있습니다.
        st.error("shutter.wav 파일을 찾을 수 없습니다. 파일이 스크립트와 같은 폴더에 있는지 확인하세요.")
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
      - 여러 사람의 얼굴 roll angle 계산
      - 타겟 각도와 비교해 사람별 유사도 계산
      - 사람별 유사도 바 표시
      - 조건 만족 시 자동 캡처 (최근 10장 저장, 캡처 사진에는 UI 없음)
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

        # 🔔 셔터 소리 트리거용 큐
        self.shutter_queue = queue.Queue()
        # 🔔 캡처 발생 신호용 큐 (main 함수에서 UI 새로고침을 위해 사용)
        self.capture_event_queue = queue.Queue()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        raw_img = img.copy()

        h, w, _ = img.shape

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(img_rgb)

        self.person_infos = []
        faces_for_capture = []

        if results and results.detections:
            # (중략: 얼굴 인식 및 유사도 계산 로직)
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

                # 눈 선 및 얼굴 박스 그리기 (중략)
                keypoints = det.location_data.relative_keypoints
                right_eye = keypoints[0]
                left_eye = keypoints[1]
                x1, y1 = right_eye.x * w, right_eye.y * h
                x2, y2 = left_eye.x * w, left_eye.y * h
                right_eye_pt = (int(x1), int(y1))
                left_eye_pt = (int(x2), int(y2))

                cv2.line(img, right_eye_pt, left_eye_pt, (0, 255, 0), 2)

                rel_box = det.location_data.relative_bounding_box
                bx = int(rel_box.xmin * w)
                by = int(rel_box.ymin * h)
                bw = int(rel_box.width * w)
                bh = int(rel_box.height * h)
                bx = max(0, bx)
                by = max(0, by)
                bw = max(0, bw)
                bh = max(0, bh)
                cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (255, 255, 0), 1)

                if sim is not None:
                    text = f"P{idx} angle:{angle_deg:.1f} deg | sim:{sim:.0f}%"
                else:
                    text = f"P{idx} angle:{angle_deg:.1f} deg"
                cv2.putText(
                    img,
                    text,
                    (bx, max(0, by - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                if sim is not None:
                    # 얼굴 아래 개인 유사도 바 그리기 (중략)
                    bar_x1 = bx
                    bar_y1 = by + bh + 10
                    bar_x2 = bx + bw
                    bar_y2 = bar_y1 + 10

                    bar_x1 = max(0, min(bar_x1, w - 1))
                    bar_x2 = max(0, min(bar_x2, w - 1))
                    bar_y1 = max(0, min(bar_y1, h - 1))
                    bar_y2 = max(0, min(bar_y2, h - 1))

                    if bar_x2 > bar_x1 and bar_y2 > bar_y1:
                        cv2.rectangle(
                            img,
                            (bar_x1, bar_y1),
                            (bar_x2, bar_y2),
                            (80, 80, 80),
                            1,
                        )
                        ratio = max(0.0, min(1.0, sim / 100.0))
                        fill_x2 = bar_x1 + int((bar_x2 - bar_x1) * ratio)
                        cv2.rectangle(
                            img,
                            (bar_x1, bar_y1),
                            (fill_x2, bar_y2),
                            (0, 200, 0),
                            -1,
                        )

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

                    # 🔔 셔터 소리 신호 보내기
                    try:
                        self.shutter_queue.put_nowait(True)
                    except queue.Full:
                        pass

                    # 🔔 캡처 발생 신호 보내기 (main 함수에서 캡처 목록 새로고침을 위해 사용)
                    try:
                        self.capture_event_queue.put_nowait(True)
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
    # 캡처 목록을 관리하기 위한 placeholder 설정
    if 'capture_placeholder' not in st.session_state:
        st.session_state.capture_placeholder = st.empty()

    st.title("타겟 포즈 유사도 기반 자동 촬영 (여러 명 + 전면/후면 지원)")
    # (중략: 안내 문구)

    # 🔔 셔터 소리 HTML 미리 준비
    SHUTTER_HTML = load_shutter_html()

    # --- 타겟 사진 업로드 --- (중략)
    st.sidebar.header("① 타겟 사진 업로드")
    ref_file = st.sidebar.file_uploader(
        "타겟 포즈 사진 (jpg, png)",
        type=["jpg", "jpeg", "png"],
        key="ref_upload",
    )

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

    # --- 촬영 조건 --- (중략)
    st.sidebar.header("② 촬영 조건")
    tolerance = st.sidebar.slider(
        "허용 각도 차 (deg)",
        min_value=2.0,
        max_value=30.0,
        value=8.0,
        step=1.0,
    )
    cooldown_sec = st.sidebar.slider(
        "촬영 간 최소 간격 (초)",
        min_value=0.0,
        max_value=10.0,
        value=3.0,
        step=1.0,
    )

    if ref_disp is not None:
        st.subheader("타겟 사진 (각도 표시)")
        st.image(ref_disp, channels="BGR")

    rtc_config = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    st.subheader("웹캠")

    # 전면 / 후면 카메라 선택 (모바일에서 유효)
    cam_mode = st.radio(
        "카메라 선택",
        ["전면", "후면"],
        horizontal=True,
    )

    # 해상도는 조금 낮게 (480x360) → 전송/디코딩 안정성 ↑
    base_constraints = {
        "width": {"ideal": 480},
        "height": {"ideal": 360},
        "frameRate": {"ideal": 15},
    }

    if cam_mode == "전면":
        video_constraints = {
            **base_constraints,
            "facingMode": {"ideal": "user"},
        }
    else:  # 후면
        video_constraints = {
            **base_constraints,
            "facingMode": {"ideal": "environment"},
        }

    webrtc_ctx = webrtc_streamer(
        key="pose-match-capture-multi",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        media_stream_constraints={
            "video": video_constraints,
            "audio": False,
        },
        video_processor_factory=PoseMatchProcessor,
        async_processing=True,
    )

    # ---------------- 캡처 목록을 다시 그리는 함수 ----------------
    def redraw_captured_images(vp: PoseMatchProcessor):
        with st.session_state.capture_placeholder.container():
            st.subheader("자동 촬영된 사진들")

            # 캡처 목록 새로고침 버튼을 누르더라도 캡처가 재발생하지 않도록 st.rerun() 없이 로직만 수행
            if st.button("캡처 목록 새로고침"):
                pass

            if vp.captured_images:
                for idx, img in enumerate(reversed(vp.captured_images), start=1):
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

    if webrtc_ctx.video_processor:
        vp: PoseMatchProcessor = webrtc_ctx.video_processor

        # 타겟 각도/설정 전달
        vp.ref_angle = ref_angle
        vp.tolerance = tolerance
        vp.cooldown_sec = cooldown_sec

        # ---------------- 셔터 소리 및 캡처 이벤트 감지 로직 ----------------
        shutter_sound_triggered = False
        capture_list_updated = False

        try:
            # 🔔 셔터 큐 확인 → 신호 있으면 소리 재생
            if vp.shutter_queue.get_nowait():
                components.html(SHUTTER_HTML, height=0)
                shutter_sound_triggered = True
        except queue.Empty:
            pass

        try:
            # 🔔 캡처 이벤트 큐 확인 → 신호 있으면 캡처 목록 업데이트를 위해 UI 새로고침
            if vp.capture_event_queue.get_nowait():
                capture_list_updated = True
        except queue.Empty:
            pass

        # 캡처 목록을 표시합니다.
        # 주의: 이 로직은 Streamlit의 UI 업데이트를 위해 메인 루프에서 실행됩니다.
        redraw_captured_images(vp)

        # 캡처가 발생한 경우, 캡처 목록을 업데이트하기 위해 Streamlit을 재실행합니다.
        # 캡처 버튼 누름으로 인한 재실행이 아니므로 캡처가 재발생하지 않습니다.
        if capture_list_updated:
            st.rerun()

        # ---------------- 현재 상태 표시 (기존 코드 유지) ----------------
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


if __name__ == "__main__":
    main()