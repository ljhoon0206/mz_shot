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


# ---------------- 셔터 소리용 HTML 생성 함수 (변경 없음) ----------------
def load_shutter_html():
    try:
        with open("shutter.wav", "rb") as f:
            audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        html = f"""
        <audio autoplay>
            <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav" />
        </audio>
        """
        return html
    except FileNotFoundError:
        return ""


# ---------------- 얼굴 각도 계산 및 그리기 함수 (변경 없음) ----------------
def get_face_roll_angle(img_bgr):
    h, w, _ = img_bgr.shape
    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6) as face_detector:
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
        angle_deg = np.degrees(np.arctan2(dy, dx))
        return angle_deg, ((int(x1), int(y1)), (int(x2), int(y2)))


def draw_angle_overlay(img_bgr, angle_deg, eye_pts, label=""):
    img = img_bgr.copy()
    if angle_deg is not None and eye_pts is not None:
        cv2.line(img, eye_pts[0], eye_pts[1], (0, 255, 0), 2)
        text = f"{label} roll: {angle_deg:.1f} deg"
    else:
        text = f"{label} No face"
    cv2.putText(img, text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
    return img


# ---------------- 비디오 프로세서 클래스 ----------------
class PoseMatchProcessor(VideoProcessorBase):
    def __init__(self):
        self.ref_angle = None
        self.tolerance = 5.0
        self.cooldown_sec = 3.0
        self.last_capture_time = 0.0

        self.person_infos = []

        # [수정 1] 내부 리스트 대신 외부에서 주입받을 저장소 변수 선언
        self.shared_gallery = None

        self.face_detector = mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.6
        )
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
            temp_list = []
            for det in results.detections:
                # ... (좌표 계산 로직 동일) ...
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
                    if diff >= self.tolerance:
                        sim = 0.0
                    else:
                        sim = max(0.0, 100.0 * (1.0 - diff / self.tolerance))

                center_x = (x1 + x2) / 2.0
                temp_list.append({"det": det, "angle": angle_deg, "sim": sim, "center_x": center_x})

            temp_list.sort(key=lambda d: d["center_x"])

            for idx, info in enumerate(temp_list, start=1):
                # ... (정보 저장 로직 동일) ...
                det = info["det"]
                angle_deg = info["angle"]
                sim = info["sim"]
                self.person_infos.append({"id": idx, "angle": angle_deg, "sim": sim})

                # 얼굴 박스 그리기 등 시각화 (간략화)
                bbox = det.location_data.relative_bounding_box
                rect_start_point = mp_face.get_rect_xmin(bbox, w, h), mp_face.get_rect_ymin(bbox, w, h)
                rect_end_point = rect_start_point[0] + mp_face.get_rect_width(bbox, w, h), rect_start_point[
                    1] + mp_face.get_rect_height(bbox, w, h)
                cv2.rectangle(img, rect_start_point, rect_end_point, (0, 255, 0), 2)

                if sim is not None:
                    faces_for_capture.append(sim)

            # 자동 캡처 로직
            if self.ref_angle is not None and faces_for_capture:
                max_sim = max(faces_for_capture)
                now = time.time()
                if max_sim >= 90.0 and now - self.last_capture_time > self.cooldown_sec:
                    self.last_capture_time = now

                    # [수정 2] 외부(Session State) 리스트에 직접 저장
                    if self.shared_gallery is not None:
                        self.shared_gallery.append(raw_img.copy())
                        # 10장 제한 유지
                        if len(self.shared_gallery) > 10:
                            self.shared_gallery.pop(0)

                    try:
                        self.shutter_queue.put(True, block=False)
                    except queue.Full:
                        pass

                    cv2.putText(img, "CAPTURED!", (30, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3,
                                cv2.LINE_AA)

            top_text = f"Faces: {len(temp_list)}"
            if self.ref_angle is not None:
                top_text += f" | target:{self.ref_angle:.1f} deg"
        else:
            top_text = "No face"
            self.person_infos = []

        cv2.putText(img, top_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    st.title("타겟 포즈 유사도 기반 자동 촬영")
    SHUTTER_HTML = load_shutter_html()

    # [수정 3] 세션 스테이트 초기화 (이 리스트를 프로세서와 공유함)
    if "captured_images_main" not in st.session_state:
        st.session_state["captured_images_main"] = []

    st.sidebar.header("① 타겟 사진 업로드")
    ref_file = st.sidebar.file_uploader("타겟 포즈 사진", type=["jpg", "png"], key="ref_upload")

    ref_angle = None
    if ref_file is not None:
        data = ref_file.read()
        arr = np.frombuffer(data, np.uint8)
        ref_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if ref_img is not None:
            ref_angle, eye_pts = get_face_roll_angle(ref_img)
            if ref_angle:
                st.sidebar.success(f"타겟 각도: {ref_angle:.1f}°")
                st.image(draw_angle_overlay(ref_img, ref_angle, eye_pts), channels="BGR")

    st.sidebar.header("② 촬영 조건")
    tolerance = st.sidebar.slider("허용 각도 차", 2.0, 30.0, 8.0)
    cooldown_sec = st.sidebar.slider("촬영 간격(초)", 0.0, 10.0, 3.0)

    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    cam_mode = st.radio("카메라 선택", ["전면", "후면"], horizontal=True)
    video_constraints = {
        "width": {"ideal": 480}, "height": {"ideal": 360}, "frameRate": {"ideal": 15},
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

    # [수정 4] 프로세서가 생성되면 즉시 세션 스테이트 리스트를 연결해줌
    if webrtc_ctx.video_processor:
        vp = webrtc_ctx.video_processor
        vp.ref_angle = ref_angle
        vp.tolerance = tolerance
        vp.cooldown_sec = cooldown_sec

        # 여기가 핵심: 세션 스테이트의 리스트 주소를 프로세서 변수에 할당
        vp.shared_gallery = st.session_state["captured_images_main"]

        # 셔터 소리 감지 루프
        while webrtc_ctx.state.playing:
            try:
                if vp.shutter_queue.get(timeout=0.1):
                    components.html(SHUTTER_HTML, height=0)
                    st.rerun()  # 캡처 즉시 화면 갱신 (선택 사항)
            except queue.Empty:
                pass
            except Exception:
                break
            time.sleep(0.01)

        # [수정 5] 루프 내부에서 정보 표시 (실시간)
        if vp.person_infos:
            st.subheader("현재 상태")
            for info in vp.person_infos:
                st.write(f"사람 {info['id']}: 각도 {info['angle']:.1f}°, 유사도 {info['sim'] if info['sim'] else 0:.0f}%")

    # [수정 6] 루프 밖(Stop 누른 후)에도 이미지는 세션 스테이트에 남아있으므로 바로 표시됨
    st.subheader("자동 촬영된 사진들 (최신순)")

    # 갤러리 표시 (Stop 눌러도 데이터가 유지됨)
    if st.session_state["captured_images_main"]:
        # 리스트 초기화 버튼
        if st.button("갤러리 비우기"):
            st.session_state["captured_images_main"] = []
            st.rerun()

        cols = st.columns(3)  # 3열로 예쁘게 표시
        for idx, img in enumerate(reversed(st.session_state["captured_images_main"])):
            col = cols[idx % 3]
            col.image(img, channels="BGR", caption=f"캡처 #{len(st.session_state['captured_images_main']) - idx}")

            # 다운로드 버튼
            success, buf = cv2.imencode(".jpg", img)
            if success:
                col.download_button(
                    label="다운로드",
                    data=buf.tobytes(),
                    file_name=f"capture_{idx}.jpg",
                    mime="image/jpeg",
                    key=f"down_{idx}"
                )
    else:
        st.info("아직 캡처된 사진이 없습니다.")


if __name__ == "__main__":
    main()