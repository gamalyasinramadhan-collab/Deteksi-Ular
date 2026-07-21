import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2
from inference_sdk import InferenceHTTPClient
import supervision as sv

# Config Tampilan
st.set_page_config(page_title="Deteksi Ular Real-time", layout="centered")
st.title("🐍 Deteksi Ular Real-Time")
st.write("Arahkan kamera ke objek untuk mendeteksi secara langsung.")

# Inisialisasi Roboflow Client
@st.cache_resource
def get_inference_client():
    return InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="K7ce7ZVzoLF5O0URWUWF"
    )

client = get_inference_client()
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Class Processor untuk menyimpan state (Frame Skipping)
class VideoProcessor:
    def __init__(self):
        self.frame_count = 0
        self.last_detections = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # Lakukan deteksi API HANYA setiap 3 frame sekali agar tidak lag
        if self.frame_count % 3 == 0:
            try:
                # Resize gambar menjadi lebih kecil khusus untuk request API (agar hemat bandwidth)
                resized_img = cv2.resize(img, (416, 416))
                
                # Inference Roboflow
                results = client.infer(resized_img, model_id="gamals-workspace-j10lj/ularlah-2-yolo11s-t1")
                
                # Bounding box dikembalikan ke skala frame asli
                detections = sv.Detections.from_inference(results)
                
                # Menyesuaikan kembali koordinat bbox dari ukuran (416x416) ke ukuran frame asli
                h_orig, w_orig = img.shape[:2]
                scale_x = w_orig / 416.0
                scale_y = h_orig / 416.0
                
                if len(detections) > 0:
                    detections.xyxy[:, [0, 2]] *= scale_x
                    detections.xyxy[:, [1, 3]] *= scale_y
                
                self.last_detections = detections
            except Exception:
                pass

        # Gambar deteksi terakhir pada frame aktif (video tetap jalan 30 FPS mulus)
        if self.last_detections is not None:
            img = bounding_box_annotator.annotate(scene=img, detections=self.last_detections)
            img = label_annotator.annotate(scene=img, detections=self.last_detections)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# Konfigurasi WebRTC STUN Server
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Widget Kamera
webrtc_streamer(
    key="snake-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": {
            "facingMode": "environment"
        },
        "audio": False
    },
    async_processing=True,
)