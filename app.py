import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2
from inference_sdk import InferenceHTTPClient

# Config Tampilan
st.set_page_config(page_title="Klasifikasi Ular Real-time", layout="centered")
st.title("🐍 Klasifikasi Spesies Ular Real-Time")
st.write("Arahkan kamera ke ular untuk mengidentifikasi jenisnya secara langsung.")

# Inisialisasi Roboflow Client dengan Serverless URL
@st.cache_resource
def get_inference_client():
    return InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key="K7ce7ZVzoLF5O0URWUWF"
    )

client = get_inference_client()

# Class Processor untuk mengolah video & menampilkan Teks Nama Ular
class VideoProcessor:
    def __init__(self):
        self.frame_count = 0
        self.last_label = ""

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # Lakukan deteksi setiap 3 frame sekali agar video tetap mulus 30 FPS
        if self.frame_count % 3 == 0:
            try:
                # Resize gambar menjadi 416x416 agar pengiriman API cepat
                resized_img = cv2.resize(img, (416, 416))
                
                # Mengirim request ke Model ID Klasifikasi Baru
                results = client.infer(
                    resized_img, 
                    model_id="snake-7obzc/2"
                )
                
                top_class = ""
                confidence = 0

                # Pembacaan struktur output klasifikasi Roboflow
                if isinstance(results, dict):
                    top_class = results.get("top", "")
                    confidence = results.get("confidence", 0)

                    # Jika format respons berupa dictionary "predictions"
                    if not top_class and "predictions" in results:
                        preds = results["predictions"]
                        if isinstance(preds, dict) and len(preds) > 0:
                            top_class = max(preds, key=lambda k: preds[k].get("confidence", 0))
                            confidence = preds[top_class].get("confidence", 0)
                        elif isinstance(preds, list) and len(preds) > 0:
                            top_class = preds[0].get("class", preds[0].get("label", ""))
                            confidence = preds[0].get("confidence", 0)

                # Menampilkan label jika nilai confidence lebih dari 30%
                if top_class and confidence > 0.30:
                    conf_percent = int(confidence * 100) if confidence <= 1.0 else int(confidence)
                    self.last_label = f"{top_class} ({conf_percent}%)"
                else:
                    self.last_label = "Mencari ular..."
            except Exception:
                pass

        # Gambarkan Banner Teks Hasil Klasifikasi di Atas Video
        if self.last_label:
            # Latar belakang hitam untuk teks
            cv2.rectangle(img, (10, 10), (480, 60), (0, 0, 0), -1)
            # Teks nama ular berwarna hijau terang
            cv2.putText(
                img, 
                self.last_label, 
                (20, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (0, 255, 0), 2, cv2.LINE_AA
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# Konfigurasi WebRTC STUN Server (Koneksi Stabil untuk HP & Data Seluler)
RTC_CONFIG = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun3.l.google.com:19302"]},
            {"urls": ["stun:global.stun.twilio.com:3478"]},
        ]
    }
)

# Widget Kamera WebRTC
webrtc_streamer(
    key="snake-classification",
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
