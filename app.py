import streamlit as st
import pandas as pd
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2
from inference_sdk import InferenceHTTPClient
import supervision as sv

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Sistem Pengurusan & Deteksi Ular",
    page_icon="🐍",
    layout="wide"
)

# ==========================================
# 2. SIDEBAR NAVIGATION (PILIHAN ROLE)
# ==========================================
st.sidebar.title("🐍 Portal Pengurusan Ular")
st.sidebar.write("Sistem Integrasi Deteksi & Tindakan Darurat")

role = st.sidebar.radio(
    "Pilih Akses Moda:",
    [
        "📱 User (Deteksi & Kecemasan)", 
        "🏥 Medical Care (Antivenom & AI)", 
        "🧑‍🚒 Firefighter (Zon Pelepasan Penang)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Status Sistem: **Mode Demo (Prototype v1.0)**")

# ==========================================
# 3. ROLE 1: USER (Pengguna Awam)
# ==========================================
if role == "📱 User (Deteksi & Kecemasan)":
    st.title("🐍 Deteksi Ular Real-Time")
    st.write("Arahkan kamera ke objek untuk mendeteksi secara langsung.")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📷 Kamera Deteksi Live")
        
        # Initialisasi Roboflow Client
        @st.cache_resource
        def get_inference_client():
            return InferenceHTTPClient(
                api_url="https://detect.roboflow.com",
                api_key="K7ce7ZVzoLF5O0URWUWF"
            )

        client = get_inference_client()
        bounding_box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        class VideoProcessor:
            def __init__(self):
                self.frame_count = 0
                self.last_detections = None

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")
                self.frame_count += 1

                if self.frame_count % 3 == 0:
                    try:
                        resized_img = cv2.resize(img, (416, 416))
                        results = client.infer(resized_img, model_id="gamals-workspace-j10lj/ularlah-2-yolo11s-t1")
                        detections = sv.Detections.from_inference(results)
                        
                        h_orig, w_orig = img.shape[:2]
                        scale_x = w_orig / 416.0
                        scale_y = h_orig / 416.0
                        
                        if len(detections) > 0:
                            detections.xyxy[:, [0, 2]] *= scale_x
                            detections.xyxy[:, [1, 3]] *= scale_y
                        
                        self.last_detections = detections
                    except Exception:
                        pass

                if self.last_detections is not None:
                    img = bounding_box_annotator.annotate(scene=img, detections=self.last_detections)
                    img = label_annotator.annotate(scene=img, detections=self.last_detections)

                return av.VideoFrame.from_ndarray(img, format="bgr24")

        RTC_CONFIG = RTCConfiguration({
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
            ]
        })

        webrtc_streamer(
            key="snake-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
            async_processing=True,
        )

    with col2:
        st.subheader("🚨 Kontak Bantuan Darurat")
        st.error("**Talian Kecemasan Utama:** 999")
        
        st.markdown("### 🚒 Balai Bomba Penang Terdekat")
        st.info("""
        * **JBPM HQ Penang:** 04-261 4444
        * **BBP George Town:** 04-224 4444
        * **BBP Bayan Lepas:** 04-643 4444
        * **BBP Butterworth:** 04-331 4444
        * **BBP Seri Balik Pulau:** 04-866 4444
        """)
        
        st.warning("""
        **⚠️ Pertolongan Cemas Gigitan Ular:**
        1. Bertenang dan kurangkan pergerakan mangsa.
        2. Jangan diikat/ditoreh/disedut tempat gigitan.
        3. Segera bawa ke Hospital Pulau Pinang / Hospital terdekat.
        """)

# ==========================================
# 4. ROLE 2: MEDICAL CARE (Tenaga Medis)
# ==========================================
elif role == "🏥 Medical Care (Antivenom & AI)":
    st.title("🏥 Panduan Perubatan & Semakan Antivenom")
    st.caption("Pusat rujukan cepat untuk jenis racun (venom), antidote, dan perkhidmatan AI.")

    tab1, tab2 = st.tabs(["📋 Senarai Antivenom & Spesies", "🤖 AI Medical Assistant"])

    # Data Dummy Ular & Antivenom
    data_antivenom = {
        "Nama Spesies / Ular": [
            "Naja kaouthia (Ular Senduk)", 
            "Ophiophagus hannah (Tedung Selar)", 
            "Trimeresurus purpureomaculatus (Kapak Bakau)", 
            "Bungarus candidus (Katam Tebu)", 
            "Malayopython reticulatus (Sawa Batik)"
        ],
        "Jenis Bisa (Venom)": [
            "Neurotoksin (Tinggi)", 
            "Neurotoksin (Sangat Tinggi)", 
            "Hemotoksin (Sederhana)", 
            "Neurotoksin (Tinggi)", 
            "Tidak Berbisa"
        ],
        "Antidote / Serum Sesuai": [
            "Monovalent Cobra Antivenom / SABU", 
            "King Cobra Antivenom", 
            "Green Pit Viper Antivenom", 
            "Malayan Krait Antivenom", 
            "Tiada (Rawatan Luka Sahaja)"
        ],
        "Stok RS Pulau Pinang": [
            "Tersedia (12 Vial)", 
            "Tersedia (5 Vial)", 
            "Tersedia (8 Vial)", 
            "Terhad (2 Vial)", 
            "N/A"
        ]
    }
    df_antivenom = pd.DataFrame(data_antivenom)

    with tab1:
        st.subheader("📊 Pangkalan Data Antivenom")
        st.dataframe(df_antivenom, use_container_width=True)

    with tab2:
        st.subheader("🤖 AI Antidote Assistant")
        st.write("Tanyakan jenis ular atau gejala untuk mendapatkan cadangan antidote secara cepat.")
        
        selected_snake = st.selectbox(
            "Pilih atau taip jenis ular hasil deteksi:",
            ["- Pilih Ular -"] + list(df_antivenom["Nama Spesies / Ular"])
        )
        
        user_query = st.text_input("Atau masukkan soalan anda (Contoh: 'Ular Kapak perlukan antivenom apa?')")

        if st.button("Cari Antidote / Tanya AI"):
            if selected_snake != "- Pilih Ular -":
                row = df_antivenom[df_antivenom["Nama Spesies / Ular"] == selected_snake].iloc[0]
                st.success(f"**Hasil AI untuk {selected_snake}:**")
                st.markdown(f"""
                * **Jenis Bisa:** {row['Jenis Bisa (Venom)']}
                * **Cadangan Antidote:** `{row['Antidote / Serum Sesuai']}`
                * **Status Stok Hospital Pulau Pinang:** {row['Stok RS Pulau Pinang']}
                """)
            elif user_query:
                st.info(f"**Simulasi Respons AI untuk:** '{user_query}'")
                st.write("🤖 *Berdasarkan analisis pantas: Untuk gigitan jenis Ular Kapak (Pit Viper), antidote yang sesuai adalah **Green Pit Viper Antivenom**. Pastikan pemantauan darah (coagulation profile) dilakukan segera.*")
            else:
                st.warning("Sila pilih jenis ular atau masukkan soalan terlebih dahulu.")

# ==========================================
# 5. ROLE 3: FIREFIGHTER (Abam Bomba)
# ==========================================
elif role == "🧑‍🚒 Firefighter (Zon Pelepasan Penang)":
    st.title("🧑‍🚒 Panduan Pelepasan Ular (Kawasan Penang)")
    st.caption("Lokasi zon pelepasan habitat semula jadi terdekat untuk operasi penangkapan di Penang.")

    # Data Lokasi Pelepasan Pulau Pinang
    release_sites = pd.DataFrame({
        'Nama Lokasi': [
            'Taman Negara Pulau Pinang (Teluk Bahang)', 
            'Hutan Simpan Bukit Mertajam (Cherok Tokun)', 
            'Hutan Lipur Bukit Panchor (Nibong Tebal)',
            'Kawasan Hutan Simpan Pantai Acheh'
        ],
        'Sesuai Untuk Spesies': [
            'Ular Sawa, Ular Tedung, Ular Pucuk',
            'Ular Kapak, Ular Katam, Ular Sawa',
            'Ular Sawa, Ular Air, Ular Tedung',
            'Ular Berbisa Tinggi & Ular Darat'
        ],
        'Radius dari Bandar': ['25 km dari George Town', '12 km dari Butterworth', '20 km dari Seberang Jaya', '30 km dari Bayan Lepas'],
        'lat': [5.4600, 5.3621, 5.1581, 5.4200],
        'lon': [100.1983, 100.4908, 100.4870, 100.1800]
    })

    col_map, col_list = st.columns([2, 1])

    with col_map:
        st.subheader("🗺️ Peta Zon Pelepasan (Penang)")
        st.map(release_sites[['lat', 'lon']], zoom=10)

    with col_list:
        st.subheader("📍 Senarai Lokasi Terdekat")
        for idx, row in release_sites.iterrows():
            with st.expander(f"🌲 {row['Nama Lokasi']}"):
                st.write(f"**Spesies Sesuai:** {row['Sesuai Untuk Spesies']}")
                st.write(f"**Jarak:** {row['Radius dari Bandar']}")
                st.button(f"Buka Navigasi (Google Maps)", key=f"btn_{idx}")

    st.markdown("---")
    st.subheader("📝 Prosedur Operasi Standard (SOP) Pelepasan")
    st.markdown("""
    1. Pastikan lokasi pelepasan sekurang-kurangnya **5 km dari kawasan pemukiman penduduk**.
    2. Lepaskan ular berbisa pada waktu yang sesuai dengan sifat spesiesnya (contoh: *Nocturnal* pada waktu malam).
    3. Catat jenis spesies dan jumlah ular yang dilepaskan ke dalam log Jabatan.
    """)
