import streamlit as st
import pandas as pd
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2
from inference_sdk import InferenceHTTPClient
import supervision as sv

# ==========================================
# 1. KONFIGURASI HALAMAN & STATE
# ==========================================
st.set_page_config(
    page_title="Sistem Pengurusan & Deteksi Ular",
    page_icon="🐍",
    layout="wide"
)

# Inisialisasi Session State untuk Navigasi Halaman
if 'role' not in st.session_state:
    st.session_state['role'] = None

def reset_role():
    st.session_state['role'] = None

# ==========================================
# 2. HALAMAN UTAMA (LANDING PAGE - 3 OPSI)
# ==========================================
if st.session_state['role'] is None:
    st.title("🐍 Portal Integrasi Deteksi & Pengurusan Ular")
    st.write("Sila pilih modul akses mengikut peranan anda untuk memulakan sistem:")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📱 Mode User / Awam")
        st.info(
            "Akses fungsi kamera deteksi ular secara *real-time*, panduan pertolongan cemas gigitan, "
            "dan pautan nombor kecemasan terdekat."
        )
        if st.button("🚀 Masuk Mode User", use_container_width=True):
            st.session_state['role'] = 'user'
            st.rerun()

    with col2:
        st.markdown("### 🏥 Mode Medical Care")
        st.success(
            "Jadual lengkap racun (venom), senarai antidote/SABU untuk pelbagai jenis ular, "
            "semakan stok di Hospital Penang, dan AI Assistant."
        )
        if st.button("🚀 Masuk Mode Medical Care", use_container_width=True):
            st.session_state['role'] = 'medical'
            st.rerun()

    with col3:
        st.markdown("### 🧑‍🚒 Mode Firefighter")
        st.warning(
            "Peta zon pelepasan ular khas di Pulau Pinang, semakan jarak hutan simpan terdekat, "
            "padanan jenis habitat, dan SOP penangkapan."
        )
        if st.button("🚀 Masuk Mode Firefighter", use_container_width=True):
            st.session_state['role'] = 'firefighter'
            st.rerun()

    st.markdown("---")
    st.caption("📌 *Sistem Prototaip Demo v2.0 - Dikuasakan oleh Streamlit & Roboflow AI*")

# ==========================================
# 3. INTERFACE OPSI 1: USER (Pengguna Awam)
# ==========================================
elif st.session_state['role'] == 'user':
    st.sidebar.button("⬅️ Kembali ke Menu Utama", on_click=reset_role, use_container_width=True)
    st.sidebar.markdown("---")
    st.sidebar.write("📌 **Status:** Mode User Aktif")

    st.title("🐍 Deteksi Ular Real-Time (Mode User)")
    st.write("Arahkan kamera ke objek untuk mendeteksi secara langsung.")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📷 Kamera Deteksi Live")
        
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
        st.subheader("🚨 Bantuan Kecemasan")
        st.error("**Talian Kecemasan Utama:** 999")
        
        st.markdown("### 🚒 Balai Bomba Penang Terdekat")
        st.info("""
        * **JBPM HQ Penang:** 04-261 4444
        * **BBP George Town:** 04-224 4444
        * **BBP Bayan Lepas:** 04-643 4444
        * **BBP Butterworth:** 04-331 4444
        """)
        
        st.warning("""
        **⚠️ Langkah Pertolongan Cemas:**
        1. Bertenang dan perlahankan pergerakan mangsa.
        2. Bawa terus ke Hospital Pulau Pinang / Seberang Jaya.
        3. JANGAN diikat ketat, dipotong, atau disedut.
        """)

# ==========================================
# 4. INTERFACE OPSI 2: MEDICAL CARE
# ==========================================
elif st.session_state['role'] == 'medical':
    st.sidebar.button("⬅️ Kembali ke Menu Utama", on_click=reset_role, use_container_width=True)
    st.sidebar.markdown("---")
    st.sidebar.write("📌 **Status:** Mode Medical Care Aktif")

    st.title("🏥 Portal Perubatan & Pengurusan Antidote / Antivenom")
    st.caption("Pusat Rujukan Klinikal Racun Ular untuk Hospital dan Petugas Kesihatan")

    # Data Lengkap Antivenom & Spesies
    medical_db = [
        {
            "Spesies Ular": "Naja kaouthia (Ular Senduk Monocled)",
            "Jenis Kategori": "Berbisa Tinggi",
            "Jenis Venom": "Neurotoksin (Lumpuh Otot/Pernafasan)",
            "Antidote / Serum Sesuai": "Monovalent Cobra Antivenom / SABU",
            "Dos Permulaan": "10 Vial (IV Infusion)",
            "Stok Hospital Penang": "Hospital P.Pinang (15 Vial), Hospital Seberang Jaya (8 Vial)"
        },
        {
            "Spesies Ular": "Ophiophagus hannah (Ular Tedung Selar)",
            "Jenis Kategori": "Berbisa Sangat Tinggi",
            "Jenis Venom": "Neurotoksin & Cardiotoksin",
            "Antidote / Serum Sesuai": "King Cobra Antivenom",
            "Dos Permulaan": "10 - 15 Vial (IV Infusion)",
            "Stok Hospital Penang": "Hospital P.Pinang (6 Vial)"
        },
        {
            "Spesies Ular": "Trimeresurus purpureomaculatus (Ular Kapak Bakau)",
            "Jenis Kategori": "Berbisa Tinggi",
            "Jenis Venom": "Hemotoksin (Pendarahan / Pendarahan Luar/Dalam)",
            "Antidote / Serum Sesuai": "Green Pit Viper Antivenom",
            "Dos Permulaan": "5 Vial (IV Infusion)",
            "Stok Hospital Penang": "Hospital Seberang Jaya (10 Vial), Hospital Kepala Batas (5 Vial)"
        },
        {
            "Spesies Ular": "Calloselasma rhodostoma (Ular Kapak Bodoh / Malayan Pit Viper)",
            "Jenis Kategori": "Berbisa Tinggi",
            "Jenis Venom": "Hemotoksin (Nekrosis Tisu & Pendarahan Severe)",
            "Antidote / Serum Sesuai": "Malayan Pit Viper Antivenom",
            "Dos Permulaan": "5 Vial (IV Infusion)",
            "Stok Hospital Penang": "Hospital Kepala Batas (8 Vial), Hospital P.Pinang (5 Vial)"
        },
        {
            "Spesies Ular": "Bungarus candidus (Ular Katam Tebu / Malayan Krait)",
            "Jenis Kategori": "Berbisa Tinggi",
            "Jenis Venom": "Neurotoksin Pre-synaptic (Sangat Bahaya)",
            "Antidote / Serum Sesuai": "Malayan Krait Antivenom / Polyvalent Elapid",
            "Dos Permulaan": "10 Vial (IV Infusion)",
            "Stok Hospital Penang": "Hospital P.Pinang (4 Vial)"
        },
        {
            "Spesies Ular": "Malayopython reticulatus (Ular Sawa Batik)",
            "Jenis Kategori": "Tidak Berbisa",
            "Jenis Venom": "Tiada Racun (Bahaya Gigitan / Jangkitan Bakteria)",
            "Antidote / Serum Sesuai": "Tiada Antivenom (Rawatan Tetanus & Antiseptik)",
            "Dos Permulaan": "Pembersihan Luka + Tetanus Toxoid",
            "Stok Hospital Penang": "Tersedia di Semua Hospital"
        },
        {
            "Spesies Ular": "Ahaetulla prasina (Ular Pucuk)",
            "Jenis Kategori": "Berbisa Ringan",
            "Jenis Venom": "Mild Neurotoksin / Cytotoksin",
            "Antidote / Serum Sesuai": "Tiada Antidote Khusus (Rawatan Simptomatik)",
            "Dos Permulaan": "Pemerhatian Klinal 4-6 Jam",
            "Stok Hospital Penang": "Tersedia di Semua Hospital"
        }
    ]
    df_med = pd.DataFrame(medical_db)

    tab_table, tab_ai = st.tabs(["📋 Jadual Antidote & Stok Hospital", "🤖 AI Medical Consultant"])

    with tab_table:
        st.subheader("🔍 Carian & Database Antivenom")
        search_query = st.text_input("Cari spesies ular atau jenis racun (contoh: 'Neurotoksin' atau 'Kapak'):")

        if search_query:
            filtered_df = df_med[df_med.apply(lambda row: row.astype(str).str.contains(search_query, case=False).any(), axis=1)]
            st.dataframe(filtered_df, use_container_width=True)
        else:
            st.dataframe(df_med, use_container_width=True)

        st.info("💡 **Nota Kesihatan:** Pemberian antivenom mesti dipantau rapi untuk mengelakkan reaksi anafilaksis (alahan teruk).")

    with tab_ai:
        st.subheader("🤖 AI Medical Consultant (Simulasi)")
        st.write("Gunakan pembantu AI ini untuk mendapatkan panduan persediaan rawatan mengikut simptom atau spesifikasi ular.")

        selected_snake_ai = st.selectbox("Pilih Spesies Ular yang Dikesan / Disyaki:", ["- Pilih Spesies -"] + list(df_med["Spesies Ular"]))
        symptoms = st.multiselect("Pilih Simptom Mangsa saat ini:", ["Pendarahan tidak berhenti", "Kesukaran bernafas / Kelopak mata layu", "Bengkak teruk / Tisu menghitam", "Sakit biasa di kawasan gigitan"])

        if st.button("Dapatkan Cadangan Rawatan AI"):
            if selected_snake_ai != "- Pilih Spesies -":
                detail = df_med[df_med["Spesies Ular"] == selected_snake_ai].iloc[0]
                st.success(f"### 📋 Panduan AI untuk {selected_snake_ai}")
                st.markdown(f"""
                * **Status Bisa:** `{detail['Jenis Kategori']}`
                * **Jenis Venom Utama:** {detail['Jenis Venom']}
                * **Cadangan Antidote Sesuai:** **{detail['Antidote / Serum Sesuai']}**
                * **Dos Recommended:** {detail['Dos Permulaan']}
                * **Lokasi Stok Terdekat (Penang):** {detail['Stok Hospital Penang']}
                """)
            else:
                st.warning("Sila pilih spesies ular terlebih dahulu.")

# ==========================================
# 5. INTERFACE OPSI 3: FIREFIGHTER
# ==========================================
elif st.session_state['role'] == 'firefighter':
    st.sidebar.button("⬅️ Kembali ke Menu Utama", on_click=reset_role, use_container_width=True)
    st.sidebar.markdown("---")
    st.sidebar.write("📌 **Status:** Mode Firefighter Aktif")

    st.title("🧑‍🚒 Panduan Operasi & Pelepasan Ular (Kawasan Penang)")
    st.caption("Modul Khusus Bomba / APM untuk Lokasi Pelepasan Habitat Selamat di Pulau Pinang")

    # Data Lokasi Hutan Simpan Penang
    penang_sites = pd.DataFrame([
        {
            "Nama Lokasi": "Taman Negara Pulau Pinang (Teluk Bahang)",
            "Kesesuaian Ular": "Ular Sawa Batik, Ular Tedung, Ular Pucuk",
            "Zon Bahaya": "Sangat Jauh dari Pemukiman (Sesuai)",
            "Jarak dari Georgetown": "22 km",
            "lat": 5.4600,
            "lon": 100.1983
        },
        {
            "Nama Lokasi": "Hutan Simpan Cherok Tokun (Bukit Mertajam)",
            "Kesesuaian Ular": "Ular Kapak, Ular Katam, Ular Sawa",
            "Zon Bahaya": "Hutan Tebal Seberang Perai",
            "Jarak dari Georgetown": "28 km",
            "lat": 5.3621,
            "lon": 100.4908
        },
        {
            "Nama Lokasi": "Hutan Lipur Bukit Panchor (Nibong Tebal)",
            "Kesesuaian Ular": "Ular Sawa, Ular Air, Ular Berbisa Sedang",
            "Zon Bahaya": "Kawasan Selatan Penang",
            "Jarak dari Georgetown": "45 km",
            "lat": 5.1581,
            "lon": 100.4870
        },
        {
            "Nama Lokasi": "Hutan Simpan Pantai Acheh (Balik Pulau)",
            "Kesesuaian Ular": "Ular Berbisa Tinggi (Tedung Selar / Kapak Bakau)",
            "Zon Bahaya": "Zon Konservasi Terpencil",
            "Jarak dari Georgetown": "30 km",
            "lat": 5.4200,
            "lon": 100.1800
        }
    ])

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("🗺️ Peta Interaktif Zon Pelepasan (Penang)")
        st.map(penang_sites[['lat', 'lon']], zoom=10)

    with col_right:
        st.subheader("📍 Senarai Lokasi Hutan Simpan")
        for idx, site in penang_sites.iterrows():
            with st.expander(f"🌲 {site['Nama Lokasi']}"):
                st.write(f"**Sesuai Untuk:** {site['Kesesuaian Ular']}")
                st.write(f"**Status Zon:** {site['Zon Bahaya']}")
                st.write(f"**Jarak:** {site['Jarak dari Georgetown']}")
                st.button(f"🧭 Buka Navigasi Google Maps", key=f"map_btn_{idx}")

    st.markdown("---")
    st.subheader("📋 Senarai Semak SOP Pelepasan Satwa (Bomba Penang)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.checkbox("1. Pengesahan Jenis Ular")
    with c2:
        st.checkbox("2. Radius Minima 5 km dari Rumah")
    with c3:
        st.checkbox("3. Rekod Log Operasi JBPM")
