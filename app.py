"""
Aplikasi Streamlit - Segmentasi Nasabah Kartu Kredit (K-Means Clustering)
Tugas Mandiri Pertemuan 4 - Implementasi Clustering dengan Metodologi CRISP-DM
"""

import os

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.decomposition import PCA

# ----------------------------------------------------------------------------
# Konfigurasi halaman
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Segmentasi Nasabah Kartu Kredit",
    page_icon="💳",
    layout="wide",
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

DESKRIPSI_SEGMEN = {
    "Nasabah Pasif (Low Engagement)": (
        "Saldo rendah, jarang bertransaksi, dan penggunaan cash advance moderat. "
        "Rekomendasi: kampanye reaktivasi (cashback/diskon transaksi pertama)."
    ),
    "VIP / Big Spender": (
        "Saldo dan nilai pembelian sangat tinggi, limit kredit besar, rasio pembayaran "
        "penuh cukup baik. Rekomendasi: program loyalti eksklusif & penawaran kenaikan limit."
    ),
    "Revolver / Pengguna Cash Advance": (
        "Saldo tinggi namun didominasi penarikan tunai (cash advance) dengan persentase "
        "pembayaran penuh sangat rendah. Segmen berisiko kredit tinggi — perlu monitoring "
        "risiko dan edukasi keuangan."
    ),
    "Transactor / Pengguna Aktif Bertanggung Jawab": (
        "Frekuensi pembelian tinggi, cash advance minim, disiplin membayar penuh. "
        "Nasabah paling sehat secara finansial — cocok ditawari program reward/poin belanja."
    ),
}


# ----------------------------------------------------------------------------
# Memuat model & metadata (di-cache agar tidak reload berulang kali)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    cluster_names = joblib.load(os.path.join(MODEL_DIR, "cluster_names.pkl"))
    cluster_profile = pd.read_csv(
        os.path.join(MODEL_DIR, "cluster_profile.csv"), index_col=0
    )
    return model, scaler, feature_columns, cluster_names, cluster_profile


model, scaler, feature_columns, cluster_names, cluster_profile = load_artifacts()

FEATURE_LABELS = {
    "BALANCE": "Saldo (Balance)",
    "BALANCE_FREQUENCY": "Frekuensi Update Saldo (0-1)",
    "PURCHASES": "Total Pembelian",
    "ONEOFF_PURCHASES": "Pembelian One-off Terbesar",
    "INSTALLMENTS_PURCHASES": "Pembelian Cicilan",
    "CASH_ADVANCE": "Cash Advance (Tarik Tunai)",
    "PURCHASES_FREQUENCY": "Frekuensi Pembelian (0-1)",
    "ONEOFF_PURCHASES_FREQUENCY": "Frekuensi Pembelian One-off (0-1)",
    "PURCHASES_INSTALLMENTS_FREQUENCY": "Frekuensi Pembelian Cicilan (0-1)",
    "CASH_ADVANCE_FREQUENCY": "Frekuensi Cash Advance (0-1)",
    "CASH_ADVANCE_TRX": "Jumlah Transaksi Cash Advance",
    "PURCHASES_TRX": "Jumlah Transaksi Pembelian",
    "CREDIT_LIMIT": "Limit Kartu Kredit",
    "PAYMENTS": "Total Pembayaran",
    "MINIMUM_PAYMENTS": "Pembayaran Minimum",
    "PRC_FULL_PAYMENT": "Persentase Pembayaran Penuh (0-1)",
    "TENURE": "Lama Menjadi Nasabah (bulan)",
}

FREQ_COLUMNS = [c for c in feature_columns if "FREQUENCY" in c or c == "PRC_FULL_PAYMENT"]


def predict_cluster(df_input: pd.DataFrame):
    """Menerima dataframe dengan kolom sesuai feature_columns, kembalikan label cluster & nama segmen."""
    df_ordered = df_input[feature_columns]
    scaled = scaler.transform(df_ordered)
    labels = model.predict(scaled)
    names = [cluster_names[label] for label in labels]
    return labels, names, scaled


def plot_pca(scaled_input: np.ndarray, label_input: int):
    """Menampilkan posisi nasabah baru relatif terhadap keempat cluster (dummy centroid-based visualization)."""
    centers = model.cluster_centers_
    pca = PCA(n_components=2, random_state=42)
    all_points = np.vstack([centers, scaled_input])
    pca_result = pca.fit_transform(all_points)

    center_2d = pca_result[: len(centers)]
    input_2d = pca_result[len(centers):]

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(centers)))
    for i, (x, y) in enumerate(center_2d):
        ax.scatter(x, y, s=300, marker="X", color=colors[i], edgecolor="black",
                   label=f"Pusat Cluster {i}: {cluster_names[i]}")
    ax.scatter(input_2d[:, 0], input_2d[:, 1], s=180, marker="*", color="red",
               edgecolor="black", label="Nasabah (input)", zorder=5)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("Posisi Nasabah Relatif terhadap Pusat Cluster")
    ax.legend(fontsize=7, loc="best")
    return fig


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("💳 Segmentasi Nasabah")
st.sidebar.markdown(
    "Aplikasi ini mengelompokkan nasabah kartu kredit ke dalam 4 segmen "
    "menggunakan model **K-Means Clustering** yang telah dilatih pada "
    "dataset [Credit Card Dataset for Clustering (Kaggle)]"
    "(https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)."
)
mode = st.sidebar.radio(
    "Pilih mode input:",
    ["Input Manual (1 Nasabah)", "Upload File CSV (Banyak Nasabah)"],
)

with st.sidebar.expander("ℹ️ Tentang Segmen"):
    for nama, desk in DESKRIPSI_SEGMEN.items():
        st.markdown(f"**{nama}**\n\n{desk}\n")

st.title("Segmentasi Nasabah Kartu Kredit dengan K-Means Clustering")
st.caption("Studi kasus CRISP-DM — dari Business Understanding hingga Deployment")

tab_predict, tab_profile = st.tabs(["🔎 Prediksi Segmen", "📊 Profil Cluster (Model)"])

# ----------------------------------------------------------------------------
# TAB 1: PREDIKSI
# ----------------------------------------------------------------------------
with tab_predict:
    if mode == "Input Manual (1 Nasabah)":
        st.subheader("Masukkan Data Perilaku Nasabah")
        st.write(
            "Isi nilai di bawah ini untuk memprediksi segmen seorang nasabah. "
            "Nilai default diambil dari rata-rata keseluruhan data."
        )

        col1, col2, col3 = st.columns(3)
        input_values = {}
        cols_cycle = [col1, col2, col3]

        for i, feat in enumerate(feature_columns):
            target_col = cols_cycle[i % 3]
            label = FEATURE_LABELS.get(feat, feat)
            if feat in FREQ_COLUMNS:
                input_values[feat] = target_col.slider(label, 0.0, 1.0, 0.3, 0.01, key=feat)
            elif feat == "TENURE":
                input_values[feat] = target_col.slider(label, 1, 12, 12, 1, key=feat)
            elif feat in ("CASH_ADVANCE_TRX", "PURCHASES_TRX"):
                input_values[feat] = target_col.number_input(label, min_value=0, value=5, step=1, key=feat)
            else:
                input_values[feat] = target_col.number_input(
                    label, min_value=0.0, value=500.0, step=50.0, key=feat
                )

        if st.button("🔍 Prediksi Segmen Nasabah", type="primary"):
            df_input = pd.DataFrame([input_values])
            labels, names, scaled = predict_cluster(df_input)

            st.success(f"Nasabah ini termasuk dalam segmen: **{names[0]}**")
            st.info(DESKRIPSI_SEGMEN[names[0]])

            fig = plot_pca(scaled, labels[0])
            st.pyplot(fig)

    else:
        st.subheader("Upload File CSV Nasabah")
        st.write(
            "File CSV harus memiliki kolom berikut (urutan bebas): "
            + ", ".join(feature_columns)
        )
        contoh = st.checkbox("Gunakan contoh dataset bawaan untuk mencoba aplikasi")

        uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

        df_batch = None
        if contoh:
            sample_path = os.path.join(os.path.dirname(__file__), "sample_data", "CC_GENERAL.csv")
            df_batch = pd.read_csv(sample_path)
        elif uploaded_file is not None:
            df_batch = pd.read_csv(uploaded_file)

        if df_batch is not None:
            missing_cols = [c for c in feature_columns if c not in df_batch.columns]
            if missing_cols:
                st.error(f"Kolom berikut tidak ditemukan di file: {missing_cols}")
            else:
                df_clean = df_batch.copy()
                for c in feature_columns:
                    if df_clean[c].isna().any():
                        df_clean[c] = df_clean[c].fillna(df_clean[c].median())

                labels, names, scaled = predict_cluster(df_clean)
                df_batch["CLUSTER"] = labels
                df_batch["SEGMEN"] = names

                st.write(f"Total nasabah diproses: **{len(df_batch)}**")
                st.dataframe(df_batch.head(20))

                st.write("Distribusi segmen:")
                dist = df_batch["SEGMEN"].value_counts()
                fig, ax = plt.subplots(figsize=(7, 4))
                dist.plot(kind="bar", color="teal", ax=ax)
                ax.set_ylabel("Jumlah Nasabah")
                ax.set_xlabel("Segmen")
                plt.xticks(rotation=20, ha="right")
                st.pyplot(fig)

                csv_hasil = df_batch.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Unduh Hasil Segmentasi (CSV)",
                    data=csv_hasil,
                    file_name="hasil_segmentasi_nasabah.csv",
                    mime="text/csv",
                )

# ----------------------------------------------------------------------------
# TAB 2: PROFIL CLUSTER
# ----------------------------------------------------------------------------
with tab_profile:
    st.subheader("Profil Rata-Rata Tiap Cluster (Hasil Training Model)")
    profile_display = cluster_profile.copy()
    profile_display.index = [f"Cluster {i} — {cluster_names[i]}" for i in profile_display.index]
    st.dataframe(profile_display)

    st.subheader("Jumlah Nasabah per Segmen (Data Training)")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(profile_display.index, cluster_profile["JUMLAH_NASABAH"], color="darkcyan")
    plt.xticks(rotation=20, ha="right")
    ax.set_ylabel("Jumlah Nasabah")
    st.pyplot(fig)

    st.markdown("---")
    st.caption(
        "Model: K-Means (k=4) · Dataset: Credit Card Dataset for Clustering (Kaggle, arjunbhasin2013/ccdata) "
        "· Dibangun mengikuti metodologi CRISP-DM."
    )
