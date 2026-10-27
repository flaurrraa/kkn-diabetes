import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Prediksi Risiko Diabetes (KNN)",
    page_icon="🩺",
    layout="wide",
)

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .hero {background: linear-gradient(135deg,#0f766e 0%,#0e7490 60%,#1d4ed8 100%);
           padding: 28px 32px; border-radius: 18px; color: white; margin-bottom: 18px;}
    .hero h1 {margin:0; font-size: 2rem;}
    .hero p {margin:6px 0 0 0; opacity:.9;}
    .result-high {background:#fef2f2; border:2px solid #ef4444; color:#991b1b;
                  padding:22px; border-radius:16px; text-align:center;}
    .result-low {background:#f0fdf4; border:2px solid #22c55e; color:#166534;
                 padding:22px; border-radius:16px; text-align:center;}
    .result-high h2, .result-low h2 {margin:0; font-size:1.8rem;}
    .result-high p, .result-low p {margin:6px 0 0 0;}
    div[data-testid="stMetric"] {background:#f8fafc; border:1px solid #e2e8f0;
                                 padding:12px 16px; border-radius:12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Data ----------
FITUR = ["Usia", "BMI", "Glukosa", "Tekanan Darah", "Riwayat Keluarga"]

DATA = pd.DataFrame(
    {
        "Pasien": [f"P{i:02d}" for i in range(1, 11)],
        "Usia": [25, 30, 35, 40, 45, 50, 55, 28, 38, 48],
        "BMI": [21.5, 23.1, 25.0, 27.5, 30.2, 31.5, 33.0, 22.5, 26.8, 29.5],
        "Glukosa": [90, 95, 105, 120, 140, 155, 165, 92, 115, 145],
        "Tekanan Darah": [110, 115, 120, 125, 135, 140, 145, 112, 122, 130],
        "Riwayat Keluarga": [0, 0, 0, 1, 1, 1, 1, 0, 0, 1],
        "Kelas": [0, 0, 0, 1, 1, 1, 1, 0, 0, 1],
    }
)
LABEL = {0: "Risiko Rendah", 1: "Risiko Tinggi"}
WARNA = {"Risiko Rendah": "#22c55e", "Risiko Tinggi": "#ef4444"}
DATA["Label"] = DATA["Kelas"].map(LABEL)

# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
      <h1>🩺 Prediksi Risiko Diabetes</h1>
      <p>Klasifikasi risiko pasien menggunakan algoritma K-Nearest Neighbors (KNN)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Pengaturan Model")
    k = st.slider("Nilai K (jumlah tetangga)", 1, 9, 3, step=2,
                  help="Gunakan K ganjil agar tidak terjadi seri suara.")
    normalisasi = st.toggle("Normalisasi Min-Max", value=True,
                            help="Menyamakan skala fitur agar glukosa tidak mendominasi jarak.")
    metrik = st.radio("Metrik jarak", ["Euclidean", "Manhattan"], horizontal=True)
    berbobot = st.toggle("Voting berbobot (1/jarak)", value=False)
    st.divider()
    st.caption("Dataset: 10 pasien klinik · 5 fitur · 2 kelas")

# ---------- Input pasien baru ----------
st.subheader("👤 Data Pasien Baru")
c1, c2, c3, c4, c5 = st.columns(5)
usia = c1.number_input("Usia (tahun)", 1, 120, 42)
bmi = c2.number_input("BMI", 10.0, 60.0, 28.5, step=0.1)
glukosa = c3.number_input("Glukosa (mg/dL)", 40, 400, 130)
td = c4.number_input("Tekanan Darah (mmHg)", 60, 220, 128)
rk = c5.selectbox("Riwayat Keluarga", [1, 0], format_func=lambda x: "Ada" if x else "Tidak ada")

baru = np.array([usia, bmi, glukosa, td, rk], dtype=float)

# ---------- KNN ----------
X = DATA[FITUR].values.astype(float)
y = DATA["Kelas"].values

if normalisasi:
    mn, mx = X.min(axis=0), X.max(axis=0)
    rng = np.where(mx - mn == 0, 1, mx - mn)
    Xs, bs = (X - mn) / rng, (baru - mn) / rng
else:
    Xs, bs = X, baru

if metrik == "Euclidean":
    jarak = np.sqrt(((Xs - bs) ** 2).sum(axis=1))
else:
    jarak = np.abs(Xs - bs).sum(axis=1)

hasil = DATA[["Pasien"] + FITUR + ["Label", "Kelas"]].copy()
hasil["Jarak"] = jarak
hasil = hasil.sort_values("Jarak").reset_index(drop=True)
hasil.insert(0, "Peringkat", hasil.index + 1)
tetangga = hasil.head(k).copy()

bobot = 1 / (tetangga["Jarak"] + 1e-9) if berbobot else pd.Series(1.0, index=tetangga.index)
skor = {c: float(bobot[tetangga["Kelas"] == c].sum()) for c in (0, 1)}
total = sum(skor.values())
prob = {c: skor[c] / total for c in skor}
prediksi = 1 if skor[1] > skor[0] else 0

# ---------- Hasil ----------
st.divider()
kolom_hasil, kolom_vote = st.columns([1.2, 1])

with kolom_hasil:
    if prediksi == 1:
        st.markdown(
            f'<div class="result-high"><h2>⚠️ Risiko Tinggi</h2>'
            f'<p>Pasien diprediksi berisiko diabetes tinggi (kelas 1) · keyakinan {prob[1]:.0%}</p></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="result-low"><h2>✅ Risiko Rendah</h2>'
            f'<p>Pasien diprediksi berisiko diabetes rendah (kelas 0) · keyakinan {prob[0]:.0%}</p></div>',
            unsafe_allow_html=True,
        )
    m1, m2, m3 = st.columns(3)
    m1.metric("K", k)
    m2.metric("Tetangga Tinggi", int((tetangga["Kelas"] == 1).sum()))
    m3.metric("Tetangga Rendah", int((tetangga["Kelas"] == 0).sum()))

with kolom_vote:
    fig_v = go.Figure(
        go.Bar(
            x=[prob[0] * 100, prob[1] * 100],
            y=["Risiko Rendah", "Risiko Tinggi"],
            orientation="h",
            marker_color=[WARNA["Risiko Rendah"], WARNA["Risiko Tinggi"]],
            text=[f"{prob[0]:.0%}", f"{prob[1]:.0%}"],
            textposition="auto",
        )
    )
    fig_v.update_layout(title="Hasil Voting Tetangga (%)", height=240,
                        xaxis=dict(range=[0, 100]), margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_v, use_container_width=True)

# ---------- Tab ----------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📏 Perhitungan Jarak", "📊 Visualisasi", "🗂️ Dataset", "📘 Cara Kerja KNN"]
)

with tab1:
    st.markdown(
        f"Jarak **{metrik}** dihitung dari pasien baru ke setiap pasien di dataset"
        f"{' (data sudah dinormalisasi Min-Max)' if normalisasi else ''}. "
        f"**{k} tetangga terdekat** disorot."
    )

    def sorot(row):
        warna = "background-color: #fef9c3; font-weight:600" if row["Peringkat"] <= k else ""
        return [warna] * len(row)

    tampil = hasil.drop(columns="Kelas")
    st.dataframe(
        tampil.style.apply(sorot, axis=1).format({"BMI": "{:.1f}", "Jarak": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )
    fig_j = px.bar(
        hasil, x="Pasien", y="Jarak", color="Label", color_discrete_map=WARNA,
        title="Jarak Pasien Baru ke Tiap Pasien (urut terdekat)",
    )
    fig_j.update_layout(height=360, xaxis_title=None, legend_title=None)
    st.plotly_chart(fig_j, use_container_width=True)

with tab2:
    a, b = st.columns(2)
    with a:
        fig1 = px.scatter(
            DATA, x="Glukosa", y="BMI", color="Label", color_discrete_map=WARNA,
            text="Pasien", size_max=14, title="Glukosa vs BMI",
        )
        fig1.update_traces(marker=dict(size=13), textposition="top center")
        fig1.add_trace(go.Scatter(
            x=[glukosa], y=[bmi], mode="markers+text", text=["Pasien Baru"],
            textposition="bottom center",
            marker=dict(symbol="star", size=22, color="#2563eb", line=dict(width=2, color="white")),
            name="Pasien Baru",
        ))
        fig1.update_layout(height=420, legend_title=None)
        st.plotly_chart(fig1, use_container_width=True)
    with b:
        fig2 = px.scatter(
            DATA, x="Usia", y="Tekanan Darah", color="Label", color_discrete_map=WARNA,
            text="Pasien", title="Usia vs Tekanan Darah",
        )
        fig2.update_traces(marker=dict(size=13), textposition="top center")
        fig2.add_trace(go.Scatter(
            x=[usia], y=[td], mode="markers+text", text=["Pasien Baru"],
            textposition="bottom center",
            marker=dict(symbol="star", size=22, color="#2563eb", line=dict(width=2, color="white")),
            name="Pasien Baru",
        ))
        fig2.update_layout(height=420, legend_title=None)
        st.plotly_chart(fig2, use_container_width=True)

    # Radar: pasien baru vs rata-rata tiap kelas (normalisasi 0-1)
    mn_all, mx_all = X.min(axis=0), X.max(axis=0)
    rg = np.where(mx_all - mn_all == 0, 1, mx_all - mn_all)
    rata = {c: ((X[y == c] - mn_all) / rg).mean(axis=0) for c in (0, 1)}
    nb = np.clip((baru - mn_all) / rg, 0, 1)
    fig_r = go.Figure()
    for nama, nilai, warna in [
        ("Rata-rata Risiko Rendah", rata[0], WARNA["Risiko Rendah"]),
        ("Rata-rata Risiko Tinggi", rata[1], WARNA["Risiko Tinggi"]),
        ("Pasien Baru", nb, "#2563eb"),
    ]:
        fig_r.add_trace(go.Scatterpolar(
            r=list(nilai) + [nilai[0]], theta=FITUR + [FITUR[0]],
            name=nama, line=dict(color=warna, width=3 if nama == "Pasien Baru" else 2),
            fill="toself" if nama == "Pasien Baru" else None, opacity=0.8,
        ))
    fig_r.update_layout(title="Profil Pasien Baru vs Rata-rata Kelas (skala 0–1)",
                        height=450, polar=dict(radialaxis=dict(range=[0, 1])))
    st.plotly_chart(fig_r, use_container_width=True)

with tab3:
    ringkas = DATA.drop(columns="Kelas").copy()
    ringkas["Riwayat Keluarga"] = ringkas["Riwayat Keluarga"].map({0: "Tidak ada", 1: "Ada"})
    st.dataframe(ringkas, use_container_width=True, hide_index=True)
    d1, d2, d3 = st.columns(3)
    d1.metric("Jumlah data", len(DATA))
    d2.metric("Risiko Rendah", int((y == 0).sum()))
    d3.metric("Risiko Tinggi", int((y == 1).sum()))
    st.download_button("⬇️ Unduh dataset (CSV)",
                       DATA.drop(columns="Label").to_csv(index=False).encode(),
                       "dataset_diabetes.csv", "text/csv")

with tab4:
    st.markdown(
        """
        **Langkah-langkah KNN pada kasus ini**
        1. **Siapkan data** – 10 pasien dengan 5 fitur dan label risiko.
        2. **Normalisasi** *(opsional)* – setiap fitur diubah ke skala 0–1:
           `x' = (x − min) / (max − min)`.
        3. **Hitung jarak** pasien baru ke semua pasien:
           - Euclidean: `d = √Σ(xᵢ − yᵢ)²`
           - Manhattan: `d = Σ|xᵢ − yᵢ|`
        4. **Urutkan** jarak dari yang terkecil.
        5. **Ambil K tetangga terdekat.**
        6. **Voting** – kelas mayoritas menjadi hasil prediksi
           (opsional dibobot 1/jarak sehingga tetangga dekat lebih berpengaruh).

        ⚠️ *Aplikasi ini hanya untuk pembelajaran dan tidak menggantikan diagnosis dokter.*
        """
    )
