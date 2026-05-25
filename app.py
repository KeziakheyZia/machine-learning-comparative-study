# =========================================================
# HEAD
# =========================================================

# IMPORT
# =========================================================
import streamlit as st
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
import ast

df_raww = pd.read_csv("Hasil Klasifikasi/Sentimen_Publik.csv")
df_data_mentah = df_raww.head(5)

df_raw = pd.read_csv("Hasil Klasifikasi/data_sentimen_labeled.csv")
df_data = df_raw.head(5)

df_raw3 = pd.read_pickle("Hasil Klasifikasi/data_sentimen_labeled.pkl")

from PIL import Image
logo = Image.open("media/logo3.png")
st.set_page_config(
    page_title="Dashboard Hasil Analisis",
    page_icon=logo,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# CUSTOM CSS
# =========================================================
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("style.css")

# HELPER FUNCTIONS
# =========================================================
def create_bar(df, x, y, title, color=None, height=500): 
    custom_colors = ["#da251d", "#5fa378", "#ededed"] 

    fig = px.bar(
        df,
        x=x,
        y=y,
        color=color if color else x,
        text=y,
        template="plotly_white",
        title=title,
        color_discrete_sequence=custom_colors if not color else None 
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=height,
        title_x=0.02,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False,
    )
    return fig

def create_horizontal_bar(df, x, y, title, height=290):
    fig = px.bar(
        df.sort_values(x),
        x=x,
        y=y,
        orientation="h",
        text=x,
        template="plotly_white",
        title=title,
    )
    fig.update_layout(
        height=height,
        title_x=0.02,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False,
    )
    return fig

def create_model_comparison(df):
    melted = df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig = px.bar(
    melted,
    x="Metric",
    y="Score",
    color="Model",
    barmode="group",
    text="Score",
    template="plotly_white",
    title="Perbandingan Kinerja Model Random Forest vs SVM",

    color_discrete_map={
        "Random Forest": "#efc14b",  
        "SVM": "#da251d"            
    }

)

    fig.update_traces(
        texttemplate="%{text:.0%}",
        textposition="outside"
    )

    fig.update_yaxes(
        tickformat=".0%",
        range=[0, 1]
    )

    fig.update_layout(
        height=420,
        title_x=0.02,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig

# =========================================================
# BODY
# =========================================================

# HERO HEADER
# =========================================================
import base64

with open("media/logo3.png", "rb") as f:
    data = base64.b64encode(f.read()).decode()

st.markdown(f"""
<div class="hero-wrap">
    <img src="data:image/png;base64,{data}" width="50"/>
    <div class="hero-badge">Dashboard Hasil Penelitian</div>
    <div class="hero-title">
        PERBANDINGAN KINERJA RANDOM FOREST DAN SUPPORT VECTOR MACHINE 
        DALAM ANALISIS SENTIMEN PUBLIK TERHADAP FENOMENA GOLONGAN PUTIH 
        PADA PEMILU 2024
    </div>
    <div class="hero-subtitle">
        Dashboard ini dirancang untuk menampilkan alur penelitian secara terstruktur mulai dari
        <b>Pengumpulan data</b>, <b>Preprocessing data</b>, <b>Pelabelan data</b>, 
        <b>Pembagian data</b>, <b>Ekstraksi fitur</b>, hingga 
        <b>Evaluasi model dan Kesimpulan</b>.
    </div>
</div>
""", unsafe_allow_html=True)

# SUMMARY METRICS
# =========================================================
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(
        """
    <div class="metric-card">
        <div class="metric-title">Jumlah Data</div>
        <div class="metric-value">1008</div>
        <div class="hero-badge2">Dari X (Twitter)</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        """
    <div class="metric-card">
        <div class="metric-title">Akurasi RF</div>
        <div class="metric-value">75%</div>
        <div class="hero-badge2">Lebih Tinggi</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with m3:
    st.markdown(
        """
    <div class="metric-card">
        <div class="metric-title">Akurasi SVM</div>
        <div class="metric-value">72%</div>
        <div class="hero-badge2">Lebih Rendah</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with m4:
    st.markdown(
        """
    <div class="metric-card">
        <div class="metric-title">Selisih</div>
        <div class="metric-value">3%</div>
        <div class="hero-badge3">Perbedaan Akurasi</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# MINI NAVIGATION
# =========================================================
st.markdown(
    """
<div class="nav-card">
    <div class="nav-title">Struktur Dashboard</div>
    <div class="nav-pills">
        <div class="pill">Pengumpulan Data</div>
        <div class="pill">Preprocessing</div>
        <div class="pill">Pelabelan Data</div>
        <div class="pill">Pembagian Data</div>
        <div class="pill">Ekstraksi Fitur</div>
        <div class="pill">Evaluasi Model</div>
        <div class="pill">Kesimpulan</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# SECTION 1 - PENGUMPULAN DATA
# =========================================================
dataset_mentah = df_data_mentah[
    [
        "conversation_id_str",
        "created_at",
        "favorite_count",
        "full_text",
        "id_str",
        "image_url",
        "in_reply_to_screen_name",
        "lang",
        "location",
        "quote_count",
        "reply_count",
        "retweet_count",
        "tweet_url",
        "user_id_str",
        "username",
    ]
].head(5)

# =========================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Pengumpulan Data </div>', unsafe_allow_html=True)
st.markdown("""<div class="section-desc"> Data diambil dari media sosial X (sebelumnya Twitter) yang mengandung opini publik mengenai fenomena golongan putih pada pemilu 2024.</div>""", unsafe_allow_html=True,)
# st.markdown("---")

st.dataframe(dataset_mentah, use_container_width=True, hide_index=True)

st.markdown("""
<div class="info-box">
    <strong>Rentang Waktu:</strong> 
    01 Juni 2023 hingga 20 Februari 2024, dipilih karena mencakup seluruh fase penting, mulai dari awal kampanye, berbagai diskusi publik, hari pemilihan, hingga periode pasca pemungutan suara. Rentang waktu ini memungkinkan peneliti untuk menangkap dinamika perubahan opini dan sentimen masyarakat dengan lebih akurat.
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# Section 2 PREPROCESSING DATA
# =========================================================
preprocessing = pd.read_csv("Hasil Klasifikasi/data_sentimen_labeled.csv")
preprocessing = df_data [
    [
        "Case Folding",
        "Text Cleaning",
        "Tokenizing",
        "Normalization",
        "Stemming",
        "Text Filtering",
    ]
].head(5)

perbandingan = df_data [["full_text", "Text Filtering"]].head(5)
# Ubah list token jadi string
perbandingan["Text Filtering"] = perbandingan["Text Filtering"].apply(
    lambda x: " ".join(eval(x)) if isinstance(x, str) else " ".join(x)
)
# Rename kolom
perbandingan = perbandingan.rename(
    columns={"full_text": "Sebelum", "Text Filtering": "Sesudah"}
)

# =========================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Preprocessing Data</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">Tahapan preprocessing mencakup <b>Case Folding</b>, <b>Text Cleaning</b>, <b>Tokenizing</b>, <b>Normalization</b>, <b>Stemming</b> dan <b>Filtering</b>. </div>', unsafe_allow_html=True,)

c1, c2 = st.columns([2, 1])
with c1:
    st.dataframe(preprocessing, use_container_width=True, hide_index=True)
with c2:
    st.dataframe(perbandingan, use_container_width=True, hide_index=True)

st.markdown("""
<div class="info-box">
    <strong>Tujuan:</strong> 
    Mengubah data teks mentah menjadi data teks yang lebih bersih, konsisten, dan siap dianalisis.
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# SECTION 3 - PELABELAN DATA
# =========================================================
label_distribution = df_raw["sentimen"].value_counts().reset_index()
label_distribution.columns = ["Sentimen", "Jumlah"]
order = ["positif", "negatif", "netral"]
label_distribution["Sentimen"] = pd.Categorical(
    label_distribution["Sentimen"], categories=order, ordered=True
)
label_distribution = label_distribution.sort_values("Sentimen")

# =========================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Pelabelan Data</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">Distribusi label sentimen, sebaran skor polaritas, serta representasi kata dominan pada setiap kelas sentimen.</div>',
unsafe_allow_html=True,)

tab1, tab2 = st.tabs(["Statistik Label", "Word Cloud"])

with tab1:
    c1, c2 = st.columns(2)
    
    with c1:
        st.plotly_chart(
            create_bar(
                label_distribution, "Sentimen", "Jumlah", "Distribusi Label Sentimen"
            ),
            use_container_width=True,
        )
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

with c2:
    polarity = df_raw["polarity_score"]
    mean_score = polarity.mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_style("white")

    sns.histplot(
        polarity,
        bins=20,
        kde=True,
        color='#da251d',        
        edgecolor='#2c2c2c',     
        ax=ax
    )

    ax.axvline(x=0, color='#efc14b', linestyle='--', linewidth=2, label='Batas Netral (0)')
    ax.axvline(x=mean_score, color='#2c2c2c', linestyle='-', linewidth=2, label=f'Mean ({mean_score:.2f})')

    ax.text(
        mean_score,
        ax.get_ylim()[1] * 0.85,
        f'Mean: {mean_score:.2f}',
        color='#111827',
        fontweight='bold',
        ha='center',
        bbox=dict(facecolor='white', edgecolor='#d1d5db', boxstyle='round,pad=0.3')
    )

    ax.set_title('Sebaran Skor Polaritas Sentimen', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Skor Polaritas')
    ax.set_ylabel('Frekuensi (Jumlah Tweet)')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend()
    sns.despine()

    st.pyplot(fig, use_container_width=True)

import streamlit as st
from PIL import Image

with tab2:

    image_path = 'media/ct.png'
    try:
        img = Image.open(image_path)
        
        st.image(
            img, 
            # caption='Perbandingan Word Cloud: Positif, Negatif, dan Netral',
            use_container_width=True
        )
        # 4. Tambahkan tombol download (Opsional agar user bisa ambil filenya)
        # with open(image_path, "rb") as file:
        #     st.download_button(
        #         label="Unduh Gambar Word Cloud",
        #         data=file,
        #         file_name="wordcloud_sentimen_final.png",
        #         mime="image/png"
        #     )
            
    except FileNotFoundError:
        st.error(f"File {image_path} tidak ditemukan. Pastikan kodingan save_fig sudah dijalankan.")


st.markdown('</div>', unsafe_allow_html=True) # Tutup section-card
st.markdown("""
<div class="info-box">
    <strong>Tujuan:</strong> 
    Sebagai inti dari metode Supervised Learning, pelabelan bertujuan untuk mentransformasi data tekstual menjadi data kategorikal yang memiliki makna informasi. Seperti positif (Pro-Golput), negatif (Anti-Golput) dan netral
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# SECTION 4 - PEMBAGIAN DATA
# =========================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Pembagian Data</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">Dataset dipisahkan menggunakan teknik <B>Stratified Sampling</B> untuk menjaga rasio distribusi kelas.</div>', unsafe_allow_html=True)

#Proses Data (Hanya sekali jalan agar efisien) ---
def join_tokens(x):
    if isinstance(x, str):
        try: return " ".join(ast.literal_eval(x))
        except: return x
    return " ".join(x) if isinstance(x, list) else x

df_raw["text_for_tfidf"] = df_raw["Text Filtering"].apply(join_tokens)

# Split Data (Memastikan stratify agar distribusi konsisten)
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df_raw["text_for_tfidf"], df_raw["sentimen"], 
    test_size=0.2, random_state=42, stratify=df_raw["sentimen"]
)

# Inisialisasi Tab
tab1, tab2 = st.tabs(["Statistik Pembagian", "Konsistensi Distribusi"])

with tab1:
    c1, c2 = st.columns([1, 1.2])
    
    with c1:
        data_manual = pd.DataFrame({"Kategori": ["Data Latih", "Data Uji"], "Jumlah": [len(y_train), len(y_test)]})
        fig_pie = px.pie(
    data_manual,
    values="Jumlah",
    names="Kategori",
    color="Kategori",  
    hole=0.3,
    template="plotly_white",
    color_discrete_map={
        "Data Latih": "#da251d",   
        "Data Uji": "#fae89d"      
    },
    title="<b>Proporsi Pembagian Data</b>"
)
        fig_pie.update_traces(textposition="outside", textinfo="percent+label", marker=dict(line=dict(color="#FFFFFF", width=2)))
        # fig_pie.update_layout(height=400, showlegend=False, margin=dict(l=10, r=10, t=100, b=10))
        fig_pie.update_layout(height=400, margin=dict(l=10, r=10, t=100, b=10), legend=dict(orientation="h", y=1.1, x=0.7, xanchor="left"))
        
        st.markdown('<div class="responsive-pie-box">', unsafe_allow_html=True)
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)


    with c2:
        # Bar Chart Distribusi Jumlah
        train_dist = y_train.value_counts().reset_index(); train_dist.columns = ["Sentimen", "Jumlah"]; train_dist["Dataset"] = "Data Latih"
        test_dist = y_test.value_counts().reset_index(); test_dist.columns = ["Sentimen", "Jumlah"]; test_dist["Dataset"] = "Data Uji"
        combined_dist = pd.concat([train_dist, test_dist])
        combined_dist["Sentimen"] = combined_dist["Sentimen"].str.capitalize()

        fig_split = px.bar(
            combined_dist, x="Sentimen", y="Jumlah", color="Dataset",
            barmode="group", text="Jumlah", template="plotly_white",
            color_discrete_map={"Data Latih": "#da251d", "Data Uji": "#fae89d"},
            title="<b>Distribusi Label Sentimen pada Data Latih dan Data Uji</b>"
        )
        fig_split.update_traces(textposition="outside")
        fig_split.update_layout(height=500, margin=dict(l=10, r=10, t=80, b=10), legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"))
        st.plotly_chart(fig_split, use_container_width=True)

# TAB 2: KONSISTENSI PEMBAGIAN DATA ---
with tab2:
    # Hitung Persentase
    train_perc = (y_train.value_counts(normalize=True) * 100).reset_index(); train_perc.columns = ["Sentimen", "Persentase"]; train_perc["Kelompok Data"] = "Data Latih (%)"
    test_perc = (y_test.value_counts(normalize=True) * 100).reset_index(); test_perc.columns = ["Sentimen", "Persentase"]; test_perc["Kelompok Data"] = "Data Uji (%)"
    combined_perc = pd.concat([train_perc, test_perc])
    combined_perc["Sentimen"] = combined_perc["Sentimen"].str.capitalize()

    fig_cons = px.bar(
        combined_perc, x="Sentimen", y="Persentase", color="Kelompok Data",
        barmode="group", text="Persentase", template="plotly_white",
        color_discrete_map={"Data Latih (%)": "#da251d", "Data Uji (%)": "#fae89d"},
        title="<b>Konsistensi Distribusi Data Latih dan Data Uji</b>"
    )
    fig_cons.update_traces(texttemplate="%{text:.1f}%", textposition="outside", textfont=dict(weight="bold"))
    fig_cons.update_layout(height=500, margin=dict(l=10, r=10, t=80, b=10), legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"))
    
    st.plotly_chart(fig_cons, use_container_width=True, config={"displayModeBar":"hover"})
  
st.markdown("""
<div class="info-box">
    <strong>Tujuan:</strong> 
    Pembagian dataset bertujuan untuk mengukur kemampuan generalisasi model terhadap data baru. Dengan memisahkan Data Latih (80%) untuk pembentukan pola sentimen dan Data Uji (20%) untuk evaluasi. Pemisahan ini penting untuk memastikan model diuji dengan data yang independen, sehingga hasil perbandingan akurasi antara Random Forest dan SVM bersifat valid dan terhindar dari adanya resiko overfitting.
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# SECTION 5 - EKSTRAKSI FITUR 
# =========================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Ekstraksi Fitur (TF-IDF)</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">Tahapan ini dilakukan untuk mengubah teks menjadi representasi numerik menggunakan metode TF-IDF sehingga dapat diproses oleh algoritma machine learning.</div>',
unsafe_allow_html=True,)
st.markdown("---")

# Baca data yang sudah diekspor dari Colab
try:
    top_terms_df = pd.read_csv("Hasil Klasifikasi/top_20_tfidf.csv")

    col_t1, col_t2 = st.columns([1, 1.2])

    with col_t1:
        st.markdown("###### Parameter TF-IDF")
        # Info statis 
        tfidf_stats = pd.DataFrame(
            {
                "Komponen": [
                    "Metode",
                    "Max Features",
                    "N-Gram Range",
                    "Min Document Frequency (min_df)",
                    "Max Document Frequency (max_df)",
                ],
                "Nilai": [
                    "TF-IDF Vectorizer",
                    "3000 Fitur",
                    "(1, 1) - Unigram",
                    "2 Dokumen",
                    "90% (0.9)",
                ],
            }
        )
        st.dataframe(tfidf_stats, use_container_width=True, hide_index=True)

        st.markdown(
            """
        <div class="info-box" style="font-size: 0.85rem;">
        Dimensi TF-IDF Data Latih (806,986) <br>  Dimensi TF-IDF Data Uji (202,986) 
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div class="info-box" style="font-size: 0.85rem;">
        Kata-kata di samping memiliki bobot tertinggi, yang berarti kata tersebut sangat berpengaruh dalam menentukan sentimen.
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_t2:
        fig_tfidf = px.bar(
            top_terms_df.sort_values("Score", ascending=True),
            x="Score",
            y="Term",
            orientation="h",
            text="Score",
            color="Score",
            color_continuous_scale="reds",
            template="plotly_white",
            title="20 Kata Kunci dengan Bobot Tertinggi",
        )

        fig_tfidf.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_tfidf.update_layout(
            height=480,
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis_title="Skor TF-IDF",
            yaxis_title=None,
            showlegend=False,
        )
        st.plotly_chart(
            fig_tfidf, use_container_width=True, config={"displayModeBar":"hover"}
        )

except FileNotFoundError:
    st.error(
        "File 'top_tfidf_features.csv' tidak ditemukan. Pastikan kamu sudah memindahkan file dari Colab ke folder VS Code."
    )
st.markdown("</div>", unsafe_allow_html=True)


# SECTION 6 - RF VS SVM
# =========================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Penerapan Random Forest dan Support Vector Machine</div>',unsafe_allow_html=True,)
st.markdown('<div class="section-desc">Evaluasi model berdasarkan metrik performa, classification report, dan confusion matrix dari data uji.</div>', unsafe_allow_html=True,)

rf_colorscale = [
    "#F9FAFB",
    "#D4D4D4",
    "#efc14b",  
    "#efc14b",  
    "#8C6B00"  
]
svm_colorscale = [
    "#F9FAFB",
    "#D4D4D4",
    "#da251d",
    "#da251d",
    "#991B1B"
]

def create_heatmap(cm_df, title, color_scale="Blues"):
    fig = go.Figure(
        data=go.Heatmap(
            z=cm_df.values,
            x=cm_df.columns,
            y=cm_df.index,
            text=cm_df.values,
            texttemplate="%{text}",
            colorscale=color_scale, 
            showscale=True,
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="Prediksi",
        yaxis_title="Aktual",
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig

try:
    # LOAD DATA DARI CSV (Hasil ekspor Colab)
    metrics_df = pd.read_csv("Hasil Klasifikasi/model_metrics.csv")
    rf_report_df = pd.read_csv("Hasil Klasifikasi/report_rf.csv")
    svm_report_df = pd.read_csv("Hasil Klasifikasi/report_svm.csv")

    # Load CM dan beri label kategori
    labels = ["Negatif", "Netral", "Positif"]
    rf_cm_values = pd.read_csv("Hasil Klasifikasi/cm_rf.csv")
    svm_cm_values = pd.read_csv("Hasil Klasifikasi/cm_svm.csv")

    rf_cm_df = pd.DataFrame(rf_cm_values.values, index=labels, columns=labels)
    svm_cm_df = pd.DataFrame(svm_cm_values.values, index=labels, columns=labels)

    tab1, tab2, tab3 = st.tabs(
        ["Evaluasi Komparatif", "Confusion Matrix", "Classification Report"]
    )

    with tab1:
        # Grafik Perbandingan Model
        st.plotly_chart(
            create_model_comparison(metrics_df),
            use_container_width=True,
            config={"displayModeBar":"hover"},
        )

        # Tabel Perbandingan Model
        df_compare = pd.DataFrame({
        "Metric": ["Accuracy %", "Precision %", "Recall %", "F1-Score %"],
        "Random Forest": [75, 75, 75, 73],
        "Support Vector Machine": [72, 69, 72, 69]
        })
        df_compare["Selisih"] = df_compare["Random Forest"] - df_compare["Support Vector Machine"]
        styled_df = df_compare.style.set_properties(**{
        'font-weight': 'bold'
            })
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

    with tab2:
        h1, h2 = st.columns(2)
        with h1:
            # Random Forest pakai warna Oranges/Cokelat
            st.plotly_chart(
                create_heatmap(
                    rf_cm_df, "Confusion Matrix - Random Forest", rf_colorscale
                ),
                use_container_width=True,
                config={"displayModeBar":"hover"},
            )
        with h2:
            # SVM pakai warna Blues
            st.plotly_chart(
                create_heatmap(
                    svm_cm_df, "Confusion Matrix - Support Vector Machine", svm_colorscale
                ),
                use_container_width=True,
                config={"displayModeBar":"hover"},
            )

    with tab3:
        st.markdown("###### Classification Report Comparison")
        try:
            with open("Hasil Klasifikasi/classification_report.txt", "r") as f:
                report_content = f.read()

            st.code(report_content, language="text")

        except FileNotFoundError:
            st.error("⚠️ File 'classification_report.txt' tidak ditemukan.")

    
    # INFO BOX OTOMATIS
    # Mencari model dengan akurasi tertinggi secara otomatis
    best_acc_model = metrics_df.loc[metrics_df["Accuracy"].idxmax(), "Model"]

    st.markdown(
        f"""
    <div class="info-box">
    Berdasarkan hasil evaluasi, model <b>{best_acc_model}</b> menunjukkan performa yang lebih unggul
    pada metrik evaluasi utama, serta menghasilkan tingkat kesalahan klasifikasi yang lebih rendah.
    </div>
    """,
        unsafe_allow_html=True,
    )

except FileNotFoundError as e:
    st.error(
        f"⚠️ Data evaluasi belum lengkap. Pastikan semua file CSV sudah ada di folder project. Error: {e}"
    )

st.markdown("</div>", unsafe_allow_html=True)


# SECTION 7 - KESIMPULAN
# =========================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Kesimpulan Penelitian</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">Interpretasi akhir berdasarkan keseluruhan tahapan dan hasil evaluasi model.</div>', unsafe_allow_html=True)

st.markdown("""
<div class="conclusion-box">
<b>Kesimpulan:</b><br><br>
Secara keseluruhan, penelitian ini menunjukkan bahwa algoritma <b>Random Forest</b> dan <b>Support Vector Machine</b> sama-sama memiliki kemampuan yang baik dalam melakukan klasifikasi sentimen pada data teks. Meskipun Random Forest menunjukkan performa yang sedikit lebih unggul, kedua algoritma tetap memberikan hasil yang kompetitif sehingga pemilihannya dapat disesuaikan dengan kebutuhan dan karakteristik data. 
<br><br>
Selain itu, penelitian ini tidak hanya menghasilkan perbandingan kinerja algoritma, tetapi juga memberikan gambaran awal mengenai pola sentimen publik terhadap fenomena golongan putih pada pemilu 2024 yang didominasi oleh sentimen positif (Pro-Golput) dengan jumlah 487 data (48,31%), diikuti oleh sentimen negatif (Anti-Golput)dengan jumlah 362 data (35,91%) dan sentimen netral dengan jumlah 159 data (15,78%). 
</div>
""", unsafe_allow_html=True)


# FOOTER
# =========================================================
st.markdown(
    '<div class="footer-note">Perbandingan Kinerja Random Forest dan Support Vector Machine dalam Analisis Sentimen Publik Terhadap Fenomena Golongan Putih Pada Pemilu 2024</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="footer-note">Kezia Cantika Saroinsong</div>', unsafe_allow_html=True
)


















