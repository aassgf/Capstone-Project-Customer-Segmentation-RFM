# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px
from io import BytesIO
import os

st.set_page_config(layout="wide", page_title="RFM Dashboard — Insight Driven")

# ---------------------------
# Helpers
# ---------------------------
@st.cache_data
def load_csv_if_exists(path="rfm_upload_format.csv"):
    """Load CSV from local path if exists; otherwise return None."""
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            return df, f"local file: {path}"
        except Exception as e:
            return None, f"failed to read {path}: {e}"
    return None, None

def clean_df(df):
    df = df.copy()
    for c in ['Cluster','MonetaryValue','Frequency','Recency']:
        if c not in df.columns:
            df[c] = np.nan
    df['Cluster'] = pd.to_numeric(df['Cluster'], errors='coerce')
    df['MonetaryValue'] = pd.to_numeric(df['MonetaryValue'], errors='coerce')
    df['Frequency'] = pd.to_numeric(df['Frequency'], errors='coerce')
    df['Recency'] = pd.to_numeric(df['Recency'], errors='coerce')
    df = df.dropna(subset=['Cluster','MonetaryValue','Frequency','Recency']).copy()
    # convert int-like floats to int
    if pd.api.types.is_float_dtype(df['Cluster']):
        if (df['Cluster'] % 1 == 0).all():
            df['Cluster'] = df['Cluster'].astype(int)
    return df

def build_palette(clusters, base_colors=None):
    if base_colors is None:
        base_colors = {}
    unique = sorted(clusters)
    palette = {}
    for c in unique:
        if c in base_colors:
            palette[c] = base_colors[c]
    missing = [c for c in unique if c not in palette]
    if missing:
        auto = sns.color_palette(n_colors=len(missing))
        for c, col in zip(missing, auto):
            palette[c] = mcolors.to_hex(col)
    # order deterministic
    return {k: palette[k] for k in sorted(palette.keys())}

def df_to_csv_bytes(df):
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf

# ---------------------------
# Base colors & cluster descriptions
# ---------------------------
BASE_COLORS = {
    0: '#1f77b4',  # Blue - Retain
    1: '#ff7f0e',  # Orange - Re-Engage
    2: '#2ca02c',  # Green - Nurture
    3: '#d62728'   # Red - Reward
}

CLUSTER_TEXT = {
    0: ("Retain", "Pelanggan bernilai tinggi. Fokus: pertahankan melalui program loyalitas & personalisasi."),
    1: ("Re-Engage", "Pelanggan berisiko churn. Fokus: kampanye re-engagement & diskon khusus."),
    2: ("Nurture", "Pelanggan baru/berpotensi. Fokus: edukasi produk & insentif untuk repeat purchase."),
    3: ("Reward", "Pelanggan paling loyal. Fokus: reward eksklusif & pengakuan loyalitas.")
}

# ---------------------------
# Load data (auto) or fallback to dummy
# ---------------------------
df_raw, src = load_csv_if_exists("rfm_upload_format.csv")
if df_raw is None:
    # create dummy sample
    rng = np.random.default_rng(42)
    n = 400
    df_raw = pd.DataFrame({
        'Cluster': rng.integers(0,4,size=n),
        'MonetaryValue': np.abs(rng.normal(200,120,size=n)).round(2),
        'Frequency': np.clip(np.abs(rng.normal(5,3,size=n)).round(0), 0, None),
        'Recency': np.abs(rng.normal(60,45,size=n)).round(0)
    })
    src = "generated dummy data (no local rfm_upload_format.csv found)"

df = clean_df(df_raw)

# ---------------------------
# Sidebar: filters & controls
# ---------------------------
st.sidebar.title("Filter & Kontrol")
st.sidebar.markdown(f"**Sumber data:** {src}")

clusters_present = sorted(df['Cluster'].unique())
selected_clusters = st.sidebar.multiselect("Pilih cluster (kosong = semua)", options=clusters_present, default=clusters_present)

mv_min, mv_max = float(df['MonetaryValue'].min()), float(df['MonetaryValue'].max())
freq_min, freq_max = float(df['Frequency'].min()), float(df['Frequency'].max())
rec_min, rec_max = float(df['Recency'].min()), float(df['Recency'].max())

mv_range = st.sidebar.slider("MonetaryValue range", mv_min, mv_max, (mv_min, mv_max))
freq_range = st.sidebar.slider("Frequency range", freq_min, freq_max, (freq_min, freq_max))
rec_range = st.sidebar.slider("Recency range (days)", rec_min, rec_max, (rec_min, rec_max))

# apply filters
df_filtered = df[
    (df['Cluster'].isin(selected_clusters)) &
    (df['MonetaryValue'] >= mv_range[0]) & (df['MonetaryValue'] <= mv_range[1]) &
    (df['Frequency'] >= freq_range[0]) & (df['Frequency'] <= freq_range[1]) &
    (df['Recency'] >= rec_range[0]) & (df['Recency'] <= rec_range[1])
].copy()

# build palette for filtered clusters
palette = build_palette(sorted(df_filtered['Cluster'].unique()), base_colors=BASE_COLORS)
# plotly wants string keys
plotly_color_map = {str(k): v for k, v in palette.items()}

# ---------------------------
# Layout: title, summary
# ---------------------------
st.title("🎯 RFM Customer Segmentation — Dashboard")
st.caption("Visualisasi dan insight otomatis per cluster")

left, right = st.columns([2,1])
with left:
    st.subheader("Preview Data")
    st.dataframe(df_filtered.head(10))
with right:
    st.subheader("Download")
    st.download_button("Unduh data (CSV)", data=df_to_csv_bytes(df_filtered), file_name="rfm_filtered_for_dashboard.csv", mime="text/csv")
    st.write("")
    st.markdown("**Cluster legend**")
    for k, v in palette.items():
        st.markdown(f"- <span style='color:{v}'>■</span> **Cluster {k}** — {CLUSTER_TEXT.get(int(k), (f'Cluster {k}',''))[0]}", unsafe_allow_html=True)

# ---------------------------
# KPIs per cluster
# ---------------------------
st.markdown("## KPI Ringkasan per Cluster")
kpi_cols = st.columns(max(1, len(sorted(df_filtered['Cluster'].unique()))))
for i, c in enumerate(sorted(df_filtered['Cluster'].unique())):
    col = kpi_cols[i]
    sub = df_filtered[df_filtered['Cluster']==c]
    count = len(sub)
    avg_m = sub['MonetaryValue'].mean() if count>0 else 0
    med_r = sub['Recency'].median() if count>0 else 0
    avg_f = sub['Frequency'].mean() if count>0 else 0
    with col:
        col.metric(label=f"Cluster {int(c)} — {CLUSTER_TEXT.get(int(c),(str(c),''))[0]}", value=int(count))
        col.write(f"Avg Monetary: £{avg_m:,.0f}")
        col.write(f"Median Recency: {med_r:.0f} days")
        col.write(f"Avg Frequency: {avg_f:.1f}")

# ---------------------------
# Charts: Count Bar + Violin (Plotly) + Boxplot Matplotlib
# ---------------------------
st.markdown("## Visualisasi")

# 1) Count bar (plotly)
st.markdown("### Jumlah Pelanggan per Cluster")
count_df = df_filtered['Cluster'].value_counts().rename_axis('Cluster').reset_index(name='Count').sort_values('Cluster')
fig_count = px.bar(count_df, x='Cluster', y='Count', text='Count',
                   color=count_df['Cluster'].astype(str), color_discrete_map=plotly_color_map)
fig_count.update_layout(showlegend=False, xaxis_title='Cluster', yaxis_title='Jumlah Pelanggan')
st.plotly_chart(fig_count, use_container_width=True)

# 2) Violin plots with Plotly (interactive)
st.markdown("### Distribusi Monetary / Frequency / Recency (Violin — Interaktif)")
if len(df_filtered) > 0:
    # convert cluster to string for plotly color mapping
    df_plot = df_filtered.copy()
    df_plot['Cluster_str'] = df_plot['Cluster'].astype(str)

    col1, col2 = st.columns(2)
    with col1:
        fig_v_mon = px.violin(df_plot, x='Cluster_str', y='MonetaryValue', color='Cluster_str',
                              color_discrete_map=plotly_color_map, points='all', hover_data=['Frequency','Recency'])
        fig_v_mon.update_layout(title='MonetaryValue by Cluster', xaxis_title='Cluster', yaxis_title='Monetary (£)', showlegend=False)
        st.plotly_chart(fig_v_mon, use_container_width=True)

    with col2:
        fig_v_freq = px.violin(df_plot, x='Cluster_str', y='Frequency', color='Cluster_str',
                               color_discrete_map=plotly_color_map, points='all')
        fig_v_freq.update_layout(title='Frequency by Cluster', xaxis_title='Cluster', yaxis_title='Frequency', showlegend=False)
        st.plotly_chart(fig_v_freq, use_container_width=True)

    # single wide violin for Recency
    fig_v_rec = px.violin(df_plot, x='Cluster_str', y='Recency', color='Cluster_str',
                         color_discrete_map=plotly_color_map, points='all')
    fig_v_rec.update_layout(title='Recency by Cluster', xaxis_title='Cluster', yaxis_title='Days since last purchase', showlegend=False)
    st.plotly_chart(fig_v_rec, use_container_width=True)
else:
    st.info("Tidak ada data untuk divisualisasikan (setelah filter).")

# 3) Boxplot matplotlib for compact comparison
st.markdown("### Perbandingan Statistik (Boxplot)")
fig = plt.figure(figsize=(10,4))
order = sorted(df_filtered['Cluster'].unique())
sns.boxplot(x='Cluster', y='MonetaryValue', data=df_filtered, order=order, palette=[palette[k] for k in order])
plt.title("Monetary Value - Boxplot per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Monetary (£)")
st.pyplot(fig)

# ---------------------------
# 3D scatter interactive (Plotly)
# ---------------------------
st.markdown("## Scatter 3D Interaktif")
if len(df_filtered) > 0:
    df_plot = df_filtered.copy()
    df_plot['Cluster_str'] = df_plot['Cluster'].astype(str)
    fig3d = px.scatter_3d(df_plot, x='MonetaryValue', y='Frequency', z='Recency',
                          color='Cluster_str',
                          color_discrete_map=plotly_color_map,
                          hover_data=['MonetaryValue','Frequency','Recency'])
    fig3d.update_traces(marker=dict(size=4, line=dict(width=0.2, color='white')))
    fig3d.update_layout(height=650)
    st.plotly_chart(fig3d, use_container_width=True)
else:
    st.info("Tidak ada titik untuk ditampilkan di scatter 3D.")

# ---------------------------
# Insights & Recommendations (auto-generated)
# ---------------------------
st.markdown("## Insight & Rekomendasi Otomatis")
for c in sorted(df_filtered['Cluster'].unique()):
    title, desc = CLUSTER_TEXT.get(int(c), (f"Cluster {c}", "Deskripsi tidak tersedia"))
    sub = df_filtered[df_filtered['Cluster']==c]
    count = len(sub)
    avg_m = sub['MonetaryValue'].mean() if count>0 else 0
    avg_f = sub['Frequency'].mean() if count>0 else 0
    med_r = sub['Recency'].median() if count>0 else 0

    st.markdown(f"### Cluster {int(c)} — {title}")
    st.write(f"- **Jumlah pelanggan:** {int(count)}")
    st.write(f"- **Rata-rata Monetary:** £{avg_m:,.0f}")
    st.write(f"- **Rata-rata Frequency:** {avg_f:.2f}")
    st.write(f"- **Median Recency:** {med_r:.0f} hari")
    st.write(f"- **Rekomendasi singkat:** {desc}")
    # small action plan
    if int(c) == 0:
        st.write("- Action plan: Program loyalitas, personalisasi penawaran, komunikasi rutin.")
    elif int(c) == 1:
        st.write("- Action plan: Kampanye re-engagement, diskon targeted, email reminder.")
    elif int(c) == 2:
        st.write("- Action plan: Onboarding email, edukasi produk, promo untuk repeat.")
    elif int(c) == 3:
        st.write("- Action plan: Reward eksklusif, early access, referral perks.")
    st.write("---")

st.write("— Selesai —")
