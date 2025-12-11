# app.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(layout="wide", page_title="RFM Dashboard")

# 1. Baca CSV (harus ada di working dir atau di repo saat deploy)
df = pd.read_csv("rfm_upload_format.csv")

# 2. Tampilkan tabel
st.header("Data (sample)")
st.dataframe(df.head(20))

# 3. Violin plots (matplotlib + seaborn)
st.header("Distribusi RFM per Cluster")
fig = plt.figure(figsize=(10, 12))

order = sorted(df['Cluster'].unique())

plt.subplot(3, 1, 1)
sns.violinplot(x='Cluster', y='MonetaryValue', data=df, hue='Cluster', legend=False, order=order)
plt.title('Monetary Value')

plt.subplot(3, 1, 2)
sns.violinplot(x='Cluster', y='Frequency', data=df, hue='Cluster', legend=False, order=order)
plt.title('Frequency')

plt.subplot(3, 1, 3)
sns.violinplot(x='Cluster', y='Recency', data=df, hue='Cluster', legend=False, order=order)
plt.title('Recency')

plt.tight_layout()
st.pyplot(fig)   # tampilkan matplotlib fig di Streamlit

# 4. 3D interactive with Plotly (keuntungan: interaktif)
st.header("Scatter 3D Interaktif")
df['Cluster_str'] = df['Cluster'].astype(str)
fig3d = px.scatter_3d(df, x='MonetaryValue', y='Frequency', z='Recency',
                      color='Cluster_str',
                      hover_data=['MonetaryValue','Frequency','Recency'])
st.plotly_chart(fig3d, use_container_width=True)
