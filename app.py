# joint_embed_streamlit/app.py

import streamlit as st
import os
import json
import numpy as np
import plotly.express as px
from utils.embedding_utils import load_clip_model, get_image_embedding, get_text_embedding
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from PIL import Image
import pandas as pd

st.set_page_config(layout="wide")
st.title("🔍 Visualizing Image vs Text Embeddings Separately")

# --- Constants ---
DATA_FOLDER = "data/images"
CAPTION_FILE = "data/captions1.json"  # updated to match generated caption file
os.makedirs("embeddings", exist_ok=True)
os.makedirs("projections", exist_ok=True)

# --- Load files and captions ---
@st.cache_data
def load_dataset():
    image_list = sorted([f for f in os.listdir(DATA_FOLDER) if f.lower().endswith(('jpg', 'jpeg', 'png'))])
    with open(CAPTION_FILE) as f:
        caption_map = json.load(f)
    return image_list, caption_map

image_files, captions = load_dataset()

# --- Load CLIP model ---
model, processor = load_clip_model()

# --- Extract Embeddings ---
@st.cache_data
def extract_embeddings():
    image_vecs, text_vecs, tags = [], [], []
    for filename in image_files:
        img_path = os.path.join(DATA_FOLDER, filename)
        text = captions.get(filename, f"An image of {filename.split('.')[0]}")
        image_vecs.append(get_image_embedding(model, processor, img_path))
        text_vecs.append(get_text_embedding(model, processor, text))
        tags.append(filename)
    return np.array(image_vecs), np.array(text_vecs), tags

image_embeds, text_embeds, labels = extract_embeddings()

# --- Sidebar projection method ---
st.sidebar.subheader("🔧 Projection Settings")
method = st.sidebar.selectbox("Select projection method:", ["PCA", "UMAP", "t-SNE"])
show_combined = st.sidebar.checkbox("Show combined embedding plot")

# --- Projection helper ---
def project(method, vecs):
    n_samples = len(vecs)
    if n_samples < 3:
        return PCA(n_components=2).fit_transform(vecs)
    if method == "PCA":
        return PCA(n_components=2).fit_transform(vecs)
    elif method == "UMAP":
        if n_samples <= 5:
            return PCA(n_components=2).fit_transform(vecs)
        return UMAP(n_components=2, n_neighbors=min(5, n_samples - 2)).fit_transform(vecs)
    elif method == "t-SNE":
        if n_samples <= 5:
            return PCA(n_components=2).fit_transform(vecs)
        return TSNE(n_components=2, perplexity=min(5, n_samples - 1), init="random", random_state=42).fit_transform(vecs)

proj_img = project(method, image_embeds)
proj_txt = project(method, text_embeds)

# --- Save projections ---
pd.DataFrame(proj_img, columns=["x1", "x2"]).to_csv("projections/image_proj.csv", index=False)
pd.DataFrame(proj_txt, columns=["x1", "x2"]).to_csv("projections/text_proj.csv", index=False)
pd.DataFrame(image_embeds).to_csv("embeddings/image_embeddings.csv", index=False)
pd.DataFrame(text_embeds).to_csv("embeddings/text_embeddings.csv", index=False)

# --- Categorization helper ---
def get_category(label):
    return "Flickr8k"  # Simplified: You can expand category logic if needed

categories = [get_category(lbl) for lbl in labels]

# Create unified dataframe with modality
proj_combined = np.concatenate([proj_img, proj_txt])
modalities = ["Image"] * len(proj_img) + ["Text"] * len(proj_txt)

df_combined = pd.DataFrame({
    f"{method} 1": proj_combined[:, 0],
    f"{method} 2": proj_combined[:, 1],
    "Filename": labels + labels,
    "Type": modalities,
    "Caption": [captions.get(lbl, "") for lbl in labels] * 2,
    "Category": categories * 2
})

# Split for clarity
df_img = df_combined[df_combined.Type == "Image"]
df_txt = df_combined[df_combined.Type == "Text"]

# --- Side-by-side layout ---
col1, col2 = st.columns(2)
with col1:
    st.subheader(f"🖼 Image Embedding Projection ({method})")
    fig1 = px.scatter(
        df_img,
        x=f"{method} 1",
        y=f"{method} 2",
        color="Category",
        hover_data=["Filename", "Caption"],
        title="Image Embeddings"
    )
    fig1.update_layout(template="plotly_dark")
    st.plotly_chart(fig1, _container_width=True)

with col2:
    st.subheader(f"📝 Text Embedding Projection ({method})")
    fig2 = px.scatter(
        df_txt,
        x=f"{method} 1",
        y=f"{method} 2",
        color="Category",
        hover_data=["Filename", "Caption"],
        title="Text Embeddings"
    )
    fig2.update_layout(template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)

# --- Combined visualization toggle ---
if show_combined:
    st.subheader("🔄 Combined Embedding Plot")
    fig3 = px.scatter(
        df_combined,
        x=f"{method} 1",
        y=f"{method} 2",
        color="Category",
        symbol="Type",
        hover_data=["Filename", "Caption"],
        title="Combined Embeddings"
    )
    fig3.update_layout(template="plotly_dark")
    st.plotly_chart(fig3, use_container_width=True)

# --- Download options ---
st.sidebar.subheader("⬇️ Download Options")
with st.sidebar.expander("Download Embeddings & Projections"):
    with open("embeddings/image_embeddings.csv", "rb") as f:
        st.download_button("📥 Download Image Embeddings", f, file_name="image_embeddings.csv")
    with open("embeddings/text_embeddings.csv", "rb") as f:
        st.download_button("📥 Download Text Embeddings", f, file_name="text_embeddings.csv")
    with open("projections/image_proj.csv", "rb") as f:
        st.download_button("📈 Download Image Projection", f, file_name="image_proj.csv")
    with open("projections/text_proj.csv", "rb") as f:
        st.download_button("📈 Download Text Projection", f, file_name="text_proj.csv")

# --- Sample Preview ---
st.subheader("🔎 Sample Preview")
selected = st.selectbox("Pick an image:", labels)
col1, col2 = st.columns(2)
with col1:
    st.image(os.path.join(DATA_FOLDER, selected), caption=selected, use_column_width=True)
with col2:
    st.markdown(f"**Caption**: {captions.get(selected, 'No caption found')}")
