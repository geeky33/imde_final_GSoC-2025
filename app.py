import streamlit as st
import os
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.embedding_utils import load_clip_model, get_image_embedding, get_text_embedding
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from PIL import Image
import pandas as pd
from scipy.spatial.distance import cosine

st.set_page_config(layout="wide")
st.title("🔍 Visualizing Image vs Text Embeddings with Improved Joint View")

# --- Constants ---
DATA_FOLDER = "data/images"
CAPTION_FILE = "data/captions1.json"
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

# Add joint embedding method selection
st.sidebar.subheader("🔗 Joint Embedding Method")
joint_method = st.sidebar.selectbox(
    "How to combine image and text embeddings:",
    ["Concatenated Projection", "Averaged Pairs", "Aligned Projections", "Connected Pairs"]
)

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
        return UMAP(n_components=2, n_neighbors=min(10, n_samples - 2), min_dist=0.3, metric="cosine").fit_transform(vecs)
    elif method == "t-SNE":
        if n_samples <= 5:
            return PCA(n_components=2).fit_transform(vecs)
        return TSNE(n_components=2, perplexity=min(30, n_samples - 1), learning_rate='auto', init="random", random_state=42).fit_transform(vecs)

# --- Improved Joint Projection Methods ---
def create_joint_projection(method, joint_method, image_embeds, text_embeds):
    if joint_method == "Concatenated Projection":
        # Original method - project concatenated embeddings
        combined_embeds = np.concatenate([image_embeds, text_embeds])
        proj_joint = project(method, combined_embeds)
        proj_img_joint = proj_joint[:len(image_embeds)]
        proj_txt_joint = proj_joint[len(image_embeds):]
        
    elif joint_method == "Averaged Pairs":
        # Average corresponding image-text pairs and project
        paired_embeds = (image_embeds + text_embeds) / 2
        proj_pairs = project(method, paired_embeds)
        # Use the same projection for both modalities but add small offset for visualization
        offset = 0.5
        proj_img_joint = proj_pairs + np.array([-offset, 0])
        proj_txt_joint = proj_pairs + np.array([offset, 0])
        
    elif joint_method == "Aligned Projections":
        # Project separately then align using Procrustes-like approach
        proj_img = project(method, image_embeds)
        proj_txt = project(method, text_embeds)
        
        # Simple alignment: translate text embeddings to minimize distance to image embeddings
        mean_img = np.mean(proj_img, axis=0)
        mean_txt = np.mean(proj_txt, axis=0)
        translation = mean_img - mean_txt
        proj_img_joint = proj_img
        proj_txt_joint = proj_txt + translation
        
    elif joint_method == "Connected Pairs":
        # Use concatenated approach but we'll add connection lines later
        combined_embeds = np.concatenate([image_embeds, text_embeds])
        proj_joint = project(method, combined_embeds)
        proj_img_joint = proj_joint[:len(image_embeds)]
        proj_txt_joint = proj_joint[len(image_embeds):]
    
    return proj_img_joint, proj_txt_joint

# Calculate projections
proj_img = project(method, image_embeds)
proj_txt = project(method, text_embeds)
proj_img_joint, proj_txt_joint = create_joint_projection(method, joint_method, image_embeds, text_embeds)

# --- Save projections ---
pd.DataFrame(proj_img, columns=["x1", "x2"]).to_csv("projections/image_proj.csv", index=False)
pd.DataFrame(proj_txt, columns=["x1", "x2"]).to_csv("projections/text_proj.csv", index=False)
pd.DataFrame(image_embeds).to_csv("embeddings/image_embeddings.csv", index=False)
pd.DataFrame(text_embeds).to_csv("embeddings/text_embeddings.csv", index=False)

# --- Categorization helper ---
def get_category(label):
    return "Flickr8k"

categories = [get_category(lbl) for lbl in labels]

# --- Prepare data for individual plots ---
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
    st.plotly_chart(fig1, use_container_width=True)

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

# --- Improved Joint Embedding ---
st.subheader(f"🔄 Joint Embedding Plot ({joint_method})")

# Prepare joint data
proj_joint_combined = np.concatenate([proj_img_joint, proj_txt_joint])
modalities_joint = ["Image"] * len(proj_img_joint) + ["Text"] * len(proj_txt_joint)

df_joint = pd.DataFrame({
    f"{method} 1": proj_joint_combined[:, 0],
    f"{method} 2": proj_joint_combined[:, 1],
    "Filename": labels + labels,
    "Type": modalities_joint,
    "Caption": [captions.get(lbl, "") for lbl in labels] * 2,
    "Category": categories * 2
})

if joint_method == "Connected Pairs":
    # Create connected pairs visualization
    fig3 = go.Figure()
    
    # Add connection lines between corresponding pairs
    for i in range(len(labels)):
        fig3.add_trace(go.Scatter(
            x=[proj_img_joint[i, 0], proj_txt_joint[i, 0]],
            y=[proj_img_joint[i, 1], proj_txt_joint[i, 1]],
            mode='lines',
            line=dict(color='gray', width=0.5),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add image points
    fig3.add_trace(go.Scatter(
        x=proj_img_joint[:, 0],
        y=proj_img_joint[:, 1],
        mode='markers',
        marker=dict(symbol='circle', size=6, color='lightblue'),
        name='Image',
        text=[f"{labels[i]}<br>{captions.get(labels[i], '')}" for i in range(len(labels))],
        hovertemplate='<b>Image</b><br>%{text}<extra></extra>'
    ))
    
    # Add text points
    fig3.add_trace(go.Scatter(
        x=proj_txt_joint[:, 0],
        y=proj_txt_joint[:, 1],
        mode='markers',
        marker=dict(symbol='diamond', size=6, color='lightcoral'),
        name='Text',
        text=[f"{labels[i]}<br>{captions.get(labels[i], '')}" for i in range(len(labels))],
        hovertemplate='<b>Text</b><br>%{text}<extra></extra>'
    ))
    
    fig3.update_layout(
        template="plotly_dark",
        title="Joint Embeddings with Connections",
        xaxis_title=f"{method} 1",
        yaxis_title=f"{method} 2"
    )
    
else:
    # Regular scatter plot for other methods
    fig3 = px.scatter(
        df_joint,
        x=f"{method} 1",
        y=f"{method} 2",
        color="Category",
        symbol="Type",
        hover_data=["Filename", "Caption"],
        title=f"Joint Embeddings ({joint_method})"
    )
    fig3.update_layout(template="plotly_dark")

st.plotly_chart(fig3, use_container_width=True)

# --- Similarity Analysis ---
st.subheader("📊 Image-Text Similarity Analysis")

# Calculate cosine similarities between corresponding pairs
similarities = []
for i in range(len(image_embeds)):
    sim = 1 - cosine(image_embeds[i], text_embeds[i])
    similarities.append(sim)

# Display statistics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Average Similarity", f"{np.mean(similarities):.3f}")
with col2:
    st.metric("Min Similarity", f"{np.min(similarities):.3f}")
    
with col3:
    st.metric("Max Similarity", f"{np.max(similarities):.3f}")

# Show similarity distribution
fig_hist = px.histogram(
    x=similarities,
    nbins=30,
    title="Distribution of Image-Text Cosine Similarities",
    labels={'x': 'Cosine Similarity', 'y': 'Count'}
)
fig_hist.update_layout(template="plotly_dark")
st.plotly_chart(fig_hist, use_container_width=True)

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
selected_idx = labels.index(selected)

col1, col2 = st.columns(2)
with col1:
    st.image(os.path.join(DATA_FOLDER, selected), caption=selected, use_column_width=True)
with col2:
    st.markdown(f"**Caption**: {captions.get(selected, 'No caption found')}")
    st.markdown(f"**Image-Text Similarity**: {similarities[selected_idx]:.3f}")
    
    # Show position in joint embedding
    if joint_method != "Connected Pairs":
        st.markdown(f"**Joint Embedding Position**:")
        st.markdown(f"- Image: ({proj_img_joint[selected_idx, 0]:.2f}, {proj_img_joint[selected_idx, 1]:.2f})")
        st.markdown(f"- Text: ({proj_txt_joint[selected_idx, 0]:.2f}, {proj_txt_joint[selected_idx, 1]:.2f})")
        
        # Calculate distance between image and text projections
        distance = np.linalg.norm(proj_img_joint[selected_idx] - proj_txt_joint[selected_idx])
        st.markdown(f"- **2D Distance**: {distance:.3f}")