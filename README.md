# 🔍 Interactive Multimodal Embedding Explorer

This project is a **Streamlit-based visualization tool** built as part of **Google Summer of Code 2025** under the Intel Corporation (OpenVINO Toolkit).  
It helps you **visualize and compare CLIP-based embeddings** for images and their captions.

## 📌 Features

- Upload a dataset of **images and captions**
- Extract **CLIP embeddings** (image, text, or both)
- Project embeddings using **PCA**, **UMAP**, or **t-SNE**
- Visualize **Image**, **Text**, and **Joint** embeddings
- Explore with **interactive Plotly** scatter plots
- Download embeddings and projections
- Supports multiple **joint projection methods**:
  - Concatenated Projection
  - Averaged Pairs
  - Aligned Projections
  - Connected Pairs
- Smart `.npy`-based caching to skip recomputation if dataset remains unchanged

## 🛠 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/embedding-explorer.git
   cd embedding-explorer
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3.Make sure your image dataset is placed inside ```data/images``` and the captions are in ```data/captions1.json```.

4.Run the app:
```bash
streamlit run app.py
```

FOLDER STRUCTURE
```
.
├── app.py
├── utils/
│   └── embedding_utils.py
├── data/
│   ├── images/
│   └── captions1.json
├── embeddings/
├── projections/
├── requirements.txt
└── README.md
```
💡 Tech Stack

- Python 

- Streamlit 

- CLIP (via HuggingFace Transformers)

- PCA / UMAP / t-SNE for projection

- Plotly for interactive plots

🚀 Project Motivation

This tool was built to provide researchers and practitioners a simple way to explore how multimodal embeddings behave—both separately and jointly. It was inspired by the need to visually debug and compare image and text representations.

🙋‍♀️ Author

Aarya Pandey

GSoC 2025 Contributor @ Intel (OpenVINO Toolkit)
