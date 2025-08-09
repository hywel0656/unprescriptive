import streamlit as st
import json
import os
from sentence_transformers import SentenceTransformer, util

# Cache the model loading
@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

model = load_model()

DATA_FILE = "data/data.jsonl"

# Load Japanese sentences from jsonl
def load_japanese_sentences(path):
    if not os.path.exists(path):
        st.error(f"Data file not found: {path}")
        return []
    sentences = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            sentences.append(item["japanese"])
    return sentences

japanese_sentences = load_japanese_sentences(DATA_FILE)

st.title("🧠 Japanese to English Meaning Similarity Practice")

if not japanese_sentences:
    st.warning("No Japanese sentences found.")
    st.stop()

# Select a sentence
selected_japanese = st.selectbox("Select a Japanese sentence:", japanese_sentences)

st.markdown("### Translate the Japanese sentence into English")

user_input = st.text_input("Enter your English translation:")

if user_input:
    # Compute embeddings and cosine similarity
    embeddings = model.encode([selected_japanese, user_input], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    score = similarity.item()

    st.write(f"**Similarity score:** {score:.2f}")

    if score > 0.8:
        st.success("✅ Your translation is very close in meaning!")
    elif score > 0.6:
        st.info("🧐 Some similarity detected. Keep trying!")
    else:
        st.warning("❌ Your translation is quite different. Try again!")
