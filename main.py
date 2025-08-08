import streamlit as st
import json
import os
from sentence_transformers import SentenceTransformer, util

# Load sentence transformer model with caching
@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")

model = load_model()

# Load translations from file
def load_translations(file_path="data/translations.json"):
    if not os.path.exists(file_path):
        st.error("Translation file not found.")
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

translations = load_translations()
japanese_sentences = [entry["japanese"] for entry in translations]

st.title("🧠 Japanese to English Translation Practice")

if not japanese_sentences:
    st.warning("No translation data found.")
    st.stop()

selected_japanese = st.selectbox("Select a Japanese sentence:", japanese_sentences)
entry = next((e for e in translations if e["japanese"] == selected_japanese), None)

st.markdown("### 📝 Your English translation:")
user_input = st.text_input("Type your translation here:")

# Button to show the reference translation(s)
show_translation = st.button("Show correct translation")

if show_translation:
    st.markdown("### 📘 Correct Translation")
    st.write(entry["english"])
    if "alternatives" in entry:
        st.markdown("### 🔄 Alternatives")
        for alt in entry["alternatives"]:
            st.write(alt)

if user_input:
    # Prepare embeddings for user input and reference sentences
    all_refs = [entry["english"]] + entry.get("alternatives", [])
    embeddings = model.encode([user_input] + all_refs, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(embeddings[0], embeddings[1:])

    best_score = scores.max().item()
    st.write(f"**Similarity score:** {best_score:.2f}")

    if best_score > 0.8:
        st.success("✅ Very good!")
    elif best_score > 0.6:
        st.info("🧐 Not bad, but try to get closer.")
    else:
        st.warning("❌ Quite different. Try rephrasing.")
