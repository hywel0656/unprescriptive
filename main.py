import streamlit as st
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load or initialize translation data
DATA_FILE = "data/translations.json"
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        translations = json.load(f)
else:
    translations = []

# Load sentence transformer model
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

st.title("Translation Practice App")

if not translations:
    st.warning("No translation entries found. Please add some examples to `data/translations.json`.")
else:
    # Select a translation entry
    entry_idx = st.selectbox("Choose a translation task", range(len(translations)), format_func=lambda i: translations[i]["japanese"])
    entry = translations[entry_idx]

    # Show Japanese sentence and input box
    st.write("**Japanese:**", entry["japanese"])
    user_input = st.text_input("Your English translation:")

    if user_input:
        # Get reference sentences
        references = [entry["english"]] + entry.get("alternatives", [])
        
        # Encode embeddings properly
        user_embedding = model.encode(user_input)
        ref_embeddings = model.encode(references)

        # Ensure proper shape (2D) for cosine similarity
        user_embedding = np.array(user_embedding).reshape(1, -1)
        ref_embeddings = np.array(ref_embeddings)

        # Calculate similarity scores
        scores = cosine_similarity(user_embedding, ref_embeddings)[0]
        best_score = max(scores)

        st.write("**Similarity Score:**", f"{best_score:.2f}")

        # Show all reference scores
        with st.expander("See all reference comparisons"):
            for ref, score in zip(references, scores):
                st.write(f"`{ref}` → {score:.2f}")

        if best_score > 0.8:
            st.success("Great job! Your translation is quite similar.")
        elif best_score > 0.6:
            st.info("Not bad, but you could get closer.")
        else:
            st.warning("Your translation seems quite different.")
