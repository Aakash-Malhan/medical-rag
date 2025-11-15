import os
import pickle

import faiss
import gradio as gr
import google.generativeai as genai
import numpy as np
from sentence_transformers import SentenceTransformer

# Config

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "faiss_index.bin"
DOCS_PATH = "docs.pkl"
TOP_K = 5

DISCLAIMER_TEXT = (
    "⚠️ **Important:** This tool provides educational summaries of medical literature "
    "about COVID-19. It is **not** a substitute for professional medical advice, diagnosis, "
    "or treatment. Always consult a licensed healthcare professional for medical decisions. "
    "If you think you may have a medical emergency, call your local emergency number immediately."
)

# Gemini setup

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set. "
        "Add it as a secret in your Hugging Face Space settings."
    )

genai.configure(api_key=GEMINI_API_KEY)

llm = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    system_instruction=(
        "You are a medical literature assistant focusing on COVID-19 research.\n"
        "You ONLY summarize and explain information that is present in the provided documents.\n"
        "You MUST NOT give personal medical advice, diagnoses, treatment plans, or drug dosages.\n"
        "Speak in general, educational terms and encourage consultation with licensed clinicians.\n"
        "Always end your answer with this exact sentence:\n"
        "\"This is an educational summary, not medical advice. Please consult a licensed healthcare "
        "professional for decisions about diagnosis or treatment.\""
    ),
)

# Load index + docs

if not os.path.exists(INDEX_PATH) or not os.path.exists(DOCS_PATH):
    raise FileNotFoundError(
        "faiss_index.bin and/or docs.pkl not found in the repo. "
        "Upload the files generated from your Colab notebook."
    )

index = faiss.read_index(INDEX_PATH)

with open(DOCS_PATH, "rb") as f:
    docs = pickle.load(f)

embed_model = SentenceTransformer(EMBED_MODEL_NAME)


def retrieve_documents(query: str, top_k: int = TOP_K):
    """Retrieve top_k most similar documents from FAISS index."""
    query_emb = embed_model.encode(query, convert_to_numpy=True)
    query_emb = query_emb.astype("float32").reshape(1, -1)
    faiss.normalize_L2(query_emb)

    scores, indices = index.search(query_emb, top_k)
    hits = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(docs):
            continue
        doc = docs[int(idx)]

        hits.append(
            {
                "score": float(score),
                "text": doc["text"],
                "title": doc.get("title", ""),
                "journal": doc.get("journal", ""),
                "publish_time": doc.get("publish_time", ""),
                "doi": doc.get("doi", ""),
                "source": doc.get("source", "CORD-19"),
                "id": doc.get("id", int(idx)),
            }
        )
    return hits


def generate_answer(question: str):
    question = question.strip()
    if not question:
        return "Please enter a question.", "_No sources retrieved yet._"

    hits = retrieve_documents(question, top_k=TOP_K)
    if not hits:
        return (
            "I could not retrieve any relevant passages from the literature. "
            "Try rephrasing your question or using more general terms.",
            "_No sources found._",
        )

    # Build context for LLM
    context_blocks = []
    sources_md = []
    for i, h in enumerate(hits, start=1):
        context_blocks.append(f"[{i}] {h['text']}")
        title = h["title"] or "Untitled article"
        journal = h["journal"] or "Unknown journal"
        year = str(h["publish_time"]) or "Unknown year"
        doi = h["doi"]
        sim = h["score"]

        source_line = f"**[{i}]** {title} — *{journal}* ({year})  \nSimilarity: `{sim:.3f}`"
        if doi and str(doi).lower() != "nan":
            source_line += f"  \nDOI: `{doi}`"

        sources_md.append(source_line)

    context_str = "\n\n".join(context_blocks)
    sources_md_str = "\n\n---\n\n".join(sources_md)

    prompt = f"""
User question:
{question}
Relevant excerpts from COVID-19 research articles:
{context_str}
Instructions:
- Use ONLY the information from the excerpts above.
- Provide a high-level, educational explanation (3–6 short paragraphs).
- DO NOT provide personal medical advice, diagnoses, treatment plans, or dosages.
- You may mention risk factors, complications, or high-level treatment themes only if supported by the text.
- Avoid telling the user what they personally should do.
- At the very end, include this exact sentence:
"This is an educational summary, not medical advice. Please consult a licensed healthcare professional for decisions about diagnosis or treatment."
"""

    try:
        response = llm.generate_content(prompt)
        answer_text = response.text
    except Exception as e:
        answer_text = (
            f"An error occurred while calling Gemini: `{e}`\n\n"
            "Please check that your GEMINI_API_KEY is valid and that the Space can access gemini-2.0-flash."
        )

    return answer_text, sources_md_str


# Gradio UI

def answer_interface(question):
    answer, sources = generate_answer(question)
    return answer, sources


with gr.Blocks(title="Medical RAG on COVID-19 Literature (Educational Only)") as demo:
    gr.Markdown("# Misdiagnosis Risk Reduction via Medical RAG – COVID-19")
    gr.Markdown(DISCLAIMER_TEXT)

    with gr.Row():
        with gr.Column(scale=3):
            question_box = gr.Textbox(
                label="Ask a question about COVID-19 (educational only)",
                placeholder=(
                    "Example: What risk factors are associated with severe COVID-19 "
                    "leading to ICU admission?"
                ),
                lines=4,
            )
            submit_btn = gr.Button("Search medical literature")
            answer_box = gr.Markdown(label="Literature-based answer")
        with gr.Column(scale=2):
            sources_box = gr.Markdown(
                label="Top retrieved sources",
                value="_No sources yet. Ask a question to see which articles were used._",
            )

    submit_btn.click(
        fn=answer_interface,
        inputs=question_box,
        outputs=[answer_box, sources_box],
    )
    question_box.submit(
        fn=answer_interface,
        inputs=question_box,
        outputs=[answer_box, sources_box],
    )

    gr.Markdown(
        "This demo uses a Retrieval-Augmented Generation (RAG) pipeline over a subset of the "
        "CORD-19 research corpus: FAISS + sentence-transformers for retrieval and Gemini 2.0 Flash "
        "for grounded summarization."
    )

if __name__ == "__main__":
    demo.launch()
