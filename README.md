# Misdiagnosis Risk Reduction via Medical RAG – COVID-19

A Retrieval-Augmented Generation (RAG) assistant that surfaces relevant COVID-19 research and generates grounded, source-linked summaries using **Gemini 2.0 Flash**.  
Deployed as a Gradio app on Hugging Face Spaces and backed by a FAISS vector store built on a curated subset of the **CORD-19** corpus.

**Demo**: https://huggingface.co/spaces/aakash-malhan/medical-rag

⚠️ **Important:** This project is for **educational and research purposes only**.  
> It does **not** provide medical advice, diagnosis, or treatment recommendations.

<img width="1919" height="940" alt="Screenshot 2025-11-15 151844" src="https://github.com/user-attachments/assets/2b117104-11db-47e6-8752-cbf36039c830" />
<img width="1677" height="877" alt="Screenshot 2025-11-15 152101" src="https://github.com/user-attachments/assets/f67dccbb-b021-4329-a85b-56842485c5bf" />
<img width="1697" height="865" alt="Screenshot 2025-11-15 152130" src="https://github.com/user-attachments/assets/e90051e3-53d1-4ad6-b015-be76c3a5e147" />


## 1. Business Problem

During the COVID-19 outbreak, clinicians and researchers faced:

- An **explosion of literature** (tens of thousands of papers in months).  
- Difficulty **keeping up with new risk factors, complications, and treatment evidence**.  
- High risk of **missing key findings** buried in papers and abstracts, which can contribute to mis-informed decisions and misdiagnosis risk.

Traditional keyword search over PDFs or PubMed requires manual triaging and reading, which is slow and error-prone when time is critical.

## 2. Project Objective

Build a **medical literature assistant** that:

- Retrieves the most relevant **peer-reviewed COVID-19 papers** for a clinical/research question.  
- Generates **concise, grounded summaries** that explicitly cite the underlying articles.  
- Enforces **safety guardrails**: educational tone only, no patient-specific advice or treatment plans.

This is positioned as a **decision-support / research helper**, not a replacement for clinicians.


## 3. Tech Stack 

    Python | Gradio | Gemini 2.0 Flash | FAISS | Sentence Transformers (all-MiniLM-L6-v2) | RAG | 


## 5. Business Impact (Prototype)

Although this is a research prototype (no clinical deployment), the design aims to:

- **Reduce manual literature search time** for a clinical question by quickly surfacing 3–5 highly relevant papers instead of paging through dozens of PubMed results.  
- **Highlight key risk factors and complications** consistently across papers, helping reduce misdiagnosis risk due to missed evidence.  
- Provide an **auditable trail of sources**, making it easier for clinicians to drill into original studies instead of relying on a black-box model.
