# DIRO.ai — AI-Powered Material Generation Assistant

DIRO.ai is an AI-enabled educational content generation platform that helps create structured learning material such as **question papers**, **worksheets**, and **lesson plans** from selected grade/chapter inputs.

It combines a **React + TypeScript frontend** with a **FastAPI backend** and a **RAG-style pipeline** to generate high-quality, context-aware academic content.

---

## 🚀 Key Features

- Grade-wise content generation workflow
- Chapter-based material generation (single/multi chapter support)
- Material type selection:
  - Question Paper
  - Worksheet
  - Lesson Plan
- Difficulty selection (where applicable)
- Stream support for higher grades
- Maximum marks customization for question papers
- Real-time generation progress updates (SSE/EventSource)
- Export generated output to:
  - PDF
  - DOCX

---

## 🧠 Tech Stack

### Frontend
- React
- TypeScript
- Vite
- CSS

### Backend
- Python
- FastAPI
- Uvicorn
- Pydantic

### AI / Processing Layer
- Retrieval-Augmented workflow (RAG-style)
- LLM inference integration
- PDF ingestion & vectorization utilities

---

## 🏗️ High-Level Architecture

1. User selects grade, chapters, material type, and difficulty in frontend UI.
2. Frontend sends request to backend API (`/api/generate` or `/api/generate_stream`).
3. Backend pipeline:
   - resolves chapter context
   - retrieves relevant academic data
   - prepares prompt
   - invokes LLM inference
4. Generated material is returned to frontend.
5. User can export the generated output as PDF or DOCX.

---

## 📁 Project Structure

```text
DIRO-ai/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── rag_pipeline.py
│   │   ├── pdf_ingest.py
│   │   ├── auto_vectorizer.py
│   │   ├── file_watcher.py
│   │   ├── deepseek_infer.py
│   │   ├── export.py
│   │   ├── models.py
│   │   └── utils.py
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── api.ts
│   │   └── components/
│   ├── package.json
│   └── .env.example
└── README.md
