# DAL Word MVP

This repository contains a practical MVP scaffold for an AI-assisted dissertation editor.

## MVP Architecture

- Frontend: React + TipTap editor + AI sidebar
- Backend API: Django REST (`documents`, `agent`)
- Async orchestration: Celery task queue for long-running dissertation generation
- Asset generation: Matplotlib chart rendering and image placeholders
- Document model: JSON-first document storage for section-aware editing

## Repository Layout

- `backend/`
- `frontend/`

## Backend Features Implemented

- `Document` model with JSON content and version snapshots
- Document CRUD API (`/api/documents/`)
- Tool-based agent actions (`/api/agent/<doc_id>/action/`):
  - `generate_outline`
  - `enhance_section`
  - `generate_chart`
  - `generate_image`
  - `insert_section`
- Async dissertation generation endpoint (`/api/agent/<doc_id>/generate-dissertation/`)
- Celery task to generate chapters sequentially and persist progress

## Frontend Features Implemented

- Three-pane layout:
  - left: document list
  - center: editor
  - right: AI action panel
- TipTap editor for text editing
- Buttons for outline generation, section enhancement, image/chart insertion, full dissertation generation

## Quick Start

### 1) Backend setup

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```

### 2) Celery worker (separate terminal)

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
celery -A config worker --loglevel=info
```

### 3) Redis

Celery is configured for Redis (`redis://localhost:6379`).
Run Redis locally (Docker or native install) before starting worker tasks.

### 4) Frontend setup

```powershell
cd frontend
npm install
npm run dev
```

Frontend runs on `http://localhost:5173`.
Backend runs on `http://127.0.0.1:8000`.

## Example Document JSON

```json
{
  "sections": [
    {
      "title": "1. Introduction",
      "content": "..."
    },
    {
      "title": "3. Methodology",
      "content": "..."
    }
  ]
}
```

## Suggested Phase 2 Additions

- Citation manager (APA/MLA/Harvard)
- Literature review ingestion from PDFs
- Plagiarism/similarity checks
- Multi-format export (`docx`, `pdf`, `pptx`, `xlsx`)
- Real undo/redo diff viewer for AI edits
- Real LLM integration in `agent/executor.py` and task pipeline
