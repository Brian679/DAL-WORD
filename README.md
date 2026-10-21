# DAL Word MVP

This repository contains a practical MVP scaffold for an AI-assisted dissertation editor.

## MVP Architecture

- Frontend: React + TipTap editor + AI sidebar
- Backend API: Django REST (`documents`, `agent`)
- Async orchestration: Celery task queue for background jobs
- Asset generation: Matplotlib chart rendering and image placeholders
- Document model: JSON-first document storage for section-aware editing

## Repository Layout

- `backend/`
- `frontend/`

## Backend Features Implemented

- `Document` model with JSON content and version snapshots
- Document CRUD API (`/api/documents/`)
- Stripe billing (`/api/billing/`): checkout, customer portal, webhooks, monthly AI request limits
- Tool-based agent actions (`/api/agent/<doc_id>/action/`):
  - `generate_outline`
  - `enhance_section`
  - `generate_chart`
  - `generate_image`
  - `insert_section`

## Frontend Features Implemented

- Three-pane layout:
  - left: document list
  - center: editor
  - right: AI action panel
- TipTap editor for text editing
- Buttons for outline generation, section enhancement, image/chart insertion

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

## Payments (Stripe)

Plans: **Free** (30 AI requests a month), **Pro** monthly or yearly, and a one-off **Thesis Pass** (30 days of Pro). Pro allows 1,000 AI requests a month. All limits are set in `.env`. AI detection, the plagiarism check and the quality check use AI requests only when run from chat.

1. In the Stripe dashboard (start in test mode), create a product called "DAL Word Pro" with two recurring prices ($12/month, $96/year). Create a product called "Thesis Pass" with a one-time $29 price. Copy the three `price_...` IDs into `STRIPE_PRICE_PRO_MONTHLY`, `STRIPE_PRICE_PRO_YEARLY` and `STRIPE_PRICE_THESIS_PASS`.
2. Put the secret key in `STRIPE_SECRET_KEY` and set `FRONTEND_URL` to where the app runs.
3. Add a webhook endpoint at `https://<your-domain>/api/billing/webhook/` for these events: `checkout.session.completed`, `customer.subscription.created`, `customer.subscription.updated` and `customer.subscription.deleted`. Put its signing secret in `STRIPE_WEBHOOK_SECRET`.
4. Turn on the customer portal (Settings → Billing → Customer portal) so users can update their card or cancel.
5. Run `python manage.py migrate`.

For local testing, run `stripe listen --forward-to localhost:8000/api/billing/webhook/` and use the `whsec_...` it prints. Pay with test card `4242 4242 4242 4242`.

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
