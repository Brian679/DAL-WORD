const API_BASE = (import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000/api').replace(/\/$/, '');

async function readApiError(res, fallbackMessage) {
    try {
        const data = await res.json();
        if (typeof data?.error === 'string' && data.error.trim()) return data.error;
        if (typeof data?.detail === 'string' && data.detail.trim()) return data.detail;
        if (Array.isArray(data?.non_field_errors) && data.non_field_errors.length) {
            return data.non_field_errors.join(', ');
        }
        if (data && typeof data === 'object') {
            const firstEntry = Object.entries(data)[0];
            if (firstEntry) {
                const [field, value] = firstEntry;
                if (Array.isArray(value) && value.length) return `${field}: ${value[0]}`;
                if (typeof value === 'string') return `${field}: ${value}`;
            }
        }
    } catch {
        // Fall through to text parsing and fallback
    }

    try {
        const text = await res.text();
        if (text && text.trim()) return text.trim();
    } catch {
        // ignore
    }
    return fallbackMessage;
}

export async function listDocuments() {
    const res = await fetch(`${API_BASE}/documents/`);
    if (!res.ok) throw new Error(await readApiError(res, 'Failed to list documents'));
    return res.json();
}

export async function createDocument(payload) {
    const res = await fetch(`${API_BASE}/documents/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(await readApiError(res, 'Failed to create document'));
    return res.json();
}

export async function updateDocument(id, payload) {
    const res = await fetch(`${API_BASE}/documents/${id}/`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(await readApiError(res, 'Failed to update document'));
    return res.json();
}

export async function getDocument(id) {
    const res = await fetch(`${API_BASE}/documents/${id}/`);
    if (!res.ok) throw new Error(await readApiError(res, 'Failed to fetch document'));
    return res.json();
}

export async function extractFileText(file) {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE}/documents/extract/`, {
        method: 'POST',
        body: formData,
    });
    if (!res.ok) throw new Error(await readApiError(res, 'Failed to extract file text'));
    return res.json();
}

// Unified agent endpoint: POST /api/agent/actions/
export async function runAgentAction(docId, action, payload) {
    const res = await fetch(`${API_BASE}/agent/${docId}/action/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ doc_id: docId, action, payload }),
    });
    if (!res.ok) throw new Error(await readApiError(res, 'Agent action failed'));
    return res.json();
}

export async function chatWithDocument(
    docId,
    message,
    model = 'grok',
    file = null,
    previewOnly = false,
    options = {},
) {
    const {
        groundedResearch = false,
        verifyCitations = false,
        syntheticMode = false,
    } = options || {};

    let body;
    let headers = {};
    if (file) {
        body = new FormData();
        body.append('message', message);
        body.append('model', model);
        if (previewOnly) body.append('preview_only', 'true');
        if (groundedResearch) body.append('grounded_research', 'true');
        if (verifyCitations) body.append('verify_citations', 'true');
        if (syntheticMode) body.append('synthetic_mode', 'true');
        body.append('file', file);
        // Let browser set Content-Type with boundary for multipart
    } else {
        body = JSON.stringify({
            message,
            model,
            preview_only: previewOnly,
            grounded_research: groundedResearch,
            verify_citations: verifyCitations,
            synthetic_mode: syntheticMode,
        });
        headers['Content-Type'] = 'application/json';
    }
    const res = await fetch(`${API_BASE}/agent/${docId}/chat/`, {
        method: 'POST',
        headers,
        body,
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || 'Chat request failed');
    }
    return res.json();
}

export async function runResearchWorkflow(docId, message, topic = '') {
    const res = await fetch(`${API_BASE}/agent/${docId}/research-workflow/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, topic }),
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || 'Research workflow request failed');
    }
    return res.json();
}

export async function detectAIContent(docId, text = '') {
    const res = await fetch(`${API_BASE}/agent/${docId}/ai-detect/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(text ? { text } : {}),
    });
    if (!res.ok) throw new Error(await readApiError(res, 'AI detection failed'));
    return res.json();
}

export async function getDissertationPlan(docId, message) {
    try {
        const res = await fetch(`${API_BASE}/agent/${docId}/dissertation-plan/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message }),
        });
        if (!res.ok) return null;
        return res.json();
    } catch {
        return null;
    }
}