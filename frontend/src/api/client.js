const API_BASE = 'http://127.0.0.1:8000/api';

export async function listDocuments() {
    const res = await fetch(`${API_BASE}/documents/`);
    if (!res.ok) throw new Error('Failed to list documents');
    return res.json();
}

export async function createDocument(payload) {
    const res = await fetch(`${API_BASE}/documents/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error('Failed to create document');
    return res.json();
}

export async function updateDocument(id, payload) {
    const res = await fetch(`${API_BASE}/documents/${id}/`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error('Failed to update document');
    return res.json();
}

export async function getDocument(id) {
    const res = await fetch(`${API_BASE}/documents/${id}/`);
    if (!res.ok) throw new Error('Failed to fetch document');
    return res.json();
}

export async function extractFileText(file) {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE}/documents/extract/`, {
        method: 'POST',
        body: formData,
    });
    if (!res.ok) throw new Error('Failed to extract file text');
    return res.json();
}

// Unified agent endpoint: POST /api/agent/actions/
export async function runAgentAction(docId, action, payload) {
    const res = await fetch(`${API_BASE}/agent/${docId}/action/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ doc_id: docId, action, payload }),
    });
    if (!res.ok) throw new Error('Agent action failed');
    return res.json();
}

export async function chatWithDocument(docId, message, model = 'grok', file = null) {
    let body;
    let headers = {};
    if (file) {
        body = new FormData();
        body.append('message', message);
        body.append('model', model);
        body.append('file', file);
        // Let browser set Content-Type with boundary for multipart
    } else {
        body = JSON.stringify({ message, model });
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
    // Returns { reply, plan, document_updated, intent, document? }
    return res.json();
}