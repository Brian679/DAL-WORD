const API_BASE = (import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000/api').replace(/\/$/, '');

// Origin only (no /api suffix) — for building <img> src URLs against media files
// served from the backend (e.g. generated charts/images), which live outside /api.
export const API_ORIGIN = API_BASE.replace(/\/api\/?$/, '');

const TOKEN_KEY = 'dalword_auth_tokens';

function getTokens() {
    try {
        return JSON.parse(localStorage.getItem(TOKEN_KEY)) || null;
    } catch {
        return null;
    }
}

function setTokens(tokens) {
    localStorage.setItem(TOKEN_KEY, JSON.stringify(tokens));
}

function clearTokens() {
    localStorage.removeItem(TOKEN_KEY);
}

export function isAuthenticated() {
    return Boolean(getTokens()?.access);
}

function notifyLoggedOut() {
    clearTokens();
    window.dispatchEvent(new Event('auth:logout'));
}

let refreshInFlight = null;

async function refreshAccessToken() {
    const tokens = getTokens();
    if (!tokens?.refresh) return null;
    try {
        const res = await fetch(`${API_BASE}/auth/refresh/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ refresh: tokens.refresh }),
        });
        if (!res.ok) return null;
        const data = await res.json();
        const next = { access: data.access, refresh: tokens.refresh };
        setTokens(next);
        return next.access;
    } catch {
        return null;
    }
}

// Drop-in replacement for fetch() that attaches the JWT access token and
// transparently retries once via the refresh token on a 401 — callers don't
// need to know tokens exist at all.
async function apiFetch(url, options = {}) {
    const tokens = getTokens();
    const headers = new Headers(options.headers || {});
    if (tokens?.access) headers.set('Authorization', `Bearer ${tokens.access}`);

    let res = await fetch(url, { ...options, headers });
    if (res.status !== 401 || !tokens?.refresh) return res;

    if (!refreshInFlight) {
        refreshInFlight = refreshAccessToken().finally(() => {
            refreshInFlight = null;
        });
    }
    const newAccess = await refreshInFlight;
    if (!newAccess) {
        notifyLoggedOut();
        return res;
    }
    const retryHeaders = new Headers(options.headers || {});
    retryHeaders.set('Authorization', `Bearer ${newAccess}`);
    return fetch(url, { ...options, headers: retryHeaders });
}

export async function signup(username, email, password) {
    const res = await fetch(`${API_BASE}/auth/signup/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, email, password }),
    });
    if (!res.ok) throw new Error(await readApiError(res, 'Signup failed'));
    const data = await res.json();
    setTokens({ access: data.access, refresh: data.refresh });
    return data.user;
}

export async function login(username, password) {
    const res = await fetch(`${API_BASE}/auth/login/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
    });
    if (!res.ok) throw new Error(await readApiError(res, 'Login failed'));
    const data = await res.json();
    setTokens({ access: data.access, refresh: data.refresh });
    return data.user;
}

export async function getMe() {
    const res = await apiFetch(`${API_BASE}/auth/me/`);
    if (!res.ok) throw new Error(await readApiError(res, 'Failed to fetch current user'));
    return res.json();
}

export function logout() {
    clearTokens();
}

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
    const res = await apiFetch(`${API_BASE}/documents/`);
    if (!res.ok) throw new Error(await readApiError(res, 'Failed to list documents'));
    return res.json();
}

export async function createDocument(payload) {
    const res = await apiFetch(`${API_BASE}/documents/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(await readApiError(res, 'Failed to create document'));
    return res.json();
}

export async function updateDocument(id, payload) {
    const res = await apiFetch(`${API_BASE}/documents/${id}/`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(await readApiError(res, 'Failed to update document'));
    return res.json();
}

export async function getDocument(id) {
    const res = await apiFetch(`${API_BASE}/documents/${id}/`);
    if (!res.ok) throw new Error(await readApiError(res, 'Failed to fetch document'));
    return res.json();
}

// Downloads the document as a real .docx file (server-rendered with python-docx
// so headings/lists/tables/images match what's shown in the editor).
export async function exportDocumentDocx(id) {
    const res = await apiFetch(`${API_BASE}/documents/${id}/export/?as_format=docx`);
    if (!res.ok) throw new Error(await readApiError(res, 'Failed to export document'));
    const blob = await res.blob();
    const disposition = res.headers.get('Content-Disposition') || '';
    const match = disposition.match(/filename="?([^"]+)"?/);
    const filename = match ? match[1] : 'document.docx';
    return { blob, filename };
}

export async function extractFileText(file) {
    const formData = new FormData();
    formData.append('file', file);
    const res = await apiFetch(`${API_BASE}/documents/extract/`, {
        method: 'POST',
        body: formData,
    });
    if (!res.ok) throw new Error(await readApiError(res, 'Failed to extract file text'));
    return res.json();
}

// Unified agent endpoint: POST /api/agent/actions/
export async function runAgentAction(docId, action, payload) {
    const res = await apiFetch(`${API_BASE}/agent/${docId}/action/`, {
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
    const res = await apiFetch(`${API_BASE}/agent/${docId}/chat/`, {
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
    const res = await apiFetch(`${API_BASE}/agent/${docId}/research-workflow/`, {
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
    const res = await apiFetch(`${API_BASE}/agent/${docId}/ai-detect/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(text ? { text } : {}),
    });
    if (!res.ok) throw new Error(await readApiError(res, 'AI detection failed'));
    return res.json();
}

export async function checkPlagiarism(docId, text = '') {
    const res = await apiFetch(`${API_BASE}/agent/${docId}/plagiarism-check/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(text ? { text } : {}),
    });
    if (!res.ok) throw new Error(await readApiError(res, 'Plagiarism check failed'));
    return res.json();
}

export async function getDissertationPlan(docId, message) {
    try {
        const res = await apiFetch(`${API_BASE}/agent/${docId}/dissertation-plan/`, {
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

// Asks the agent to design a section plan for a non-dissertation document
// (assignment, essay, report, ...) and decide whether it's substantial
// enough to need a visible todo list before writing starts.
export async function getDocumentPlan(docId, message) {
    try {
        const res = await apiFetch(`${API_BASE}/agent/${docId}/document-plan/`, {
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