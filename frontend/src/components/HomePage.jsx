import { useRef, useState } from 'react';
import { 
  Clock, Star, Share2, Inbox, Cloud, Triangle, Box, Monitor, Clipboard,
  Home, Plus, ChevronDown, RefreshCw, MoreHorizontal, Image as ImageIcon, FileText, X
} from 'lucide-react';
import { extractFileText } from '../api/client';

/* ─── File Type Icon ─────────────────────────────────────────── */
function FileTypeIcon({ type = 'docx', large = false }) {
  const map = {
    docx: { label: 'W',   cls: 'ftype-docx' },
    pptx: { label: 'P',   cls: 'ftype-pptx' },
    xlsx: { label: 'S',   cls: 'ftype-xlsx' },
    pdf:  { label: 'PDF', cls: 'ftype-pdf'  },
  };
  const { label, cls } = map[type] ?? map.docx;
  return (
    <span className={`ftype-icon ${cls}${large ? ' ftype-icon--lg' : ''}`}>
      {label}
    </span>
  );
}

/* ─── Date grouping helpers ──────────────────────────────────── */
function groupByDate(docs) {
  const todayStr     = new Date().toDateString();
  const yestStr      = new Date(Date.now() - 86_400_000).toDateString();
  const groups = [
    { label: 'Today',     docs: [] },
    { label: 'Yesterday', docs: [] },
    { label: 'Earlier',   docs: [] },
  ];
  docs.forEach(doc => {
    const d = doc.updated_at ? new Date(doc.updated_at).toDateString() : '';
    if (d === todayStr)  groups[0].docs.push(doc);
    else if (d === yestStr) groups[1].docs.push(doc);
    else groups[2].docs.push(doc);
  });
  return groups.filter(g => g.docs.length > 0);
}

function timeAgo(iso) {
  if (!iso) return '';
  const ms   = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(ms / 60_000);
  if (mins < 1)  return 'Just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

/* ─── Main Component ─────────────────────────────────────────── */
/* ─── Document templates ─────────────────────────────────────── */
const TEMPLATES = [
  {
    id: 'blank', label: 'Blank Document', color: '#2563eb', abbr: 'W',
    sections: [],
  },
];

export default function HomePage({ documents, onOpenDocument, onNewDocument, onRefresh, onImportFile, onNewFromTemplate }) {
  const [selectedDoc, setSelectedDoc]   = useState(null);
  const [activeNav,   setActiveNav]     = useState('recent');
  const [checkedIds,  setCheckedIds]    = useState(new Set());
  const [showAll,     setShowAll]       = useState(false);
  const fileInputRef = useRef(null);

  function handleThisPC() {
    fileInputRef.current?.click();
  }

  async function handleFileChange(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    e.target.value = '';
    const name = file.name.replace(/\.[^.]+$/, '');
    const ext  = file.name.split('.').pop().toLowerCase();
    
    // For DOCX and PDF, send to backend for server-side extraction
    if (ext === 'docx' || ext === 'pdf') {
      try {
        const res = await extractFileText(file);
        if (res.text) {
          const chunks = res.text.split(/\n\s*\n/).map(s => s.trim()).filter(Boolean);
          let sections = chunks.map((chunk, i) => {
            const lines = chunk.split('\n');
            const title = lines.length > 1 && lines[0].length < 80 ? lines.shift() : (i === 0 ? name : `Section ${i + 1}`);
            return { title, content: lines.join('\n').trim() || chunk };
          });
          if (!sections.length) sections = [{ title: name, content: res.text }];
          onImportFile?.({ title: name, sections });
        } else {
          console.warn("Backend returned empty text for", file.name);
          onImportFile?.({ title: name, sections: [{ title: name, content: "" }] });
        }
      } catch (err) {
        console.error("Error extracting file server-side", err);
        // Create an empty doc so user can at least start
        onImportFile?.({ title: name, sections: [{ title: name, content: "Error loading file content." }] });
      }
      return;
    }

    // For local text/json reading
    const reader = new FileReader();
    reader.onload = (ev) => {
      const raw = ev.target.result;
      let sections = [];
      if (ext === 'json') {
        try {
          const parsed = JSON.parse(raw);
          sections = parsed.sections ?? parsed.content?.sections ?? [];
        } catch {
          sections = [{ title: name, content: raw }];
        }
      } else {
        // .txt or any plain text: split on blank lines → sections
        const chunks = raw.split(/\n\s*\n/).map(s => s.trim()).filter(Boolean);
        sections = chunks.map((chunk, i) => {
          const lines = chunk.split('\n');
          const title = lines.length > 1 && lines[0].length < 80 ? lines.shift() : (i === 0 ? name : `Section ${i + 1}`);
          return { title, content: lines.join('\n').trim() || chunk };
        });
        if (!sections.length) sections = [{ title: name, content: raw }];
      }
      onImportFile?.({ title: name, sections });
    };
    reader.readAsText(file);
  }

  const groups = groupByDate(documents);

  function toggleCheck(id, e) {
    e.stopPropagation();
    setCheckedIds(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }

  const navItems = [
    { id: 'recent',   label: 'Recent',   icon: <Clock size={16} /> },
    { id: 'starred',  label: 'Starred',  icon: <Star size={16} /> },
    { id: 'share',    label: 'Share',    icon: <Share2 size={16} /> },
    { id: 'received', label: 'Received', icon: <Inbox size={16} /> },
  ];
  const driveItems = [
    { id: 'wps',     label: 'WPS Drive',    icon: <Cloud size={16} /> },
    { id: 'google',  label: 'Google Drive', icon: <Triangle size={16} /> },
    { id: 'dropbox', label: 'Dropbox',      icon: <Box size={16} /> },
  ];
  const localItems = [
    { id: 'pc',      label: 'This PC',  icon: <Monitor size={16} /> },
    { id: 'desktop', label: 'Desktop',  icon: <Clipboard size={16} /> },
  ];

  return (
    <div className="home-body">
      <input
        ref={fileInputRef}
        type="file"
        accept=".txt,.json,.docx,.doc,.md,.pdf"
        style={{ display: 'none' }}
        onChange={handleFileChange}
      />

      {/* ── Narrow icon strip ─────────────────────────────────── */}
      <aside className="icon-strip">
        <button className="istrip-btn istrip-btn--active" title="Home">
          <span className="istrip-icon"><Home size={20} /></span>
          <span className="istrip-label">Home</span>
        </button>
        <button className="istrip-btn" title="New" onClick={onNewDocument}>
          <span className="istrip-icon" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}><Plus size={22} strokeWidth={1.5} /></span>
          <span className="istrip-label">New</span>
        </button>
        <button className="istrip-btn" title="WPS AI">
          <span className="istrip-ai">AI</span>
          <span className="istrip-label">WPS AI</span>
        </button>

        <div className="istrip-divider" />

        <button className="istrip-btn" title="Docs">
          <span className="istrip-app docs-app">W</span>
          <span className="istrip-label">Docs</span>
        </button>
        <button className="istrip-btn" title="Slides">
          <span className="istrip-app slides-app">P</span>
          <span className="istrip-label">Slides</span>
        </button>
        <button className="istrip-btn" title="Sheets">
          <span className="istrip-app sheets-app">S</span>
          <span className="istrip-label">Sheets</span>
        </button>
        <button className="istrip-btn" title="PDF">
          <span className="istrip-app pdf-app">PDF</span>
          <span className="istrip-label">PDF</span>
        </button>
      </aside>

      {/* ── Left nav panel ────────────────────────────────────── */}
      <nav className="nav-panel">
        <ul className="nav-list">
          {navItems.map(item => (
            <li key={item.id}>
              <button
                className={`nav-item${activeNav === item.id ? ' nav-item--active' : ''}`}
                onClick={() => setActiveNav(item.id)}
              >
                <span className="nav-glyph">{item.icon}</span>
                {item.label}
              </button>
            </li>
          ))}
        </ul>

        <div className="nav-group-label">Drive</div>
        <ul className="nav-list">
          {driveItems.map(item => (
            <li key={item.id}>
              <button className="nav-item">
                <span className="nav-glyph">{item.icon}</span>
                {item.label}
              </button>
            </li>
          ))}
        </ul>

        <div className="nav-group-label">Local</div>
        <ul className="nav-list">
          {localItems.map(item => (
            <li key={item.id}>
              <button
                className="nav-item"
                onClick={item.id === 'pc' ? handleThisPC : undefined}
              >
                <span className="nav-glyph">{item.icon}</span>
                {item.label}
              </button>
            </li>
          ))}
        </ul>

        <div className="nav-storage">
          <div className="storage-track">
            <div className="storage-fill" style={{ width: '1%' }} />
          </div>
          <div className="storage-label">15.4M of 1G used · 1%</div>
          <button className="try-cloud-btn">Try 20 GB Cloud Storage</button>
        </div>
      </nav>

      {/* ── File list ─────────────────────────────────────────── */}
      <main className="file-main">
        <div className="file-main-header">
          <h2 className="file-main-title">
            Recent
            <button className="refresh-btn" onClick={onRefresh} title="Refresh"><RefreshCw size={16} /></button>
          </h2>
        </div>

        {/* ── Template strip ──────────────────────────────────── */}
        <div className="tpl-section">
          <div className="tpl-header">
            <span className="tpl-heading">New</span>
            {TEMPLATES.length > 1 && (
              <button className="tpl-toggle" onClick={() => setShowAll(v => !v)}>
                {showAll ? 'Show less' : 'More templates'}
              </button>
            )}
          </div>
          <div className={`tpl-grid${showAll ? ' tpl-grid--all' : ''}`}>
            {(showAll ? TEMPLATES : TEMPLATES.slice(0, 5)).map(tpl => (
              <button
                key={tpl.id}
                className="tpl-card"
                onClick={() => onNewFromTemplate?.(tpl)}
                title={`New ${tpl.label}`}
              >
                <span className="tpl-card-thumb" style={{ background: tpl.color }}>
                  {tpl.abbr}
                </span>
                <span className="tpl-card-label">{tpl.label}</span>
              </button>
            ))}
          </div>
        </div>


        <div className="file-cols-header">
          <span className="fcol-check" />
          <span className="fcol-name">
            <span className="filter-all">All Types <span className="filter-caret"><ChevronDown size={14} style={{ display: 'inline', verticalAlign: 'middle' }} /></span></span>
          </span>
          <span className="fcol-location">Location</span>
          <span className="fcol-time">Last modified</span>
        </div>

        {groups.length === 0 && (
          <div className="empty-state">
            <p>No recent documents.</p>
            <button className="empty-new-btn" onClick={onNewDocument}>
              Create New Document
            </button>
          </div>
        )}

        {groups.map(({ label, docs }) => (
          <div key={label}>
            <div className="file-date-group">{label}</div>
            {docs.map(doc => (
              <div
                key={doc.id}
                className={`file-row${selectedDoc?.id === doc.id ? ' file-row--selected' : ''}`}
                onClick={() => setSelectedDoc(doc)}
                onDoubleClick={() => onOpenDocument(doc)}
              >
                <span className="fcol-check">
                  <input
                    type="checkbox"
                    checked={checkedIds.has(doc.id)}
                    onChange={(e) => toggleCheck(doc.id, e)}
                    onClick={(e) => e.stopPropagation()}
                  />
                </span>
                <span className="fcol-name">
                  <FileTypeIcon type="docx" />
                  <span className="file-row-name">{doc.title}</span>
                  <span className="file-badge-local">Local</span>
                  <span className="file-star" style={{ display: 'flex', alignItems: 'center' }}><Star size={14} /></span>
                  <span className="file-more" style={{ display: 'flex', alignItems: 'center' }}><MoreHorizontal size={14} /></span>
                </span>
                <span className="fcol-location">Documents</span>
                <span className="fcol-time">{timeAgo(doc.updated_at)}</span>
              </div>
            ))}
          </div>
        ))}
      </main>

      {/* ── Right info panel ──────────────────────────────────── */}
      {selectedDoc && (
        <aside className="info-panel">
          <div className="info-header">
            <span className="info-title">File Information</span>
            <button className="info-close" onClick={() => setSelectedDoc(null)}><X size={14} /></button>
          </div>

          <div className="info-preview">
            <FileTypeIcon type="docx" large />
          </div>

          <div className="info-filename">{selectedDoc.title}.docx</div>

          <div className="info-block">
            <div className="info-block-title">Sharing settings</div>
            <div className="info-share-row">
              <span className="info-not-shared">Not shared</span>
              <button className="info-share-btn"><Share2 size={13} style={{ marginRight: 4, verticalAlign: 'middle' }} /> Share</button>
            </div>
          </div>

          <div className="info-block">
            <div className="info-block-title">Recommendations</div>
            <button className="info-rec-btn"><ImageIcon size={14} style={{ marginRight: 6, verticalAlign: 'middle' }} />Export as Image-only PDF</button>
            <button className="info-rec-btn"><FileText size={14} style={{ marginRight: 6, verticalAlign: 'middle' }} />Export to PDF</button>
          </div>
        </aside>
      )}
    </div>
  );
}
