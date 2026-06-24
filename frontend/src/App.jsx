import { useEffect, useMemo, useState } from 'react';
import HomePage from './components/HomePage';
import DocumentEditorPage from './components/DocumentEditorPage';
import LoginPage from './components/LoginPage';
import SignupPage from './components/SignupPage';
import { useAuth } from './context/AuthContext';
import { createDocument, getDocument, listDocuments, runAgentAction, updateDocument } from './api/client';
import './styles.css';

function getRouteFromPath(pathname) {
  if (pathname === '/login') return { name: 'login', docId: null };
  if (pathname === '/signup') return { name: 'signup', docId: null };
  const match = pathname.match(/^\/document\/([^/]+)$/);
  if (match) {
    return { name: 'document', docId: decodeURIComponent(match[1]) };
  }
  return { name: 'home', docId: null };
}

export default function App() {
  const { user, loading: authLoading, logout } = useAuth();
  const [documents,   setDocuments]   = useState([]);
  const [message,     setMessage]     = useState('');
  const [route,       setRoute]       = useState(() => getRouteFromPath(window.location.pathname));
  const [chatHint,    setChatHint]    = useState(null);
  const [activeDoc,   setActiveDoc]   = useState(null);
  const [userMenuOpen, setUserMenuOpen] = useState(false);

  async function refreshDocuments() {
    try {
      const docs = await listDocuments();
      setDocuments(docs);
    } catch {
      // backend may not be running yet
    }
  }

  useEffect(() => {
    if (user) refreshDocuments();
  }, [user]);

  useEffect(() => {
    function onPopState() {
      setRoute(getRouteFromPath(window.location.pathname));
    }
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, []);

  function navigate(pathname) {
    window.history.pushState({}, '', pathname);
    setRoute(getRouteFromPath(pathname));
  }

  useEffect(() => {
    if (authLoading) return;
    if (!user && route.name !== 'login' && route.name !== 'signup') {
      navigate('/login');
    } else if (user && (route.name === 'login' || route.name === 'signup')) {
      navigate('/');
    }
  }, [authLoading, user, route.name]);

  async function handleNewDocument() {
    try {
      const doc = await createDocument({ title: 'Untitled Document', content: { sections: [] } });
      await refreshDocuments();
      openDocument(doc);
    } catch (e) {
      setMessage(e?.message || 'Could not create new document');
    }
  }

  function openDocument(doc, hint) {
    setChatHint(hint || null);
    setActiveDoc(doc || null);
    navigate(`/document/${encodeURIComponent(doc.id)}`);
  }

  const currentDoc = useMemo(
    () => {
      if (route.name !== 'document') return null;
      if (activeDoc && String(activeDoc.id) === String(route.docId)) {
        return activeDoc;
      }
      return documents.find((doc) => String(doc.id) === String(route.docId)) ?? null;
    },
    [activeDoc, documents, route.docId, route.name]
  );

  useEffect(() => {
    if (route.name !== 'document' || !route.docId) return;
    if (currentDoc && String(currentDoc.id) === String(route.docId)) return;

    let cancelled = false;
    (async () => {
      try {
        const doc = await getDocument(route.docId);
        if (!cancelled) {
          setActiveDoc(doc);
        }
      } catch {
        if (!cancelled) {
          setMessage('Could not open document');
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [route.name, route.docId, currentDoc]);

  async function runAction(action, payload) {
    if (!currentDoc) return;
    try {
      await runAgentAction(currentDoc.id, action, payload);
      await refreshDocuments();
      setMessage(`\u2713 ${action}`);
    } catch (e) {
      setMessage(e.message);
    }
  }

  if (authLoading) {
    return <div className="auth-page"><div className="auth-loading">Loading\u2026</div></div>;
  }

  if (!user) {
    return route.name === 'signup' ? (
      <SignupPage onNavigateLogin={() => navigate('/login')} />
    ) : (
      <LoginPage onNavigateSignup={() => navigate('/signup')} />
    );
  }

  if (route.name === 'login' || route.name === 'signup') {
    return null;
  }

  const initials = (user.username || '?').slice(0, 2).toUpperCase();

  return (
    <div className={`app-root${route.name === 'document' ? ' app-root--document' : ''}`}>

      {/* ── Header ─────────────────────────────────────────────── */}
      <header className="wps-header">
        <div className="wps-brand">
          <span className="wps-logo-w">W</span>
          <span className="wps-brand-name">DAL Word</span>
        </div>
        <div className="wps-searchbar">
          <span className="srch-icon">&#9909;</span>
          <input className="srch-input" placeholder="Search files and tools" readOnly />
        </div>
        <div className="wps-header-right">
          {message && <span className="status-pill">{message}</span>}
          <span className="header-icon-btn" title="Headphones">&#127911;</span>
          <span className="header-icon-btn" title="Settings">&#9881;</span>
          <div className="user-menu">
            <button className="user-avatar" onClick={() => setUserMenuOpen((open) => !open)}>
              {initials}
            </button>
            {userMenuOpen && (
              <div className="user-menu-dropdown">
                <div className="user-menu-name">{user.username}</div>
                <button
                  className="user-menu-logout"
                  onClick={() => {
                    setUserMenuOpen(false);
                    logout();
                  }}
                >
                  Log out
                </button>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* ── Tab bar ────────────────────────────────────────────── */}
      <div className="wps-tabbar">
        <button
          className={`wtab wtab--home${route.name === 'home' ? ' wtab--active' : ''}`}
          onClick={() => navigate('/')}
        >
          Home
        </button>
        {route.name === 'document' && currentDoc && (
          <button className="wtab wtab--doc wtab--active">
            <span className="wtab-ficon">W</span>
            <span className="wtab-label">{currentDoc.title}</span>
          </button>
        )}
        <button className="wtab-new" onClick={handleNewDocument}>+ New</button>
        <span className="tabbar-spacer" />
        <button className="upgrade-now-btn">&#9889; <span>Upgrade Now</span></button>
      </div>

      {/* ── View ───────────────────────────────────────────────── */}
      {route.name === 'home' ? (
        <HomePage
          documents={documents}
          onOpenDocument={openDocument}
          onNewDocument={handleNewDocument}
          onRefresh={refreshDocuments}
          onImportFile={async ({ title, sections }) => {
            try {
              const doc = await createDocument({ title, content: { sections } });
              await refreshDocuments();
              openDocument(doc);
            } catch (e) {
              setMessage(e?.message || 'Could not import file');
            }
          }}
          onNewFromTemplate={async (tpl) => {
            try {
              const doc = await createDocument({ title: tpl.label, content: { sections: [] } });
              await refreshDocuments();
              openDocument(doc);
            } catch (e) {
              setMessage(e?.message || 'Could not create document');
            }
          }}
        />
      ) : (
        <DocumentEditorPage
          document={currentDoc}
          initialChatHint={chatHint}
          onBackHome={() => navigate('/')}
          onGenerateOutline={(topic) => runAction('generate_outline', { topic })}
          onEnhanceSection={(query) => runAction('enhance_section', { query, instruction: 'Improve academic clarity' })}
          onGenerateImage={(query) => runAction('generate_image', {
            query,
            prompt: `${query} conceptual framework diagram with clear stages and outcome`,
          })}
          onGenerateChart={(query) => runAction('generate_chart', {
            query,
            series: [1.8, 2.6, 3.9, 4.7, 5.4],
            chart_type: 'line',
            title: `${query} trend overview`,
          })}
          onGenerateDissertation={(topic) => runAction('generate_dissertation', { topic })}
          onManualSave={async (sections) => {
            if (!currentDoc) return;
            const nextContent = {
              ...(currentDoc.content || {}),
              sections,
            };
            await updateDocument(currentDoc.id, { content: nextContent });
            await refreshDocuments();
            setMessage('✓ manual save');
          }}
          onDocumentChanged={refreshDocuments}
        />
      )}
    </div>
  );
}
