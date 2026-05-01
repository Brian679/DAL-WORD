import { useMemo, useRef, useState } from 'react';
import {
  ArrowLeft,
  Bold,
  Italic,
  Underline,
  AlignLeft,
  AlignCenter,
  AlignRight,
  List,
  ListOrdered,
  Check,
  Globe,
  Play,
  Menu,
  ZoomOut,
  ZoomIn,
  // AI panel icons
  Plus,
  Settings2,
  RotateCcw,
  Copy,
  ThumbsDown,
  Send,
} from 'lucide-react';
import { chatWithDocument } from '../api/client';

const sampleParagraph = `An analysis of revenue streams focusing on rates as the main source of income at city level.

Local authorities rely heavily on rates as a basic fiscal tool for service delivery and investment. This section discusses financial sustainability and limitations in over-reliance on property tax collections.`;

const INITIAL_MESSAGES = [];
const INITIAL_CHAT_ID = 'chat-initial';
const INITIAL_CHATS = [{ id: INITIAL_CHAT_ID, name: 'New Chat', messages: INITIAL_MESSAGES }];

function normalizeStep(step = '') {
  return step.replace(/^[-\s]+/, '').trim();
}

function buildSummaryFromResult(result = {}) {
  if (result.orchestration?.todo_required === false) {
    return null;
  }
  if (result.chat_summary) {
    return result.chat_summary;
  }
  const plan = Array.isArray(result.plan) ? result.plan : [];
  const todoList = plan.map((item) => normalizeStep(item.step || '')).filter(Boolean);
  const completed = plan.filter((item) => item.status === 'done').length;
  const pending = Math.max(todoList.length - completed, 0);
  const completionPercent = todoList.length ? Math.round((completed / todoList.length) * 100) : 0;
  const nextTasks = plan
    .filter((item) => item.status !== 'done')
    .map((item) => normalizeStep(item.step || ''))
    .filter(Boolean)
    .slice(0, 6);
  return {
    stage: result.document_updated ? 'Completed and applied to document' : 'Completed',
    intent: result.intent || 'chat',
    todo_list: todoList,
    completion_percent: completionPercent,
    tasks_completed: completed,
    tasks_pending: pending,
    next_tasks: nextTasks,
    done_brief: todoList.filter((_, idx) => idx < 3).join(', ') || 'No task details',
  };
}

function flattenSections(content) {
  if (!content?.sections?.length) return sampleParagraph;
  return content.sections
    .map((s) => `${s.title}\n${s.content || ''}`)
    .join('\n\n');
}

export default function DocumentEditorPage({
  document,
  onBackHome,
  onGenerateOutline,
  onEnhanceSection,
  onGenerateImage,
  onGenerateChart,
  onGenerateDissertation,
  onDocumentChanged,
}) {
  const [chats,        setChats]        = useState(INITIAL_CHATS);
  const [activeChatId, setActiveChatId] = useState(INITIAL_CHAT_ID);
  const [showChatList, setShowChatList] = useState(false);
  const [inputValue,   setInputValue]   = useState('');
  const [isThinking,   setIsThinking]   = useState(false);
  const [selectedModel, setSelectedModel] = useState('gemini');
  const [activeModel,  setActiveModel]  = useState('Gemini 1.5 Flash');
  const bottomRef = useRef(null);

  const activeChat = useMemo(
    () => chats.find((chat) => chat.id === activeChatId) || chats[0],
    [chats, activeChatId]
  );
  const messages = activeChat?.messages || [];

  const docBody   = useMemo(() => flattenSections(document?.content), [document]);
  const wordCount = useMemo(() => {
    const clean = docBody.replace(/\s+/g, ' ').trim();
    return clean ? clean.split(' ').length : 0;
  }, [docBody]);

  function chatNameFromMessage(text) {
    const trimmed = text.trim();
    if (!trimmed) return 'New Chat';
    return trimmed.length > 46 ? `${trimmed.slice(0, 46)}...` : trimmed;
  }

  function createNewChat() {
    const id = `chat-${Date.now()}`;
    setChats((prev) => [{ id, name: 'New Chat', messages: [] }, ...prev]);
    setActiveChatId(id);
    setShowChatList(false);
    setInputValue('');
  }

  async function sendMessage(text) {
    if (!text.trim() || isThinking) return;
    const userText = text.trim();
    const userMsg = { id: Date.now(), role: 'user', text: text.trim() };
    setChats((prev) =>
      prev.map((chat) => {
        if (chat.id !== activeChatId) return chat;
        const isFirstMessage = chat.messages.length === 0;
        return {
          ...chat,
          name: isFirstMessage ? chatNameFromMessage(userText) : chat.name,
          messages: [...chat.messages, userMsg],
        };
      })
    );
    setInputValue('');
    setIsThinking(true);
    try {
      const result = await chatWithDocument(document?.id, userText, selectedModel);
      if (result?.model) {
        setActiveModel(result.model);
      }
      setChats((prev) =>
        prev.map((chat) =>
          chat.id === activeChatId
            ? {
                ...chat,
                messages: [
                  ...chat.messages,
                  {
                    id: Date.now() + 1,
                    role: 'assistant',
                    text: result.reply,
                    summary: buildSummaryFromResult(result),
                    plan: Array.isArray(result.plan) ? result.plan : [],
                  },
                ],
              }
            : chat
        )
      );
      if (result.document_updated) {
        onDocumentChanged?.();
      }
    } catch (err) {
      setChats((prev) =>
        prev.map((chat) =>
          chat.id === activeChatId
            ? {
                ...chat,
                messages: [...chat.messages, { id: Date.now() + 1, role: 'assistant', text: `Error: ${err.message}` }],
              }
            : chat
        )
      );
    } finally {
      setIsThinking(false);
      setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: 'smooth' }), 50);
    }
  }

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage(inputValue);
    }
  }

  function quickAction(label) {
    sendMessage(label);
  }

  return (
    <div className="doc-editor-root">
    <div className="doc-page-root">
      {/* ── Left column: ribbon + paper ── */}
      <div className="doc-left-col">
        {/* Ribbon */}
        <div className="doc-ribbon-shell">
          <div className="doc-ribbon-row doc-ribbon-row--tabs">
            <button className="doc-back-btn" onClick={onBackHome}>
              <ArrowLeft size={16} /> Home
            </button>
            <span className="doc-title-pill">{document?.title || 'Document'}.docx</span>
            <span className="doc-ribbon-tab doc-ribbon-tab--active">Home</span>
            <span className="doc-ribbon-tab">Insert</span>
            <span className="doc-ribbon-tab">Page Layout</span>
            <span className="doc-ribbon-tab">References</span>
            <span className="doc-ribbon-tab">Review</span>
            <span className="doc-ribbon-tab">View</span>
            <span className="doc-ribbon-tab">Tools</span>
            <span className="doc-ribbon-tab">WPS AI</span>
          </div>
          <div className="doc-ribbon-row doc-ribbon-row--tools">
            <div className="doc-tool-group">
              <button className="doc-tool-btn"><Bold size={15} /></button>
              <button className="doc-tool-btn"><Italic size={15} /></button>
              <button className="doc-tool-btn"><Underline size={15} /></button>
            </div>
            <div className="doc-tool-group">
              <button className="doc-tool-btn"><AlignLeft size={15} /></button>
              <button className="doc-tool-btn"><AlignCenter size={15} /></button>
              <button className="doc-tool-btn"><AlignRight size={15} /></button>
            </div>
            <div className="doc-tool-group">
              <button className="doc-tool-btn"><List size={15} /></button>
              <button className="doc-tool-btn"><ListOrdered size={15} /></button>
            </div>
          </div>
        </div>

        {/* Paper */}
        <section className="doc-paper-zone">
          <div className="doc-paper">
            <h1>{document?.title || 'Untitled Document'}</h1>
            {docBody.split('\n\n').map((block, i) => (
              <p key={`${i}-${block.slice(0, 10)}`}>{block}</p>
            ))}
          </div>
        </section>

      </div>

      {/* ── Right column: full-height AI chat ── */}
      <aside className="doc-ai-panel">

        {/* ── Panel header ── */}
        <div className="dap-header">
          <div className="dap-header-left">
            <button type="button" className="dap-head-icon-btn" onClick={createNewChat} title="New Chat">
              <Plus size={16} className="dap-icon-btn" />
            </button>
            <span className="dap-title">CHAT</span>
          </div>
          <div className="dap-header-right">
            <Settings2 size={15} className="dap-icon-btn" />
            <span className="dap-icon-btn dap-dots">···</span>
            <span className="dap-icon-btn">⤢</span>
          </div>
        </div>

        {/* ── Back row ── */}
        <div className="dap-back-row">
          <button className="dap-back-btn" onClick={() => setShowChatList((prev) => !prev)}>
            <ArrowLeft size={14} />
            <span className="dap-back-label">
              {showChatList ? 'Chats' : (activeChat?.name || 'New Chat')}
            </span>
          </button>
        </div>

        {/* ── Messages area ── */}
        <div className="dap-messages">
          {showChatList ? (
            <div className="dap-chat-list">
              {chats.map((chat) => (
                <button
                  key={chat.id}
                  type="button"
                  className={`dap-chat-item${chat.id === activeChatId ? ' dap-chat-item--active' : ''}`}
                  onClick={() => {
                    setActiveChatId(chat.id);
                    setShowChatList(false);
                  }}
                >
                  <span className="dap-chat-name">{chat.name}</span>
                  <span className="dap-chat-meta">{chat.messages.length} message{chat.messages.length === 1 ? '' : 's'}</span>
                </button>
              ))}
            </div>
          ) : (
            messages.map((msg) =>
              msg.role === 'assistant' ? (
                <div key={msg.id} className="dap-msg dap-msg--ai">
                  <div className="dap-msg-body">
                    {msg.summary ? (
                      <div className="dap-summary-card">
                        <p className="dap-summary-row"><strong>Stage:</strong> {msg.summary.stage}</p>
                        <p className="dap-summary-row"><strong>Done (brief):</strong> {msg.summary.done_brief}</p>
                        <p className="dap-summary-row"><strong>To-do list:</strong> {msg.summary.todo_list?.length || 0} task(s)</p>
                        <p className="dap-summary-row"><strong>Completion:</strong> {msg.summary.completion_percent || 0}%</p>
                        <p className="dap-summary-row"><strong>Completed:</strong> {msg.summary.tasks_completed || 0}</p>
                        <p className="dap-summary-row"><strong>Not completed:</strong> {msg.summary.tasks_pending || 0}</p>
                        <p className="dap-summary-row"><strong>Next tasks:</strong> {(msg.summary.next_tasks || []).length ? msg.summary.next_tasks.join(', ') : 'None (all completed)'}</p>
                        {!!(msg.summary.todo_list || []).length && (
                          <div className="dap-summary-todo">
                            {((msg.plan || []).length ? (msg.plan || []).slice(0, 20) : (msg.summary.todo_list || []).slice(0, 12).map((task) => ({ step: task, status: 'pending' }))).map((item, idx) => {
                              const stepLabel = normalizeStep(item.step || '');
                              const done = item.status === 'done';
                              return (
                                <div key={`${msg.id}-todo-${idx}`} className={`dap-bullet ${done ? 'dap-bullet--done' : 'dap-bullet--pending'}`}>
                                  {done ? '✓' : '○'} {stepLabel}
                                </div>
                              );
                            })}
                          </div>
                        )}
                      </div>
                    ) : (
                      msg.text.split('\n').map((line, li) =>
                        line.startsWith('•') ? (
                          <div key={li} className="dap-bullet">{line}</div>
                        ) : line.trim() ? (
                          <p key={li}>{line}</p>
                        ) : (
                          <br key={li} />
                        )
                      )
                    )}
                  </div>
                  <div className="dap-msg-actions">
                    <button className="dap-msg-act-btn"><RotateCcw size={12} /></button>
                    <button className="dap-msg-act-btn"><Copy size={12} /></button>
                    <button className="dap-msg-act-btn"><ThumbsDown size={12} /></button>
                  </div>
                  <div className="dap-model-tag">{activeModel} · 1×</div>
                </div>
              ) : (
                <div key={msg.id} className="dap-msg dap-msg--user">
                  {msg.text}
                </div>
              )
            )
          )}

          {!showChatList && (
            <>
              <div ref={bottomRef} />
              {isThinking && (
                <div className="dap-msg dap-msg--ai dap-thinking">
                  <div className="dap-msg-body"><p>Thinking…</p></div>
                </div>
              )}
            </>
          )}

        </div>

        {/* ── Composer ── */}
        {!showChatList && (
          <div className="dap-composer">
          <textarea
            className="dap-composer-input"
            placeholder="Ask anything about this document…"
            rows={2}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKey}
          />
          <div className="dap-model-row">
            <span className="dap-model-label">Model</span>
            <select
              className="dap-model-select"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              disabled={isThinking}
            >
              <option value="gemini">Gemini</option>
              <option value="grok">Grok</option>
            </select>
          </div>
          <div className="dap-composer-footer">
            <div className="dap-composer-left" />
            <button
              className="dap-send-btn"
              onClick={() => sendMessage(inputValue)}
              disabled={!inputValue.trim() || isThinking}
            >
              <Send size={14} />
            </button>
          </div>
          </div>
        )}
      </aside>
    </div>

    {/* ── Status bar — full width, below both columns ── */}
    <footer className="doc-status-bar">
      <div className="doc-status-left">
        <span className="doc-status-item">Page: 3/53</span>
        <span className="doc-status-item">Words: {wordCount}</span>
        <button type="button" className="doc-spell-btn">
          <span className="doc-spell-indicator"><Check size={11} /></span>
          AI Spell Check
        </button>
      </div>
      <div className="doc-status-right">
        <button type="button" className="doc-status-icon-btn"><Menu size={14} /></button>
        <button type="button" className="doc-status-icon-btn"><Play size={14} /></button>
        <button type="button" className="doc-status-icon-btn"><Globe size={14} /></button>
        <span className="doc-zoom-value">80%</span>
        <ZoomOut size={12} className="doc-zoom-icon" />
        <div className="doc-zoom-track" aria-hidden="true">
          <span className="doc-zoom-fill" style={{ width: '80%' }} />
        </div>
        <ZoomIn size={12} className="doc-zoom-icon" />
      </div>
    </footer>
    </div>
  );
}
