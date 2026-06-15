import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ArrowLeft,
  Bold,
  Italic,
  Underline,
  Strikethrough,
  Subscript,
  Superscript,
  AlignLeft,
  AlignCenter,
  AlignRight,
  AlignJustify,
  List,
  ListOrdered,
  Check,
  Globe,
  Play,
  Menu,
  ZoomOut,
  ZoomIn,
  Scissors,
  Paintbrush,
  Eraser,
  Search,
  ChevronDown,
  Highlighter,
  Palette,
  IndentIncrease,
  IndentDecrease,
  // AI panel icons
  Plus,
  Settings2,
  RotateCcw,
  Copy,
  ThumbsDown,
  Send,
  Paperclip,
  X,
  // Other tab icons
  Table,
  Image,
  Link,
  FileText,
  Columns,
  Type,
  MessageSquare,
  StickyNote,
  BookOpen,
  SpellCheck,
  Eye,
  EyeOff,
  Maximize,
  Ruler,
  Hash,
  FileSearch,
  Printer,
  Download,
  Upload,
  LayoutTemplate,
  PanelLeft,
  RefreshCw,
  Mic,
  Star,
  Wand2,
  ChevronRight,
  Minus,
  MoreHorizontal,
  MessageCircle,
  CheckCircle2,
  ShieldCheck,
  AlertTriangle,
  ShieldAlert,
} from 'lucide-react';
import { chatWithDocument, getDocument, getDissertationPlan, detectAIContent } from '../api/client';

const sampleParagraph = `An analysis of revenue streams focusing on rates as the main source of income at city level.

Local authorities rely heavily on rates as a basic fiscal tool for service delivery and investment. This section discusses financial sustainability and limitations in over-reliance on property tax collections.`;

const INITIAL_MESSAGES = [];
const INITIAL_CHAT_ID = 'chat-initial';
const INITIAL_CHATS = [{ id: INITIAL_CHAT_ID, name: 'New Chat', messages: INITIAL_MESSAGES }];
const MIN_EDITOR_HEIGHT = 900;
const PAGE_CYCLE_PX = 1120;
const DISSERTATION_REQUEST_RE = /(full|complete|entire).{0,30}(dissertation|thesis|project)|write.{0,20}(dissertation|thesis|project)|generate.{0,20}(dissertation|thesis|project)/i;

function normalizeStep(step = '') {
  return step.replace(/^[-\s]+/, '').trim();
}

function looksLikeDissertationRequest(text = '') {
  return DISSERTATION_REQUEST_RE.test((text || '').trim());
}

function createFallbackPreviewPlan() {
  // Minimal fallback used only when the dynamic plan endpoint is unavailable.
  return [
    { step: 'Creating dissertation to-do list', status: 'done' },
    { step: 'Writing Chapter 1: Introduction', status: 'in_progress' },
    { step: 'Writing Chapter 2: Literature Review', status: 'pending' },
    { step: 'Writing Chapter 3: Methodology', status: 'pending' },
    { step: 'Writing Chapter 4: Results and Discussion', status: 'pending' },
    { step: 'Writing Chapter 5: Conclusion and Recommendations', status: 'pending' },
    { step: 'Writing Chapter 6: References and Appendices', status: 'pending' },
  ];
}

function escapeRegExp(value = '') {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function derivePlanFromDocument(previewPlan = [], sections = []) {
  const chapterMap = new Map(
    (sections || []).map((section) => [String(section?.title || '').toLowerCase(), String(section?.content || '')])
  );
  const normalized = previewPlan.map((item) => ({ ...item, status: 'pending' }));

  if (!normalized.length) return normalized;
  normalized[0].status = 'done';

  for (let idx = 1; idx < normalized.length; idx += 1) {
    const stepLabel = normalizeStep(normalized[idx].step || '');
    const chapterMatch = stepLabel.match(/^writing\s+(chapter\s*\d+:[^]+)$/i);
    const subsectionMatch = stepLabel.match(/^writing\s+(\d+\.\d+(?:\.\d+)?\s+.+)$/i);

    if (chapterMatch) {
      const chapterTitle = chapterMatch[1].toLowerCase();
      const chapterContent = chapterMap.get(chapterTitle) || '';
      if (chapterContent.trim().length > 24) {
        normalized[idx].status = 'done';
      }
      continue;
    }

    if (subsectionMatch) {
      const subsectionTitle = subsectionMatch[1];
      const chapterNoMatch = subsectionTitle.match(/^(\d+)\./);
      const chapterNo = chapterNoMatch ? chapterNoMatch[1] : null;
      const chapterEntry = Array.from(chapterMap.entries()).find(([title]) =>
        chapterNo ? title.startsWith(`chapter ${chapterNo}:`) : false
      );
      const chapterContent = chapterEntry ? chapterEntry[1] : '';
      const subsectionRe = new RegExp(`(^|\\n)${escapeRegExp(subsectionTitle)}(\\n|$)`, 'i');
      if (subsectionRe.test(chapterContent)) {
        normalized[idx].status = 'done';
      }
    }
  }

  const firstPending = normalized.findIndex((item, index) => index > 0 && item.status !== 'done');
  if (firstPending >= 0) {
    normalized[firstPending].status = 'in_progress';
    for (let idx = firstPending + 1; idx < normalized.length; idx += 1) {
      if (normalized[idx].status !== 'done') normalized[idx].status = 'pending';
    }
  }

  return normalized;
}

function buildSummaryFromPlan(plan = [], stage = 'Execution in progress') {
  const todoList = plan.map((item) => normalizeStep(item.step || '')).filter(Boolean);
  const completed = plan.filter((item) => item.status === 'done').length;
  const pending = Math.max(todoList.length - completed, 0);
  const completionPercent = todoList.length ? Math.round((completed / todoList.length) * 100) : 0;
  const nextTasks = plan
    .filter((item) => item.status === 'in_progress' || item.status === 'pending')
    .map((item) => normalizeStep(item.step || ''))
    .slice(0, 6);

  return {
    stage,
    intent: 'write_dissertation',
    todo_list: todoList,
    completion_percent: completionPercent,
    tasks_completed: completed,
    tasks_pending: pending,
    next_tasks: nextTasks,
    done_brief: todoList.filter((_, idx) => idx < completed).slice(0, 3).join(', ') || 'No task details',
  };
}

function statusLabel(step = '') {
  return normalizeStep(step).replace(/^Writing\s+/i, '');
}

function buildWorkflowFromPlan(plan = [], previous = null) {
  const current = plan.find((item) => item.status === 'in_progress');
  const completedItems = plan.filter((item) => item.status === 'done').map((item) => statusLabel(item.step || ''));
  const previousStatuses = Array.isArray(previous?.statuses) ? previous.statuses : [];
  const updates = Array.isArray(previous?.updates) ? [...previous.updates] : [];

  plan.forEach((item, idx) => {
    const prevStatus = previousStatuses[idx];
    const nowStatus = item.status;
    const label = statusLabel(item.step || '');
    if (prevStatus !== 'done' && nowStatus === 'done' && label !== 'Creating dissertation to-do list') {
      updates.unshift(`Completed: ${label}`);
    }
  });

  const prevCurrent = previous?.currentStep;
  const nowCurrent = current ? statusLabel(current.step || '') : '';
  if (nowCurrent && nowCurrent !== prevCurrent) {
    updates.unshift(`Now doing: ${nowCurrent}`);
  }

  const capped = updates.filter(Boolean).slice(0, 12);
  return {
    statuses: plan.map((item) => item.status || 'pending'),
    currentStep: nowCurrent,
    completedCount: completedItems.length,
    totalCount: plan.length,
    updates: capped,
  };
}

/* ── Inline markdown renderer ────────────────────────────────── */
function renderInline(text) {
  const parts = text.split(/(\*\*[\s\S]+?\*\*|\*[\s\S]+?\*|`[^`]+`)/g);
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**'))
      return <strong key={i}>{part.slice(2, -2)}</strong>;
    if (part.startsWith('*') && part.endsWith('*'))
      return <em key={i}>{part.slice(1, -1)}</em>;
    if (part.startsWith('`') && part.endsWith('`'))
      return <code key={i} className="dap-inline-code">{part.slice(1, -1)}</code>;
    return part;
  });
}

function MdText({ text }) {
  if (!text) return null;
  const segments = text.split(/(```[\s\S]*?```)/g);
  return (
    <>
      {segments.map((seg, si) => {
        if (seg.startsWith('```')) {
          const code = seg.replace(/^```\w*\n?/, '').replace(/\n?```$/, '');
          return (
            <div key={si} className="dap-code-block">
              <pre><code>{code}</code></pre>
            </div>
          );
        }
        const lines = seg.split('\n');
        const elements = [];
        let listBuf = [];
        let olBuf = [];
        const flushUl = () => {
          if (listBuf.length) { elements.push(<ul key={`ul${elements.length}`}>{listBuf}</ul>); listBuf = []; }
        };
        const flushOl = () => {
          if (olBuf.length) { elements.push(<ol key={`ol${elements.length}`}>{olBuf}</ol>); olBuf = []; }
        };
        lines.forEach((line, li) => {
          const hm = line.match(/^(#{1,4})\s+(.*)/);
          if (hm) {
            flushUl(); flushOl();
            const lvl = hm[1].length;
            const Tag = lvl <= 2 ? 'h3' : 'h4';
            elements.push(<Tag key={li} className="dap-md-heading">{renderInline(hm[2])}</Tag>);
          } else if (line.match(/^[-*]\s+/)) {
            flushOl();
            listBuf.push(<li key={li}>{renderInline(line.replace(/^[-*]\s+/, ''))}</li>);
          } else if (line.match(/^\d+\.\s+/)) {
            flushUl();
            olBuf.push(<li key={li}>{renderInline(line.replace(/^\d+\.\s+/, ''))}</li>);
          } else if (!line.trim()) {
            flushUl(); flushOl();
            if (elements.length && elements[elements.length - 1]?.type !== 'br')
              elements.push(<br key={`br${li}`} />);
          } else {
            flushUl(); flushOl();
            elements.push(<p key={li}>{renderInline(line)}</p>);
          }
        });
        flushUl(); flushOl();
        return <span key={si}>{elements}</span>;
      })}
    </>
  );
}

function CopilotWorkflowCard({ summary, planItems, msgId, workflow }) {
  const [expanded, setExpanded] = useState(false);
  const done = summary?.tasks_completed ?? 0;
  
  return (
    <div className="copilot-workflow-collapsible">
      <button type="button" className="copilot-collapsible-toggle" onClick={() => setExpanded(e => !e)}>
        <span className="copilot-collapsible-icon">✓</span>
        <span className="copilot-collapsible-text">Completed {done} steps</span>
        <span className="copilot-collapsible-chevron">{expanded ? '▴' : '▾'}</span>
      </button>
      
      {expanded && (
        <div className="copilot-collapsible-body">
          <DissertationPlan planItems={planItems || []} todoList={summary?.todo_list || []} msgId={msgId} chapterPlan={summary?.chapter_plan || []} />
        </div>
      )}
    </div>
  );
}


// ── Agent Todo Panel — expandable, lives just above the composer ─────────────
function AgentTodoPanel({ plan, isActive }) {
  const [expanded, setExpanded] = useState(false);
  if (!plan?.length) return null;

  const done = plan.filter((i) => i.status === 'done').length;
  const inProgress = plan.find((i) => i.status === 'in_progress');
  const total = plan.length;
  const pct = total ? Math.round((done / total) * 100) : 0;

  return (
    <div className={`agent-todo-panel${isActive ? ' agent-todo-panel--active' : ''}`}>
      <button type="button" className="agent-todo-toggle" onClick={() => setExpanded((v) => !v)}>
        <span className={`agent-todo-icon${isActive ? ' agent-todo-icon--spin' : ''}`}>
          {isActive ? '◌' : '☑'}
        </span>
        <span className="agent-todo-head">
          Todo
          {inProgress && (
            <span className="agent-todo-current">
              {' · '}{normalizeStep(inProgress.step).replace(/^Writing\s+/i, '')}
            </span>
          )}
        </span>
        <span className="agent-todo-progress">
          <span className="agent-todo-bar">
            <span className="agent-todo-bar-fill" style={{ width: `${pct}%` }} />
          </span>
          <span className="agent-todo-frac">{done}/{total}</span>
        </span>
        <span className="agent-todo-chevron">{expanded ? '▾' : '▸'}</span>
      </button>
      {expanded && (
        <div className="agent-todo-list">
          {plan.map((item, idx) => {
            const st = item.status || 'pending';
            const label = normalizeStep(item.step || '');
            return (
              <div key={idx} className={`agent-todo-item agent-todo-item--${st}`}>
                <span className={`agent-todo-dot${st === 'in_progress' ? ' agent-todo-dot--spin' : ''}`}>
                  {st === 'done' ? '✓' : st === 'in_progress' ? '◌' : '○'}
                </span>
                <span className="agent-todo-item-label">{label}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ── Agent Activity Bar — shows current subsection being written ───────────────
function AgentActivityBar({ activity }) {
  if (!activity) return null;
  return (
    <div className="agent-activity-bar">
      <span className="agent-activity-spin">◌</span>
      <span className="agent-activity-text">{activity}</span>
    </div>
  );
}

function extractChapterRows(planItems = []) {
  return planItems
    .filter((item) => /^writing chapter\s*\d+/i.test(normalizeStep(item.step || '')))
    .map((item) => ({
      title: normalizeStep((item.step || '').replace(/^writing\s+/i, '')),
      status: item.status || 'pending',
    }));
}

function renderFigureBlock(blk, key) {
  if (!blk || (blk.type !== 'image' && blk.type !== 'chart')) return null;
  return (
    <figure key={key} className="doc-figure">
      <img
        src={`http://127.0.0.1:8000${blk.src}`}
        alt={blk.caption || 'Generated image'}
        className="doc-figure-img"
        onError={(e) => {
          e.currentTarget.style.display = 'none';
          const fb = e.currentTarget.nextSibling;
          if (fb) fb.textContent = 'Image generation failed';
        }}
      />
      {blk.caption && (
        <figcaption className="doc-figure-caption">{blk.caption}</figcaption>
      )}
    </figure>
  );
}

const _UL_RE = /^[-*•◦▪]\s+/;
const _OL_RE = /^\d+[.):]\s+/;

function _isListLine(l) { return _UL_RE.test(l) || _OL_RE.test(l); }

function renderParagraph(para, key) {
  const lines = para.split('\n').map(l => l.trimEnd());
  const nonEmpty = lines.filter(l => l.trim());

  if (nonEmpty.length === 0) return null;

  // Check if this block is mostly list items (allow one intro line at top)
  const listItemLines = nonEmpty.filter(l => _isListLine(l.trimStart()));
  const hasIntro = nonEmpty.length > 1 && !_isListLine(nonEmpty[0].trimStart()) && listItemLines.length >= nonEmpty.length - 1;
  const isList = listItemLines.length >= 2 && (listItemLines.length === nonEmpty.length || hasIntro);

  if (isList) {
    const intro = hasIntro ? nonEmpty[0].trim() : null;
    const itemLines = nonEmpty.filter((l, i) => !(i === 0 && hasIntro) && l.trim());
    const isOrdered = itemLines.filter(l => _OL_RE.test(l.trimStart())).length > itemLines.filter(l => _UL_RE.test(l.trimStart())).length;
    const items = itemLines.map(l => l.trimStart().replace(_UL_RE, '').replace(_OL_RE, '').trim());
    const ListTag = isOrdered ? 'ol' : 'ul';
    const result = [];
    if (intro) result.push(<p key={`${key}-intro`} className="doc-list-intro">{intro}</p>);
    result.push(
      <ListTag key={`${key}-list`} className="doc-content-list">
        {items.map((item, i) => <li key={i}>{item}</li>)}
      </ListTag>
    );
    return result;
  }

  // Regular paragraph
  if (nonEmpty.length === 1) return <p key={key}>{para.trim()}</p>;
  return (
    <p key={key}>
      {lines.map((line, i) => (
        <span key={i}>
          {line}
          {i < lines.length - 1 && <br />}
        </span>
      ))}
    </p>
  );
}

function normalizeContentText(raw) {
  // Convert <br> / <br/> / <BR> etc. to newlines so they render correctly.
  return (raw || '').replace(/<br\s*\/?>/gi, '\n');
}

function renderContentWithMarkers(section, sectionIndex) {
  const rawContent = normalizeContentText(section?.content || '');
  const blocks = Array.isArray(section?.blocks) ? section.blocks : [];
  const parts = [];
  const markerRe = /\[\[BLOCK:([^\]]+)\]\]/g;
  let last = 0;
  let match;
  const placed = new Set();

  while ((match = markerRe.exec(rawContent)) !== null) {
    const textPart = rawContent.slice(last, match.index);
    if (textPart.trim()) {
      textPart
        .split('\n\n')
        .filter(Boolean)
        .forEach((para, idx) => {
          parts.push(renderParagraph(para.trim(), `s${sectionIndex}-t${parts.length}-${idx}`));
        });
    }

    const blockId = (match[1] || '').trim();
    const block = blocks.find((b) => (b.block_id || '').trim() === blockId);
    if (block) {
      placed.add(blockId);
      parts.push(renderFigureBlock(block, `s${sectionIndex}-b${parts.length}`));
    }
    last = markerRe.lastIndex;
  }

  const tail = rawContent.slice(last);
  if (tail.trim()) {
    tail
      .split('\n\n')
      .filter(Boolean)
      .forEach((para, idx) => {
        parts.push(renderParagraph(para.trim(), `s${sectionIndex}-tail-${idx}`));
      });
  }

  // Fallback: render any remaining blocks not referenced by marker.
  blocks.forEach((blk, idx) => {
    const blockId = (blk.block_id || '').trim();
    if (!blockId || !placed.has(blockId)) {
      parts.push(renderFigureBlock(blk, `s${sectionIndex}-fallback-${idx}`));
    }
  });

  return parts;
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

function toChatReply(result = {}) {
  const raw = typeof result?.reply === 'string' ? result.reply.trim() : '';
  if (!raw) {
    return result?.document_updated
      ? 'Update complete. The content has been written to the document.'
      : 'Done.';
  }
  // For document updates, keep chat output concise and avoid dumping long write payloads.
  if (result?.document_updated && raw.length > 1200) {
    return 'Update complete. The content has been written to the document.';
  }
  return raw;
}

function flattenSections(content) {
  if (!content?.sections?.length) return sampleParagraph;
  return content.sections
    .map((s) => `${s.title}\n${s.content || ''}`)
    .join('\n\n');
}

function cloneSections(sections = []) {
  return (Array.isArray(sections) ? sections : []).map((section) => ({
    title: section?.title || '',
    content: section?.content || '',
    blocks: Array.isArray(section?.blocks) ? section.blocks : [],
  }));
}

function sectionHash(section = {}) {
  return JSON.stringify({
    title: section?.title || '',
    content: section?.content || '',
    blocks: Array.isArray(section?.blocks) ? section.blocks : [],
  });
}

function detectEditedSections(beforeSections = [], afterSections = []) {
  const beforeMap = new Map(
    (beforeSections || []).map((section) => [
      String(section?.title || '').trim().toLowerCase(),
      sectionHash(section),
    ])
  );
  const edited = [];

  for (const section of afterSections || []) {
    const title = String(section?.title || '').trim() || 'Untitled section';
    const key = title.toLowerCase();
    const nextHash = sectionHash(section);
    const prevHash = beforeMap.get(key);
    if (!prevHash || prevHash !== nextHash) {
      edited.push(title);
    }
  }

  return Array.from(new Set(edited));
}

// ── Hierarchical dissertation plan component ─────────────────────────────
function DissertationPlan({ planItems, todoList, msgId, chapterPlan }) {
  const [expanded, setExpanded] = useState(true);
  const sourceItems = planItems.length ? planItems : todoList.map((t) => ({ step: t, status: 'pending' }));
  const chapterRows = extractChapterRows(sourceItems);

  // Build a tree from the flat plan (indented with spaces)
  const tree = buildPlanTree(sourceItems);

  return (
    <div className="dplan">
      <button
        type="button"
        className="dplan-toggle"
        onClick={() => setExpanded((v) => !v)}
      >
        {expanded ? '▾' : '▸'} Dissertation Plan ({sourceItems.length} steps)
      </button>
      {expanded && (
        <>
          {!!chapterRows.length && (
            <table className="dplan-table">
              <thead>
                <tr>
                  <th>Chapter</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {chapterRows.map((row, idx) => (
                  <tr key={`${msgId}-row-${idx}`}>
                    <td>{row.title}</td>
                    <td>
                      <span className={`dplan-chip dplan-chip--${row.status}`}>
                        {row.status === 'done' ? 'Completed' : row.status === 'in_progress' ? 'Loading...' : 'Pending'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <div className="dplan-tree">
            {tree.map((node, ni) => (
              <PlanNode key={`${msgId}-n${ni}`} node={node} depth={0} />
            ))}
          </div>
        </>
      )}
    </div>
  );
}

function buildPlanTree(flatItems) {
  // Each item's level = leading spaces / 2
  const roots = [];
  const stack = []; // [{node, level}]
  for (const item of flatItems) {
    const raw = item.step || '';
    const match = raw.match(/^(\s+)/);
    const level = match ? Math.floor(match[1].length / 2) : 0;
    const node = {
      label: normalizeStep(raw),
      status: item.status || 'pending',
      level,
      children: [],
    };
    // Pop stack until we find a parent at level-1
    while (stack.length && stack[stack.length - 1].level >= level) {
      stack.pop();
    }
    if (stack.length === 0) {
      roots.push(node);
    } else {
      stack[stack.length - 1].node.children.push(node);
    }
    stack.push({ node, level });
  }
  return roots;
}

function PlanNode({ node, depth }) {
  const [open, setOpen] = useState(depth < 1); // chapters start expanded; sub-sub collapsed
  const done = node.status === 'done';
  const inProgress = node.status === 'in_progress';
  const isChapter = depth === 0 && /^writing chapter/i.test(node.label);
  const hasChildren = node.children.length > 0;

  // Step-type icon for Copilot-style steps
  const stepIcon = (() => {
    const lbl = node.label.toLowerCase();
    if (lbl.startsWith('reading')) return '📄 ';
    if (lbl.startsWith('editing')) return '✏️ ';
    if (lbl.startsWith('saving')) return '💾 ';
    if (lbl.startsWith('identifying')) return '🔍 ';
    return '';
  })();

  return (
    <div className={`pnode pnode--d${depth}`}>
      <div
        className={`pnode-row${isChapter ? ' pnode-row--chapter' : ''} ${done ? 'pnode-row--done' : inProgress ? 'pnode-row--in-progress' : 'pnode-row--pending'}`}
        onClick={hasChildren ? () => setOpen((v) => !v) : undefined}
        style={{ cursor: hasChildren ? 'pointer' : 'default' }}
      >
        <span className={`pnode-tick${inProgress ? ' pnode-tick--spin' : ''}`}>{done ? '✓' : inProgress ? '◌' : '○'}</span>
        {hasChildren && (
          <span className="pnode-arrow">{open ? '▾' : '▸'}</span>
        )}
        <span className="pnode-label">{stepIcon}{node.label}</span>
        {hasChildren && (
          <span className="pnode-count">
            {node.children.filter((c) => c.status === 'done').length}/{node.children.length}
          </span>
        )}
      </div>
      {hasChildren && open && (
        <div className="pnode-children">
          {node.children.map((child, ci) => (
            <PlanNode key={ci} node={child} depth={depth + 1} />
          ))}
        </div>
      )}
    </div>
  );
}

export default function DocumentEditorPage({
  document,
  initialChatHint,
  onBackHome,
  onGenerateOutline,
  onEnhanceSection,
  onGenerateImage,
  onGenerateChart,
  onGenerateDissertation,
  onManualSave,
  onDocumentChanged,
}) {
  const INITIAL_CHATS_LOCAL = [{ id: INITIAL_CHAT_ID, name: 'New Chat', messages: [] }];
  const [chats,        setChats]        = useState(INITIAL_CHATS_LOCAL);
  const [activeChatId, setActiveChatId] = useState(INITIAL_CHAT_ID);
  const [showChatList, setShowChatList] = useState(false);
  const [inputValue,   setInputValue]   = useState('');
  const [attachedFile, setAttachedFile] = useState(null);
  const [groundedResearch, setGroundedResearch] = useState(false);
  const fileInputRef = useRef(null);
  const [isThinking,   setIsThinking]   = useState(false);
  const [isSavingManual, setIsSavingManual] = useState(false);
  const [isDirty,      setIsDirty]      = useState(false);
  const [autoSaved,    setAutoSaved]    = useState(false);
  const [manualError,  setManualError]  = useState('');
  const [draftSections, setDraftSections] = useState([]);
  const [selectedModel, setSelectedModel] = useState('grok');
  const [activeModel,  setActiveModel]  = useState('Grok');
  const [liveProgressMsgId, setLiveProgressMsgId] = useState(null);
  const [editorHeight, setEditorHeight] = useState(MIN_EDITOR_HEIGHT);
  const [highlightedSections, setHighlightedSections] = useState([]);
  // AI panel visibility
  const [aiPanelOpen, setAiPanelOpen] = useState(true);
  // Comments panel (shown inside the AI panel)
  const [showCommentsPanel, setShowCommentsPanel] = useState(false);
  // AI detection panel
  const [showAiDetectPanel, setShowAiDetectPanel] = useState(false);
  const [aiDetecting, setAiDetecting] = useState(false);
  const [aiDetectResult, setAiDetectResult] = useState(null);
  // Ribbon state
  const [activeRibbonTab, setActiveRibbonTab] = useState('Home');
  // View state
  const [showRuler, setShowRuler] = useState(false);
  const [showFormattingMarks, setShowFormattingMarks] = useState(false);
  const [trackChanges, setTrackChanges] = useState(false);
  const [spellCheckEnabled, setSpellCheckEnabled] = useState(true);
  const [lineSpacing, setLineSpacing] = useState('1.5');
  const [pageMarginsInch, setPageMarginsInch] = useState({
    top: '1',
    bottom: '1',
    left: '1.25',
    right: '1.25',
  });
  const imageInputRef = useRef(null);
  const [fontFamily, setFontFamily] = useState('Times New Roman');
  const [fontSize, setFontSize] = useState('12');
  const [activeFormats, setActiveFormats] = useState({});
  const fontColorInputRef = useRef(null);
  const highlightColorInputRef = useRef(null);
  const richEditorRef = useRef(null);
  const ignoreNextInputRef = useRef(false);
  const bottomRef    = useRef(null);
  const autoSaveTimer = useRef(null);
  const progressPollRef = useRef(null);
  const editorTextareaRef = useRef(null);
  const progressPollBusyRef = useRef(false);
  const workflowStateRef = useRef({});
  const aiDetectResultRef = useRef(null);

  function clearProgressPolling() {
    if (progressPollRef.current) {
      clearInterval(progressPollRef.current);
      progressPollRef.current = null;
    }
    progressPollBusyRef.current = false;
  }

  // ── Rich text helpers ──────────────────────────────────────────
  function sectionsToHtml(sections) {
    return (sections || []).map((s) => {
      let html = '';
      if (s.title) {
        let tag = 'h2';
        if (/^chapter\s+\d/i.test(s.title) || /^chapter\s+[ivxlc]/i.test(s.title)) tag = 'h1';
        else if (/^\d+\.\d+\.\d+/.test(s.title)) tag = 'h3';
        html += `<${tag} data-section-title="true">${s.title}</${tag}>`;
      }
      if (s.content) {
        const normalized = s.content.replace(/<br\s*\/?>/gi, '\n');
        const paras = normalized.split(/\n\n+/);
        html += paras.map((p) => {
          const pLines = p.split('\n').map(l => l.trimEnd()).filter(l => l.trim());
          if (pLines.length === 0) return '';
          const listItems = pLines.filter(l => _isListLine(l.trimStart()));
          const hasIntro = pLines.length > 1 && !_isListLine(pLines[0].trimStart()) && listItems.length >= pLines.length - 1;
          const isList = listItems.length >= 2 && (listItems.length === pLines.length || hasIntro);
          if (isList) {
            const intro = hasIntro ? `<p>${pLines[0].trim().replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}</p>` : '';
            const items = pLines.filter((l, i) => !(i === 0 && hasIntro) && l.trim());
            const orderedCount = items.filter(l => _OL_RE.test(l.trimStart())).length;
            const tag = orderedCount > items.length - orderedCount ? 'ol' : 'ul';
            const lis = items.map(l => {
              const text = l.trimStart().replace(_UL_RE,'').replace(_OL_RE,'').trim();
              const esc = text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
              const withComments = esc.replace(
                /\[Comment:\s*([^\]]+)\]/gi,
                (_, txt) => `<span class="doc-comment-inline" contenteditable="false">[Comment: ${txt}]</span>`
              );
              return `<li>${withComments}</li>`;
            }).join('');
            return `${intro}<${tag}>${lis}</${tag}>`;
          }
          const escaped = p.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
          const withComments = escaped.replace(
            /\[Comment:\s*([^\]]+)\]/gi,
            (_, txt) => `<span class="doc-comment-inline" contenteditable="false">[Comment: ${txt}]</span>`
          );
          return `<p>${withComments.replace(/\n/g, '<br>')}</p>`;
        }).join('');
      }
      return html;
    }).join('');
  }

  function htmlToSections(html) {
    const div = window.document.createElement('div');
    div.innerHTML = html;
    const sections = [];
    let current = { title: '', content: '', blocks: [] };
    for (const node of Array.from(div.childNodes)) {
      const tag = node.nodeName.toLowerCase();
      if (['h1', 'h2', 'h3', 'h4'].includes(tag)) {
        if (current.title || current.content) sections.push(current);
        current = { title: node.textContent.trim(), content: '', blocks: [] };
      } else if (tag === 'ul' || tag === 'ol') {
        const liNodes = Array.from(node.querySelectorAll('li'));
        if (liNodes.length) {
          const prefix = tag === 'ol' ? (i) => `${i + 1}. ` : () => '- ';
          const listText = liNodes.map((li, i) => `${prefix(i)}${li.textContent.trim()}`).join('\n');
          current.content += (current.content ? '\n\n' : '') + listText;
        }
      } else if (tag !== '#comment') {
        const text = (node.textContent || '').trim();
        if (text) {
          current.content += (current.content ? '\n\n' : '') + text;
        }
      }
    }
    if (current.title || current.content) sections.push(current);
    return sections.length ? sections : [{ title: '', content: div.textContent || '', blocks: [] }];
  }

  // Apply execCommand formatting to the contenteditable editor
  function execFmt(cmd, value = null) {
    const editor = richEditorRef.current;
    if (!editor) return;
    editor.focus();
    document.execCommand('styleWithCSS', false, true);
    document.execCommand(cmd, false, value);
    updateActiveFormats();
  }

  function applyFontFamily(family) {
    setFontFamily(family);
    execFmt('fontName', family);
  }

  function applyFontSize(pt) {
    setFontSize(pt);
    const editor = richEditorRef.current;
    if (!editor) return;
    editor.focus();
    document.execCommand('styleWithCSS', false, true);
    document.execCommand('fontSize', false, '7');
    editor.querySelectorAll('font[size="7"]').forEach((el) => {
      el.removeAttribute('size');
      el.style.fontSize = pt + 'pt';
    });
  }

  function updateActiveFormats() {
    setActiveFormats({
      bold: document.queryCommandState('bold'),
      italic: document.queryCommandState('italic'),
      underline: document.queryCommandState('underline'),
      strikethrough: document.queryCommandState('strikeThrough'),
      justifyLeft: document.queryCommandState('justifyLeft'),
      justifyCenter: document.queryCommandState('justifyCenter'),
      justifyRight: document.queryCommandState('justifyRight'),
      justifyFull: document.queryCommandState('justifyFull'),
      insertUnorderedList: document.queryCommandState('insertUnorderedList'),
      insertOrderedList: document.queryCommandState('insertOrderedList'),
    });
  }

  function handleEditorKeyDown(e) {
    if (e.key === 'b' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); execFmt('bold'); }
    if (e.key === 'i' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); execFmt('italic'); }
    if (e.key === 'u' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); execFmt('underline'); }
    if (e.key === 'z' && (e.ctrlKey || e.metaKey) && !e.shiftKey) { e.preventDefault(); document.execCommand('undo'); }
    if (e.key === 'z' && (e.ctrlKey || e.metaKey) && e.shiftKey) { e.preventDefault(); document.execCommand('redo'); }
    if (e.key === 'y' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); document.execCommand('redo'); }
    if (e.key === 'a' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); document.execCommand('selectAll'); }
    updateActiveFormats();
  }

  function handleEditorInput(e) {
    ignoreNextInputRef.current = true;
    const sections = htmlToSections(e.currentTarget.innerHTML);
    setDraftSections(sections);
    setIsDirty(true);
    setAutoSaved(false);
    clearTimeout(autoSaveTimer.current);
    autoSaveTimer.current = setTimeout(() => triggerSave(sections), 1500);
  }

  // ── End rich text helpers ──────────────────────────────────────

  // ── Insert helpers ────────────────────────────────────────────
  function insertTable(rows, cols) {
    const editor = richEditorRef.current;
    if (!editor) return;
    editor.focus();
    let html = '<table style="border-collapse:collapse;width:100%;margin:8px 0">';
    for (let r = 0; r < rows; r++) {
      html += '<tr>';
      for (let c = 0; c < cols; c++) {
        html += `<td style="border:1px solid #ccc;padding:6px 8px;min-width:40px">${r === 0 ? `<strong>Col ${c + 1}</strong>` : ''}</td>`;
      }
      html += '</tr>';
    }
    html += '</table><p><br></p>';
    document.execCommand('insertHTML', false, html);
    handleEditorInput({ currentTarget: editor });
  }

  function insertImage(src, alt) {
    const editor = richEditorRef.current;
    if (!editor) return;
    editor.focus();
    document.execCommand('insertHTML', false, `<img src="${src}" alt="${alt || 'image'}" style="max-width:100%;height:auto;margin:4px 0;" />`);
    handleEditorInput({ currentTarget: editor });
  }

  function insertLink() {
    const url = window.prompt('Enter URL:', 'https://');
    if (!url) return;
    const text = window.getSelection()?.toString() || url;
    const editor = richEditorRef.current;
    if (!editor) return;
    editor.focus();
    if (window.getSelection()?.isCollapsed) {
      document.execCommand('insertHTML', false, `<a href="${url}" target="_blank">${text}</a>`);
    } else {
      document.execCommand('createLink', false, url);
      const a = editor.querySelector(`a[href="${url}"]`);
      if (a) a.target = '_blank';
    }
    handleEditorInput({ currentTarget: editor });
  }

  function insertPageBreak() {
    const editor = richEditorRef.current;
    if (!editor) return;
    editor.focus();
    document.execCommand('insertHTML', false, '<hr style="page-break-after:always;border:none;border-top:2px dashed #ccc;margin:16px 0;" /><p><br></p>');
    handleEditorInput({ currentTarget: editor });
  }

  function insertHorizontalRule() {
    const editor = richEditorRef.current;
    if (!editor) return;
    editor.focus();
    document.execCommand('insertHorizontalRule');
    handleEditorInput({ currentTarget: editor });
  }

  function applyLineSpacing(value) {
    setLineSpacing(value);
    const editor = richEditorRef.current;
    if (!editor) return;
    editor.focus();
    document.execCommand('insertHTML', false, '');
    const sel = window.getSelection();
    if (!sel || sel.rangeCount === 0) return;
    const range = sel.getRangeAt(0);
    let el = range.startContainer;
    if (el.nodeType === Node.TEXT_NODE) el = el.parentElement;
    while (el && el !== editor && !['P','H1','H2','H3','H4','LI','DIV'].includes(el.nodeName)) el = el.parentElement;
    if (el && el !== editor) el.style.lineHeight = value;
  }

  function applyPageMargins(top, right, bottom, left) {
    const paper = richEditorRef.current?.closest('.doc-page-body-zone');
    if (!paper) return;
    paper.style.padding = `${top}px ${right}px ${bottom}px ${left}px`;
  }

  function applyPageMarginsInches(next) {
    setPageMarginsInch(next);
    const toPx = (v, fallback) => {
      const num = parseFloat(v);
      return Number.isFinite(num) ? Math.max(0, num) * 96 : fallback;
    };
    applyPageMargins(
      toPx(next.top, 96),
      toPx(next.right, 120),
      toPx(next.bottom, 96),
      toPx(next.left, 120)
    );
  }

  function applyColumnLayout(cols) {
    const editor = richEditorRef.current;
    if (!editor) return;
    editor.style.columnCount = cols > 1 ? cols : '';
    editor.style.columnGap = cols > 1 ? '24px' : '';
  }
  // ── End insert helpers ─────────────────────────────────────────

  useEffect(() => () => clearProgressPolling(), []);

  const activeChat = useMemo(
    () => chats.find((chat) => chat.id === activeChatId) || chats[0],
    [chats, activeChatId]
  );
  const messages = activeChat?.messages || [];
  const plainDocText = useMemo(
    () => (draftSections || []).map((s) => s?.content || '').join('\n\n').trim(),
    [draftSections]
  );
  const wordCount = useMemo(() => {
    const clean = plainDocText.replace(/\s+/g, ' ').trim();
    return clean ? clean.split(' ').length : 0;
  }, [plainDocText]);
  const pageCount = useMemo(
    () => Math.max(1, Math.ceil(editorHeight / PAGE_CYCLE_PX)),
    [editorHeight]
  );

  // Derive the active plan from the live progress message (or last message with a plan)
  const activePlan = useMemo(() => {
    if (liveProgressMsgId) {
      const liveMsg = messages.find((m) => m.id === liveProgressMsgId);
      if (liveMsg?.plan?.length) return liveMsg.plan;
    }
    const withPlan = [...messages].reverse().find((m) => m.plan?.length);
    return withPlan?.plan || [];
  }, [messages, liveProgressMsgId]);

  // Current subsection/step the agent is actively writing
  const currentActivity = useMemo(() => {
    if (!liveProgressMsgId) return null;
    const liveMsg = messages.find((m) => m.id === liveProgressMsgId);
    return liveMsg?.workflow?.currentActivity
      || liveMsg?.workflow?.currentStep
      || liveMsg?.summary?.stage
      || null;
  }, [messages, liveProgressMsgId]);

  // Extract all [Comment: ...] annotations from the current draft sections
  const docComments = useMemo(() => {
    const re = /\[Comment:\s*([^\]]+)\]/gi;
    const results = [];
    for (const section of draftSections) {
      const body = section.content || '';
      let m;
      re.lastIndex = 0;
      while ((m = re.exec(body)) !== null) {
        results.push({ sectionTitle: section.title, text: m[1].trim(), fullMatch: m[0] });
      }
    }
    return results;
  }, [draftSections]);

  // Auto-resize textarea to content height (no inner scrollbar)
  useEffect(() => {
    const el = editorTextareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    const nextHeight = Math.max(el.scrollHeight, MIN_EDITOR_HEIGHT);
    el.style.height = nextHeight + 'px';
    setEditorHeight(nextHeight);
  }, [plainDocText]);

  // Sync draftSections → contenteditable (AI updates only; user edits skip via flag)
  useEffect(() => {
    if (ignoreNextInputRef.current) {
      ignoreNextInputRef.current = false;
      return;
    }
    const editor = richEditorRef.current;
    if (!editor) return;
    editor.innerHTML = sectionsToHtml(draftSections);
    // Re-apply AI highlights after editor reset so they survive section updates
    if (aiDetectResultRef.current) {
      setTimeout(() => _applyAiHighlights(aiDetectResultRef.current), 0);
    }
  }, [draftSections]); // eslint-disable-line react-hooks/exhaustive-deps

  // Ghost-highlight updated sections inline in the editor
  useEffect(() => {
    const editor = richEditorRef.current;
    if (!editor || !highlightedSections.length) {
      // Clear any existing highlights when list is empty
      editor?.querySelectorAll('[data-ghost-highlight]').forEach((el) => {
        el.removeAttribute('data-ghost-highlight');
      });
      return;
    }
    // Remove stale highlights first
    editor.querySelectorAll('[data-ghost-highlight]').forEach((el) => {
      el.removeAttribute('data-ghost-highlight');
    });
    const headings = editor.querySelectorAll('h1,h2,h3');
    headings.forEach((heading) => {
      const titleText = heading.textContent.trim().toLowerCase();
      const matched = highlightedSections.some(
        (t) => titleText.includes(t.trim().toLowerCase()) || t.trim().toLowerCase().includes(titleText)
      );
      if (!matched) return;
      heading.setAttribute('data-ghost-highlight', 'true');
      // Also mark sibling paragraphs that follow this heading (until next heading)
      let sib = heading.nextElementSibling;
      while (sib && !['H1', 'H2', 'H3'].includes(sib.tagName)) {
        sib.setAttribute('data-ghost-highlight', 'content');
        sib = sib.nextElementSibling;
      }
    });
    // Auto-clear after 6 seconds
    const timer = setTimeout(() => setHighlightedSections([]), 6000);
    return () => clearTimeout(timer);
  }, [highlightedSections, draftSections]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const sections = Array.isArray(document?.content?.sections) ? document.content.sections : [];
    setDraftSections(
      sections.map((section) => ({
        title: section?.title || '',
        content: section?.content || '',
        blocks: Array.isArray(section?.blocks) ? section.blocks : [],
      }))
    );
    setIsDirty(false);
    setAutoSaved(false);
    setManualError('');
  }, [document?.id, document?.updated_at]);

  // Reset chat only when switching to a different document, not on every save/update.
  // This preserves frontend-only state (changeSet.pending for Keep/Undo) across agent saves.
  useEffect(() => {
    if (document?.chat_messages?.length) {
      const persistedMessages = document.chat_messages.map((msg) => ({
        id: msg.id,
        role: msg.role,
        text: msg.content,
      }));
      setChats([{ id: 'chat-history', name: 'Document History', messages: persistedMessages }]);
      setActiveChatId('chat-history');
    } else {
      setChats([{ id: INITIAL_CHAT_ID, name: 'New Chat', messages: [] }]);
      setActiveChatId(INITIAL_CHAT_ID);
    }
  }, [document?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  function updateDraftSection(index, field, value) {
    setDraftSections((prev) => prev.map((s, i) => (i === index ? { ...s, [field]: value } : s)));
    setIsDirty(true);
    setAutoSaved(false);
    // debounced auto-save
    clearTimeout(autoSaveTimer.current);
    autoSaveTimer.current = setTimeout(() => triggerSave(), 1500);
  }

  function addDraftSection() {
    setDraftSections((prev) => {
      const next = [...prev, { title: `Section ${prev.length + 1}`, content: '', blocks: [] }];
      return next;
    });
    setIsDirty(true);
  }

  function removeDraftSection(index) {
    setDraftSections((prev) => prev.filter((_, i) => i !== index));
    setIsDirty(true);
  }

  const triggerSave = useCallback(async (sections) => {
    const target = sections || draftSections;
    await persistSectionsNow(target);
  }, [draftSections, onManualSave, onDocumentChanged]);

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

  function updateAssistantMessage(chatId, messageId, updater) {
    setChats((prev) =>
      prev.map((chat) => {
        if (chat.id !== chatId) return chat;
        return {
          ...chat,
          messages: chat.messages.map((msg) => {
            if (msg.id !== messageId) return msg;
            return typeof updater === 'function' ? updater(msg) : { ...msg, ...updater };
          }),
        };
      })
    );
  }

  async function persistSectionsNow(sections, onSuccess) {
    if (!onManualSave) return;
    setIsSavingManual(true);
    setManualError('');
    try {
      const cleaned = (Array.isArray(sections) ? sections : [])
        .map((s) => ({
          title: (s?.title || '').trim() || 'Untitled section',
          content: s?.content || '',
          ...(Array.isArray(s?.blocks) && s.blocks.length ? { blocks: s.blocks } : {}),
        }))
        .filter((s) => s.title || s.content);

      await onManualSave(cleaned);
      setIsDirty(false);
      setAutoSaved(true);
      onDocumentChanged?.();
      if (typeof onSuccess === 'function') onSuccess();
      setTimeout(() => setAutoSaved(false), 2500);
    } catch (err) {
      setManualError(err?.message || 'Save failed');
    } finally {
      setIsSavingManual(false);
    }
  }

  async function keepAgentChanges(chatId, messageId) {
    updateAssistantMessage(chatId, messageId, (msg) => ({
      ...msg,
      changeSet: msg.changeSet ? { ...msg.changeSet, pending: false } : msg.changeSet,
    }));
    setHighlightedSections([]);
  }

  async function undoAgentChanges(chatId, messageId, beforeSections) {
    const restored = cloneSections(beforeSections);
    setDraftSections(restored);
    setHighlightedSections([]);
    await persistSectionsNow(restored, () => {
      updateAssistantMessage(chatId, messageId, (msg) => ({
        ...msg,
        changeSet: msg.changeSet
          ? { ...msg.changeSet, pending: false, undone: true }
          : msg.changeSet,
      }));
    });
  }

  function startDissertationProgressPolling(docId, chatId, messageId, previewPlan) {
    clearProgressPolling();
    workflowStateRef.current[messageId] = buildWorkflowFromPlan(previewPlan, null);

    progressPollRef.current = setInterval(async () => {
      if (progressPollBusyRef.current) return;
      progressPollBusyRef.current = true;
      try {
        const latestDoc = await getDocument(docId);
        const sections = Array.isArray(latestDoc?.content?.sections) ? latestDoc.content.sections : [];
        setDraftSections(
          sections.map((section) => ({
            title: section?.title || '',
            content: section?.content || '',
            blocks: Array.isArray(section?.blocks) ? section.blocks : [],
          }))
        );
        setIsDirty(false);

        const planFromDoc = derivePlanFromDocument(previewPlan, sections);
        const activeStep = planFromDoc.find((item) => item.status === 'in_progress');
        const stageLabel = activeStep
          ? `Generating ${normalizeStep(activeStep.step).replace(/^Writing\s+/i, '')}...`
          : 'Finalizing dissertation...';

        // Read subsection-level activity written by backend before each node
        const subsectionActivity = latestDoc?.content?._current_activity || null;

        const previousWorkflow = workflowStateRef.current[messageId] || null;
        const nextWorkflow = buildWorkflowFromPlan(planFromDoc, previousWorkflow);
        // Attach current subsection activity so AgentActivityBar can display it
        if (subsectionActivity) nextWorkflow.currentActivity = subsectionActivity;
        workflowStateRef.current[messageId] = nextWorkflow;

        updateAssistantMessage(chatId, messageId, (msg) => ({
          ...msg,
          plan: planFromDoc,
          summary: buildSummaryFromPlan(planFromDoc, stageLabel),
          workflow: nextWorkflow,
        }));
      } catch {
        // Continue polling; transient network errors are expected while backend is busy.
      } finally {
        progressPollBusyRef.current = false;
      }
    }, 1200);
  }

  async function playbackDissertationResult(chatId, messageId, result, previewPlan, beforeSections = []) {
    const generatedSections = Array.isArray(result?.document?.content?.sections)
      ? result.document.content.sections
      : [];

    if (!generatedSections.length) {
      updateAssistantMessage(chatId, messageId, (msg) => ({
        ...msg,
        text: result.reply,
        summary: buildSummaryFromResult(result),
        plan: Array.isArray(result.plan) ? result.plan : msg.plan || [],
        research: result?.research || null,
      }));
      return;
    }

    setDraftSections(
      generatedSections.map((section) => ({
        title: section?.title || '',
        content: section?.content || '',
        blocks: Array.isArray(section?.blocks) ? section.blocks : [],
      }))
    );
    setIsDirty(false);
    const editedTitles = detectEditedSections(beforeSections, generatedSections);
    setHighlightedSections(editedTitles);

    const finalPlan = (Array.isArray(result.plan) && result.plan.length)
      ? result.plan.map((item) => ({ ...item, status: 'done' }))
      : derivePlanFromDocument(previewPlan, generatedSections).map((item) => ({ ...item, status: 'done' }));
    const finalWorkflow = buildWorkflowFromPlan(finalPlan, workflowStateRef.current[messageId] || null);
    finalWorkflow.updates = [`Completed: Dissertation generation finished`, ...finalWorkflow.updates].slice(0, 12);
    workflowStateRef.current[messageId] = finalWorkflow;
    updateAssistantMessage(chatId, messageId, (msg) => ({
      ...msg,
      text: result.reply,
      plan: finalPlan,
      summary: buildSummaryFromPlan(finalPlan, 'All planned tasks completed; document updated'),
      workflow: finalWorkflow,
      research: result?.research || null,
      changeSet: {
        pending: true,
        editedSections: editedTitles,
        beforeSections: cloneSections(beforeSections),
      },
    }));
  }

  function _applyAiHighlights(result) {
    const editor = richEditorRef.current;
    if (!editor) return;

    // Remove any existing AI highlight spans (unwrap them back to plain text)
    // Note: `document` prop shadows the global; use window.document for DOM APIs.
    const doc = window.document;
    editor.querySelectorAll('[data-ai-sent]').forEach((el) => {
      const txt = doc.createTextNode(el.textContent);
      el.parentNode.replaceChild(txt, el);
    });
    editor.normalize(); // merge adjacent text nodes

    if (!result?.sentences?.length) return;
    const flagged = result.sentences.filter((s) => s.label !== 'likely_human');
    if (!flagged.length) return;

    // TreeWalker approach: find text nodes and wrap matching sentences in-place.
    // This avoids innerHTML.replace() which breaks on HTML tag boundaries and
    // gets wiped by React reconciliation.
    let appliedCount = 0;
    for (const { text, label, ai_probability } of flagged) {
      try {
        const bg = label === 'likely_ai' ? 'rgba(239,68,68,0.28)' : 'rgba(251,191,36,0.35)';
        const borderColor = label === 'likely_ai' ? '#dc2626' : '#d97706';
        const pct = Math.round(ai_probability * 100);

        const walker = doc.createTreeWalker(editor, NodeFilter.SHOW_TEXT);
        let node;
        while ((node = walker.nextNode())) {
          const idx = node.textContent.indexOf(text);
          if (idx === -1) continue;

          const before = node.textContent.slice(0, idx);
          const after  = node.textContent.slice(idx + text.length);
          const parent = node.parentNode;

          const span = doc.createElement('span');
          span.setAttribute('data-ai-sent', label);
          span.style.cssText = [
            `background:${bg}`,
            `border-bottom:2px solid ${borderColor}`,
            'border-radius:3px',
            'padding:0 1px',
            'cursor:help',
          ].join(';');
          span.title = `AI probability: ${pct}% — ${label === 'likely_ai' ? 'Likely AI-generated' : 'Uncertain'}`;
          span.textContent = text;

          if (before) parent.insertBefore(doc.createTextNode(before), node);
          parent.insertBefore(span, node);
          if (after)  parent.insertBefore(doc.createTextNode(after), node);
          parent.removeChild(node);
          appliedCount++;
          break; // only first occurrence per sentence
        }
      } catch (e) {
        console.error('[AI-HIGHLIGHT] error applying sentence highlight:', e, text?.slice(0, 40));
      }
    }
  }

  // Re-apply highlights after every React render when detection result is live.
  // Must be a useEffect so it runs AFTER React finishes reconciling the DOM.
  useEffect(() => {
    if (aiDetectResult && !aiDetecting && !aiDetectResult.error) {
      // Small defer so any concurrent DOM sync effects settle first
      const id = setTimeout(() => _applyAiHighlights(aiDetectResult), 80);
      return () => clearTimeout(id);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [aiDetectResult, aiDetecting, draftSections]);

  async function runAiDetect() {
    const docId = document?.id;
    if (!docId) return;
    setAiDetecting(true);
    setShowAiDetectPanel(true);
    setShowCommentsPanel(false);
    setAiPanelOpen(true);
    try {
      const result = await detectAIContent(docId);
      aiDetectResultRef.current = result;
      setAiDetectResult(result);
      // highlights applied by useEffect above after re-render
    } catch (err) {
      aiDetectResultRef.current = null;
      setAiDetectResult({ error: String(err?.message || 'Detection failed') });
    } finally {
      setAiDetecting(false);
    }
  }

  function clearAiHighlights() {
    aiDetectResultRef.current = null;
    const editor = richEditorRef.current;
    if (!editor) return;
    editor.querySelectorAll('[data-ai-sent]').forEach((el) => {
      const txt = window.document.createTextNode(el.textContent);
      el.parentNode.replaceChild(txt, el);
    });
    editor.normalize();
    setAiDetectResult(null);
    setShowAiDetectPanel(false);
  }

  async function sendMessage(text) {
    if (!text.trim() && (!attachedFile)) return;
    if (isThinking) return;
    const userText = text.trim();
    // If a file is attached, the message text should reflect the file, not the internal content
    let displayText = userText;
    if (attachedFile) {
        if (!displayText) {
             displayText = `Uploaded file: ${attachedFile.name}`;
        } else {
             displayText += `\n[Attached File: ${attachedFile.name}]`;
        }
    }
    const userMsg = { id: Date.now(), role: 'user', text: displayText };
    const currentChatId = activeChatId;
    const beforeAgentSections = cloneSections(draftSections);
    const dissertationRequest = looksLikeDissertationRequest(userText);
    const progressMessageId = Date.now() + 1;
    setChats((prev) =>
      prev.map((chat) => {
        if (chat.id !== currentChatId) return chat;
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

    let previewPlan = [];
    if (dissertationRequest) {
      // Show an immediate placeholder so the user sees something straight away.
      const placeholderPlan = createFallbackPreviewPlan();
      setChats((prev) =>
        prev.map((chat) =>
          chat.id === currentChatId
            ? {
                ...chat,
                messages: [
                  ...chat.messages,
                  {
                    id: progressMessageId,
                    role: 'assistant',
                    text: 'Generating your personalised dissertation plan...',
                    summary: buildSummaryFromPlan(placeholderPlan, 'AI is creating your todo list...'),
                    plan: placeholderPlan,
                    workflow: {
                      currentStep: 'Generating todo list',
                      completedCount: 0,
                      totalCount: placeholderPlan.length,
                      statuses: placeholderPlan.map((item) => item.status),
                      updates: ['AI is generating a tailored chapter plan for your topic...'],
                    },
                  },
                ],
              }
            : chat
        )
      );
      setLiveProgressMsgId(progressMessageId);

      // Call the LLM plan endpoint — this is where the AI generates the tailored todo structure.
      const planResult = await getDissertationPlan(document?.id, userText);
      previewPlan = planResult?.plan?.length ? planResult.plan : createFallbackPreviewPlan();

      const firstInProgress = previewPlan.find((item) => item.status === 'in_progress');
      const stageLabel = firstInProgress
        ? `Starting: ${statusLabel(firstInProgress.step || '')}...`
        : 'Writing dissertation...';

      // Replace placeholder with the real AI-generated plan.
      updateAssistantMessage(currentChatId, progressMessageId, (msg) => ({
        ...msg,
        text: 'Dissertation plan ready. Starting to write...',
        summary: buildSummaryFromPlan(previewPlan, stageLabel),
        plan: previewPlan,
        workflow: {
          currentStep: statusLabel(firstInProgress?.step || ''),
          completedCount: previewPlan.filter((item) => item.status === 'done').length,
          totalCount: previewPlan.length,
          statuses: previewPlan.map((item) => item.status),
          updates: [
            'Todo list generated by AI',
            `Now doing: ${statusLabel(firstInProgress?.step || '')}`,
          ],
        },
      }));

      startDissertationProgressPolling(document?.id, currentChatId, progressMessageId, previewPlan);
    }

    try {
      // ── Step 1: Preview (non-dissertation only) ──────────────────────────
      // Send directly without preview_only so the backend executes immediately
      const result = await chatWithDocument(
        document?.id, userText, selectedModel,
        dissertationRequest ? attachedFile : null,
        /* previewOnly = */ false,
        { groundedResearch, verifyCitations: groundedResearch },
      );

      setAttachedFile(null);
      if (result?.model) {
        setActiveModel(result.model);
      }
      clearProgressPolling();

      if (dissertationRequest) {
        await playbackDissertationResult(currentChatId, progressMessageId, result, previewPlan, beforeAgentSections);
      } else {
        const assistantMsgId = Date.now() + 1;
        setChats((prev) =>
          prev.map((chat) =>
            chat.id === currentChatId
              ? {
                  ...chat,
                  messages: [
                    ...chat.messages,
                    {
                      id: assistantMsgId,
                      role: 'assistant',
                      text: toChatReply(result),
                      summary: buildSummaryFromResult(result),
                      plan: Array.isArray(result.plan) ? result.plan : [],
                      research: result?.research || null,
                    },
                  ],
                }
              : chat
          )
        );

        if (result.document_updated && Array.isArray(result?.document?.content?.sections)) {
          const nextSections = cloneSections(result.document.content.sections);
          const editedTitles = detectEditedSections(beforeAgentSections, nextSections);
          setDraftSections(nextSections);
          setIsDirty(false);
          setHighlightedSections(editedTitles);
          updateAssistantMessage(currentChatId, assistantMsgId, (msg) => ({
            ...msg,
            changeSet: {
              pending: true,
              editedSections: editedTitles,
              beforeSections: beforeAgentSections,
            },
          }));
        }
      }
      if (result.document_updated) {
        onDocumentChanged?.();
      }
    } catch (err) {
      clearProgressPolling();
      setChats((prev) =>
        prev.map((chat) =>
          chat.id === currentChatId
            ? {
                ...chat,
                messages: [...chat.messages, { id: Date.now() + 1, role: 'assistant', text: `Error: ${err.message}` }],
              }
            : chat
        )
      );
    } finally {
      clearProgressPolling();
      setLiveProgressMsgId(null);
      delete workflowStateRef.current[progressMessageId];
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


  return (
    <div className="doc-editor-root">
      <div className="doc-page-root">
        {/* ── Left column: ribbon + paper ── */}
        <div className="doc-left-col">
          {/* Ribbon */}
          <div className="doc-ribbon-shell">
            {/* ── Tab bar ── */}
            <div className="doc-ribbon-tabs">
              <div className="doc-quick-access">
                <button className="doc-qa-btn" title="Menu"><Menu size={15} /></button>
                <button className="doc-qa-file" onClick={onBackHome} title="Back to Home">File</button>
                <button className="doc-qa-btn" title="Save"><Download size={14} /></button>
                <button className="doc-qa-btn" title="Print"><Printer size={14} /></button>
                <button className="doc-qa-btn" title="Undo"><RotateCcw size={14} /></button>
                <button className="doc-qa-btn" title="More"><ChevronDown size={14} /></button>
              </div>
              {['Home', 'Insert', 'Page Layout', 'References', 'Review', 'View', 'Tools', 'WPS AI'].map((tab) => (
                <span
                  key={tab}
                  className={`doc-ribbon-tab${activeRibbonTab === tab ? ' doc-ribbon-tab--active' : ''}`}
                  onClick={() => setActiveRibbonTab(tab)}
                >
                  {tab}
                </span>
              ))}
            </div>

            {/* ── Home toolbar ── */}
            {activeRibbonTab === 'Home' && (
              <div className="doc-ribbon-toolbar">

                {/* Clipboard group */}
                <div className="doc-tool-group">
                  <div className="doc-tool-group-rows">
                    <div className="doc-tool-group-row">
                      <button
                        className="doc-tool-btn doc-tool-btn--tall"
                        title="Format Painter"
                        onClick={() => { const editor = richEditorRef.current; editor?.focus(); document.execCommand('copy'); }}
                      >
                        <Paintbrush size={16} />
                      </button>
                      <div className="doc-tool-col">
                        <button className="doc-tool-btn doc-tool-btn--wide" title="Paste (Ctrl+V)" onClick={() => execFmt('paste')}>
                          <span style={{fontSize:'10px',fontWeight:'600'}}>Paste</span><ChevronDown size={9}/>
                        </button>
                        <div className="doc-tool-group-row" style={{gap:'1px'}}>
                          <button className="doc-tool-btn" title="Cut (Ctrl+X)" onClick={() => execFmt('cut')}><Scissors size={13}/></button>
                          <button className="doc-tool-btn" title="Copy (Ctrl+C)" onClick={() => execFmt('copy')}><Copy size={13}/></button>
                        </div>
                      </div>
                    </div>
                  </div>
                  <span className="doc-tool-group-label">Clipboard</span>
                </div>

                {/* Font group */}
                <div className="doc-tool-group">
                  <div className="doc-tool-group-rows">
                    <div className="doc-tool-group-row">
                      <select
                        className="doc-font-select doc-font-select--family"
                        value={fontFamily}
                        onChange={(e) => applyFontFamily(e.target.value)}
                        title="Font Family"
                      >
                        {['Times New Roman','Arial','Calibri','Cambria','Georgia','Helvetica','Tahoma','Verdana','Courier New','Trebuchet MS'].map(f => (
                          <option key={f} value={f}>{f}</option>
                        ))}
                      </select>
                      <select
                        className="doc-font-select doc-font-select--size"
                        value={fontSize}
                        onChange={(e) => applyFontSize(e.target.value)}
                        title="Font Size"
                      >
                        {['8','9','10','11','12','14','16','18','20','22','24','26','28','36','48','72'].map(s => (
                          <option key={s} value={s}>{s}</option>
                        ))}
                      </select>
                      <button className="doc-tool-btn" title="Increase Font Size" onClick={() => applyFontSize(String(Math.min(72, parseInt(fontSize)+2)))}><span style={{fontSize:'12px',fontWeight:'bold'}}>A</span><span style={{fontSize:'8px',verticalAlign:'super',lineHeight:1}}>+</span></button>
                      <button className="doc-tool-btn" title="Decrease Font Size" onClick={() => applyFontSize(String(Math.max(6, parseInt(fontSize)-2)))}><span style={{fontSize:'12px',fontWeight:'bold'}}>A</span><span style={{fontSize:'8px',verticalAlign:'super',lineHeight:1}}>-</span></button>
                      <button className="doc-tool-btn" title="Change Case" onClick={() => {
                        const sel = window.getSelection();
                        if (sel && !sel.isCollapsed) {
                          const txt = sel.toString();
                          document.execCommand('insertText', false, txt === txt.toUpperCase() ? txt.toLowerCase() : txt.toUpperCase());
                        }
                      }}><span style={{fontSize:'11px'}}>Aa</span></button>
                      <button className="doc-tool-btn" title="Clear Formatting" onClick={() => { execFmt('removeFormat'); }}><Eraser size={13}/></button>
                    </div>
                    <div className="doc-tool-group-row">
                      <button className={`doc-tool-btn doc-tool-btn--fmt${activeFormats.bold ? ' doc-tool-btn--active' : ''}`} title="Bold (Ctrl+B)" onClick={() => execFmt('bold')}><Bold size={14}/></button>
                      <button className={`doc-tool-btn doc-tool-btn--fmt${activeFormats.italic ? ' doc-tool-btn--active' : ''}`} title="Italic (Ctrl+I)" onClick={() => execFmt('italic')}><Italic size={14}/></button>
                      <button className={`doc-tool-btn doc-tool-btn--fmt${activeFormats.underline ? ' doc-tool-btn--active' : ''}`} title="Underline (Ctrl+U)" onClick={() => execFmt('underline')}><Underline size={14}/></button>
                      <button className={`doc-tool-btn doc-tool-btn--fmt${activeFormats.strikethrough ? ' doc-tool-btn--active' : ''}`} title="Strikethrough" onClick={() => execFmt('strikeThrough')}><Strikethrough size={14}/></button>
                      <button className="doc-tool-btn" title="Subscript" onClick={() => execFmt('subscript')}><Subscript size={14}/></button>
                      <button className="doc-tool-btn" title="Superscript" onClick={() => execFmt('superscript')}><Superscript size={14}/></button>
                      {/* Font Color */}
                      <button className="doc-tool-btn doc-tool-btn--color" title="Font Color" onClick={() => fontColorInputRef.current?.click()}>
                        <span style={{fontSize:'11px',fontWeight:'bold',borderBottom:'2px solid #1a56db'}}>A</span>
                        <input ref={fontColorInputRef} type="color" style={{display:'none'}} defaultValue="#000000" onChange={(e) => execFmt('foreColor', e.target.value)} />
                      </button>
                      {/* Highlight */}
                      <button className="doc-tool-btn doc-tool-btn--color" title="Highlight Color" onClick={() => highlightColorInputRef.current?.click()}>
                        <Highlighter size={13}/>
                        <input ref={highlightColorInputRef} type="color" style={{display:'none'}} defaultValue="#ffff00" onChange={(e) => execFmt('hiliteColor', e.target.value)} />
                      </button>
                    </div>
                  </div>
                  <span className="doc-tool-group-label">Font</span>
                </div>

                {/* Paragraph group */}
                <div className="doc-tool-group">
                  <div className="doc-tool-group-rows">
                    <div className="doc-tool-group-row">
                      <button className={`doc-tool-btn${activeFormats.insertUnorderedList ? ' doc-tool-btn--active' : ''}`} title="Bullet List" onClick={() => execFmt('insertUnorderedList')}><List size={14}/></button>
                      <button className={`doc-tool-btn${activeFormats.insertOrderedList ? ' doc-tool-btn--active' : ''}`} title="Numbered List" onClick={() => execFmt('insertOrderedList')}><ListOrdered size={14}/></button>
                      <button className="doc-tool-btn" title="Decrease Indent" onClick={() => execFmt('outdent')}><IndentDecrease size={14}/></button>
                      <button className="doc-tool-btn" title="Increase Indent" onClick={() => execFmt('indent')}><IndentIncrease size={14}/></button>
                      <button className="doc-tool-btn" title="Sort" onClick={() => {}}><span style={{fontSize:'11px'}}>↕</span></button>
                      <button className="doc-tool-btn" title="Show Formatting Marks" onClick={() => {}}><span style={{fontSize:'13px',fontWeight:'bold'}}>¶</span></button>
                    </div>
                    <div className="doc-tool-group-row">
                      <button className={`doc-tool-btn${activeFormats.justifyLeft ? ' doc-tool-btn--active' : ''}`} title="Align Left (Ctrl+L)" onClick={() => execFmt('justifyLeft')}><AlignLeft size={14}/></button>
                      <button className={`doc-tool-btn${activeFormats.justifyCenter ? ' doc-tool-btn--active' : ''}`} title="Center (Ctrl+E)" onClick={() => execFmt('justifyCenter')}><AlignCenter size={14}/></button>
                      <button className={`doc-tool-btn${activeFormats.justifyRight ? ' doc-tool-btn--active' : ''}`} title="Align Right (Ctrl+R)" onClick={() => execFmt('justifyRight')}><AlignRight size={14}/></button>
                      <button className={`doc-tool-btn${activeFormats.justifyFull ? ' doc-tool-btn--active' : ''}`} title="Justify" onClick={() => execFmt('justifyFull')}><AlignJustify size={14}/></button>
                      <button className="doc-tool-btn" title="Line Spacing" onClick={() => {}}><span style={{fontSize:'10px',lineHeight:'1'}}>≡↕</span></button>
                      <button className="doc-tool-btn" title="Borders & Shading" onClick={() => {}}><span style={{fontSize:'12px',border:'1px solid currentColor',padding:'0 2px'}}>▣</span></button>
                    </div>
                  </div>
                  <span className="doc-tool-group-label">Paragraph</span>
                </div>

                {/* Styles / Headings group */}
                <div className="doc-tool-group" style={{minWidth:'140px'}}>
                  <div className="doc-tool-group-rows">
                    <div className="doc-tool-group-row">
                      <button
                        className="doc-tool-btn doc-heading-btn"
                        title="Heading 1"
                        onClick={() => {
                          const editor = richEditorRef.current;
                          if (!editor) return;
                          editor.focus();
                          document.execCommand('formatBlock', false, 'h1');
                          handleEditorInput({ currentTarget: editor });
                        }}
                        style={{fontWeight:'bold',fontSize:'13px'}}
                      >H1</button>
                      <button
                        className="doc-tool-btn doc-heading-btn"
                        title="Heading 2"
                        onClick={() => {
                          const editor = richEditorRef.current;
                          if (!editor) return;
                          editor.focus();
                          document.execCommand('formatBlock', false, 'h2');
                          handleEditorInput({ currentTarget: editor });
                        }}
                        style={{fontWeight:'bold',fontSize:'12px'}}
                      >H2</button>
                      <button
                        className="doc-tool-btn doc-heading-btn"
                        title="Heading 3"
                        onClick={() => {
                          const editor = richEditorRef.current;
                          if (!editor) return;
                          editor.focus();
                          document.execCommand('formatBlock', false, 'h3');
                          handleEditorInput({ currentTarget: editor });
                        }}
                        style={{fontWeight:'bold',fontSize:'11px'}}
                      >H3</button>
                      <button
                        className="doc-tool-btn doc-heading-btn"
                        title="Heading 4"
                        onClick={() => {
                          const editor = richEditorRef.current;
                          if (!editor) return;
                          editor.focus();
                          document.execCommand('formatBlock', false, 'h4');
                          handleEditorInput({ currentTarget: editor });
                        }}
                        style={{fontWeight:'bold',fontSize:'10px'}}
                      >H4</button>
                      <button
                        className="doc-tool-btn doc-heading-btn"
                        title="Normal text"
                        onClick={() => {
                          const editor = richEditorRef.current;
                          if (!editor) return;
                          editor.focus();
                          document.execCommand('formatBlock', false, 'p');
                          handleEditorInput({ currentTarget: editor });
                        }}
                        style={{fontSize:'11px'}}
                      >T</button>
                    </div>
                  </div>
                  <span className="doc-tool-group-label">Styles</span>
                </div>

                {/* Find & Replace */}
                <div className="doc-tool-group">
                  <div className="doc-tool-group-rows" style={{alignItems:'center',justifyContent:'center',gap:'4px'}}>
                    <button
                      className="doc-tool-btn doc-tool-btn--labeled"
                      title="Find & Replace (Ctrl+H)"
                      onClick={() => {
                        const term = window.prompt('Find:');
                        if (!term) return;
                        const replace = window.prompt('Replace with:');
                        if (replace === null) return;
                        const editor = richEditorRef.current;
                        if (!editor) return;
                        editor.innerHTML = editor.innerHTML.replace(new RegExp(term.replace(/[.*+?^${}()|[\]\\]/g,'\\$&'), 'g'), replace);
                        handleEditorInput({ currentTarget: editor });
                      }}
                    >
                      <Search size={13}/>
                      <span>Find &amp; Replace</span>
                    </button>
                    <button
                      className="doc-tool-btn doc-tool-btn--labeled"
                      title="Select All (Ctrl+A)"
                      onClick={() => { richEditorRef.current?.focus(); document.execCommand('selectAll'); }}
                    >
                      <span style={{fontSize:'11px'}}>Select</span><ChevronDown size={9}/>
                    </button>
                  </div>
                  <span className="doc-tool-group-label">Editing</span>
                </div>

              </div>
            )}

            {/* ── INSERT tab ── */}
            {activeRibbonTab === 'Insert' && (
              <div className="doc-ribbon-toolbar">
                {/* Pages group */}
                <div className="doc-tool-group">
                  <div className="doc-tool-group-rows">
                    <div className="doc-tool-group-row">
                      <button className="doc-tool-btn doc-tool-btn--labeled" title="Insert Page Break (Ctrl+Enter)" onClick={insertPageBreak}><FileText size={13}/><span>Page Break</span></button>
                      <button className="doc-tool-btn doc-tool-btn--labeled" title="Insert Blank Page" onClick={() => { insertPageBreak(); insertPageBreak(); }}><LayoutTemplate size={13}/><span>Blank Page</span></button>
                    </div>
                  </div>
                  <span className="doc-tool-group-label">Pages</span>
                </div>
                {/* Table group */}
                <div className="doc-tool-group">
                  <div className="doc-tool-group-rows">
                    <div className="doc-tool-group-row">
                      <button className="doc-tool-btn doc-tool-btn--labeled" title="Insert Table" onClick={() => {
                        const rows = parseInt(window.prompt('Rows:', '3') || '3');
                        const cols = parseInt(window.prompt('Columns:', '3') || '3');
                        if (rows > 0 && cols > 0) insertTable(rows, cols);
                      }}><Table size={13}/><span>Table</span></button>
                    </div>
                  </div>
                  <span className="doc-tool-group-label">Tables</span>
                </div>
                {/* Illustrations group */}
                <div className="doc-tool-group">
                  <div className="doc-tool-group-rows">
                    <div className="doc-tool-group-row">
                      <button className="doc-tool-btn doc-tool-btn--labeled" title="Insert Picture from file" onClick={() => imageInputRef.current?.click()}>
                        <Image size={13}/><span>Picture</span>
                        <input ref={imageInputRef} type="file" accept="image/*" style={{display:'none'}} onChange={(e) => {
                          const file = e.target.files?.[0];
                          if (!file) return;
                          const reader = new FileReader();
                          reader.onload = (ev) => insertImage(ev.target.result, file.name);
                          reader.readAsDataURL(file);
                          e.target.value = '';
                        }}/>
                      </button>
                      <button className="doc-tool-btn doc-tool-btn--labeled" title="Insert Horizontal Rule" onClick={insertHorizontalRule}><Minus size={13}/><span>Line</span></button>
                    </div>
                  </div>
                  <span className="doc-tool-group-label">Illustrations</span>
                </div>
                {/* Links group */}
                <div className="doc-tool-group">
                  <div className="doc-tool-group-rows">
                    <div className="doc-tool-group-row">
                      <button className="doc-tool-btn doc-tool-btn--labeled" title="Insert Hyperlink (Ctrl+K)" onClick={insertLink}><Link size={13}/><span>Hyperlink</span></button>
                      <button className="doc-tool-btn doc-tool-btn--labeled" title="Insert Bookmark" onClick={() => {
                        const name = window.prompt('Bookmark name:');
                        if (name) {
                          const editor = richEditorRef.current; if (!editor) return;
                          editor.focus();
                          document.execCommand('insertHTML', false, `<a id="${name}" style="color:inherit;text-decoration:none">[${name}]</a>`);
                          handleEditorInput({ currentTarget: editor });
                        }
                      }}><Hash size={13}/><span>Bookmark</span></button>
                    </div>
                  </div>
                  <span className="doc-tool-group-label">Links</span>
                </div>
                {/* Header & Footer group */}
                <div className="doc-tool-group">
                  <div className="doc-tool-group-rows">
                    <div className="doc-tool-group-row">
                      <button className="doc-tool-btn doc-tool-btn--labeled" title="Edit Header" onClick={() => {
                        const text = window.prompt('Header text:', '');
                        if (text === null) return;
                        const existing = window.document.querySelector('.doc-header-band');
                        if (existing) { existing.textContent = text; return; }
                        const scroll = richEditorRef.current?.closest('.doc-paper-scroll');
                        if (scroll) {
                          const el = window.document.createElement('div');
                          el.className = 'doc-header-band'; el.textContent = text;
                          scroll.insertBefore(el, scroll.firstChild);
                        }
                      }}><LayoutTemplate size={13}/><span>Header</span></button>
                      <button className="doc-tool-btn doc-tool-btn--labeled" title="Edit Footer" onClick={() => {
                        const text = window.prompt('Footer text:', '');
                        if (text === null) return;
                        const existing = window.document.querySelector('.doc-footer-band');
                        if (existing) { existing.textContent = text; return; }
                        const scroll = richEditorRef.current?.closest('.doc-paper-scroll');
                        if (scroll) {
                          const el = window.document.createElement('div');
                          el.className = 'doc-footer-band'; el.textContent = text;
                          scroll.appendChild(el);
                        }
                      }}><LayoutTemplate size={13}/><span>Footer</span></button>
                    </div>
                    <div className="doc-tool-group-row">
                      <button className="doc-tool-btn doc-tool-btn--labeled" title="Insert Page Number" onClick={() => {
                        const editor = richEditorRef.current; if (!editor) return;
                        editor.focus();
                        document.execCommand('insertHTML', false, '<span class="doc-page-num" style="color:#1a56db;font-size:10pt">[Page]</span>');
                        handleEditorInput({ currentTarget: editor });
                      }}><Hash size={12}/><span>Page #</span></button>
                    </div>
                  </div>
                  <span className="doc-tool-group-label">Header &amp; Footer</span>
                </div>
                {/* Text group */}
                <div className="doc-tool-group">
                  <div className="doc-tool-group-rows">
                    <div className="doc-tool-group-row">
                      <button className="doc-tool-btn doc-tool-btn--labeled" title="Insert Text Box" onClick={() => {
                        const editor = richEditorRef.current; if (!editor) return;
                        editor.focus();
                        document.execCommand('insertHTML', false, '<div style="border:1px solid #ccc;padding:8px 12px;margin:8px 0;min-height:40px;display:inline-block;min-width:200px">Text box</div>');
                        handleEditorInput({ currentTarget: editor });
                      }}><Type size={13}/><span>Text Box</span></button>
                      <button className="doc-tool-btn doc-tool-btn--labeled" title="Insert special character" onClick={() => {
                        const chars = ['©','®','™','€','£','¥','°','±','×','÷','→','←','↑','↓','•','…','—','–'];
                        const ch = window.prompt('Paste or type a special character, or pick:\n' + chars.join(' '), '©');
                        if (ch) { const editor = richEditorRef.current; if (editor) { editor.focus(); document.execCommand('insertText', false, ch); handleEditorInput({ currentTarget: editor }); }}
                      }}><Star size={13}/><span>Symbol</span></button>
                    </div>
                  </div>
                  <span className="doc-tool-group-label">Text</span>
                </div>
              </div>
            )}

            {/* ── PAGE LAYOUT tab ── */}
            {activeRibbonTab === 'Page Layout' && (
              <div className="doc-ribbon-toolbar">
                <div className="doc-tool-group doc-layout-group-page-setup">
                  <div className="doc-layout-page-setup-wrap">
                    <div className="doc-layout-icon-title" title="Margins">
                      <LayoutTemplate size={24} />
                      <span>Margins</span>
                      <ChevronDown size={11} />
                    </div>

                    <div className="doc-layout-margins-grid">
                      <div className="doc-layout-spin-row">
                        <label>Top:</label>
                        <input
                          type="number"
                          min="0"
                          step="0.05"
                          value={pageMarginsInch.top}
                          onChange={(e) => applyPageMarginsInches({ ...pageMarginsInch, top: e.target.value })}
                        />
                        <span>in</span>
                      </div>
                      <div className="doc-layout-spin-row">
                        <label>Bottom:</label>
                        <input
                          type="number"
                          min="0"
                          step="0.05"
                          value={pageMarginsInch.bottom}
                          onChange={(e) => applyPageMarginsInches({ ...pageMarginsInch, bottom: e.target.value })}
                        />
                        <span>in</span>
                      </div>
                      <div className="doc-layout-spin-row">
                        <label>Left:</label>
                        <input
                          type="number"
                          min="0"
                          step="0.05"
                          value={pageMarginsInch.left}
                          onChange={(e) => applyPageMarginsInches({ ...pageMarginsInch, left: e.target.value })}
                        />
                        <span>in</span>
                      </div>
                      <div className="doc-layout-spin-row">
                        <label>Right:</label>
                        <input
                          type="number"
                          min="0"
                          step="0.05"
                          value={pageMarginsInch.right}
                          onChange={(e) => applyPageMarginsInches({ ...pageMarginsInch, right: e.target.value })}
                        />
                        <span>in</span>
                      </div>
                    </div>

                    <div className="doc-layout-mini-icons">
                      <button className="doc-tool-btn" title="Portrait" onClick={() => {
                        const scroll = richEditorRef.current?.closest('.doc-paper-scroll');
                        if (!scroll) return;
                        scroll.style.width = '816px';
                        scroll.style.minHeight = '1056px';
                      }}>
                        <FileText size={15} />
                      </button>
                      <button className="doc-tool-btn" title="Landscape" onClick={() => {
                        const scroll = richEditorRef.current?.closest('.doc-paper-scroll');
                        if (!scroll) return;
                        scroll.style.width = '1056px';
                        scroll.style.minHeight = '816px';
                      }}>
                        <FileText size={15} style={{ transform: 'rotate(90deg)' }} />
                      </button>
                      <div className="doc-layout-mini-select-wrap" title="Columns">
                        <button className="doc-tool-btn">
                          <Columns size={15} />
                          <ChevronDown size={10} />
                        </button>
                        <select onChange={(e) => applyColumnLayout(parseInt(e.target.value, 10))}>
                          <option value="1">1</option>
                          <option value="2">2</option>
                          <option value="3">3</option>
                        </select>
                      </div>
                    </div>
                  </div>
                  <span className="doc-tool-group-label">Page Setup</span>
                </div>

                <div className="doc-tool-group doc-layout-group-background">
                  <div className="doc-layout-quick-stack">
                    <button className="doc-layout-text-btn" title="Themes">
                      <Type size={14} />
                      <span>Themes</span>
                      <ChevronDown size={10} />
                    </button>
                    <button className="doc-layout-text-btn" title="Cover Page">
                      <Image size={14} />
                      <span>Cover Page</span>
                      <ChevronDown size={10} />
                    </button>
                  </div>
                  <button className="doc-tool-btn doc-tool-btn--tall" title="Page Borders" onClick={() => {
                    const scroll = richEditorRef.current?.closest('.doc-paper-scroll');
                    if (!scroll) return;
                    scroll.style.border = scroll.style.border ? '' : '1px solid #1a1a1a';
                  }}>
                    <LayoutTemplate size={22} />
                    <span>Page Borders</span>
                  </button>
                  <button className="doc-tool-btn doc-tool-btn--tall" title="Page Color" onClick={() => {
                    const el = window.document.createElement('input');
                    el.type = 'color';
                    el.value = '#ffffff';
                    el.onchange = (ev) => {
                      const scroll = richEditorRef.current?.closest('.doc-paper-scroll');
                      if (scroll) scroll.style.backgroundColor = ev.target.value;
                    };
                    el.click();
                  }}>
                    <Palette size={22} />
                    <span>Page Color</span>
                  </button>
                  <button className="doc-layout-text-btn" title="Watermark">
                    <Globe size={14} />
                    <span>Watermark</span>
                    <ChevronDown size={10} />
                  </button>
                  <button className="doc-layout-text-btn" title="Line Numbers">
                    <ListOrdered size={14} />
                    <span>Line Numbers</span>
                    <ChevronDown size={10} />
                  </button>
                  <span className="doc-tool-group-label">Page Background</span>
                </div>

                <div className="doc-tool-group doc-layout-group-sections">
                  <button className="doc-layout-text-btn" title="Blank Page" onClick={() => insertPageBreak()}>
                    <FileText size={14} />
                    <span>Blank Page</span>
                    <ChevronDown size={10} />
                  </button>
                  <button className="doc-layout-text-btn" title="Breaks" onClick={() => insertPageBreak()}>
                    <Minus size={14} />
                    <span>Breaks</span>
                    <ChevronDown size={10} />
                  </button>
                  <button className="doc-layout-text-btn" title="Table Of Contents">
                    <BookOpen size={14} />
                    <span>Table Of Contents</span>
                    <ChevronDown size={10} />
                  </button>
                  <button className="doc-layout-text-btn" title="Section Pane">
                    <PanelLeft size={14} />
                    <span>Section Pane</span>
                  </button>
                  <button className="doc-layout-text-btn doc-layout-text-btn--disabled" title="Delete Section" disabled>
                    <X size={14} />
                    <span>Delete Section</span>
                  </button>
                  <span className="doc-tool-group-label">Sections</span>
                </div>

                <div className="doc-tool-group doc-layout-group-settings">
                  <button className="doc-tool-btn doc-tool-btn--tall" title="Settings">
                    <Settings2 size={20} />
                    <span>Settings</span>
                    <ChevronDown size={10} />
                  </button>
                  <span className="doc-tool-group-label">Settings</span>
                </div>
              </div>
            )}

            {/* ── REFERENCES tab ── */}
            {activeRibbonTab === 'References' && (
              <div className="doc-ribbon-toolbar">
                {/* TOC */}
                <div className="doc-tool-group">
                  <div className="doc-tool-group-rows">
                    <div className="doc-tool-group-row">
                      <button className="doc-tool-btn doc-tool-btn--labeled" title="Insert Table of Contents" onClick={() => {
                        const editor = richEditorRef.current; if (!editor) return;
                        editor.focus();
                        const headings = Array.from(editor.querySelectorAll('h1,h2,h3'));
                        if (!headings.length) { window.alert('No headings found. Add headings first.'); return; }
                        let toc = '<div style="border:1px solid #e2e8f0;padding:12px 16px;margin:12px 0;background:#f8fafc"><strong>Table of Contents</strong><br/>';
                        headings.forEach((h, i) => {
                          const indent = h.tagName === 'H1' ? 0 : h.tagName === 'H2' ? 16 : 32;
                          toc += `<div style="margin-left:${indent}px;padding:2px 0;font-size:11pt">${i+1}. ${h.textContent}</div>`;
                        });
                        toc += '</div><p><br></p>';
                        editor.insertAdjacentHTML('afterbegin', toc);
                        handleEditorInput({ currentTarget: editor });
                      }}><BookOpen size={13}/><span>Table of Contents</span></button>
                    </div>
                  </div>
                  <span className="doc-tool-group-label">Table of Contents</span>
                </div>
                {/* Footnotes */}
                <div className="doc-tool-group">
                  <div className="doc-tool-group-rows">
                    <div className="doc-tool-group-row">
                      <button className="doc-tool-btn doc-tool-btn--labeled" title="Insert Footnote" onClick={() => {
                        const text = window.prompt('Footnote text:');
                        if (!text) return;
                        const editor = richEditorRef.current; if (!editor) return;
                        editor.focus();
                        const num = editor.querySelectorAll('.doc-footnote-ref').length + 1;
                        document.execCommand('insertHTML', false, `<sup class="doc-footnote-ref" style="color:#1a56db;cursor:pointer" title="${text}">[${num}]</sup>`);
                        let fns = editor.querySelector('.doc-footnotes');
                        if (!fns) {
                          fns = window.document.createElement('div');
                          fns.className = 'doc-footnotes';
                          fns.style.cssText = 'border-top:1px solid #ccc;margin-top:24px;padding-top:8px;font-size:10pt';
                          editor.appendChild(fns);
                        }
                        const fn = window.document.createElement('p');
                        fn.style.margin = '2px 0'; fn.style.fontSize = '10pt';
                        fn.textContent = `[${num}] ${text}`;
                        fns.appendChild(fn);
                        handleEditorInput({ currentTarget: editor });
                      }}><MessageSquare size={13}/><span>Insert Footnote</span></button>
                    </div>
                  </div>
                  <span className="doc-tool-group-label">Footnotes</span>
                </div>
                {/* Citations */}
                <div className="doc-tool-group">
                  <div className="doc-tool-group-rows">
                    <div className="doc-tool-group-row">
                      <button className="doc-tool-btn doc-tool-btn--labeled" title="Insert Citation" onClick={() => {
                        const author = window.prompt('Author(s):') || 'Author';
                        const year = window.prompt('Year:') || new Date().getFullYear();
                        const editor = richEditorRef.current; if (!editor) return;
                        editor.focus();
                        document.execCommand('insertHTML', false, `<span style="color:#374151">(${author}, ${year})</span>`);
                        handleEditorInput({ currentTarget: editor });
                      }}><BookOpen size={13}/><span>Insert Citation</span></button>
                      <button className="doc-tool-btn doc-tool-btn--labeled" title="Insert Bibliography" onClick={() => {
                        const editor = richEditorRef.current; if (!editor) return;
                        editor.focus();
                        document.execCommand('insertHTML', false, '<h2>References</h2><p style="font-size:11pt">Add your references here in APA/MLA format.</p>');
                        handleEditorInput({ currentTarget: editor });
                      }}><FileSearch size={13}/><span>Bibliography</span></button>
                    </div>
                  </div>
                  <span className="doc-tool-group-label">Citations &amp; Bibliography</span>
                </div>
                {/* Index */}
                <div className="doc-tool-group">
                  <div className="doc-tool-group-rows">
                    <div className="doc-tool-group-row">
                      <button className="doc-tool-btn doc-tool-btn--labeled" title="Mark entry for index" onClick={() => {
                        const sel = window.getSelection()?.toString();
                        if (!sel) { window.alert('Select text first.'); return; }
                        const editor = richEditorRef.current; if (!editor) return;
                        editor.focus();
                        document.execCommand('insertHTML', false, `<span style="background:#fef9c3" title="Index entry: ${sel}">${sel}</span>`);
                        handleEditorInput({ currentTarget: editor });
                      }}><Hash size={13}/><span>Mark Entry</span></button>
                    </div>
                  </div>
                  <span className="doc-tool-group-label">Index</span>
                </div>
              </div>
            )}

            {/* ── REVIEW tab ── */}
            {activeRibbonTab === 'Review' && (
              <div className="doc-ribbon-toolbar">
                {/* Proofing */}
                <div className="doc-tool-group">
                  <div className="doc-tool-group-rows">
                    <div className="doc-tool-group-row">
                      <button className={`doc-tool-btn doc-tool-btn--labeled${spellCheckEnabled ? ' doc-tool-btn--active' : ''}`}
                        title="Toggle spell check"
                        onClick={() => {
                          const next = !spellCheckEnabled;
                          setSpellCheckEnabled(next);
                          if (richEditorRef.current) richEditorRef.current.spellcheck = next;
                        }}>
                        <SpellCheck size={13}/><span>Spell Check</span>
                      </button>
                      <button className="doc-tool-btn doc-tool-btn--labeled" title="Word count" onClick={() => {
                        const text = richEditorRef.current?.textContent || '';
                        const wc = text.trim().split(/\s+/).filter(Boolean).length;
                        const cc = text.replace(/\s/g,'').length;
                        const pc = richEditorRef.current?.querySelectorAll('p,h1,h2,h3').length || 0;
                        window.alert(`Words: ${wc}\nCharacters (no spaces): ${cc}\nParagraphs: ${pc}`);
                      }}><Hash size={13}/><span>Word Count</span></button>
                    </div>
                  </div>
                  <span className="doc-tool-group-label">Proofing</span>
                </div>
                {/* Comments */}
                <div className="doc-tool-group">
                  <div className="doc-tool-group-rows">
                    <div className="doc-tool-group-row">
                      <button className="doc-tool-btn doc-tool-btn--labeled" title="Insert a comment on the selected text" onClick={() => {
                        const sel = window.getSelection()?.toString();
                        const comment = window.prompt(`Comment on "${sel || 'cursor position'}":`) ;
                        if (!comment) return;
                        const editor = richEditorRef.current; if (!editor) return;
                        editor.focus();
                        const marker = `[Comment: ${comment}]`;
                        if (sel) {
                          // Wrap selected text in a highlight mark AND append the comment marker
                          document.execCommand('insertHTML', false,
                            `<mark class="doc-comment-mark" data-comment="${comment.replace(/"/g,'&quot;')}" title="Comment: ${comment}">${sel}</mark><span class="doc-comment-inline" contenteditable="false">${marker}</span>`);
                        } else {
                          document.execCommand('insertHTML', false,
                            `<span class="doc-comment-inline" contenteditable="false">${marker}</span>`);
                        }
                        handleEditorInput({ currentTarget: editor });
                      }}><StickyNote size={13}/><span>New Comment</span></button>
                      <button className="doc-tool-btn doc-tool-btn--labeled" title="Ask AI to address all comments in the document"
                        disabled={docComments.length === 0 || isThinking}
                        onClick={() => {
                          setAiPanelOpen(true);
                          sendMessage('Address all comments in the document');
                        }}>
                        <CheckCircle2 size={13}/><span>Address Comments</span>
                      </button>
                    </div>
                    <div className="doc-tool-group-row">
                      <button className="doc-tool-btn doc-tool-btn--labeled" title="View all comments"
                        onClick={() => { setAiPanelOpen(true); setShowCommentsPanel(true); }}>
                        <MessageCircle size={13}/><span>View Comments {docComments.length > 0 ? `(${docComments.length})` : ''}</span>
                      </button>
                    </div>
                  </div>
                  <span className="doc-tool-group-label">Comments</span>
                </div>
                {/* Tracking */}
                <div className="doc-tool-group">
                  <div className="doc-tool-group-rows">
                    <div className="doc-tool-group-row">
                      <button className={`doc-tool-btn doc-tool-btn--labeled${trackChanges ? ' doc-tool-btn--active' : ''}`}
                        title="Track changes made to the document"
                        onClick={() => setTrackChanges(t => !t)}>
                        <Eye size={13}/><span>Track Changes</span>
                      </button>
                    </div>
                  </div>
                  <span className="doc-tool-group-label">Tracking</span>
                </div>
                {/* Compare */}
                <div className="doc-tool-group">
                  <div className="doc-tool-group-rows">
                    <div className="doc-tool-group-row">
                      <button className="doc-tool-btn doc-tool-btn--labeled" title="Undo all changes" onClick={() => { document.execCommand('undo'); }}><RotateCcw size={13}/><span>Undo All</span></button>
                    </div>
                  </div>
                  <span className="doc-tool-group-label">Changes</span>
                </div>
              </div>
            )}

            {/* ── VIEW tab ── */}
            {activeRibbonTab === 'View' && (
              <div className="doc-ribbon-toolbar">
                {/* Document Views */}
                <div className="doc-tool-group">
                  <div className="doc-tool-group-rows">
                    <div className="doc-tool-group-row">
                      <button className="doc-tool-btn doc-tool-btn--labeled doc-tool-btn--active" title="Print Layout view"><FileText size={13}/><span>Print Layout</span></button>
                      <button className="doc-tool-btn doc-tool-btn--labeled" title="Full screen reading" onClick={() => {
                        const el = window.document.documentElement;
                        if (window.document.fullscreenElement) window.document.exitFullscreen?.();
                        else el.requestFullscreen?.();
                      }}><Maximize size={13}/><span>Full Screen</span></button>
                    </div>
                  </div>
                  <span className="doc-tool-group-label">Document Views</span>
                </div>
                {/* Show/Hide */}
                <div className="doc-tool-group">
                  <div className="doc-tool-group-rows">
                    <div className="doc-tool-group-row">
                      <button className={`doc-tool-btn doc-tool-btn--labeled${showRuler ? ' doc-tool-btn--active' : ''}`} title="Toggle ruler" onClick={() => setShowRuler(r => !r)}><Ruler size={13}/><span>Ruler</span></button>
                      <button className={`doc-tool-btn doc-tool-btn--labeled${showFormattingMarks ? ' doc-tool-btn--active' : ''}`} title="Show formatting marks" onClick={() => setShowFormattingMarks(f => !f)}><span style={{fontWeight:'bold',fontSize:'13px'}}>¶</span><span>Formatting</span></button>
                    </div>
                    <div className="doc-tool-group-row">
                      <button className={`doc-tool-btn doc-tool-btn--labeled${aiPanelOpen ? ' doc-tool-btn--active' : ''}`} title="Toggle AI panel" onClick={() => setAiPanelOpen(a => !a)}><PanelLeft size={13}/><span>AI Panel</span></button>
                    </div>
                  </div>
                  <span className="doc-tool-group-label">Show</span>
                </div>
                {/* Zoom */}
                <div className="doc-tool-group">
                  <div className="doc-tool-group-rows">
                    <div className="doc-tool-group-row">
                      {['75%','100%','125%','150%'].map(z => (
                        <button key={z} className="doc-tool-btn doc-tool-btn--labeled" title={`Zoom to ${z}`} onClick={() => {
                          const canvas = richEditorRef.current?.closest('.doc-page-canvas');
                          if (canvas) { canvas.style.transform = `scale(${parseInt(z)/100})`; canvas.style.transformOrigin = 'top center'; }
                        }}><span style={{fontSize:'11px'}}>{z}</span></button>
                      ))}
                    </div>
                  </div>
                  <span className="doc-tool-group-label">Zoom</span>
                </div>
                {/* Window */}
                <div className="doc-tool-group">
                  <div className="doc-tool-group-rows">
                    <div className="doc-tool-group-row">
                      <button className="doc-tool-btn doc-tool-btn--labeled" title="Print document" onClick={() => window.print()}><Printer size={13}/><span>Print</span></button>
                    </div>
                  </div>
                  <span className="doc-tool-group-label">Window</span>
                </div>
              </div>
            )}

            {/* ── TOOLS tab ── */}
            {activeRibbonTab === 'Tools' && (
              <div className="doc-ribbon-toolbar">
                {/* Proofing Tools */}
                <div className="doc-tool-group">
                  <div className="doc-tool-group-rows">
                    <div className="doc-tool-group-row">
                      <button className="doc-tool-btn doc-tool-btn--labeled" title="Find (Ctrl+F)" onClick={() => {
                        const term = window.prompt('Find:');
                        if (!term) return;
                        const editor = richEditorRef.current; if (!editor) return;
                        const text = editor.innerHTML;
                        editor.innerHTML = text.replace(new RegExp(term.replace(/[.*+?^${}()|[\]\\]/g,'\\$&'),'gi'), `<mark style="background:#fde047">$&</mark>`);
                        handleEditorInput({ currentTarget: editor });
                      }}><Search size={13}/><span>Find</span></button>
                      <button className="doc-tool-btn doc-tool-btn--labeled" title="Find & Replace (Ctrl+H)" onClick={() => {
                        const term = window.prompt('Find:');
                        if (!term) return;
                        const replace = window.prompt('Replace with:');
                        if (replace === null) return;
                        const editor = richEditorRef.current; if (!editor) return;
                        editor.innerHTML = editor.innerHTML.replace(new RegExp(term.replace(/[.*+?^${}()|[\]\\]/g,'\\$&'),'gi'), replace);
                        handleEditorInput({ currentTarget: editor });
                      }}><Search size={13}/><span>Replace</span></button>
                    </div>
                  </div>
                  <span className="doc-tool-group-label">Find</span>
                </div>
                {/* Export */}
                <div className="doc-tool-group">
                  <div className="doc-tool-group-rows">
                    <div className="doc-tool-group-row">
                      <button className="doc-tool-btn doc-tool-btn--labeled" title="Download document as text" onClick={() => {
                        const text = richEditorRef.current?.textContent || '';
                        const blob = new Blob([text], { type: 'text/plain' });
                        const a = window.document.createElement('a');
                        a.href = URL.createObjectURL(blob);
                        a.download = (document?.title || 'document') + '.txt';
                        a.click();
                        URL.revokeObjectURL(a.href);
                      }}><Download size={13}/><span>Export TXT</span></button>
                      <button className="doc-tool-btn doc-tool-btn--labeled" title="Download document as HTML" onClick={() => {
                        const html = `<!DOCTYPE html><html><head><meta charset="utf-8"><title>${document?.title || 'Document'}</title></head><body style="max-width:800px;margin:0 auto;font-family:Times New Roman,serif;font-size:12pt;line-height:1.6">${richEditorRef.current?.innerHTML || ''}</body></html>`;
                        const blob = new Blob([html], { type: 'text/html' });
                        const a = window.document.createElement('a');
                        a.href = URL.createObjectURL(blob);
                        a.download = (document?.title || 'document') + '.html';
                        a.click();
                        URL.revokeObjectURL(a.href);
                      }}><Download size={13}/><span>Export HTML</span></button>
                    </div>
                  </div>
                  <span className="doc-tool-group-label">Export</span>
                </div>
                {/* Document Info */}
                <div className="doc-tool-group">
                  <div className="doc-tool-group-rows">
                    <div className="doc-tool-group-row">
                      <button className="doc-tool-btn doc-tool-btn--labeled" title="Document statistics" onClick={() => {
                        const editor = richEditorRef.current;
                        const text = editor?.textContent || '';
                        const wc = text.trim().split(/\s+/).filter(Boolean).length;
                        const cc = text.length;
                        const hc = editor?.querySelectorAll('h1,h2,h3').length || 0;
                        const tc = editor?.querySelectorAll('table').length || 0;
                        const ic = editor?.querySelectorAll('img').length || 0;
                        window.alert(`Document Statistics\n──────────────────\nWords: ${wc}\nCharacters: ${cc}\nHeadings: ${hc}\nTables: ${tc}\nImages: ${ic}`);
                      }}><FileSearch size={13}/><span>Doc Info</span></button>
                    </div>
                  </div>
                  <span className="doc-tool-group-label">Information</span>
                </div>
                {/* Undo/Redo */}
                <div className="doc-tool-group">
                  <div className="doc-tool-group-rows">
                    <div className="doc-tool-group-row">
                      <button className="doc-tool-btn doc-tool-btn--labeled" title="Undo (Ctrl+Z)" onClick={() => document.execCommand('undo')}><RotateCcw size={13}/><span>Undo</span></button>
                      <button className="doc-tool-btn doc-tool-btn--labeled" title="Redo (Ctrl+Y)" onClick={() => document.execCommand('redo')}><RefreshCw size={13}/><span>Redo</span></button>
                    </div>
                  </div>
                  <span className="doc-tool-group-label">History</span>
                </div>
              </div>
            )}

            {/* ── WPS AI tab ── */}
            {activeRibbonTab === 'WPS AI' && (
              <div className="doc-ribbon-toolbar">
                {/* Write */}
                <div className="doc-tool-group">
                  <div className="doc-tool-group-rows">
                    <div className="doc-tool-group-row">
                      <button className="doc-tool-btn doc-tool-btn--labeled" title="Ask AI to continue writing at cursor"
                        disabled={isThinking}
                        onClick={() => { setAiPanelOpen(true); sendMessage('Continue writing from where I left off.'); }}
                      ><Wand2 size={13}/><span>Continue Writing</span></button>
                      <button className="doc-tool-btn doc-tool-btn--labeled" title="Ask AI to rewrite selected text"
                        disabled={isThinking}
                        onClick={() => {
                          const sel = window.getSelection()?.toString().trim();
                          const msg = sel ? `Rewrite this more clearly: "${sel}"` : 'Rewrite the last paragraph more clearly.';
                          setAiPanelOpen(true);
                          setInputValue(msg);
                        }}
                      ><Wand2 size={13}/><span>Rewrite</span></button>
                    </div>
                    <div className="doc-tool-group-row">
                      <button className="doc-tool-btn doc-tool-btn--labeled" title="Ask AI to improve selected text"
                        disabled={isThinking}
                        onClick={() => {
                          const sel = window.getSelection()?.toString().trim();
                          const msg = sel ? `Improve this text: "${sel}"` : 'Improve the writing quality of the document.';
                          setAiPanelOpen(true);
                          setInputValue(msg);
                        }}
                      ><Star size={13}/><span>Improve</span></button>
                      <button className="doc-tool-btn doc-tool-btn--labeled" title="Summarise the document with AI"
                        disabled={isThinking}
                        onClick={() => { setAiPanelOpen(true); sendMessage('Summarise the entire document in 3-5 bullet points.'); }}
                      ><FileSearch size={13}/><span>Summarise</span></button>
                    </div>
                  </div>
                  <span className="doc-tool-group-label">Writing Assistance</span>
                </div>
                {/* Research */}
                <div className="doc-tool-group">
                  <div className="doc-tool-group-rows">
                    <div className="doc-tool-group-row">
                      <button className="doc-tool-btn doc-tool-btn--labeled" title="Ask AI a research question"
                        disabled={isThinking}
                        onClick={() => {
                          const q = window.prompt('Research question:');
                          if (!q) return;
                          setAiPanelOpen(true);
                          sendMessage(q);
                        }}
                      ><BookOpen size={13}/><span>Research</span></button>
                    </div>
                  </div>
                  <span className="doc-tool-group-label">Research</span>
                </div>
                {/* Translate */}
                <div className="doc-tool-group">
                  <div className="doc-tool-group-rows">
                    <div className="doc-tool-group-row">
                      <button className="doc-tool-btn doc-tool-btn--labeled" title="Translate selected text"
                        disabled={isThinking}
                        onClick={() => {
                          const sel = window.getSelection()?.toString().trim();
                          const lang = window.prompt('Translate to language:', 'French');
                          if (!lang) return;
                          setAiPanelOpen(true);
                          sendMessage(sel ? `Translate this to ${lang}: "${sel}"` : `Translate the document to ${lang}.`);
                        }}
                      ><Globe size={13}/><span>Translate</span></button>
                    </div>
                  </div>
                  <span className="doc-tool-group-label">Language</span>
                </div>
                {/* Grammar */}
                <div className="doc-tool-group">
                  <div className="doc-tool-group-rows">
                    <div className="doc-tool-group-row">
                      <button className="doc-tool-btn doc-tool-btn--labeled" title="Check grammar and style with AI"
                        disabled={isThinking}
                        onClick={() => {
                          const sel = window.getSelection()?.toString().trim();
                          setAiPanelOpen(true);
                          sendMessage(sel ? `Check and fix the grammar in: "${sel}"` : 'Check the grammar and academic style of the entire document and suggest improvements.');
                        }}
                      ><SpellCheck size={13}/><span>Grammar Check</span></button>
                    </div>
                  </div>
                  <span className="doc-tool-group-label">Grammar</span>
                </div>
                {/* AI Detection */}
                <div className="doc-tool-group">
                  <div className="doc-tool-group-rows">
                    <div className="doc-tool-group-row">
                      <button
                        className={`doc-tool-btn doc-tool-btn--labeled${aiDetectResult && !aiDetecting ? ' doc-tool-btn--active' : ''}`}
                        title="Detect AI-generated content (perplexity + burstiness analysis)"
                        disabled={aiDetecting}
                        onClick={runAiDetect}
                      >
                        <ShieldCheck size={13}/>
                        <span>{aiDetecting ? 'Scanning…' : 'Detect AI'}</span>
                      </button>
                    </div>
                    <div className="doc-tool-group-row">
                      <button
                        className="doc-tool-btn doc-tool-btn--labeled"
                        title="Clear AI detection highlights"
                        disabled={!aiDetectResult}
                        onClick={clearAiHighlights}
                      >
                        <X size={13}/>
                        <span>Clear</span>
                      </button>
                    </div>
                  </div>
                  <span className="doc-tool-group-label">AI Detection</span>
                </div>
                {/* Tone */}
                <div className="doc-tool-group">
                  <div className="doc-tool-group-rows">
                    <div className="doc-tool-group-row">
                      <button className="doc-tool-btn doc-tool-btn--labeled" title="Make writing more formal/academic"
                        disabled={isThinking}
                        onClick={() => { setAiPanelOpen(true); sendMessage('Make the writing style more formal and academic throughout the document.'); }}
                      ><Star size={13}/><span>Formalise</span></button>
                    </div>
                    <div className="doc-tool-group-row">
                      <button className="doc-tool-btn doc-tool-btn--labeled" title="Expand the document with more detail"
                        disabled={isThinking}
                        onClick={() => { setAiPanelOpen(true); sendMessage('Expand each section with more detail and supporting analysis.'); }}
                      ><Wand2 size={13}/><span>Expand</span></button>
                    </div>
                  </div>
                  <span className="doc-tool-group-label">Tone &amp; Style</span>
                </div>
              </div>
            )}

            {/* Other tabs (fallback) */}
            {!['Home','Insert','Page Layout','References','Review','View','Tools','WPS AI'].includes(activeRibbonTab) && (
              <div className="doc-ribbon-toolbar doc-ribbon-toolbar--placeholder">
                <span style={{color:'#9aa0a6',fontSize:'12px',padding:'4px 12px'}}>{activeRibbonTab} tools coming soon</span>
              </div>
            )}
          </div>

          {/* Paper */}
          <section className="doc-paper-zone">
            <div className="doc-page-canvas">
              <div className="doc-paper-scroll">
                <div className="doc-page-body-zone">
                  {!!manualError && <p className="doc-manual-error">{manualError}</p>}
                  {!!highlightedSections.length && (
                    <div className="doc-change-highlight-strip" role="status" aria-live="polite">
                      <span className="doc-change-highlight-label">Agent updated:</span>
                      {highlightedSections.map((title, idx) => (
                        <span className="doc-change-highlight-chip" key={`${title}-${idx}`}>{title}</span>
                      ))}
                    </div>
                  )}
                  <div
                    ref={richEditorRef}
                    className="doc-rich-editor"
                    contentEditable
                    suppressContentEditableWarning
                    spellCheck
                    onInput={handleEditorInput}
                    onKeyDown={handleEditorKeyDown}
                    onMouseUp={updateActiveFormats}
                    onKeyUp={updateActiveFormats}
                  />
                </div>
              </div>
            </div>
          </section>
        </div>

        {/* ── Right column: full-height AI chat ── */}
        {aiPanelOpen && <aside className="doc-ai-panel">

          {/* ── Panel header ── */}
          <div className="dap-header">
            <div className="dap-header-left">
              <div className="dap-header-logo">
                <Wand2 size={14} />
              </div>
              <span className="dap-title">{showAiDetectPanel ? 'AI Detection' : showCommentsPanel ? 'Comments' : 'Copilot'}</span>
            </div>
            <div className="dap-header-right">
              <button
                type="button"
                className={`dap-head-icon-btn${showAiDetectPanel ? ' dap-head-icon-btn--active' : ''}`}
                title="AI Detection results"
                onClick={() => { setShowAiDetectPanel(p => !p); setShowCommentsPanel(false); }}
              >
                <ShieldCheck size={13} className="dap-icon-btn" />
                {aiDetectResult && !showAiDetectPanel && (
                  <span className="dap-comment-badge" style={{background: aiDetectResult.verdict === 'likely_ai' ? '#ef4444' : aiDetectResult.verdict === 'mixed' ? '#ca8a04' : '#16a34a'}}>!</span>
                )}
              </button>
              <button
                type="button"
                className={`dap-head-icon-btn${showCommentsPanel ? ' dap-head-icon-btn--active' : ''}`}
                title={showCommentsPanel ? 'Back to Chat' : `Comments${docComments.length ? ` (${docComments.length})` : ''}`}
                onClick={() => { setShowCommentsPanel(p => !p); setShowAiDetectPanel(false); }}
              >
                <MessageCircle size={13} className="dap-icon-btn" />
                {docComments.length > 0 && !showCommentsPanel && (
                  <span className="dap-comment-badge">{docComments.length}</span>
                )}
              </button>
              <button type="button" className="dap-head-icon-btn" onClick={createNewChat} title="New Chat">
                <Plus size={15} className="dap-icon-btn" />
              </button>
              <button type="button" className="dap-head-icon-btn" title="Chat history" onClick={() => setShowChatList((prev) => !prev)}>
                <RotateCcw size={13} className="dap-icon-btn" />
              </button>
              <button
                type="button"
                className="dap-head-icon-btn dap-close-btn"
                onClick={() => setAiPanelOpen(false)}
                title="Close"
              >
                <X size={14} />
              </button>
            </div>
          </div>

          {/* ── AI Detection panel ── */}
          {showAiDetectPanel ? (
            <div className="dap-comments-panel" style={{padding:'12px 14px',overflowY:'auto'}}>
              {aiDetecting ? (
                <div style={{display:'flex',flexDirection:'column',alignItems:'center',gap:10,padding:'32px 0',color:'#6b7280'}}>
                  <ShieldCheck size={32} strokeWidth={1.5} style={{animation:'spin 1.5s linear infinite'}}/>
                  <span style={{fontSize:13}}>Scanning for AI content…</span>
                </div>
              ) : aiDetectResult?.error ? (
                <div style={{color:'#ef4444',fontSize:12,padding:8}}>{aiDetectResult.error}</div>
              ) : aiDetectResult ? (
                <>
                  {/* Score gauge */}
                  <div style={{textAlign:'center',marginBottom:14}}>
                    {(() => {
                      const pct = aiDetectResult.overall_ai_percentage ?? 0;
                      const color = pct >= 65 ? '#ef4444' : pct >= 30 ? '#ca8a04' : '#16a34a';
                      const label = aiDetectResult.verdict === 'likely_ai' ? 'Likely AI-Generated'
                        : aiDetectResult.verdict === 'mixed' ? 'Mixed (Human + AI)'
                        : 'Likely Human-Written';
                      const Icon = aiDetectResult.verdict === 'likely_ai' ? ShieldAlert
                        : aiDetectResult.verdict === 'mixed' ? AlertTriangle
                        : ShieldCheck;
                      return (
                        <>
                          <Icon size={28} color={color} strokeWidth={1.8} style={{marginBottom:6}}/>
                          <div style={{fontSize:28,fontWeight:700,color,lineHeight:1}}>{pct}%</div>
                          <div style={{fontSize:11,color:'#6b7280',marginTop:3}}>AI Probability</div>
                          <div style={{fontSize:12,fontWeight:600,color,marginTop:4}}>{label}</div>
                          <div style={{marginTop:8,height:6,borderRadius:4,background:'#e5e7eb',overflow:'hidden'}}>
                            <div style={{height:'100%',width:`${pct}%`,background:color,transition:'width 0.6s ease',borderRadius:4}}/>
                          </div>
                          <div style={{display:'flex',justifyContent:'space-between',fontSize:10,color:'#9ca3af',marginTop:3}}>
                            <span>Human</span><span>AI</span>
                          </div>
                          <div style={{fontSize:11,color:'#6b7280',marginTop:8}}>
                            Burstiness: <strong>{aiDetectResult.burstiness?.toFixed(2)}</strong>
                            <span style={{marginLeft:6,color:'#9ca3af'}}>(low = uniform = AI)</span>
                          </div>
                        </>
                      );
                    })()}
                  </div>
                  {/* Legend */}
                  <div style={{display:'flex',gap:10,fontSize:10,color:'#6b7280',marginBottom:10,flexWrap:'wrap'}}>
                    <span style={{display:'flex',alignItems:'center',gap:3}}><span style={{width:10,height:10,borderRadius:2,background:'rgba(239,68,68,0.3)',border:'1px solid #ef4444',display:'inline-block'}}/> Likely AI</span>
                    <span style={{display:'flex',alignItems:'center',gap:3}}><span style={{width:10,height:10,borderRadius:2,background:'rgba(234,179,8,0.3)',border:'1px solid #ca8a04',display:'inline-block'}}/> Uncertain</span>
                    <span style={{display:'flex',alignItems:'center',gap:3}}><span style={{width:10,height:10,borderRadius:2,background:'transparent',border:'1px solid #d1d5db',display:'inline-block'}}/> Human</span>
                  </div>
                  {/* Flagged sentences */}
                  {aiDetectResult.sentences?.filter(s => s.label !== 'likely_human').length > 0 ? (
                    <div>
                      <div style={{fontSize:11,fontWeight:600,color:'#374151',marginBottom:6}}>
                        Flagged passages ({aiDetectResult.sentences.filter(s=>s.label!=='likely_human').length})
                      </div>
                      <div style={{display:'flex',flexDirection:'column',gap:6}}>
                        {aiDetectResult.sentences.filter(s=>s.label!=='likely_human').map((s,i)=>{
                          const c = s.label==='likely_ai' ? '#ef4444' : '#ca8a04';
                          const bg = s.label==='likely_ai' ? 'rgba(239,68,68,0.07)' : 'rgba(234,179,8,0.07)';
                          return (
                            <div key={i} style={{background:bg,border:`1px solid ${c}33`,borderRadius:5,padding:'6px 8px'}}>
                              <div style={{fontSize:10,color:c,fontWeight:600,marginBottom:2}}>
                                {Math.round(s.ai_probability*100)}% AI · {s.label==='likely_ai'?'Likely AI':'Uncertain'}
                              </div>
                              <div style={{fontSize:11,color:'#374151',lineHeight:1.5}}>
                                {s.text.length > 140 ? s.text.slice(0,140)+'…' : s.text}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  ) : (
                    <div style={{fontSize:12,color:'#16a34a',textAlign:'center',padding:'12px 0'}}>
                      <ShieldCheck size={20} style={{marginBottom:4}}/><br/>No AI passages flagged
                    </div>
                  )}
                  <div style={{fontSize:10,color:'#9ca3af',marginTop:12,lineHeight:1.5}}>
                    Uses perplexity approximation + burstiness analysis, mirroring Turnitin's published AI detection methodology.
                  </div>
                </>
              ) : (
                <div style={{display:'flex',flexDirection:'column',alignItems:'center',gap:8,padding:'32px 0',color:'#9ca3af'}}>
                  <ShieldCheck size={28} strokeWidth={1.5}/>
                  <span style={{fontSize:12}}>Click <strong>Detect AI</strong> in the WPS AI tab to scan the document.</span>
                </div>
              )}
            </div>
          ) : null}

          {/* ── Comments panel ── */}
          {!showAiDetectPanel && showCommentsPanel ? (
            <div className="dap-comments-panel">
              {docComments.length === 0 ? (
                <div className="dap-comments-empty">
                  <MessageCircle size={28} strokeWidth={1.5} />
                  <p className="dap-comments-empty-title">No comments yet</p>
                  <p className="dap-comments-empty-sub">Select text and use the <strong>New Comment</strong> button in the Review tab to annotate your document.</p>
                </div>
              ) : (
                <>
                  <div className="dap-comments-header">
                    <span className="dap-comments-count">{docComments.length} comment{docComments.length !== 1 ? 's' : ''}</span>
                    <button
                      type="button"
                      className="dap-comments-address-btn"
                      title="Ask AI to address all comments"
                      disabled={isThinking}
                      onClick={() => {
                        setShowCommentsPanel(false);
                        sendMessage('Address all comments in the document');
                      }}
                    >
                      <CheckCircle2 size={12} />
                      Address all with AI
                    </button>
                  </div>
                  <div className="dap-comments-list">
                    {docComments.map((c, i) => (
                      <div key={i} className="dap-comment-card">
                        <div className="dap-comment-card-section">{c.sectionTitle || 'Untitled'}</div>
                        <div className="dap-comment-card-text">{c.text}</div>
                        <div className="dap-comment-card-actions">
                          <button
                            type="button"
                            className="dap-comment-action-btn"
                            title="Ask AI to address this comment"
                            disabled={isThinking}
                            onClick={() => {
                              setShowCommentsPanel(false);
                              sendMessage(`Address the comment "${c.text}" in the "${c.sectionTitle}" section`);
                            }}
                          >
                            <Wand2 size={10} /> Fix with AI
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>
          ) : (
          <>
          {/* ── Messages area ── */}
          <div className="dap-messages">
            {showChatList ? (
              /* Chat history list */
              <div className="dap-chat-list">
                <p className="dap-chat-list-label">Recent chats</p>
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
                    <MessageSquare size={13} className="dap-chat-item-icon" />
                    <span className="dap-chat-name">{chat.name}</span>
                    <span className="dap-chat-meta">{chat.messages.length}</span>
                  </button>
                ))}
              </div>
            ) : messages.length === 0 && !isThinking ? (
              /* Empty / welcome state */
              <div className="dap-empty-state">
                <div className="dap-empty-icon-wrap">
                  <Wand2 size={28} />
                </div>
                <h3 className="dap-empty-title">How can I help?</h3>
                <p className="dap-empty-sub">Ask anything about your document</p>
                <div className="dap-suggestion-chips">
                  {[
                    'Summarise this document',
                    'Improve the writing',
                    'Check grammar & style',
                    docComments.length > 0 ? `Address ${docComments.length} comment${docComments.length !== 1 ? 's' : ''}` : 'Continue writing',
                  ].map((prompt) => (
                    <button
                      key={prompt}
                      className="dap-suggestion-chip"
                      onClick={() => sendMessage(prompt)}
                      disabled={isThinking}
                    >
                      {prompt}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              /* Message thread */
              messages.map((msg) =>
                msg.role === 'assistant' ? (
                  <div key={msg.id} className="dap-msg dap-msg--ai">
                    <div className="dap-ai-avatar">
                      <Wand2 size={11} />
                    </div>
                    <div className="dap-msg-content">
                      <div className="dap-msg-body">
                        <MdText text={msg.text} />
                        {msg.summary && (
                          <CopilotWorkflowCard
                            summary={msg.summary}
                            planItems={msg.plan || []}
                            msgId={msg.id}
                            workflow={msg.workflow || null}
                          />
                        )}
                        {!!msg?.research?.enabled && !!msg?.research?.top_sources?.length && (
                          <details className="dap-research-badge">
                            <summary>
                              Grounded on {msg.research.top_sources.length} real paper{msg.research.top_sources.length !== 1 ? 's' : ''}
                            </summary>
                            <ul>
                              {msg.research.top_sources.map((source, idx) => (
                                <li key={`${msg.id}-src-${idx}`}>
                                  {source?.title || 'Untitled'}
                                  {source?.year ? ` (${source.year})` : ''}
                                  {source?.doi ? ` - DOI: ${source.doi}` : ''}
                                </li>
                              ))}
                            </ul>
                          </details>
                        )}
                      </div>
                      {!!msg?.changeSet?.editedSections?.length && (
                        <div className="dap-change-summary">
                          <span className="dap-change-summary-label">Updated sections:</span>
                          {msg.changeSet.editedSections.map((title, idx) => (
                            <span className="dap-change-chip" key={`${msg.id}-${idx}`}>{title}</span>
                          ))}
                        </div>
                      )}
                      {!!msg?.changeSet?.pending && (
                        <div className="dap-task-actions">
                          <button type="button" className="dap-keep-btn" onClick={() => keepAgentChanges(activeChatId, msg.id)}>Keep</button>
                          <button type="button" className="dap-undo-btn" onClick={() => undoAgentChanges(activeChatId, msg.id, msg.changeSet.beforeSections || [])} disabled={isSavingManual}>Undo</button>
                        </div>
                      )}
                      <div className="dap-msg-actions">
                        <button className="dap-msg-act-btn" title="Retry" disabled={isThinking} onClick={() => {
                          const msgIdx = messages.findIndex((m) => m.id === msg.id);
                          const lastUser = [...messages].slice(0, msgIdx).reverse().find((m) => m.role === 'user');
                          if (lastUser) sendMessage(lastUser.text);
                        }}><RotateCcw size={11} /></button>
                        <button className="dap-msg-act-btn" title="Copy" onClick={() => navigator.clipboard?.writeText(msg.text)}><Copy size={11} /></button>
                        <button className="dap-msg-act-btn" title="Dislike"><ThumbsDown size={11} /></button>
                        <span className="dap-model-tag">{activeModel}</span>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div key={msg.id} className="dap-msg dap-msg--user">
                    <div className="dap-user-bubble">{msg.text}</div>
                  </div>
                )
              )
            )}

            {!showChatList && (
              <>
                {isThinking && liveProgressMsgId && currentActivity && (
                  <AgentActivityBar activity={currentActivity} />
                )}
                {isThinking && !liveProgressMsgId && (
                  <div className="dap-msg dap-msg--ai">
                    <div className="dap-ai-avatar"><Wand2 size={11} /></div>
                    <div className="dap-msg-content">
                      <div className="dap-typing-dots">
                        <span /><span /><span />
                      </div>
                    </div>
                  </div>
                )}
                <div ref={bottomRef} />
              </>
            )}
          </div>

          {/* ── Agent Todo Panel — expandable, above composer ── */}
          {!showChatList && (
            <AgentTodoPanel
              plan={activePlan}
              isActive={!!liveProgressMsgId && isThinking}
            />
          )}

          {/* ── Composer ── */}
          {!showChatList && (
            <div className="dap-composer">
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.docx,.txt"
                style={{ display: 'none' }}
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) setAttachedFile(f);
                  e.target.value = '';
                }}
              />
              {attachedFile && (
                <div className="dap-attachment-chip">
                  <Paperclip size={11} />
                  <span className="dap-attachment-name">{attachedFile.name}</span>
                  <button type="button" className="dap-attachment-remove" onClick={() => setAttachedFile(null)} title="Remove"><X size={11} /></button>
                </div>
              )}
              {groundedResearch && (
                <div className="dap-research-banner">
                  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>
                  <span>Research Mode — grounded in real academic papers</span>
                  <button type="button" onClick={() => setGroundedResearch(false)} title="Disable Research Mode"><X size={10} /></button>
                </div>
              )}
              <div className="dap-composer-box">
                <textarea
                  className="dap-composer-input"
                  placeholder={groundedResearch ? 'Ask Copilot (research mode)…' : 'Ask Copilot…'}
                  rows={1}
                  value={inputValue}
                  onChange={(e) => {
                    setInputValue(e.target.value);
                    e.target.style.height = 'auto';
                    e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
                  }}
                  onKeyDown={handleKey}
                />
                <div className="dap-composer-actions">
                  <button type="button" className="dap-attach-btn" onClick={() => fileInputRef.current?.click()} title="Attach file" disabled={isThinking}>
                    <Paperclip size={13} />
                  </button>
                  <button
                    type="button"
                    className={`dap-research-btn${groundedResearch ? ' active' : ''}`}
                    onClick={() => setGroundedResearch((v) => !v)}
                    title={groundedResearch ? 'Research Mode ON — click to disable' : 'Enable Research Mode (grounds responses in real academic papers)'}
                    disabled={isThinking}
                  >
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>
                  </button>
                  <select
                    className="dap-model-pill"
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    disabled={isThinking}
                    title="Select model"
                  >
                    <option value="gemini">Gemini</option>
                    <option value="grok">Grok</option>
                  </select>
                  <button
                    className="dap-send-btn"
                    onClick={() => sendMessage(inputValue)}
                    disabled={!inputValue.trim() || isThinking}
                    title="Send (Enter)"
                  >
                    <Send size={13} />
                  </button>
                </div>
              </div>
              <p className="dap-composer-hint">Enter to send · Shift+Enter for new line</p>
            </div>
          )}
          </>
          )}
        </aside>}
      </div>

      {/* ── Floating chat toggle ── */}
      {!aiPanelOpen && (
        <button
          type="button"
          className="doc-chat-fab"
          onClick={() => setAiPanelOpen(true)}
          title="Open AI Chat"
        >
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
          </svg>
        </button>
      )}

      {/* ── Status bar — full width, below both columns ── */}
      <footer className="doc-status-bar">
        <div className="doc-status-left">
          <span className="doc-status-item">Page: {pageCount}</span>
          <span className="doc-status-item">Words: {wordCount.toLocaleString()}</span>
          {wordCount > 0 && (
            <span className="doc-status-item">{Math.max(1, Math.ceil(wordCount / 200))} min read</span>
          )}
          {isSavingManual ? (
            <span className="doc-status-item doc-status-saving">Saving…</span>
          ) : autoSaved && !isDirty ? (
            <span className="doc-status-item doc-status-saved">✓ Saved</span>
          ) : isDirty ? (
            <span className="doc-status-item doc-status-unsaved">● Unsaved</span>
          ) : null}
          <button type="button" className="doc-spell-btn">
            <span className="doc-spell-indicator"><Check size={11} /></span>
            AI Spell Check
          </button>
        </div>
        <div className="doc-status-right">
          <button type="button" className="doc-status-icon-btn"><Menu size={14} /></button>
          <button type="button" className="doc-status-icon-btn"><Play size={14} /></button>
          <button type="button" className="doc-status-icon-btn"><Globe size={14} /></button>
          <span className="doc-zoom-value">100%</span>
          <ZoomOut size={12} className="doc-zoom-icon" />
          <div className="doc-zoom-track" aria-hidden="true">
            <span className="doc-zoom-fill" style={{ width: '100%' }} />
          </div>
          <ZoomIn size={12} className="doc-zoom-icon" />
        </div>
      </footer>
    </div>
  );
}
