import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
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
import { chatWithDocument, getDocument } from '../api/client';

const sampleParagraph = `An analysis of revenue streams focusing on rates as the main source of income at city level.

Local authorities rely heavily on rates as a basic fiscal tool for service delivery and investment. This section discusses financial sustainability and limitations in over-reliance on property tax collections.`;

const INITIAL_MESSAGES = [];
const INITIAL_CHAT_ID = 'chat-initial';
const INITIAL_CHATS = [{ id: INITIAL_CHAT_ID, name: 'New Chat', messages: INITIAL_MESSAGES }];
const MIN_EDITOR_HEIGHT = 900;
const PAGE_CYCLE_PX = 1120;
const DISSERTATION_REQUEST_RE = /(full|complete|entire).{0,30}(dissertation|thesis|project)|write.{0,20}(dissertation|thesis|project)|generate.{0,20}(dissertation|thesis|project)/i;
const DISSERTATION_TODO_TEMPLATE = [
  {
    chapter: 'Chapter 1: Introduction',
    items: [
      '1.1 Background of the Study',
      '1.2 Statement of the Problem',
      '1.3 Research Objectives',
      '1.4 Research Questions',
      '1.5 Research Hypotheses',
      '1.6 Significance of the Study',
      '1.7 Scope and Delimitations',
      '1.8 Definition of Key Terms',
    ],
  },
  {
    chapter: 'Chapter 2: Literature Review',
    items: [
      '2.1 Introduction',
      '2.2 Conceptual Review',
      '2.2.1 Key Concepts',
      '2.2.2 Variables and Relationships',
      '2.3 Theoretical Framework',
      '2.3.1 Supporting Theories',
      '2.3.2 Applicability to the Study',
      '2.4 Empirical Review',
      '2.4.1 Evidence from Developed Economies',
      '2.4.2 Evidence from Developing Economies',
      '2.4.3 Synthesis of Empirical Findings',
      '2.5 Research Gap',
      '2.6 Chapter Summary',
    ],
  },
  {
    chapter: 'Chapter 3: Methodology',
    items: [
      '3.1 Introduction',
      '3.2 Research Design',
      '3.3 Target Population',
      '3.4 Sampling Techniques and Sample Size',
      '3.5 Data Collection Methods',
      '3.6 Data Analysis Techniques',
      '3.7 Reliability and Validity',
      '3.8 Ethical Considerations',
      '3.9 Chapter Summary',
    ],
  },
  {
    chapter: 'Chapter 4: Results and Discussion',
    items: [
      '4.1 Data Presentation',
      '4.2 Objective-wise Findings',
      '4.3 Discussion of Findings',
      '4.4 Chapter Summary',
    ],
  },
  {
    chapter: 'Chapter 5: Conclusion and Recommendations',
    items: [
      '5.1 Introduction',
      '5.2 Summary of Findings',
      '5.3 Conclusions',
      '5.4 Recommendations',
      '5.5 Limitations of the Study',
      '5.6 Areas for Further Research',
    ],
  },
  {
    chapter: 'Chapter 6: References and Appendices',
    items: ['6.1 References', '6.2 Appendices'],
  },
];

function normalizeStep(step = '') {
  return step.replace(/^[-\s]+/, '').trim();
}

function looksLikeDissertationRequest(text = '') {
  return DISSERTATION_REQUEST_RE.test((text || '').trim());
}

function createDissertationPreviewPlan() {
  const steps = [{ step: 'Creating dissertation to-do list', status: 'done' }];
  for (const chapter of DISSERTATION_TODO_TEMPLATE) {
    steps.push({ step: `Writing ${chapter.chapter}`, status: 'pending' });
    for (const item of chapter.items) {
      steps.push({ step: `  Writing ${item}`, status: 'pending' });
    }
  }
  const firstPending = steps.findIndex((step) => step.status === 'pending');
  if (firstPending >= 0) {
    steps[firstPending] = { ...steps[firstPending], status: 'in_progress' };
  }
  return steps;
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

function CopilotWorkflowCard({ summary, planItems, msgId, workflow }) {
  const pct = summary?.completion_percent ?? 0;
  const done = summary?.tasks_completed ?? 0;
  const total = (summary?.tasks_completed || 0) + (summary?.tasks_pending || 0);
  const updates = workflow?.updates || [];
  return (
    <div className="copilot-workflow-card">
      <div className="copilot-workflow-head">
        <span className="copilot-badge">Agent Workflow</span>
        <span className="copilot-stage">{summary?.stage || 'Working'}</span>
      </div>
      <div className="copilot-progress-row">
        <div className="copilot-progress-track">
          <span className="copilot-progress-fill" style={{ width: `${Math.max(0, Math.min(100, pct))}%` }} />
        </div>
        <span className="copilot-progress-text">{done}/{total || planItems.length || 0}</span>
      </div>
      {!!workflow?.currentStep && (
        <p className="copilot-current-step">Now doing: {workflow.currentStep}</p>
      )}
      {!!updates.length && (
        <div className="copilot-updates">
          {updates.slice(0, 6).map((line, idx) => (
            <p key={`${msgId}-upd-${idx}`} className="copilot-update-line">{line}</p>
          ))}
        </div>
      )}
      <DissertationPlan planItems={planItems || []} todoList={summary?.todo_list || []} msgId={msgId} chapterPlan={summary?.chapter_plan || []} />
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

function renderContentWithMarkers(section, sectionIndex) {
  const rawContent = section?.content || '';
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
          parts.push(<p key={`s${sectionIndex}-t${parts.length}-${idx}`}>{para}</p>);
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
        parts.push(<p key={`s${sectionIndex}-tail-${idx}`}>{para}</p>);
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

function flattenSections(content) {
  if (!content?.sections?.length) return sampleParagraph;
  return content.sections
    .map((s) => `${s.title}\n${s.content || ''}`)
    .join('\n\n');
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
        <span className="pnode-label">{node.label}</span>
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
  const bottomRef    = useRef(null);
  const autoSaveTimer = useRef(null);
  const progressPollRef = useRef(null);
  const editorTextareaRef = useRef(null);
  const progressPollBusyRef = useRef(false);
  const workflowStateRef = useRef({});

  function clearProgressPolling() {
    if (progressPollRef.current) {
      clearInterval(progressPollRef.current);
      progressPollRef.current = null;
    }
    progressPollBusyRef.current = false;
  }

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

  // Auto-resize textarea to content height (no inner scrollbar)
  useEffect(() => {
    const el = editorTextareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    const nextHeight = Math.max(el.scrollHeight, MIN_EDITOR_HEIGHT);
    el.style.height = nextHeight + 'px';
    setEditorHeight(nextHeight);
  }, [plainDocText]);

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
    if (!onManualSave) return;
    setIsSavingManual(true);
    setManualError('');
    try {
      // use latest draft via closure or argument
      setDraftSections((current) => {
        const target = sections || current;
        const cleaned = target
          .map((s) => ({
            title: (s.title || '').trim() || 'Untitled section',
            content: s.content || '',
            ...(Array.isArray(s.blocks) && s.blocks.length ? { blocks: s.blocks } : {}),
          }))
          .filter((s) => s.title || s.content);
        onManualSave(cleaned).then(() => {
          setIsDirty(false);
          setAutoSaved(true);
          onDocumentChanged?.();
          setTimeout(() => setAutoSaved(false), 2500);
        }).catch((err) => {
          setManualError(err?.message || 'Save failed');
        }).finally(() => {
          setIsSavingManual(false);
        });
        return current;
      });
    } catch (err) {
      setManualError(err?.message || 'Save failed');
      setIsSavingManual(false);
    }
  }, [onManualSave, onDocumentChanged]);

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

        const previousWorkflow = workflowStateRef.current[messageId] || null;
        const nextWorkflow = buildWorkflowFromPlan(planFromDoc, previousWorkflow);
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

  async function playbackDissertationResult(chatId, messageId, result, previewPlan) {
    const generatedSections = Array.isArray(result?.document?.content?.sections)
      ? result.document.content.sections
      : [];

    if (!generatedSections.length) {
      updateAssistantMessage(chatId, messageId, (msg) => ({
        ...msg,
        text: result.reply,
        summary: buildSummaryFromResult(result),
        plan: Array.isArray(result.plan) ? result.plan : msg.plan || [],
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
    }));
  }

  async function sendMessage(text) {
    if (!text.trim() || isThinking) return;
    const userText = text.trim();
    const userMsg = { id: Date.now(), role: 'user', text: text.trim() };
    const currentChatId = activeChatId;
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
      previewPlan = createDissertationPreviewPlan();
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
                    text: 'Starting dissertation workflow...',
                    summary: buildSummaryFromPlan(previewPlan, `Generating ${DISSERTATION_TODO_TEMPLATE[0].chapter}...`),
                    plan: previewPlan,
                    workflow: {
                      currentStep: statusLabel(previewPlan.find((item) => item.status === 'in_progress')?.step || ''),
                      completedCount: previewPlan.filter((item) => item.status === 'done').length,
                      totalCount: previewPlan.length,
                      statuses: previewPlan.map((item) => item.status),
                      updates: [
                        'Created dissertation to-do list',
                        `Now doing: ${statusLabel(previewPlan.find((item) => item.status === 'in_progress')?.step || '')}`,
                      ],
                    },
                  },
                ],
              }
            : chat
        )
      );
      setLiveProgressMsgId(progressMessageId);
      startDissertationProgressPolling(document?.id, currentChatId, progressMessageId, previewPlan);
    }

    try {
      const result = await chatWithDocument(document?.id, userText, selectedModel);
      if (result?.model) {
        setActiveModel(result.model);
      }
      clearProgressPolling();

      if (dissertationRequest) {
        await playbackDissertationResult(currentChatId, progressMessageId, result, previewPlan);
      } else {
        setChats((prev) =>
          prev.map((chat) =>
            chat.id === currentChatId
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
              {!!manualError && <p className="doc-manual-error">{manualError}</p>}
              <div className="doc-manual-editor doc-manual-editor--plain">
                <textarea
                  ref={editorTextareaRef}
                  className="doc-paper-editor"
                  value={plainDocText}
                  placeholder=""
                  autoFocus
                  onChange={(e) => {
                    const next = e.target.value;
                    // auto-resize inline
                    e.target.style.height = 'auto';
                    const nextHeight = Math.max(e.target.scrollHeight, MIN_EDITOR_HEIGHT);
                    e.target.style.height = nextHeight + 'px';
                    setEditorHeight(nextHeight);
                    setDraftSections([{ title: '', content: next, blocks: [] }]);
                    setIsDirty(true);
                    setAutoSaved(false);
                    clearTimeout(autoSaveTimer.current);
                    autoSaveTimer.current = setTimeout(() => {
                      triggerSave([{ title: '', content: next, blocks: [] }]);
                    }, 1500);
                  }}
                />
                <div className="doc-page-guides" aria-hidden="true">
                  {Array.from({ length: Math.max(pageCount - 1, 0) }).map((_, i) => (
                    <div
                      key={`break-${i + 1}`}
                      className="doc-page-break-line"
                      style={{ top: `${(i + 1) * PAGE_CYCLE_PX - 12}px` }}
                    />
                  ))}
                  {Array.from({ length: pageCount }).map((_, i) => (
                    <span
                      key={`page-${i + 1}`}
                      className="doc-page-number-chip"
                      style={{ top: `${i * PAGE_CYCLE_PX + 10}px` }}
                    >
                      Page {i + 1}
                    </span>
                  ))}
                </div>
              </div>
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
                        <CopilotWorkflowCard
                          summary={msg.summary}
                          planItems={msg.plan || []}
                          msgId={msg.id}
                          workflow={msg.workflow || null}
                        />
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
                {isThinking && !liveProgressMsgId && (
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
          <span className="doc-status-item">Page: 1</span>
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
