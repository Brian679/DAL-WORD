import sys

file_path = "frontend/src/components/DocumentEditorPage.jsx"
with open(file_path, encoding='utf-8') as f:
    code = f.read()

old_editor = """
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
"""

new_editor = """
              <div className="doc-manual-editor doc-manual-editor--rich">
                <div 
                  ref={editorTextareaRef} 
                  className="doc-paper-editor" 
                  style={{ display: 'flex', flexDirection: 'column', gap: '8px', minHeight: MIN_EDITOR_HEIGHT }}
                >
                  {draftSections.map((s, i) => {
                    let HeadingTag = 'h2';
                    const titleLowerCase = (s.title || '').toLowerCase();
                    if (/^chapter\\s+\\d/i.test(s.title) || titleLowerCase.startsWith('chapter')) {
                      HeadingTag = 'h1';
                    } else if (/(^\\d+\\.\\d+\\.\\d+)/.test(s.title)) {
                      HeadingTag = 'h3';
                    } else if (/(^\\d+\\.\\d+)/.test(s.title) || titleLowerCase.includes("introduction") || titleLowerCase.includes("review")) {
                      HeadingTag = 'h2';
                    }

                    return (
                      <div key={i} className="doc-section-block">
                        {s.title && <HeadingTag className={`doc-heading-${HeadingTag}`} style={{ margin: '10px 0 5px 0' }}>{s.title}</HeadingTag>}
                        <textarea
                          style={{
                            width: '100%',
                            minHeight: '40px',
                            border: 'none',
                            outline: 'none',
                            resize: 'none',
                            fontFamily: 'inherit',
                            fontSize: '14px',
                            lineHeight: '1.6',
                            overflow: 'hidden'
                          }}
                          value={s.content}
                          onChange={(e) => {
                            const next = e.target.value;
                            e.target.style.height = 'auto';
                            e.target.style.height = e.target.scrollHeight + 'px';
                            
                            const newSections = [...draftSections];
                            newSections[i] = { ...newSections[i], content: next };
                            setDraftSections(newSections);
                            setIsDirty(true);
                            setAutoSaved(false);
                            clearTimeout(autoSaveTimer.current);
                            autoSaveTimer.current = setTimeout(() => {
                              triggerSave(newSections);
                            }, 1500);
                          }}
                          onInput={(e) => {
                            e.target.style.height = 'auto';
                            e.target.style.height = e.target.scrollHeight + 'px';
                          }}
                          ref={(el) => {
                            if (el) {
                              el.style.height = 'auto';
                              el.style.height = el.scrollHeight + 'px';
                            }
                          }}
                        />
                      </div>
                    );
                  })}
                  {draftSections.length === 0 && (
                    <textarea
                      style={{ width: '100%', minHeight: MIN_EDITOR_HEIGHT, border: 'none', outline: 'none', resize: 'none' }}
                      onChange={e => {
                        const next = e.target.value;
                        setDraftSections([{ title: '', content: next, blocks: [] }]);
                      }}
                    />
                  )}
                </div>
"""

code = code.replace(old_editor.strip(), new_editor.strip())

old_paper_zone = """
          {/* Paper */}
          <section className="doc-paper-zone">
"""

new_paper_zone = """
          <div className="doc-workspace-row" style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
            <div className="doc-nav-pane" style={{ width: '250px', background: '#f3f4f6', borderRight: '1px solid #e5e7eb', padding: '16px', overflowY: 'auto' }}>
              <h3 style={{fontSize: '13px', textTransform: 'uppercase', color: '#6b7280', margin: '0 0 12px 0', letterSpacing: '0.05em'}}>Navigation</h3>
              {draftSections.filter(s => s.title).map((s, i) => {
                 let indent = 0;
                 let weight = 'normal';
                 if (/^chapter\\s+\\d/i.test(s.title)) { indent = 0; weight = 'bold'; }
                 else if (/^\\d+\\.\\d+\\.\\d+/.test(s.title)) { indent = 24; }
                 else if (/^\\d+\\.\\d+/.test(s.title)) { indent = 12; weight = '500'; }
                 return (
                   <div key={i} style={{ 
                     marginLeft: indent, 
                     fontWeight: weight,
                     fontSize: '13px', 
                     padding: '4px 0', 
                     cursor: 'pointer',
                     color: '#374151',
                     whiteSpace: 'nowrap',
                     overflow: 'hidden',
                     textOverflow: 'ellipsis'
                   }}>
                     {s.title}
                   </div>
                 );
              })}
            </div>
          {/* Paper */}
          <section className="doc-paper-zone" style={{ flex: 1 }}>
"""

code = code.replace(old_paper_zone.strip(), new_paper_zone.strip())

old_paper_close = """
          </section>
        </div>

        {/* ── Right column: AI Panel ── */}
"""

new_paper_close = """
          </section>
          </div>
        </div>

        {/* ── Right column: AI Panel ── */}
"""

code = code.replace(old_paper_close.strip(), new_paper_close.strip())

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(code)
print("Replaced!")