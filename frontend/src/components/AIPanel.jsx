import { useState } from "react";

export default function AIPanel({
  documentId,
  onEnhanceSection,
  onGenerateOutline,
  onGenerateChart,
  onGenerateImage,
  onGenerateDissertation,
}) {
  const [topic, setTopic] = useState("Impact of AI on education");
  const [sectionQuery, setSectionQuery] = useState("Methodology");

  return (
    <aside className="ai-panel">
      <h2>AI Agent</h2>
      <p className="status">Tool-based actions on structured sections</p>

      <label>Topic</label>
      <input value={topic} onChange={(e) => setTopic(e.target.value)} />

      <label>Section Query</label>
      <input value={sectionQuery} onChange={(e) => setSectionQuery(e.target.value)} />

      <div className="actions">
        <button onClick={() => onGenerateOutline(topic)}>Generate Outline</button>
        <button onClick={() => onEnhanceSection(sectionQuery)}>Enhance Section</button>
        <button onClick={() => onGenerateImage(sectionQuery)}>Insert Image</button>
        <button onClick={() => onGenerateChart(sectionQuery)}>Insert Chart</button>
        <button onClick={() => onGenerateDissertation(topic)}>Run Full Dissertation</button>
      </div>
    </aside>
  );
}
