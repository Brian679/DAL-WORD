import { useEditor, EditorContent } from "@tiptap/react";
import Document from "@tiptap/extension-document";
import Paragraph from "@tiptap/extension-paragraph";
import Text from "@tiptap/extension-text";
import Heading from "@tiptap/extension-heading";

function toTextContent(documentContent) {
  if (!documentContent?.sections?.length) {
    return "";
  }

  return documentContent.sections
    .map((section) => `${section.title}\n${section.content || ""}`)
    .join("\n\n");
}

export default function EditorPane({ documentContent, onSyncFromEditor }) {
  const editor = useEditor({
    extensions: [Document, Paragraph, Text, Heading],
    content: `<p>${toTextContent(documentContent)}</p>`,
    onUpdate: ({ editor: activeEditor }) => {
      onSyncFromEditor(activeEditor.getText());
    },
    editorProps: {
      attributes: {
        class: "editor-surface",
      },
    },
  });

  return (
    <div className="editor-pane">
      <div className="editor-toolbar">Document Editor</div>
      <EditorContent editor={editor} />
    </div>
  );
}
