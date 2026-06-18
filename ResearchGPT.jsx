import { useCallback, useEffect, useRef, useState } from "react";

const API_BASE = import.meta.env?.VITE_API_URL || "http://localhost:8000";

const css = `
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --border: #2a2f3e;
    --text: #e8eaed;
    --muted: #9aa0b0;
    --accent: #6c9eff;
    --accent-hover: #8ab4ff;
    --success: #4ade80;
    --error: #f87171;
    --radius: 10px;
    --font: "Inter", system-ui, -apple-system, sans-serif;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  .rgpt {
    font-family: var(--font);
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  .rgpt-header {
    padding: 1.25rem 2rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }

  .rgpt-header h1 { font-size: 1.35rem; font-weight: 600; }
  .rgpt-header span { color: var(--muted); font-size: 0.85rem; }

  .rgpt-body {
    flex: 1;
    display: grid;
    grid-template-columns: 320px 1fr;
    gap: 0;
    max-width: 1200px;
    width: 100%;
    margin: 0 auto;
    padding: 1.5rem;
  }

  @media (max-width: 768px) {
    .rgpt-body { grid-template-columns: 1fr; }
  }

  .rgpt-sidebar {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    padding-right: 1.5rem;
    border-right: 1px solid var(--border);
  }

  @media (max-width: 768px) {
    .rgpt-sidebar { border-right: none; padding-right: 0; border-bottom: 1px solid var(--border); padding-bottom: 1.5rem; }
  }

  .rgpt-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem;
  }

  .rgpt-card h3 {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted);
    margin-bottom: 0.75rem;
  }

  .rgpt-input {
    width: 100%;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    padding: 0.6rem 0.75rem;
    font-size: 0.9rem;
    outline: none;
    transition: border-color 0.15s;
  }

  .rgpt-input:focus { border-color: var(--accent); }

  .rgpt-btn {
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    font-size: 0.85rem;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.15s;
    width: 100%;
    margin-top: 0.5rem;
  }

  .rgpt-btn:hover:not(:disabled) { background: var(--accent-hover); }
  .rgpt-btn:disabled { opacity: 0.5; cursor: not-allowed; }

  .rgpt-btn-secondary {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text);
  }

  .rgpt-btn-secondary:hover:not(:disabled) { border-color: var(--accent); }

  .rgpt-paper-list { list-style: none; display: flex; flex-direction: column; gap: 0.4rem; }

  .rgpt-paper-item {
    padding: 0.5rem 0.6rem;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.85rem;
    transition: background 0.15s;
    border: 1px solid transparent;
  }

  .rgpt-paper-item:hover { background: var(--bg); }
  .rgpt-paper-item.active { background: var(--bg); border-color: var(--accent); }

  .rgpt-paper-item small { display: block; color: var(--muted); font-size: 0.75rem; margin-top: 2px; }

  .rgpt-chat {
    display: flex;
    flex-direction: column;
    padding-left: 1.5rem;
    min-height: 0;
  }

  @media (max-width: 768px) {
    .rgpt-chat { padding-left: 0; padding-top: 1.5rem; }
  }

  .rgpt-messages {
    flex: 1;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 1rem;
    padding-bottom: 1rem;
    min-height: 300px;
    max-height: calc(100vh - 280px);
  }

  .rgpt-msg {
    max-width: 85%;
    padding: 0.85rem 1rem;
    border-radius: var(--radius);
    font-size: 0.9rem;
    line-height: 1.55;
    white-space: pre-wrap;
  }

  .rgpt-msg.user {
    align-self: flex-end;
    background: var(--accent);
    color: #fff;
    border-bottom-right-radius: 2px;
  }

  .rgpt-msg.assistant {
    align-self: flex-start;
    background: var(--surface);
    border: 1px solid var(--border);
    border-bottom-left-radius: 2px;
  }

  .rgpt-msg.system {
    align-self: center;
    background: transparent;
    color: var(--muted);
    font-size: 0.8rem;
    padding: 0.25rem;
  }

  .rgpt-msg-meta {
    font-size: 0.7rem;
    color: var(--muted);
    margin-top: 0.4rem;
  }

  .rgpt-input-row {
    display: flex;
    gap: 0.5rem;
    margin-top: auto;
    padding-top: 0.75rem;
    border-top: 1px solid var(--border);
  }

  .rgpt-input-row .rgpt-input { flex: 1; }
  .rgpt-input-row .rgpt-btn { width: auto; margin-top: 0; white-space: nowrap; }

  .rgpt-status {
    font-size: 0.8rem;
    padding: 0.5rem 0.75rem;
    border-radius: 6px;
    margin-top: 0.5rem;
  }

  .rgpt-status.error { background: rgba(248,113,113,0.12); color: var(--error); }
  .rgpt-status.success { background: rgba(74,222,128,0.12); color: var(--success); }

  .rgpt-file-label {
    display: block;
    text-align: center;
    padding: 0.6rem;
    border: 1px dashed var(--border);
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.85rem;
    color: var(--muted);
    transition: border-color 0.15s;
  }

  .rgpt-file-label:hover { border-color: var(--accent); color: var(--text); }
  .rgpt-file-label input { display: none; }

  .rgpt-loading { display: inline-block; animation: pulse 1.2s ease-in-out infinite; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
`;

async function api(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
  return data;
}

export default function ResearchGPT() {
  const [papers, setPapers] = useState([]);
  const [activePaper, setActivePaper] = useState(null);
  const [url, setUrl] = useState("");
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState(null);
  const messagesEnd = useRef(null);

  const refreshPapers = useCallback(async () => {
    try {
      const list = await api("/papers");
      setPapers(list);
    } catch {
      /* backend may not be running yet */
    }
  }, []);

  useEffect(() => {
    refreshPapers();
  }, [refreshPapers]);

  useEffect(() => {
    messagesEnd.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleLoad = async () => {
    if (!url.trim()) return;
    setLoading(true);
    setStatus(null);
    try {
      const data = await api("/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: url.trim() }),
      });
      setActivePaper(data);
      setMessages([{ role: "system", text: `Loaded "${data.title}" — ${data.chunks} chunks indexed.` }]);
      setStatus({ type: "success", text: data.message });
      await refreshPapers();
    } catch (err) {
      setStatus({ type: "error", text: err.message });
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setLoading(true);
    setStatus(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const data = await api("/upload", { method: "POST", body: form });
      setActivePaper(data);
      setMessages([{ role: "system", text: `Loaded "${data.title}" — ${data.chunks} chunks indexed.` }]);
      setStatus({ type: "success", text: data.message });
      await refreshPapers();
    } catch (err) {
      setStatus({ type: "error", text: err.message });
    } finally {
      setLoading(false);
      e.target.value = "";
    }
  };

  const handleAsk = async () => {
    if (!question.trim() || !activePaper) return;
    const q = question.trim();
    setQuestion("");
    setMessages((prev) => [...prev, { role: "user", text: q }]);
    setLoading(true);
    try {
      const data = await api("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ paper_id: activePaper.paper_id, question: q }),
      });
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: data.answer, meta: `${data.sources_used} source(s) used` },
      ]);
    } catch (err) {
      setMessages((prev) => [...prev, { role: "assistant", text: `Error: ${err.message}` }]);
    } finally {
      setLoading(false);
    }
  };

  const selectPaper = (paper) => {
    setActivePaper({
      paper_id: paper.paper_id,
      title: paper.title,
      chunks: paper.chunks,
    });
    setMessages([{ role: "system", text: `Switched to "${paper.title}".` }]);
  };

  return (
    <div className="rgpt">
      <style>{css}</style>

      <header className="rgpt-header">
        <h1>ResearchGPT</h1>
        <span>Ask questions about research papers</span>
      </header>

      <div className="rgpt-body">
        <aside className="rgpt-sidebar">
          <div className="rgpt-card">
            <h3>Load from URL</h3>
            <input
              className="rgpt-input"
              placeholder="https://arxiv.org/abs/1706.03762"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleLoad()}
            />
            <button className="rgpt-btn" onClick={handleLoad} disabled={loading}>
              {loading ? "Loading…" : "Load Paper"}
            </button>
          </div>

          <div className="rgpt-card">
            <h3>Upload PDF</h3>
            <label className="rgpt-file-label">
              <input type="file" accept=".pdf" onChange={handleUpload} disabled={loading} />
              Choose a PDF file
            </label>
          </div>

          {status && (
            <div className={`rgpt-status ${status.type}`}>{status.text}</div>
          )}

          {papers.length > 0 && (
            <div className="rgpt-card">
              <h3>Loaded Papers</h3>
              <ul className="rgpt-paper-list">
                {papers.map((p) => (
                  <li
                    key={p.paper_id}
                    className={`rgpt-paper-item ${activePaper?.paper_id === p.paper_id ? "active" : ""}`}
                    onClick={() => selectPaper(p)}
                  >
                    {p.title}
                    <small>{p.chunks} chunks</small>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </aside>

        <main className="rgpt-chat">
          <div className="rgpt-messages">
            {messages.length === 0 && (
              <div className="rgpt-msg system">
                Load a paper from arXiv or upload a PDF to get started.
              </div>
            )}
            {messages.map((msg, i) => (
              <div key={i} className={`rgpt-msg ${msg.role}`}>
                {msg.text}
                {msg.meta && <div className="rgpt-msg-meta">{msg.meta}</div>}
              </div>
            ))}
            {loading && messages.length > 0 && messages[messages.length - 1].role === "user" && (
              <div className="rgpt-msg assistant">
                <span className="rgpt-loading">Thinking…</span>
              </div>
            )}
            <div ref={messagesEnd} />
          </div>

          <div className="rgpt-input-row">
            <input
              className="rgpt-input"
              placeholder={activePaper ? "Ask a question about the paper…" : "Load a paper first"}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleAsk()}
              disabled={!activePaper || loading}
            />
            <button className="rgpt-btn" onClick={handleAsk} disabled={!activePaper || loading || !question.trim()}>
              Ask
            </button>
          </div>
        </main>
      </div>
    </div>
  );
}
