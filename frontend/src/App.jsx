import { useCallback, useEffect, useRef, useState } from "react";
import "./App.css";

const API_BASE = import.meta.env?.VITE_API_URL || "http://localhost:8000";

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
      <header className="rgpt-header">
        <div className="rgpt-brand">
          <div className="rgpt-logo" />
          <h1>ResearchGPT</h1>
        </div>
        {activePaper ? (
          <div className="rgpt-active-paper">
            <div className="rgpt-active-dot" />
            <span>{activePaper.title}</span>
          </div>
        ) : (
          <span className="rgpt-tagline">Research paper Q&amp;A</span>
        )}
      </header>

      <div className="rgpt-body">
        <aside className="rgpt-sidebar">
          <div className="rgpt-section">
            <span className="rgpt-section-label">From URL</span>
            <input
              className="rgpt-input"
              placeholder="arxiv.org/abs/..."
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleLoad()}
            />
            <button className="rgpt-btn" onClick={handleLoad} disabled={loading}>
              {loading ? "Loading…" : "Load"}
            </button>
          </div>

          <div className="rgpt-section">
            <span className="rgpt-section-label">Upload</span>
            <label className="rgpt-file-label">
              <input type="file" accept=".pdf" onChange={handleUpload} disabled={loading} />
              Choose PDF
            </label>
          </div>

          {status && (
            <div className={`rgpt-status ${status.type}`}>{status.text}</div>
          )}

          {papers.length > 0 && (
            <div className="rgpt-section">
              <span className="rgpt-section-label">Papers</span>
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
          {messages.length === 0 ? (
            <div className="rgpt-empty">
              <div className="rgpt-empty-icon">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
                  <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
                </svg>
              </div>
              <h2>Ask anything about a paper</h2>
              <p>Load from arXiv or upload a PDF to start asking questions.</p>
            </div>
          ) : (
            <div className="rgpt-messages">
              <div className="rgpt-messages-inner">
                {messages.map((msg, i) => (
                  <div key={i} className={`rgpt-msg ${msg.role}`}>
                    {msg.text}
                    {msg.meta && <div className="rgpt-msg-meta">{msg.meta}</div>}
                  </div>
                ))}
                {loading && messages.length > 0 && messages[messages.length - 1].role === "user" && (
                  <div className="rgpt-msg assistant">
                    <span className="rgpt-loading">Thinking</span>
                  </div>
                )}
                <div ref={messagesEnd} />
              </div>
            </div>
          )}

          <div className="rgpt-input-bar">
            <div className="rgpt-input-row">
              <input
                className="rgpt-input"
                placeholder={activePaper ? "Ask a question…" : "Load a paper first"}
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleAsk()}
                disabled={!activePaper || loading}
              />
              <button
                className="rgpt-btn rgpt-btn-ghost"
                onClick={handleAsk}
                disabled={!activePaper || loading || !question.trim()}
              >
                Send
              </button>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
