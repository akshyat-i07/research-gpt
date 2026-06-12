import { useState, useRef, useEffect } from "react";

const BACKEND = "http://localhost:8000";
const API_KEY_STORAGE = "researchgpt_api_key";

function getStoredApiKey() {
  try { return localStorage.getItem(API_KEY_STORAGE) || ""; } catch { return ""; }
}

function setStoredApiKey(key) {
  try { localStorage.setItem(API_KEY_STORAGE, key); } catch { /* ignore */ }
}

function withApiKey(payload = {}) {
  const key = getStoredApiKey().trim();
  return key ? { ...payload, api_key: key } : payload;
}

function parseApiError(data, fallback) {
  const detail = data?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((d) => d.msg || d.message || String(d)).join(", ");
  return fallback;
}

const SAMPLE_PAPERS = [
  { label: "Attention Is All You Need", url: "https://arxiv.org/abs/1706.03762" },
  { label: "BERT: Pre-training Deep Bidirectional Transformers", url: "https://arxiv.org/abs/1810.04805" },
  { label: "GPT-3: Language Models are Few-Shot Learners", url: "https://arxiv.org/abs/2005.14165" },
];

const SUGGESTED = [
  "What is the main contribution of this paper?",
  "What methodology do the authors use?",
  "What are the key results and findings?",
  "What are the limitations mentioned?",
  "How does this compare to prior work?",
];

// ── Icons ─────────────────────────────────────────────────────────────────────
const Icon = ({ d, size = 18, stroke = 1.8 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth={stroke} strokeLinecap="round" strokeLinejoin="round">
    {Array.isArray(d) ? d.map((p, i) => <path key={i} d={p} />) : <path d={d} />}
  </svg>
);

const UploadIcon = () => <Icon size={40} stroke={1.4} d={["M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4","M17 8l-5-5-5 5","M12 3v12"]} />;
const SendIcon = () => <Icon size={16} d={["M22 2L11 13","M22 2l-7 20-4-9-9-4 20-7"]} />;
const BookIcon = () => <Icon size={16} d={["M4 19.5A2.5 2.5 0 0 1 6.5 17H20","M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"]} />;
const ChevronLeft = () => <Icon size={16} d="M15 18l-6-6 6-6" />;
const ChevronRight = () => <Icon size={16} d="M9 18l6-6-6-6" />;
const CheckIcon = () => <Icon size={16} d="M20 6L9 17l-5-5" />;
const XIcon = () => <Icon size={14} d="M18 6L6 18M6 6l12 12" />;
const BrainIcon = () => <Icon size={48} stroke={1.2} d={["M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2z","M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2z"]} />;

// ── Markdown renderer ─────────────────────────────────────────────────────────
function renderMd(text) {
  return text
    .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
    .replace(/`([^`]+)`/g, '<code style="background:#1e2028;padding:2px 6px;border-radius:4px;font-size:13px;font-family:monospace">$1</code>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/^#{1,3} (.+)$/gm, '<p style="font-weight:600;font-size:15px;margin:12px 0 4px">$1</p>')
    .replace(/^\* (.+)$/gm, '<li style="margin:4px 0;padding-left:4px">$1</li>')
    .replace(/^- (.+)$/gm, '<li style="margin:4px 0;padding-left:4px">$1</li>')
    .replace(/(<li[\s\S]*?<\/li>)/g, '<ul style="list-style:disc;padding-left:20px;margin:8px 0">$1</ul>')
    .replace(/\n\n/g, '</p><p style="margin:8px 0">')
    .replace(/^/, '<p style="margin:0">')
    .replace(/$/, '</p>');
}

// ── Typing indicator ──────────────────────────────────────────────────────────
function TypingDots() {
  return (
    <div style={{ display: "flex", gap: 5, alignItems: "center", padding: "12px 16px" }}>
      {[0, 1, 2].map(i => (
        <div key={i} style={{
          width: 7, height: 7, borderRadius: "50%", background: "#3b82f6",
          animation: `bounce 1.2s ease-in-out ${i * 0.2}s infinite`,
        }} />
      ))}
    </div>
  );
}

// ── Progress bar ──────────────────────────────────────────────────────────────
function ProgressBar({ value, label }) {
  return (
    <div style={{ width: "100%" }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
        <span style={{ fontSize: 12, color: "#9ca3af" }}>{label}</span>
        <span style={{ fontSize: 12, color: "#3b82f6" }}>{value}%</span>
      </div>
      <div style={{ height: 4, background: "#1e2028", borderRadius: 2, overflow: "hidden" }}>
        <div style={{
          height: "100%", width: `${value}%`, background: "#3b82f6",
          borderRadius: 2, transition: "width 0.3s ease",
        }} />
      </div>
    </div>
  );
}

// ── HeroSection ───────────────────────────────────────────────────────────────
function HeroSection() {
  return (
    <div style={{ textAlign: "center", padding: "48px 24px 32px" }}>
      <div style={{ display: "inline-flex", alignItems: "center", gap: 8, background: "#13161d", border: "1px solid #2a2d35", borderRadius: 20, padding: "4px 14px", marginBottom: 20 }}>
        <div style={{ width: 7, height: 7, borderRadius: "50%", background: "#3b82f6", boxShadow: "0 0 8px #3b82f6" }} />
        <span style={{ fontSize: 12, color: "#6b7280", letterSpacing: "0.05em" }}>RAG-POWERED RESEARCH ASSISTANT</span>
      </div>
      <h1 style={{ fontSize: "clamp(28px, 5vw, 44px)", fontWeight: 700, color: "#f9fafb", margin: "0 0 14px", letterSpacing: "-0.02em", lineHeight: 1.2 }}>
        ResearchGPT
      </h1>
      <p style={{ fontSize: "clamp(14px, 2vw, 17px)", color: "#6b7280", maxWidth: 560, margin: "0 auto", lineHeight: 1.7 }}>
        Upload scientific papers and get grounded answers, summaries, and insights powered by Retrieval-Augmented Generation.
      </p>
    </div>
  );
}

// ── ApiKeyField ───────────────────────────────────────────────────────────────
function ApiKeyField({ apiKey, onChange }) {
  const hint = apiKey.startsWith("AQ.")
    ? "AQ. keys often fail — use an AIza key from AI Studio"
    : apiKey.startsWith("AIza")
      ? "AI Studio key detected"
      : "Get a key at aistudio.google.com (must start with AIza)";

  return (
    <div style={{ marginBottom: 16 }}>
      <label style={{ display: "block", fontSize: 12, color: "#6b7280", marginBottom: 6 }}>
        Gemini API key (optional if set in backend .env)
      </label>
      <input
        type="password"
        placeholder="AIza…"
        value={apiKey}
        onChange={(e) => onChange(e.target.value)}
        style={{ width: "100%", padding: "10px 14px", background: "#13161d", border: "1px solid #2a2d35", borderRadius: 10, color: "#f9fafb", fontSize: 14, outline: "none" }}
      />
      <p style={{ margin: "6px 0 0", fontSize: 11, color: apiKey.startsWith("AQ.") ? "#f87171" : "#4b5563" }}>{hint}</p>
    </div>
  );
}

// ── UploadZone ────────────────────────────────────────────────────────────────
function UploadZone({ onLoad, apiKey, onApiKeyChange }) {
  const [dragging, setDragging] = useState(false);
  const [url, setUrl] = useState("");
  const [status, setStatus] = useState("idle");
  const [progress, setProgress] = useState(0);
  const [errorMsg, setErrorMsg] = useState("");
  const [progressLabel, setProgressLabel] = useState("");

  async function loadPaper(paperUrl) {
    if (!paperUrl.trim()) { setErrorMsg("Enter an arxiv URL."); setStatus("error"); return; }
    setStatus("loading"); setErrorMsg(""); setProgress(10); setProgressLabel("Fetching PDF…");
    try {
      await new Promise(r => setTimeout(r, 400));
      setProgress(30); setProgressLabel("Extracting text…");
      await new Promise(r => setTimeout(r, 300));
      setProgress(55); setProgressLabel("Chunking document…");

      const res = await fetch(`${BACKEND}/load`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(withApiKey({ url: paperUrl.trim() })),
      });
      setProgress(80); setProgressLabel("Building FAISS index…");
      const data = await res.json();
      if (!res.ok) throw new Error(parseApiError(data, "Failed to load paper"));
      setProgress(100); setProgressLabel("Done!");
      await new Promise(r => setTimeout(r, 500));
      setStatus("success");
      onLoad(data);
    } catch (e) {
      setErrorMsg(e.message);
      setStatus("error");
      setProgress(0);
    }
  }

  async function handleDrop(e) {
    e.preventDefault(); setDragging(false);
    const file = e.dataTransfer.files[0];
    if (!file || !file.name.endsWith(".pdf")) {
      setErrorMsg("Please drop a valid PDF file.");
      setStatus("error");
      return;
    }
    setStatus("loading"); setErrorMsg(""); setProgress(20); setProgressLabel("Uploading PDF…");
    try {
      const formData = new FormData();
      formData.append("file", file);
      const key = getStoredApiKey().trim();
      if (key) formData.append("api_key", key);
      setProgress(50); setProgressLabel("Indexing chunks…");
      const res = await fetch(`${BACKEND}/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(parseApiError(data, "Upload failed"));
      setProgress(100); setProgressLabel("Done!");
      await new Promise(r => setTimeout(r, 400));
      setStatus("success");
      onLoad(data);
    } catch (e) {
      setErrorMsg(e.message);
      setStatus("error");
      setProgress(0);
    }
  }

  return (
    <div style={{ maxWidth: 640, margin: "0 auto", padding: "0 24px 32px" }}>

      <ApiKeyField apiKey={apiKey} onChange={onApiKeyChange} />

      {/* Drop zone */}
      <div
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        style={{
          border: `2px dashed ${dragging ? "#3b82f6" : "#2a2d35"}`,
          borderRadius: 16, padding: "40px 24px", textAlign: "center",
          background: dragging ? "#13161d" : "#0d1017",
          transition: "all 0.2s", cursor: "default", marginBottom: 16,
        }}
      >
        <div style={{ color: dragging ? "#3b82f6" : "#374151", marginBottom: 16 }}><UploadIcon /></div>
        <p style={{ color: "#9ca3af", fontSize: 15, margin: "0 0 6px" }}>Drag & drop a PDF, or use an arxiv link below</p>
        <p style={{ color: "#4b5563", fontSize: 13, margin: 0 }}>Supports arxiv.org/abs/ and arxiv.org/pdf/ links</p>
      </div>

      {/* URL input + load button */}
      <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
        <input
          type="url"
          placeholder="https://arxiv.org/abs/1706.03762"
          value={url}
          onChange={e => setUrl(e.target.value)}
          onKeyDown={e => e.key === "Enter" && loadPaper(url)}
          style={{ flex: 1, padding: "10px 14px", background: "#13161d", border: "1px solid #2a2d35", borderRadius: 10, color: "#f9fafb", fontSize: 14, outline: "none" }}
        />
        <button
          onClick={() => loadPaper(url)}
          disabled={status === "loading"}
          style={{ padding: "10px 20px", background: "#3b82f6", color: "#fff", border: "none", borderRadius: 10, fontSize: 14, fontWeight: 500, cursor: status === "loading" ? "not-allowed" : "pointer", opacity: status === "loading" ? 0.7 : 1, whiteSpace: "nowrap" }}
        >
          {status === "loading" ? "Loading…" : "Load Paper"}
        </button>
      </div>

      {/* Sample papers */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 16 }}>
        {SAMPLE_PAPERS.map(p => (
          <button key={p.url} onClick={() => setUrl(p.url)}
            style={{ fontSize: 12, padding: "5px 10px", background: "#13161d", border: "1px solid #2a2d35", borderRadius: 20, color: "#6b7280", cursor: "pointer" }}>
            {p.label}
          </button>
        ))}
      </div>

      {/* Progress */}
      {status === "loading" && <ProgressBar value={progress} label={progressLabel} />}

      {/* Success */}
      {status === "success" && (
        <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "10px 14px", background: "#0d1f0f", border: "1px solid #166534", borderRadius: 10, color: "#4ade80", fontSize: 13 }}>
          <CheckIcon /> Paper indexed successfully
        </div>
      )}

      {/* Error */}
      {status === "error" && (
        <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "10px 14px", background: "#1f0d0d", border: "1px solid #991b1b", borderRadius: 10, color: "#f87171", fontSize: 13 }}>
          <XIcon /> {errorMsg}
        </div>
      )}
    </div>
  );
}

// ── ResearchSummaryCard ───────────────────────────────────────────────────────
function ResearchSummaryCard({ meta }) {
  return (
    <div style={{ maxWidth: 640, margin: "0 auto 32px", padding: "0 24px" }}>
      <div style={{ background: "#13161d", border: "1px solid #2a2d35", borderRadius: 16, padding: "20px 24px" }}>
        <div style={{ display: "flex", alignItems: "flex-start", gap: 12, marginBottom: 14 }}>
          <div style={{ color: "#3b82f6", flexShrink: 0, marginTop: 2 }}><BookIcon /></div>
          <div style={{ minWidth: 0 }}>
            <h3 style={{ margin: "0 0 4px", fontSize: 15, fontWeight: 600, color: "#f9fafb", lineHeight: 1.4 }}>{meta.title}</h3>
            <p style={{ margin: 0, fontSize: 13, color: "#6b7280" }}>{meta.author}</p>
          </div>
        </div>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          {[
            { label: "Pages", value: meta.pages },
            { label: "Chunks indexed", value: meta.chunks },
            { label: "Loaded", value: new Date().toLocaleDateString() },
          ].map(s => (
            <div key={s.label} style={{ background: "#0d1017", border: "1px solid #1e2028", borderRadius: 8, padding: "8px 14px" }}>
              <p style={{ margin: "0 0 2px", fontSize: 11, color: "#4b5563", letterSpacing: "0.04em" }}>{s.label.toUpperCase()}</p>
              <p style={{ margin: 0, fontSize: 14, fontWeight: 600, color: "#d1d5db" }}>{s.value}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── ChatMessage ───────────────────────────────────────────────────────────────
function ChatMessage({ msg }) {
  const isUser = msg.role === "user";
  return (
    <div style={{ display: "flex", justifyContent: isUser ? "flex-end" : "flex-start", marginBottom: 16, animation: "fadeUp 0.2s ease" }}>
      {!isUser && (
        <div style={{ width: 28, height: 28, borderRadius: "50%", background: "#1e2875", border: "1px solid #3b82f6", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, marginRight: 10, marginTop: 2 }}>
          <span style={{ fontSize: 11, color: "#93c5fd", fontWeight: 700 }}>R</span>
        </div>
      )}
      <div style={{
        maxWidth: "78%", padding: "12px 16px",
        borderRadius: isUser ? "18px 18px 4px 18px" : "4px 18px 18px 18px",
        background: isUser ? "#1d4ed8" : "#13161d",
        border: isUser ? "none" : "1px solid #2a2d35",
        color: "#f9fafb", fontSize: 14, lineHeight: 1.65,
      }}>
        {isUser
          ? <p style={{ margin: 0 }}>{msg.content}</p>
          : <div dangerouslySetInnerHTML={{ __html: renderMd(msg.content) }} />
        }
        {msg.sources > 0 && (
          <p style={{ margin: "10px 0 0", fontSize: 11, color: "#4b5563", borderTop: "1px solid #1e2028", paddingTop: 8 }}>
            ↳ {msg.sources} source section{msg.sources !== 1 ? "s" : ""} retrieved
          </p>
        )}
        {msg.citations?.length > 0 && (
          <div style={{ marginTop: 10, borderTop: "1px solid #1e2028", paddingTop: 10 }}>
            <p style={{ margin: "0 0 8px", fontSize: 11, color: "#6b7280", letterSpacing: "0.04em" }}>SOURCES</p>
            {msg.citations.map((c, i) => (
              <div key={i} style={{ marginBottom: 8, padding: "8px 10px", background: "#0d1017", borderRadius: 8, border: "1px solid #1e2028" }}>
                <p style={{ margin: 0, fontSize: 12, color: "#9ca3af", lineHeight: 1.5 }}>{c.preview}</p>
                <p style={{ margin: "4px 0 0", fontSize: 10, color: "#4b5563" }}>relevance: {c.score}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ── ChatInput ─────────────────────────────────────────────────────────────────
function ChatInput({ onSend, loading }) {
  const [val, setVal] = useState("");
  const taRef = useRef();

  function send() {
    if (!val.trim() || loading) return;
    onSend(val.trim());
    setVal("");
    if (taRef.current) { taRef.current.style.height = "auto"; }
  }

  return (
    <div style={{ padding: "16px 24px 20px", borderTop: "1px solid #1a1d24", background: "#0F1115" }}>
      <div style={{ maxWidth: 760, margin: "0 auto", display: "flex", gap: 10, alignItems: "flex-end" }}>
        <textarea
          ref={taRef}
          value={val}
          onChange={e => { setVal(e.target.value); e.target.style.height = "auto"; e.target.style.height = Math.min(e.target.scrollHeight, 140) + "px"; }}
          onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
          placeholder="Ask anything about this paper…"
          rows={1}
          style={{ flex: 1, resize: "none", padding: "12px 16px", background: "#13161d", border: "1px solid #2a2d35", borderRadius: 12, color: "#f9fafb", fontSize: 14, lineHeight: 1.5, outline: "none", overflowY: "hidden", fontFamily: "inherit" }}
        />
        <button
          onClick={send}
          disabled={!val.trim() || loading}
          style={{ width: 42, height: 42, borderRadius: 10, background: val.trim() && !loading ? "#3b82f6" : "#1e2028", border: "none", cursor: val.trim() && !loading ? "pointer" : "not-allowed", display: "flex", alignItems: "center", justifyContent: "center", color: val.trim() && !loading ? "#fff" : "#4b5563", transition: "background 0.15s", flexShrink: 0 }}
        >
          <SendIcon />
        </button>
      </div>
      <p style={{ textAlign: "center", margin: "8px 0 0", fontSize: 11, color: "#374151" }}>Enter to send · Shift+Enter for new line</p>
    </div>
  );
}

// ── Sidebar ───────────────────────────────────────────────────────────────────
function Sidebar({ papers, activePaperId, onSelect, collapsed, onToggle }) {
  return (
    <div style={{
      width: collapsed ? 44 : 240, minWidth: collapsed ? 44 : 240,
      background: "#0a0d12", borderRight: "1px solid #1a1d24",
      display: "flex", flexDirection: "column",
      transition: "width 0.2s ease, min-width 0.2s ease", overflow: "hidden",
    }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: collapsed ? "center" : "space-between", padding: collapsed ? "14px 0" : "14px 16px", borderBottom: "1px solid #1a1d24" }}>
        {!collapsed && <span style={{ fontSize: 12, fontWeight: 600, color: "#6b7280", letterSpacing: "0.06em" }}>PAPERS</span>}
        <button onClick={onToggle} style={{ background: "none", border: "none", color: "#4b5563", cursor: "pointer", display: "flex", padding: 4 }}>
          {collapsed ? <ChevronRight /> : <ChevronLeft />}
        </button>
      </div>
      {!collapsed && (
        <div style={{ flex: 1, overflowY: "auto", padding: "8px" }}>
          {papers.length === 0
            ? <p style={{ fontSize: 12, color: "#374151", padding: "8px", textAlign: "center" }}>No papers loaded</p>
            : papers.map(p => (
              <button key={p.paper_id} onClick={() => onSelect(p.paper_id)}
                style={{ width: "100%", textAlign: "left", padding: "10px 12px", borderRadius: 8, background: activePaperId === p.paper_id ? "#13161d" : "transparent", border: activePaperId === p.paper_id ? "1px solid #2a2d35" : "1px solid transparent", cursor: "pointer", marginBottom: 4 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  {activePaperId === p.paper_id && <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#3b82f6", flexShrink: 0 }} />}
                  <p style={{ margin: 0, fontSize: 12, color: activePaperId === p.paper_id ? "#d1d5db" : "#6b7280", overflow: "hidden", textOverflow: "ellipsis", display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", lineHeight: 1.4 }}>
                    {p.title}
                  </p>
                </div>
              </button>
            ))
          }
        </div>
      )}
    </div>
  );
}

// ── ChatWindow ────────────────────────────────────────────────────────────────
function ChatWindow({ messages, loading, onSend }) {
  const bottomRef = useRef();

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <div style={{ flex: 1, overflowY: "auto", padding: "24px" }}>
        <div style={{ maxWidth: 760, margin: "0 auto" }}>
          {messages.length === 0 && (
            <div style={{ textAlign: "center", padding: "60px 0" }}>
              <div style={{ color: "#1e2875", marginBottom: 16 }}><BrainIcon /></div>
              <p style={{ color: "#4b5563", fontSize: 15 }}>Ask anything about this paper</p>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8, justifyContent: "center", marginTop: 20 }}>
                {SUGGESTED.map(q => (
                  <button key={q} onClick={() => onSend(q)}
                    style={{ fontSize: 12, padding: "7px 13px", background: "#13161d", border: "1px solid #2a2d35", borderRadius: 20, color: "#6b7280", cursor: "pointer" }}>
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}
          {messages.map((msg, i) => <ChatMessage key={i} msg={msg} />)}
          {loading && (
            <div style={{ display: "flex", justifyContent: "flex-start", marginBottom: 16 }}>
              <div style={{ width: 28, height: 28, borderRadius: "50%", background: "#1e2875", border: "1px solid #3b82f6", display: "flex", alignItems: "center", justifyContent: "center", marginRight: 10, flexShrink: 0 }}>
                <span style={{ fontSize: 11, color: "#93c5fd", fontWeight: 700 }}>R</span>
              </div>
              <div style={{ background: "#13161d", border: "1px solid #2a2d35", borderRadius: "4px 18px 18px 18px" }}>
                <TypingDots />
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>
      </div>
      <ChatInput onSend={onSend} loading={loading} />
    </div>
  );
}

// ── EmptyState ────────────────────────────────────────────────────────────────
function EmptyState() {
  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "40px 24px", textAlign: "center" }}>
      <div style={{ color: "#1e2875", marginBottom: 20 }}><BrainIcon /></div>
      <h2 style={{ fontSize: 20, fontWeight: 600, color: "#d1d5db", margin: "0 0 10px" }}>Upload a research paper to begin</h2>
      <p style={{ fontSize: 14, color: "#4b5563", maxWidth: 360, lineHeight: 1.6 }}>
        Load any arxiv paper above. ResearchGPT will index it and let you ask questions grounded in the source material.
      </p>
    </div>
  );
}

// ── App ───────────────────────────────────────────────────────────────────────
export default function ResearchGPT() {
  const [papers, setPapers] = useState([]);
  const [activePaperId, setActivePaperId] = useState(null);
  const [chatsByPaper, setChatsByPaper] = useState({});
  const [loading, setLoading] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [apiKey, setApiKey] = useState(() => getStoredApiKey());

  function handleApiKeyChange(value) {
    setApiKey(value);
    setStoredApiKey(value);
  }

  const activeMeta = papers.find(p => p.paper_id === activePaperId);
  const activeMessages = activePaperId ? (chatsByPaper[activePaperId] || []) : [];

  function handleLoad(data) {
    setPapers(prev => {
      if (prev.find(p => p.paper_id === data.paper_id)) return prev;
      return [...prev, data];
    });
    setActivePaperId(data.paper_id);
    if (!chatsByPaper[data.paper_id]) {
      setChatsByPaper(prev => ({
        ...prev,
        [data.paper_id]: [{
          role: "assistant",
          content: `**${data.title}**\n\n*${data.author}* · ${data.pages} pages · ${data.chunks} chunks indexed\n\nPaper loaded and ready. What would you like to know?`,
        }]
      }));
    }
  }

  async function handleQuery(question) {
    if (!question || loading || !activePaperId) return;
    setChatsByPaper(prev => ({ ...prev, [activePaperId]: [...(prev[activePaperId] || []), { role: "user", content: question }] }));
    setLoading(true);
    try {
      const res = await fetch(`${BACKEND}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(withApiKey({ paper_id: activePaperId, question })),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(parseApiError(data, "Query failed"));
      setChatsByPaper(prev => ({
        ...prev,
        [activePaperId]: [...(prev[activePaperId] || []), {
          role: "assistant",
          content: data.answer,
          sources: data.sources_used,
          citations: data.sources || [],
        }]
      }));
    } catch (e) {
      setChatsByPaper(prev => ({
        ...prev,
        [activePaperId]: [...(prev[activePaperId] || []), { role: "assistant", content: `⚠️ Error: ${e.message}`, isError: true }]
      }));
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <style>{`
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #0F1115; color: #f9fafb; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #0a0d12; }
        ::-webkit-scrollbar-thumb { background: #2a2d35; border-radius: 3px; }
        input::placeholder, textarea::placeholder { color: #374151; }
        @keyframes shimmer { 0%{background-position:200% 0} 100%{background-position:-200% 0} }
        @keyframes bounce { 0%,80%,100%{transform:translateY(0)} 40%{transform:translateY(-6px)} }
        @keyframes fadeUp { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
        button { font-family: inherit; }
      `}</style>

      <div style={{ height: "100vh", display: "flex", flexDirection: "column", background: "#0F1115" }}>
        {/* Top bar */}
        <div style={{ height: 52, borderBottom: "1px solid #1a1d24", display: "flex", alignItems: "center", padding: "0 20px", flexShrink: 0, background: "#0a0d12" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{ width: 26, height: 26, borderRadius: 6, background: "#1e2875", border: "1px solid #3b82f6", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <span style={{ fontSize: 12, color: "#93c5fd", fontWeight: 700 }}>R</span>
            </div>
            <span style={{ fontSize: 15, fontWeight: 600, color: "#f9fafb" }}>ResearchGPT</span>
          </div>
          {activeMeta && (
            <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
              <div style={{ width: 7, height: 7, borderRadius: "50%", background: "#22c55e" }} />
              <span style={{ fontSize: 12, color: "#6b7280", maxWidth: 280, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{activeMeta.title}</span>
            </div>
          )}
        </div>

        {/* Body */}
        <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
          <Sidebar papers={papers} activePaperId={activePaperId} onSelect={setActivePaperId} collapsed={sidebarCollapsed} onToggle={() => setSidebarCollapsed(v => !v)} />
          <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
            {!activePaperId ? (
              <div style={{ flex: 1, overflowY: "auto" }}>
                <HeroSection />
                <UploadZone onLoad={handleLoad} apiKey={apiKey} onApiKeyChange={handleApiKeyChange} />
                <EmptyState />
              </div>
            ) : (
              <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
                {activeMeta && <ResearchSummaryCard meta={activeMeta} />}
                <ChatWindow messages={activeMessages} loading={loading} onSend={handleQuery} />
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}