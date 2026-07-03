import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";

const BACKEND = import.meta.env.VITE_API_URL ?? "";

const SAMPLE_PAPERS = [
  { label: "Attention Is All You Need", url: "https://arxiv.org/abs/1706.03762" },
  { label: "BERT: Pre-training of Deep Bidirectional Transformers", url: "https://arxiv.org/abs/1810.04805" },
  { label: "GPT-3: Language Models are Few-Shot Learners", url: "https://arxiv.org/abs/2005.14165" },
  { label: "ResNet: Deep Residual Learning", url: "https://arxiv.org/abs/1512.03385" },
];

const SUGGESTIONS = [
  "Summarize this paper",
  "Explain the methodology",
  "What are the key contributions?",
  "Analyze the results",
  "What are the limitations?",
];

const LOADING_STEPS = [
  "Extracting paper...",
  "Creating embeddings...",
  "Building semantic index...",
  "Finalizing...",
];

// ── Theme tokens ──────────────────────────────────────────────────────────────
const DARK = {
  bg:          "#0B0B0B",
  bg2:         "#0e0e0e",
  surface:     "#141414",
  surface2:    "#1A1A1A",
  border:      "rgba(255,255,255,0.08)",
  border2:     "rgba(255,255,255,0.13)",
  text:        "#FFFFFF",
  text2:       "#B3B3B3",
  text3:       "#4a4a4a",
  accent:      "#00BFFF",
  accentDark:  "#0090cc",
  purple:      "#7C3AED",
  userBubble:  "linear-gradient(135deg,#00BFFF,#0090cc)",
  userText:    "#000",
  aiBubble:    "#1A1A1A",
  aiBorder:    "rgba(255,255,255,0.07)",
  inputBg:     "#141414",
  inputBorder: "rgba(255,255,255,0.1)",
  pillBg:      "rgba(0,191,255,0.06)",
  pillBorder:  "rgba(0,191,255,0.15)",
  codeBg:      "rgba(0,191,255,0.08)",
  preBg:       "#0e0e0e",
  preBorder:   "rgba(255,255,255,0.08)",
  sidebarBg:   "#0e0e0e",
  headerBg:    "#0B0B0B",
  scrollThumb: "rgba(255,255,255,0.08)",
  chipBg:      "rgba(255,255,255,0.03)",
  chipBorder:  "rgba(255,255,255,0.07)",
  mdText:      "#E8E8E8",
  mdHeading:   "#fff",
  mdMuted:     "#B3B3B3",
};

// ── Icons ─────────────────────────────────────────────────────────────────────
const Ic = ({ d, size = 18, sw = 1.7 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round">
    {(Array.isArray(d) ? d : [d]).map((p, i) => <path key={i} d={p} />)}
  </svg>
);

const UploadCloud = () => <Ic size={36} sw={1.4} d={["M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4","M17 8l-5-5-5 5","M12 3v12"]} />;
const SendIcon   = () => <Ic size={16} d={["M22 2L11 13","M22 2l-7 20-4-9-9-4 20-7"]} />;
const BookIcon   = () => <Ic size={14} d={["M4 19.5A2.5 2.5 0 0 1 6.5 17H20","M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"]} />;
const PlusIcon   = () => <Ic size={16} d="M12 5v14M5 12h14" />;
const XIcon      = () => <Ic size={15} d="M18 6L6 18M6 6l12 12" />;
const CheckIcon  = () => <Ic size={15} d="M20 6L9 17l-5-5" />;
const LinkIcon   = () => <Ic size={15} d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />;
const MenuIcon   = () => <Ic size={18} d={["M3 12h18","M3 6h18","M3 18h18"]} />;
const FileIcon   = () => <Ic size={16} d={["M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z","M14 2v6h6"]} />;
// ── Markdown ──────────────────────────────────────────────────────────────────
function Md({ text, t }) {
  const mdComponents = {
    p: ({ children }) => <p style={{ margin: "0 0 10px", color: t.mdText }}>{children}</p>,
    strong: ({ children }) => <strong style={{ color: t.mdHeading }}>{children}</strong>,
    em: ({ children }) => <em style={{ color: t.text2 }}>{children}</em>,
    h1: ({ children }) => <h1 style={{ color: t.mdHeading, fontWeight: 700, fontSize: 17, margin: "18px 0 10px" }}>{children}</h1>,
    h2: ({ children }) => <h2 style={{ color: t.mdHeading, fontWeight: 600, fontSize: 15, margin: "16px 0 8px" }}>{children}</h2>,
    h3: ({ children }) => <h3 style={{ color: t.mdHeading, fontWeight: 600, fontSize: 14, margin: "14px 0 6px" }}>{children}</h3>,
    ul: ({ children }) => <ul style={{ paddingLeft: 20, margin: "8px 0" }}>{children}</ul>,
    ol: ({ children }) => <ol style={{ paddingLeft: 20, margin: "8px 0" }}>{children}</ol>,
    li: ({ children }) => <li style={{ margin: "5px 0", color: t.mdMuted }}>{children}</li>,
    pre: ({ children }) => (
      <pre style={{ background: t.preBg, border: `1px solid ${t.preBorder}`, borderRadius: 10, padding: 14, overflowX: "auto", margin: "12px 0", fontSize: 13, lineHeight: 1.6 }}>
        {children}
      </pre>
    ),
    code: ({ className, children }) => {
      const isBlock = Boolean(className);
      if (isBlock) {
        return <code className={className} style={{ fontFamily: "monospace", color: t.mdText }}>{children}</code>;
      }
      return <code style={{ background: t.codeBg, color: t.accent, padding: "2px 6px", borderRadius: 4, fontSize: 13, fontFamily: "monospace" }}>{children}</code>;
    },
  };

  return (
    <div className="rgpt-md">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={mdComponents}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}

// ── Typing dots ───────────────────────────────────────────────────────────────
function Dots({ t }) {
  return (
    <div style={{ display: "flex", gap: 5, padding: "4px 0" }}>
      {[0,1,2].map(i => (
        <div key={i} style={{ width: 6, height: 6, borderRadius: "50%", background: t.accent,
          animation: `rgptBounce 1.2s ease-in-out ${i*0.2}s infinite` }} />
      ))}
    </div>
  );
}

// ── Progress bar ──────────────────────────────────────────────────────────────
function Progress({ step, total, label, t }) {
  const pct = Math.round((step / total) * 100);
  return (
    <div style={{ width: "100%" }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
        <span style={{ fontSize: 13, color: t.text2 }}>{label}</span>
        <span style={{ fontSize: 13, color: t.accent }}>{pct}%</span>
      </div>
      <div style={{ height: 3, background: t.border, borderRadius: 2 }}>
        <div style={{ height: "100%", width: `${pct}%`, background: `linear-gradient(90deg,${t.accent},${t.purple})`,
          borderRadius: 2, transition: "width 0.4s ease" }} />
      </div>
    </div>
  );
}

// ── Landing ───────────────────────────────────────────────────────────────────
function Landing({ onLoad, t }) {
  const [dragging, setDragging] = useState(false);
  const [url, setUrl]           = useState("");
  const [status, setStatus]     = useState("idle");
  const [loadStep, setLoadStep] = useState(0);
  const [errorMsg, setErrorMsg] = useState("");
  const fileRef = useRef();

  async function loadUrl(paperUrl) {
    if (!paperUrl.trim()) { setErrorMsg("Please enter a URL."); setStatus("error"); return; }
    setStatus("loading"); setErrorMsg(""); setLoadStep(0);
    try {
      for (let i = 0; i < LOADING_STEPS.length - 1; i++) {
        setLoadStep(i); await new Promise(r => setTimeout(r, 600));
      }
      const res = await fetch(`${BACKEND}/load`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: paperUrl.trim() }),
      });
      setLoadStep(LOADING_STEPS.length - 1);
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed");
      await new Promise(r => setTimeout(r, 400));
      setStatus("success"); onLoad(data);
    } catch (e) { setErrorMsg(e.message); setStatus("error"); }
  }

  async function handleDrop(e) {
    e.preventDefault(); setDragging(false);
    const file = e.dataTransfer.files[0];
    if (!file?.name.endsWith(".pdf")) { setErrorMsg("Please drop a PDF file."); setStatus("error"); return; }
    await uploadFile(file);
  }

  async function uploadFile(file) {
    setStatus("loading"); setErrorMsg(""); setLoadStep(0);
    try {
      for (let i = 0; i < LOADING_STEPS.length - 1; i++) {
        setLoadStep(i); await new Promise(r => setTimeout(r, 500));
      }
      const fd = new FormData(); fd.append("file", file);
      const res = await fetch(`${BACKEND}/upload`, { method: "POST", body: fd });
      setLoadStep(LOADING_STEPS.length - 1);
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Upload failed");
      await new Promise(r => setTimeout(r, 400));
      setStatus("success"); onLoad(data);
    } catch (e) { setErrorMsg(e.message); setStatus("error"); }
  }

  return (
    <div style={{ minHeight: "100vh", background: t.bg, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "40px 20px", transition: "background 0.2s" }}>

      {/* Hero */}
      <div style={{ textAlign: "center", marginBottom: 48 }}>
        <div style={{ display: "inline-flex", alignItems: "center", gap: 8, background: t.pillBg, border: `1px solid ${t.pillBorder}`, borderRadius: 20, padding: "5px 14px", marginBottom: 24 }}>
          <div style={{ width: 6, height: 6, borderRadius: "50%", background: t.accent, boxShadow: `0 0 10px ${t.accent}` }} />
          <span style={{ fontSize: 12, color: t.accent, letterSpacing: "0.08em", fontWeight: 500 }}>AI-POWERED RESEARCH ASSISTANT</span>
        </div>
        <h1 style={{ fontSize: "clamp(36px,6vw,64px)", fontWeight: 700, color: t.text, margin: "0 0 16px", letterSpacing: "-0.03em", lineHeight: 1.1 }}>
          Research<span style={{ background: `linear-gradient(135deg,${t.accent},${t.purple})`, WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>GPT</span>
        </h1>
        <p style={{ fontSize: "clamp(15px,2vw,18px)", color: t.text2, maxWidth: 520, margin: "0 auto", lineHeight: 1.7 }}>
          Upload research papers or import directly from arXiv and get instant AI-powered insights.
        </p>
      </div>

      {/* Upload card */}
      <div style={{ width: "100%", maxWidth: 580, background: t.surface, border: `1px solid ${t.border}`, borderRadius: 20, padding: 32, marginBottom: 20, boxShadow: "none", transition: "background 0.2s" }}>
        {status === "loading" ? (
          <div style={{ padding: "20px 0" }}>
            <Progress step={loadStep + 1} total={LOADING_STEPS.length} label={LOADING_STEPS[loadStep]} t={t} />
          </div>
        ) : status === "success" ? (
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 10, padding: "20px 0", color: "#4ade80", fontSize: 15 }}>
            <CheckIcon /> Paper indexed successfully
          </div>
        ) : (
          <>
            {/* Drop zone */}
            <div
              onDragOver={e => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              onClick={() => fileRef.current?.click()}
              style={{
                border: `2px dashed ${dragging ? t.accent : t.border2}`,
                borderRadius: 14, padding: "36px 24px", textAlign: "center",
                background: dragging ? t.pillBg : t.surface2,
                cursor: "pointer", transition: "all 0.2s", marginBottom: 20,
              }}>
              <input ref={fileRef} type="file" accept=".pdf" style={{ display: "none" }}
                onChange={e => e.target.files[0] && uploadFile(e.target.files[0])} />
              <div style={{ color: dragging ? t.accent : t.text3, marginBottom: 14 }}><UploadCloud /></div>
              <p style={{ color: t.text2, fontSize: 15, margin: "0 0 6px", fontWeight: 500 }}>Drag & drop your PDF here</p>
              <p style={{ color: t.text3, fontSize: 13, margin: "0 0 16px" }}>or</p>
              <button style={{ padding: "8px 20px", background: t.pillBg, border: `1px solid ${t.pillBorder}`, borderRadius: 8, color: t.accent, fontSize: 13, fontWeight: 500, cursor: "pointer" }}>
                Browse Files
              </button>
              <p style={{ color: t.text3, fontSize: 12, margin: "12px 0 0" }}>PDF supported</p>
            </div>

            {/* Divider */}
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 20 }}>
              <div style={{ flex: 1, height: 1, background: t.border }} />
              <span style={{ fontSize: 12, color: t.text3, fontWeight: 500 }}>OR</span>
              <div style={{ flex: 1, height: 1, background: t.border }} />
            </div>

            {/* URL input */}
            <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
              <div style={{ flex: 1, display: "flex", alignItems: "center", gap: 10, background: t.surface2, border: `1px solid ${t.border}`, borderRadius: 10, padding: "0 14px" }}>
                <div style={{ color: t.text3, flexShrink: 0 }}><LinkIcon /></div>
                <input type="url" placeholder="Paste arXiv URL or research paper link" value={url}
                  onChange={e => setUrl(e.target.value)}
                  onKeyDown={e => e.key === "Enter" && loadUrl(url)}
                  style={{ flex: 1, background: "none", border: "none", outline: "none", color: t.text, fontSize: 14, padding: "12px 0" }} />
              </div>
              <button onClick={() => loadUrl(url)}
                style={{ padding: "12px 20px", background: `linear-gradient(135deg,${t.accent},${t.accentDark})`, border: "none", borderRadius: 10, color: "#000", fontSize: 14, fontWeight: 600, cursor: "pointer", whiteSpace: "nowrap" }}>
                Import
              </button>
            </div>

            {/* Sample links */}
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {SAMPLE_PAPERS.map(p => (
                <button key={p.url} onClick={() => setUrl(p.url)}
                  style={{ fontSize: 11, padding: "4px 10px", background: t.chipBg, border: `1px solid ${t.chipBorder}`, borderRadius: 20, color: t.text2, cursor: "pointer" }}>
                  {p.label}
                </button>
              ))}
            </div>

            {status === "error" && (
              <div style={{ marginTop: 14, display: "flex", alignItems: "center", gap: 8, padding: "10px 14px", background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.2)", borderRadius: 8, color: "#f87171", fontSize: 13 }}>
                <XIcon /> {errorMsg}
              </div>
            )}
          </>
        )}
      </div>

      {/* Suggestion chips */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 8, justifyContent: "center", maxWidth: 560 }}>
        {SUGGESTIONS.map(s => (
          <div key={s} style={{ fontSize: 12, padding: "6px 14px", background: t.chipBg, border: `1px solid ${t.chipBorder}`, borderRadius: 20, color: t.text2 }}>
            {s}
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Workspace ─────────────────────────────────────────────────────────────────
function Workspace({ papers, activePaperId, setActivePaperId, chatsByPaper, onQuery, loading, onNewPaper, t }) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [input, setInput]             = useState("");
  const bottomRef = useRef();
  const taRef     = useRef();

  const activeMeta = papers.find(p => p.paper_id === activePaperId);
  const messages   = chatsByPaper[activePaperId] || [];

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  function send() {
    if (!input.trim() || loading) return;
    onQuery(input.trim()); setInput("");
    if (taRef.current) taRef.current.style.height = "auto";
  }

  return (
    <div style={{ height: "100vh", background: t.bg, display: "flex", flexDirection: "column", transition: "background 0.2s" }}>

      {/* Header */}
      <div style={{ height: 56, borderBottom: `1px solid ${t.border}`, display: "flex", alignItems: "center", padding: "0 20px", gap: 14, flexShrink: 0, background: t.headerBg, position: "sticky", top: 0, zIndex: 20, boxShadow: "none" }}>
        <button onClick={() => setSidebarOpen(v => !v)}
          style={{ background: "none", border: "none", color: t.text2, cursor: "pointer", display: "flex", padding: 4 }}>
          <MenuIcon />
        </button>
        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <span style={{ fontSize: 16, fontWeight: 700, color: t.text }}>Research</span>
          <span style={{ fontSize: 16, fontWeight: 700, background: `linear-gradient(135deg,${t.accent},${t.purple})`, WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>GPT</span>
        </div>
        <span style={{ fontSize: 12, color: t.text3 }}>AI-Powered Research Assistant</span>

        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 10 }}>
          <button onClick={onNewPaper}
            style={{ display: "flex", alignItems: "center", gap: 6, padding: "6px 14px", background: t.pillBg, border: `1px solid ${t.pillBorder}`, borderRadius: 8, color: t.accent, fontSize: 12, fontWeight: 500, cursor: "pointer" }}>
            <PlusIcon /> New Paper
          </button>
        </div>
      </div>

      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>

        {/* Sidebar */}
        {sidebarOpen && (
          <div style={{ width: 260, borderRight: `1px solid ${t.border}`, background: t.sidebarBg, display: "flex", flexDirection: "column", flexShrink: 0, transition: "background 0.2s" }}>
            <div style={{ padding: "16px 16px 10px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <span style={{ fontSize: 11, fontWeight: 600, color: t.text3, letterSpacing: "0.08em" }}>PAPER LIBRARY</span>
              <button onClick={() => setSidebarOpen(false)} style={{ background: "none", border: "none", color: t.text3, cursor: "pointer", display: "flex" }}><XIcon /></button>
            </div>
            <div style={{ flex: 1, overflowY: "auto", padding: "4px 8px" }}>
              {papers.length === 0
                ? <p style={{ fontSize: 12, color: t.text3, padding: "8px", textAlign: "center" }}>No papers loaded</p>
                : papers.map(p => (
                  <button key={p.paper_id} onClick={() => { setActivePaperId(p.paper_id); setSidebarOpen(false); }}
                    style={{ width: "100%", textAlign: "left", padding: "10px 12px", borderRadius: 10, background: activePaperId === p.paper_id ? t.pillBg : "transparent", border: `1px solid ${activePaperId === p.paper_id ? t.pillBorder : "transparent"}`, cursor: "pointer", marginBottom: 4 }}>
                    <div style={{ display: "flex", alignItems: "flex-start", gap: 8 }}>
                      <div style={{ color: activePaperId === p.paper_id ? t.accent : t.text3, marginTop: 1, flexShrink: 0 }}><FileIcon /></div>
                      <div>
                        <p style={{ margin: 0, fontSize: 12, fontWeight: 500, color: activePaperId === p.paper_id ? t.text : t.text2, lineHeight: 1.4, display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", overflow: "hidden" }}>
                          {p.title}
                        </p>
                        <p style={{ margin: "4px 0 0", fontSize: 11, color: t.text3 }}>{p.pages} pages · {p.chunks} chunks</p>
                      </div>
                    </div>
                  </button>
                ))
              }
            </div>
          </div>
        )}

        {/* Chat */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>

          {/* Active paper pill */}
          {activeMeta && (
            <div style={{ padding: "10px 24px", borderBottom: `1px solid ${t.border}`, flexShrink: 0 }}>
              <div style={{ display: "inline-flex", alignItems: "center", gap: 8, background: t.pillBg, border: `1px solid ${t.pillBorder}`, borderRadius: 20, padding: "5px 14px" }}>
                <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#4ade80", boxShadow: "0 0 6px #4ade80" }} />
                <span style={{ fontSize: 12, color: t.text2 }}>
                  <span style={{ color: t.accent, fontWeight: 500 }}>✓ Indexed</span>
                  {" · "}{activeMeta.title}{" · "}{activeMeta.pages} pages
                </span>
              </div>
            </div>
          )}

          {/* Messages */}
          <div style={{ flex: 1, overflowY: "auto", padding: "24px 0" }}>
            <div style={{ maxWidth: 780, margin: "0 auto", padding: "0 24px" }}>

              {messages.length === 0 && (
                <div style={{ textAlign: "center", padding: "60px 0 40px" }}>
                  <div style={{ width: 56, height: 56, borderRadius: 16, background: t.pillBg, border: `1px solid ${t.pillBorder}`, display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 20px" }}>
                    <div style={{ color: t.accent }}><BookIcon /></div>
                  </div>
                  <h3 style={{ color: t.text, fontSize: 18, fontWeight: 600, margin: "0 0 8px" }}>Ready to explore</h3>
                  <p style={{ color: t.text2, fontSize: 14, margin: "0 0 28px" }}>Ask anything about this research paper</p>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 8, justifyContent: "center" }}>
                    {SUGGESTIONS.map(s => (
                      <button key={s} onClick={() => onQuery(s)}
                        style={{ fontSize: 13, padding: "8px 16px", background: t.surface2, border: `1px solid ${t.border}`, borderRadius: 20, color: t.text2, cursor: "pointer" }}>
                        {s}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {messages.map((msg, i) => (
                <div key={i} style={{ display: "flex", justifyContent: msg.role === "user" ? "flex-end" : "flex-start", marginBottom: 20, animation: "rgptFade 0.2s ease" }}>
                  {msg.role === "assistant" && (
                    <div style={{ width: 32, height: 32, borderRadius: 10, background: t.pillBg, border: `1px solid ${t.pillBorder}`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, marginRight: 12, marginTop: 2 }}>
                      <span style={{ fontSize: 11, fontWeight: 700, color: t.accent }}>AI</span>
                    </div>
                  )}
                  <div style={{
                    maxWidth: "78%", padding: "12px 16px",
                    borderRadius: msg.role === "user" ? "18px 18px 4px 18px" : "4px 18px 18px 18px",
                    background: msg.role === "user" ? t.userBubble : t.aiBubble,
                    border: msg.role === "assistant" ? `1px solid ${t.aiBorder}` : "none",
                    color: msg.role === "user" ? t.userText : t.text,
                    fontSize: 14, lineHeight: 1.7,
                    boxShadow: "none",
                  }}>
                    {msg.role === "user"
                      ? <p style={{ margin: 0, fontWeight: 500 }}>{msg.content}</p>
                      : <Md text={msg.content} t={t} />
                    }
                    {msg.sources > 0 && (
                      <p style={{ margin: "10px 0 0", fontSize: 11, color: t.text3, borderTop: `1px solid ${t.border}`, paddingTop: 8 }}>
                        {msg.sources} source sections retrieved
                      </p>
                    )}
                  </div>
                </div>
              ))}

              {loading && (
                <div style={{ display: "flex", justifyContent: "flex-start", marginBottom: 20 }}>
                  <div style={{ width: 32, height: 32, borderRadius: 10, background: t.pillBg, border: `1px solid ${t.pillBorder}`, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, marginRight: 12 }}>
                    <span style={{ fontSize: 11, fontWeight: 700, color: t.accent }}>AI</span>
                  </div>
                  <div style={{ padding: "12px 16px", background: t.aiBubble, border: `1px solid ${t.aiBorder}`, borderRadius: "4px 18px 18px 18px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <Dots t={t} />
                      <span style={{ fontSize: 12, color: t.text2 }}>Generating answer...</span>
                    </div>
                  </div>
                </div>
              )}

              <div ref={bottomRef} />
            </div>
          </div>

          {/* Input */}
          <div style={{ padding: "16px 24px 24px", flexShrink: 0, borderTop: `1px solid ${t.border}` }}>
            <div style={{ maxWidth: 780, margin: "0 auto" }}>
              <div style={{ background: t.inputBg, border: `1px solid ${t.inputBorder}`, borderRadius: 16, padding: "4px 4px 4px 16px", display: "flex", alignItems: "flex-end", gap: 8, boxShadow: "0 0 40px rgba(0,0,0,0.4)" }}>
                <textarea ref={taRef} value={input}
                  onChange={e => { setInput(e.target.value); e.target.style.height = "auto"; e.target.style.height = Math.min(e.target.scrollHeight, 150) + "px"; }}
                  onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
                  placeholder="Ask anything about this research paper..."
                  rows={1}
                  style={{ flex: 1, background: "none", border: "none", outline: "none", color: t.text, fontSize: 15, lineHeight: 1.6, resize: "none", padding: "12px 0", fontFamily: "inherit", overflowY: "hidden" }} />
                <button onClick={send} disabled={!input.trim() || loading}
                  style={{ width: 40, height: 40, borderRadius: 12, background: input.trim() && !loading ? `linear-gradient(135deg,${t.accent},${t.accentDark})` : t.surface2, border: "none", cursor: input.trim() && !loading ? "pointer" : "not-allowed", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, transition: "all 0.15s" }}>
                  <div style={{ color: input.trim() && !loading ? "#000" : t.text3 }}><SendIcon /></div>
                </button>
              </div>
              <p style={{ textAlign: "center", margin: "8px 0 0", fontSize: 11, color: t.text3 }}>
                Enter to send · Shift+Enter for new line
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── App ───────────────────────────────────────────────────────────────────────
export default function ResearchGPT() {
  const [screen, setScreen]               = useState("landing");
  const [papers, setPapers]               = useState([]);
  const [activePaperId, setActivePaperId] = useState(null);
  const [chatsByPaper, setChatsByPaper]   = useState({});
  const [loading, setLoading]             = useState(false);

  const t = DARK;

  function handleLoad(data) {
    setPapers(prev => prev.find(p => p.paper_id === data.paper_id) ? prev : [...prev, data]);
    setActivePaperId(data.paper_id);
    if (!chatsByPaper[data.paper_id]) {
      setChatsByPaper(prev => ({
        ...prev,
        [data.paper_id]: [{
          role: "assistant",
          content: `**${data.title}**\n\n*${data.author}* · ${data.pages} pages · ${data.chunks} chunks indexed\n\nThe paper has been fully indexed and is ready for analysis. What would you like to know?`,
        }]
      }));
    }
    setScreen("workspace");
  }

  async function handleQuery(question) {
    if (!question || loading || !activePaperId) return;
    setChatsByPaper(prev => ({ ...prev, [activePaperId]: [...(prev[activePaperId] || []), { role: "user", content: question }] }));
    setLoading(true);
    try {
      const res = await fetch(`${BACKEND}/query`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ paper_id: activePaperId, question }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Query failed");
      setChatsByPaper(prev => ({
        ...prev,
        [activePaperId]: [...(prev[activePaperId] || []), { role: "assistant", content: data.answer, sources: data.sources_used }]
      }));
    } catch (e) {
      setChatsByPaper(prev => ({
        ...prev,
        [activePaperId]: [...(prev[activePaperId] || []), { role: "assistant", content: `⚠️ ${e.message}`, isError: true }]
      }));
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        html, body, #root { height: 100%; font-family: 'Inter', -apple-system, sans-serif; background: ${t.bg}; transition: background 0.2s; }
        ::-webkit-scrollbar { width: 5px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: ${t.scrollThumb}; border-radius: 3px; }
        input, textarea, button { font-family: inherit; }
        input::placeholder, textarea::placeholder { color: ${t.text3}; }
        @keyframes rgptBounce { 0%,80%,100%{transform:translateY(0)} 40%{transform:translateY(-5px)} }
        @keyframes rgptFade { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:translateY(0)} }
        .rgpt-md .katex { color: ${t.mdText}; }
        .rgpt-md .katex-display { margin: 12px 0; overflow-x: auto; }
      `}</style>

      {screen === "landing"
        ? <Landing onLoad={handleLoad} t={t} />
        : <Workspace
            papers={papers} activePaperId={activePaperId} setActivePaperId={setActivePaperId}
            chatsByPaper={chatsByPaper} onQuery={handleQuery} loading={loading}
            onNewPaper={() => setScreen("landing")}
            t={t}
          />
      }
    </>
  );
}
