import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import Markdown from "react-markdown";
import {
  ArrowUp,
  ArrowUpRight,
  BookOpen,
  Check,
  ChevronDown,
  Copy,
  ExternalLink,
  FileText,
  Menu,
  MessageSquare,
  Plus,
  RotateCcw,
  Sparkles,
} from "lucide-react";
import type { Citation, Conversation, ManagerResponse } from "./types";
import "./styles.css";

const STORAGE_KEY = "cmpe256.manager-chat.v1";
const specialistNames: Record<string, string> = {
  clinical: "Clinical",
  corporate_responsibility: "Corporate responsibility",
  technology: "Technology",
  sec: "SEC & financial filings",
};
const starters = [
  {
    label: "Explore technology",
    text: "Which Catalyst 8300 has two 10Gbps ports?",
    icon: "01",
  },
  {
    label: "Review clinical evidence",
    text: "What clinical evidence describes trial recruitment?",
    icon: "02",
  },
  {
    label: "Understand food security",
    text: "What does SOFI say about food affordability?",
    icon: "03",
  },
];
const newConversation = (): Conversation => ({
  id: crypto.randomUUID(),
  turns: [],
});
function restore(): Conversation[] {
  try {
    const saved = JSON.parse(sessionStorage.getItem(STORAGE_KEY) || "null");
    if (
      Array.isArray(saved) &&
      saved.length &&
      saved.every(
        (c) =>
          c &&
          typeof c.id === "string" &&
          Array.isArray(c.turns) &&
          c.turns.every(
            (t: {
              id?: unknown;
              question?: unknown;
              response?: ManagerResponse;
            }) =>
              t &&
              typeof t.id === "string" &&
              typeof t.question === "string" &&
              (!t.response || isResponse(t.response)),
          ),
      )
    ) {
      return saved.map((c) => ({
        ...c,
        turns: c.turns.map(
          (t: object & { response?: ManagerResponse; error?: string }) =>
            !t.response && !t.error
              ? {
                  ...t,
                  error:
                    "This request was interrupted. Retry to ask the manager again.",
                }
              : t,
        ),
      }));
    }
  } catch {
    /* A blocked or malformed session store must not prevent chatting. */
  }
  return [newConversation()];
}
function isResponse(value: unknown): value is ManagerResponse {
  if (!value || typeof value !== "object") return false;
  const r = value as ManagerResponse;
  return (
    typeof r.answer === "string" &&
    Array.isArray(r.agents_used) &&
    r.agents_used.every((a) => typeof a === "string") &&
    Array.isArray(r.citations) &&
    r.citations.every(
      (c) =>
        c &&
        typeof c.id === "string" &&
        typeof c.source === "string" &&
        typeof c.excerpt === "string",
    ) &&
    Array.isArray(r.limitations) &&
    r.limitations.every((l) => typeof l === "string") &&
    Array.isArray(r.findings) &&
    r.findings.every(
      (f) =>
        f &&
        typeof f.text === "string" &&
        Array.isArray(f.evidence_ids) &&
        f.evidence_ids.every((id) => typeof id === "string"),
    ) &&
    Array.isArray(r.recommendations) &&
    r.recommendations.every(
      (rec) =>
        rec &&
        typeof rec.action === "string" &&
        typeof rec.rationale === "string" &&
        Array.isArray(rec.evidence_ids) &&
        rec.evidence_ids.every((id) => typeof id === "string"),
    ) &&
    !!r.citation_labels &&
    typeof r.citation_labels === "object" &&
    Object.values(r.citation_labels).every((label) => typeof label === "string")
  );
}
function webSource(source: string) {
  try {
    const url = new URL(source);
    return ["https:", "http:"].includes(url.protocol) &&
      !url.username &&
      !url.password
      ? url.href
      : undefined;
  } catch {
    return undefined;
  }
}
function Source({ citation, label }: { citation: Citation; label: string }) {
  const href = webSource(citation.source);
  return (
    <details className="source">
      <summary>
        <FileText size={16} />
        <span>
          <strong>{label}</strong>
          <small>
            {citation.id}
            {citation.report_year ? ` · Report ${citation.report_year}` : ""}
          </small>
        </span>
        <ChevronDown size={15} />
      </summary>
      <div className="source-body">
        <blockquote>{citation.excerpt}</blockquote>
        {href ? (
          <a href={href} target="_blank" rel="noreferrer">
            Open original source <ExternalLink size={12} />
          </a>
        ) : (
          <span className="local-source">
            Local document · {citation.source_id}
          </span>
        )}
      </div>
    </details>
  );
}
function Answer({ response }: { response: ManagerResponse }) {
  const [copied, setCopied] = useState(false);
  const [copyError, setCopyError] = useState(false);
  async function copy() {
    try {
      await navigator.clipboard.writeText(response.answer);
      setCopied(true);
      setCopyError(false);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopyError(true);
    }
  }
  return (
    <article className="assistant-message">
      <div className="assistant-heading">
        <span className="avatar small">
          <Sparkles size={17} />
        </span>
        <strong>Manager Agent</strong>
        <span className="eyebrow">MANAGER RESPONSE</span>
      </div>
      <div className="answer-content">
        <Markdown
          components={{
            a: ({ href, children }) => {
              const safe = href && webSource(href);
              return safe ? (
                <a href={safe} target="_blank" rel="noreferrer">
                  {children}
                </a>
              ) : (
                <span>{children}</span>
              );
            },
          }}
        >
          {response.answer}
        </Markdown>
      </div>
      {response.agents_used.length > 0 && (
        <div className="specialists">
          <span>Consulted</span>
          {response.agents_used.map((name) => (
            <span className="specialist" key={name}>
              <span className="tiny-dot" />
              {specialistNames[name] || name}
            </span>
          ))}
        </div>
      )}
      {response.limitations.length > 0 && (
        <div className="limitations">
          <BookOpen size={15} />
          <div>
            <strong>Evidence & limitations</strong>
            <ul>
              {response.limitations.map((item, i) => (
                <li key={i}>{item}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
      {(response.findings.length > 0 ||
        response.recommendations.length > 0) && (
        <details className="analysis">
          <summary>
            Findings & recommendations <ChevronDown size={14} />
          </summary>
          {response.findings.map((f, i) => (
            <p key={`f${i}`}>
              {f.text} <small>[{f.evidence_ids.join(", ")}]</small>
            </p>
          ))}
          {response.recommendations.map((r, i) => (
            <p key={`r${i}`}>
              <strong>{r.action}</strong> — {r.rationale}{" "}
              <small>[{r.evidence_ids.join(", ")}]</small>
            </p>
          ))}
        </details>
      )}
      {response.citations.length > 0 && (
        <section className="sources">
          <h3>
            Sources <span>{response.citations.length}</span>
          </h3>
          {response.citations.map((c) => (
            <Source
              key={c.id}
              citation={c}
              label={response.citation_labels[c.id] || c.source_id}
            />
          ))}
        </section>
      )}
      <button
        className="copy-button"
        onClick={copy}
        aria-label="Copy manager answer"
      >
        {copied ? <Check size={14} /> : <Copy size={14} />}
        {copied ? "Copied" : "Copy answer"}
      </button>
      {copyError && (
        <small role="status">
          {" "}
          Clipboard unavailable. Select the answer to copy it.
        </small>
      )}
    </article>
  );
}
function App() {
  const [conversations, setConversations] = useState<Conversation[]>(restore);
  const [activeId, setActiveId] = useState(conversations[0].id);
  const [draft, setDraft] = useState("");
  const [pendingId, setPendingId] = useState<string | null>(null);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [storageError, setStorageError] = useState(false);
  const bottom = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  const inFlight = useRef(false);
  const conversation =
    conversations.find((c) => c.id === activeId) || conversations[0];
  const hasMessages = conversation.turns.length > 0;
  useEffect(() => {
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
      setStorageError(false);
    } catch {
      setStorageError(true);
    }
  }, [conversations]);
  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [conversations, pendingId, activeId]);
  useEffect(() => {
    if (input.current) {
      input.current.style.height = "auto";
      input.current.style.height = `${Math.min(input.current.scrollHeight, 180)}px`;
    }
  }, [draft]);
  function startNew() {
    const next = newConversation();
    setConversations((all) => [next, ...all.filter((c) => c.turns.length)]);
    setActiveId(next.id);
    setDraft("");
    setMobileOpen(false);
    input.current?.focus();
  }
  async function send(question = draft, retryId?: string) {
    const trimmed = question.trim();
    if (inFlight.current || !trimmed || trimmed.length > 8000) return;
    inFlight.current = true;
    const chatId = conversation.id;
    const turnId = retryId || crypto.randomUUID();
    setPendingId(turnId);
    setDraft("");
    setConversations((all) =>
      all.map((c) =>
        c.id !== chatId
          ? c
          : {
              ...c,
              turns: retryId
                ? c.turns.map((t) =>
                    t.id === turnId ? { id: t.id, question: trimmed } : t,
                  )
                : [...c.turns, { id: turnId, question: trimmed }],
            },
      ),
    );
    try {
      const result = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed }),
      });
      if (!result.ok) {
        let detail = "";
        try {
          const body = await result.json();
          if (typeof body.detail === "string") detail = body.detail;
        } catch {
          /* Proxy errors may return HTML. */
        }
        throw new Error(
          detail ||
            `The manager is unavailable (HTTP ${result.status}). Check that the backend is running, then retry.`,
        );
      }
      const response: unknown = await result.json();
      if (!isResponse(response))
        throw new Error(
          "The backend returned an unexpected response. Check the backend version, then retry.",
        );
      setConversations((all) =>
        all.map((c) =>
          c.id !== chatId
            ? c
            : {
                ...c,
                turns: c.turns.map((t) =>
                  t.id === turnId ? { ...t, response, error: undefined } : t,
                ),
              },
        ),
      );
    } catch (error) {
      const message =
        error instanceof TypeError
          ? "Could not reach the backend. Check your connection and make sure FastAPI is running on port 8000, then retry."
          : error instanceof Error
            ? error.message
            : "Something went wrong. Please retry.";
      setConversations((all) =>
        all.map((c) =>
          c.id !== chatId
            ? c
            : {
                ...c,
                turns: c.turns.map((t) =>
                  t.id === turnId ? { ...t, error: message } : t,
                ),
              },
        ),
      );
    } finally {
      inFlight.current = false;
      setPendingId(null);
      input.current?.focus();
    }
  }
  return (
    <div className="app-shell">
      {mobileOpen && (
        <button
          className="sidebar-overlay"
          aria-label="Close navigation"
          onClick={() => setMobileOpen(false)}
        />
      )}
      <aside className={`sidebar ${mobileOpen ? "open" : ""}`}>
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            startNew();
          }}
        >
          <span className="brand-mark">
            <Sparkles size={22} />
          </span>
          <span>
            Consult<span className="brand-period">.</span>
            <small>CMPE 256 / HACKATHON</small>
          </span>
        </a>
        <button className="new-chat" onClick={startNew}>
          <Plus size={17} /> New conversation{" "}
          <span>
            <ArrowUpRight size={14} />
          </span>
        </button>
        <div className="history-label">THIS SESSION</div>
        <nav aria-label="Session conversations">
          {conversations
            .filter((c) => c.turns.length)
            .map((c) => (
              <button
                key={c.id}
                className={`history-item ${c.id === activeId ? "selected" : ""}`}
                onClick={() => {
                  setActiveId(c.id);
                  setDraft("");
                  setMobileOpen(false);
                }}
              >
                <MessageSquare size={15} />
                <span>{c.turns[0].question}</span>
              </button>
            ))}
          {!conversations.some((c) => c.turns.length) && (
            <p className="history-empty">
              Your conversations will
              <br />
              appear here.
            </p>
          )}
        </nav>
        <div className="sidebar-bottom">
          <div className="team-note">
            <span className="tiny-dot" />
            <strong>One manager. Connected expertise.</strong>
            <p>
              Connected expertise, grounded
              <br />
              in your project’s evidence.
            </p>
          </div>
          <div className="project">
            <span className="project-avatar">256</span>
            <div>
              <strong>CMPE 256</strong>
              <small>Hackathon workspace</small>
            </div>
            <span className="project-tag">TEAM</span>
          </div>
        </div>
      </aside>
      <main>
        <header className="topbar">
          <div>
            <button
              className="mobile-toggle icon-button"
              aria-label="Open navigation"
              onClick={() => setMobileOpen(true)}
            >
              <Menu size={20} />
            </button>
            <span>Manager Chat</span>
            <span className="header-divider">/</span>
            <span className="header-caption">Your research, connected</span>
          </div>
          <span className="workspace-badge">
            <span className="tiny-dot" />
            HACKATHON PROJECT
          </span>
        </header>
        <div
          className={`conversation-scroll ${hasMessages ? "" : "welcome-scroll"}`}
        >
          {!hasMessages ? (
            <section className="welcome">
              <div className="welcome-kicker">
                <span className="line" /> A LITTLE EXPERTISE GOES A LONG WAY
              </div>
              <span className="avatar welcome-avatar">
                <Sparkles size={29} strokeWidth={1.5} />
              </span>
              <h1>
                Big questions.
                <br />
                <span>Connected answers.</span>
              </h1>
              <p>
                Your Manager Agent brings the right specialists together.
                <br className="desktop-break" /> Ask a question. Explore the
                evidence. Find a way forward.
              </p>
              <div className="starter-grid">
                {starters.map((item) => (
                  <button
                    className="starter"
                    key={item.icon}
                    onClick={() => {
                      setDraft(item.text);
                      input.current?.focus();
                    }}
                  >
                    <span className="starter-top">
                      <span>{item.icon}</span>
                      <ArrowUpRight size={17} />
                    </span>
                    <strong>{item.label}</strong>
                    <span>{item.text}</span>
                  </button>
                ))}
              </div>
              <div className="welcome-foot">
                <BookOpen size={14} /> Built around your sources. Guided by your
                manager.
              </div>
            </section>
          ) : (
            <div
              className="transcript"
              role="log"
              aria-label="Conversation history"
            >
              {conversation.turns.map((turn) => (
                <section className="turn" key={turn.id}>
                  <div className="user-message">
                    <span className="message-label">YOU</span>
                    <div>{turn.question}</div>
                  </div>
                  {turn.response && <Answer response={turn.response} />}
                  {turn.error && (
                    <div className="error-message" role="alert">
                      <strong>We couldn’t complete that request</strong>
                      <p>{turn.error}</p>
                      <button
                        disabled={!!pendingId}
                        onClick={() => send(turn.question, turn.id)}
                      >
                        <RotateCcw size={14} /> Retry question
                      </button>
                    </div>
                  )}
                  {pendingId === turn.id && (
                    <div className="loading" role="status">
                      <span className="avatar small">
                        <Sparkles size={17} />
                      </span>
                      <span>
                        Manager is consulting the evidence
                        <span className="loading-dots">…</span>
                      </span>
                    </div>
                  )}
                </section>
              ))}
              <div ref={bottom} />
            </div>
          )}
        </div>
        <div className="composer-area">
          {storageError && (
            <p className="storage-warning" role="status">
              Session storage is unavailable. History will last until this page
              is refreshed.
            </p>
          )}
          <form
            className="composer"
            onSubmit={(e) => {
              e.preventDefault();
              void send();
            }}
          >
            <label className="sr-only" htmlFor="question">
              Ask the Manager Agent
            </label>
            <textarea
              id="question"
              ref={input}
              placeholder="Ask anything about your project’s evidence…"
              rows={1}
              maxLength={8000}
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (
                  e.key === "Enter" &&
                  !e.shiftKey &&
                  !e.nativeEvent.isComposing
                ) {
                  e.preventDefault();
                  void send();
                }
              }}
            />
            <div className="composer-bottom">
              <span>
                <Sparkles size={14} /> Manager Agent{" "}
                <span className="composer-dot">·</span>{" "}
                <span className="evidence-label">Evidence-led answers</span>
              </span>
              <button
                className="send-button"
                type="submit"
                disabled={!draft.trim() || !!pendingId}
                aria-label="Send question"
              >
                <ArrowUp size={19} />
              </button>
            </div>
          </form>
          <p className="composer-note">
            History stays in this browser tab. Each question is answered
            independently.
            <span className="keyboard-hint">Shift + Enter for a new line</span>
          </p>
        </div>
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
