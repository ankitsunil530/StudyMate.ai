import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  ArrowRight,
  BarChart3,
  BookOpen,
  CheckCircle2,
  CircleHelp,
  FileText,
  MessageSquare,
  RefreshCw,
  Sparkles,
  Target,
  TrendingUp,
} from "lucide-react";
import ThemeToggle from "../components/ThemeToggle";
import { API_BASE_URL } from "../config/api";

function formatTime(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
}

function scoreTone(accuracy) {
  if (accuracy >= 80) return "bg-emerald-500";
  if (accuracy >= 50) return "bg-amber-500";
  return "bg-rose-500";
}

export default function Profile() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [conversations, setConversations] = useState([]);
  const [dashboard, setDashboard] = useState(null);

  const userName = useMemo(() => localStorage.getItem("userName") || "User", []);

  const fetchConversations = async () => {
    const token = localStorage.getItem("userToken");
    if (!token) {
      navigate("/login");
      return;
    }

    setLoading(true);
    setError("");
    try {
      const [chatRes, dashRes] = await Promise.all([
        fetch(`${API_BASE_URL}/api/conversations`, {
          headers: { Authorization: `Bearer ${token}` },
        }),
        fetch(`${API_BASE_URL}/api/learning-dashboard`, {
          headers: { Authorization: `Bearer ${token}` },
        }),
      ]);
      const chatData = await chatRes.json().catch(() => ({}));
      const dashData = await dashRes.json().catch(() => ({}));
      if (chatRes.status === 401 || dashRes.status === 401) {
        localStorage.removeItem("userToken");
        navigate("/login");
        return;
      }
      if (!chatRes.ok) throw new Error(chatData.error || "Failed to load chats");
      setConversations(chatData.conversations || []);
      if (dashRes.ok) setDashboard(dashData);
    } catch (e) {
      setError(e?.message || "Failed to load dashboard");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchConversations();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const summary = dashboard?.summary || {};
  const hasAttempts = (summary.quizAttempts || 0) > 0;
  const stats = [
    { label: "PDFs studied", value: summary.pdfsStudied ?? 0, icon: <FileText size={17} /> },
    { label: "Pages parsed", value: summary.parsedPages ?? 0, icon: <BookOpen size={17} /> },
    { label: "Doubts cleared", value: summary.doubtsAsked ?? 0, icon: <CircleHelp size={17} /> },
    { label: "Quiz attempts", value: summary.quizAttempts ?? 0, icon: <BarChart3 size={17} /> },
    {
      label: "Quiz accuracy",
      value: hasAttempts ? `${summary.quizAccuracy ?? 0}%` : "Not started",
      icon: <Target size={17} />,
    },
  ];

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="sticky top-0 z-40 border-b border-border/70 bg-background/80 backdrop-blur-xl">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6">
          <button onClick={() => navigate("/")} className="inline-flex items-center gap-2 text-sm font-bold text-muted-foreground transition hover:text-foreground">
            <ArrowLeft size={17} /> Home
          </button>
          <div className="flex items-center gap-2">
            <button onClick={fetchConversations} className="sm-btn-secondary px-3 py-2" disabled={loading}>
              <RefreshCw size={16} className={loading ? "animate-spin" : ""} />
              <span className="hidden sm:inline">Refresh</span>
            </button>
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 sm:py-12">
        <section className="mb-8 flex flex-col justify-between gap-5 sm:flex-row sm:items-end">
          <div>
            <p className="sm-eyebrow text-primary">Learning dashboard</p>
            <h1 className="mt-2 text-3xl font-black tracking-tight sm:text-5xl">
              Welcome back, {userName}
            </h1>
            <p className="mt-2 max-w-2xl text-muted-foreground">
              See what is improving, what needs attention, and continue where you left off.
            </p>
          </div>
          <button onClick={() => navigate("/upload")} className="sm-btn self-start px-5">
            <FileText size={17} /> Study a new PDF
          </button>
        </section>

        {error && <div className="mb-6 rounded-2xl border border-destructive/30 bg-destructive/10 p-4 text-sm font-semibold text-destructive">{error}</div>}

        <section className="mb-8 grid grid-cols-2 gap-3 lg:grid-cols-5">
          {stats.map(({ label, value, icon }) => (
            <div key={label} className="sm-panel p-4 sm:p-5">
              <div className="mb-5 flex items-center justify-between">
                <span className="sm-eyebrow">{label}</span>
                <span className="rounded-xl bg-primary/10 p-2 text-primary">{icon}</span>
              </div>
              <p className={`font-black tracking-tight ${typeof value === "string" && value.length > 7 ? "text-lg" : "text-3xl"}`}>{value}</p>
            </div>
          ))}
        </section>

        {dashboard && (
          <section className="mb-8 grid grid-cols-1 gap-5 lg:grid-cols-[1.08fr_0.92fr]">
            <div className="sm-card p-5 sm:p-6">
              <div className="mb-5 flex items-start justify-between gap-4">
                <div>
                  <div className="flex items-center gap-2"><Target size={19} className="text-primary" /><h2 className="text-lg font-black">Topics needing attention</h2></div>
                  <p className="mt-1 text-sm text-muted-foreground">Accuracy is based only on questions you attempted.</p>
                </div>
                <span className="rounded-full bg-rose-500/10 px-3 py-1 text-xs font-black text-rose-500">{dashboard.weakTopics?.length || 0} topics</span>
              </div>
              {dashboard.weakTopics?.length ? (
                <div className="space-y-3">
                  {dashboard.weakTopics.map((topic) => (
                    <div key={topic.topic} className="rounded-2xl border border-border/70 bg-background/45 p-4">
                      <div className="flex items-start justify-between gap-4">
                        <div className="min-w-0">
                          <p className="truncate text-sm font-bold">{topic.topic}</p>
                          <p className="mt-1 text-xs text-muted-foreground">{topic.correct} of {topic.total} correct</p>
                        </div>
                        <div className="text-right">
                          <p className="text-lg font-black">{topic.accuracy}%</p>
                          <p className="text-[11px] font-bold uppercase tracking-wider text-rose-500">Needs review</p>
                        </div>
                      </div>
                      <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-muted">
                        <div className={`h-full rounded-full ${scoreTone(topic.accuracy)}`} style={{ width: `${topic.accuracy}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="rounded-2xl border border-dashed border-border p-8 text-center">
                  <CheckCircle2 className="mx-auto text-emerald-500" size={28} />
                  <p className="mt-3 font-bold">No weak topics detected yet</p>
                  <p className="mt-1 text-sm text-muted-foreground">Take a quiz to start measuring topic mastery.</p>
                </div>
              )}
            </div>

            <div className="sm-card p-5 sm:p-6">
              <div className="mb-5 flex items-center gap-2"><TrendingUp size={19} className="text-primary" /><h2 className="text-lg font-black">Recommended next steps</h2></div>
              <div className="space-y-3">
                {(dashboard.recommendations || []).map((item, index) => {
                  const recommendation = typeof item === "string" ? { title: item, action: "" } : item;
                  return (
                    <div key={`${recommendation.title}-${index}`} className="rounded-2xl border border-border/70 bg-background/45 p-4">
                      <div className="flex items-start gap-3">
                        <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-sm font-black text-primary">{index + 1}</span>
                        <div>
                          <div className="flex flex-wrap items-center gap-2">
                            <p className="text-sm font-black">{recommendation.title}</p>
                            {recommendation.priority && <span className="rounded-full bg-primary/10 px-2 py-0.5 text-[10px] font-black uppercase tracking-wider text-primary">{recommendation.priority}</span>}
                          </div>
                          {recommendation.action && <p className="mt-1 text-sm leading-relaxed text-muted-foreground">{recommendation.action}</p>}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
              <button onClick={() => navigate("/upload")} className="sm-btn-secondary mt-4 w-full">Open study workspace <ArrowRight size={16} /></button>
            </div>
          </section>
        )}

        {!!dashboard?.recentAttempts?.length && (
          <section className="sm-card mb-8 p-5 sm:p-6">
            <div className="mb-5 flex items-center justify-between">
              <div><p className="sm-eyebrow">Performance</p><h2 className="mt-1 text-lg font-black">Recent quiz attempts</h2></div>
              <Sparkles size={20} className="text-primary" />
            </div>
            <div className="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-4">
              {dashboard.recentAttempts.map((attempt) => (
                <div key={attempt.id} className="rounded-2xl border border-border/70 bg-background/45 p-4">
                  <p className="truncate text-sm font-bold">{attempt.pdfFileName || "PDF"}</p>
                  <div className="mt-3 flex items-end justify-between">
                    <p className="text-xs text-muted-foreground">{attempt.score} / {attempt.total} correct</p>
                    <p className="text-xl font-black">{attempt.accuracy}%</p>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        <section className="sm-card p-5 sm:p-6">
          <div className="mb-5 flex items-center justify-between gap-4">
            <div><p className="sm-eyebrow">Saved workspace</p><h2 className="mt-1 text-xl font-black">Your conversations</h2></div>
            <span className="inline-flex items-center gap-2 rounded-full bg-muted px-3 py-1.5 text-xs font-bold text-muted-foreground"><MessageSquare size={14} /> {conversations.length} saved</span>
          </div>
          {loading ? (
            <div className="py-12 text-center text-sm font-medium text-muted-foreground">Loading your learning history...</div>
          ) : conversations.length === 0 ? (
            <div className="rounded-2xl border border-dashed border-border py-12 text-center">
              <MessageSquare className="mx-auto text-primary" size={28} />
              <p className="mt-3 font-bold">No saved conversations yet</p>
              <p className="mt-1 text-sm text-muted-foreground">Ask a doubt in Study and it will appear here.</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
              {conversations.map((c) => (
                <button key={c.id} onClick={() => navigate(`/study/${c.pdfId}?conversationId=${c.id}`)} className="group rounded-2xl border border-border/70 bg-background/45 p-4 text-left transition hover:-translate-y-0.5 hover:border-primary/30 hover:bg-card">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <p className="truncate font-black">{c.title || "Chat"}</p>
                      <p className="mt-1 truncate text-sm text-muted-foreground">{c.pdfFileName || "PDF"}{c.lastPageNo ? ` · Last page ${c.lastPageNo}` : ""}</p>
                      <p className="mt-3 text-xs text-muted-foreground">{formatTime(c.updatedAt)}</p>
                    </div>
                    <span className="rounded-xl border border-border bg-card p-2 text-muted-foreground transition group-hover:text-primary"><ArrowRight size={16} /></span>
                  </div>
                </button>
              ))}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
