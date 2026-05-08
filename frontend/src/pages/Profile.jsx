import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, MessageSquare, RefreshCw } from "lucide-react";
import ThemeToggle from "../components/ThemeToggle";
import { API_BASE_URL } from "../config/api";

function formatTime(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleString();
}

export default function Profile() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [conversations, setConversations] = useState([]);

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
      const res = await fetch(`${API_BASE_URL}/api/conversations`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await res.json().catch(() => ({}));
      if (res.status === 401) {
        localStorage.removeItem("userToken");
        setError("Session expired. Please login again.");
        navigate("/login");
        return;
      }
      if (!res.ok) throw new Error(data.error || "Failed to load chats");
      setConversations(data.conversations || []);
    } catch (e) {
      setError(e?.message || "Failed to load chats");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchConversations();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="min-h-screen bg-background text-foreground p-3 sm:p-6">
      <div className="max-w-5xl mx-auto">
        <div className="flex items-center justify-between gap-3 mb-6 sm:mb-8">
          <button
            onClick={() => navigate("/")}
            className="flex items-center gap-2 text-sm sm:text-base text-muted-foreground hover:text-primary transition-colors"
          >
            <ArrowLeft size={18} /> Back
          </button>
          <div className="flex items-center gap-2">
            <button
              onClick={fetchConversations}
              className="inline-flex items-center gap-2 rounded-xl border border-border bg-card/60 px-3 py-2 text-sm font-bold hover:bg-card transition"
            >
              <RefreshCw size={16} /> <span className="hidden sm:inline">Refresh</span>
            </button>
            <ThemeToggle />
          </div>
        </div>

        <div className="sm-card p-4 sm:p-8">
          <div className="flex items-start justify-between gap-4 mb-6">
            <div className="min-w-0">
              <h1 className="text-2xl sm:text-3xl font-black tracking-tight">
                {userName}'s Chats
              </h1>
              <p className="text-sm sm:text-base text-muted-foreground mt-1">
                Continue any saved conversation with your study material.
              </p>
            </div>
            <div className="hidden sm:flex items-center gap-2 text-muted-foreground">
              <MessageSquare size={18} />
              <span className="text-sm font-semibold">
                {conversations.length} saved
              </span>
            </div>
          </div>

          {error && (
            <div className="mb-6 rounded-2xl border border-destructive/30 bg-destructive/10 p-4 text-destructive text-sm font-semibold">
              {error}
            </div>
          )}

          {loading ? (
            <div className="py-10 text-center text-muted-foreground">
              Loading chats...
            </div>
          ) : conversations.length === 0 ? (
            <div className="py-10 text-center">
              <p className="text-muted-foreground font-medium">
                No saved chats yet.
              </p>
              <p className="text-muted-foreground/80 text-sm mt-1">
                Start a chat in Study and it will appear here.
              </p>
              <button
                onClick={() => navigate("/upload")}
                className="sm-btn mt-6 px-6 py-3"
              >
                Upload a PDF
              </button>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {conversations.map((c) => (
                <button
                  key={c.id}
                  onClick={() =>
                    navigate(`/study/${c.pdfId}?conversationId=${c.id}`)
                  }
                  className="text-left rounded-2xl sm:rounded-3xl border border-border bg-card/40 hover:bg-card/70 transition p-4 sm:p-5"
                >
                  <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-2 sm:gap-3">
                    <div className="min-w-0">
                      <p className="font-black truncate">{c.title || "Chat"}</p>
                      <p className="text-sm text-muted-foreground truncate mt-1">
                        {c.pdfFileName || "PDF"}{" "}
                        {c.lastPageNo ? `• Last page ${c.lastPageNo}` : ""}
                      </p>
                    </div>
                    <div className="shrink-0 text-xs text-muted-foreground">
                      {formatTime(c.updatedAt)}
                    </div>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
