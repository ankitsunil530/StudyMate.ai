import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  ChevronRight,
  FileUp,
  GraduationCap,
  LogOut,
  MessageSquare,
  Sparkles,
  X,
  Zap,
} from "lucide-react";
import ThemeToggle from "../components/ThemeToggle";

export default function Landing() {
  const [isLoggedIn, setIsLoggedIn] = useState(() =>
    Boolean(localStorage.getItem("userToken"))
  );
  const [profileOpen, setProfileOpen] = useState(false);
  const [userName, setUserName] = useState(() =>
    localStorage.getItem("userToken")
      ? localStorage.getItem("userName") || "User"
      : ""
  );
  const [showWelcome, setShowWelcome] = useState(() =>
    Boolean(localStorage.getItem("userToken"))
  );
  const navigate = useNavigate();

  useEffect(() => {
    if (!showWelcome) return;
    const t = setTimeout(() => setShowWelcome(false), 2500);
    return () => clearTimeout(t);
  }, [showWelcome]);

  const steps = useMemo(
    () => [
      {
        icon: <FileUp className="text-primary" />,
        title: "Upload PDF",
        desc: "Books, notes, or research papers — upload anything.",
      },
      {
        icon: <Sparkles className="text-amber-500" />,
        title: "Page-by-Page Logic",
        desc: "AI breaks down each page with real-life examples.",
      },
      {
        icon: <MessageSquare className="text-emerald-500" />,
        title: "Clear Doubts",
        desc: "Stuck on a line? Ask the AI Tutor instantly.",
      },
      {
        icon: <GraduationCap className="text-violet-500" />,
        title: "Quiz + Revision",
        desc: "Generate quizzes and revision packs from selected pages.",
      },
    ],
    []
  );

  const logout = () => {
    localStorage.removeItem("userToken");
    localStorage.removeItem("userName");
    setIsLoggedIn(false);
    setUserName("");
    setProfileOpen(false);
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Soft background */}
      <div className="fixed inset-0 -z-10">
        <div className="absolute -top-24 -left-24 h-[420px] w-[420px] rounded-full bg-primary/15 blur-[110px]" />
        <div className="absolute top-24 -right-24 h-[520px] w-[520px] rounded-full bg-sky-500/10 blur-[120px]" />
        <div className="absolute bottom-[-12%] left-1/2 -translate-x-1/2 h-[520px] w-[820px] rounded-full bg-violet-500/10 blur-[140px]" />
      </div>

      {/* Welcome toast */}
      {showWelcome && (
        <div className="fixed top-20 right-6 z-[60] bg-card/70 border border-border backdrop-blur-xl rounded-2xl p-5 shadow-xl max-w-sm">
          <div className="flex items-start gap-3">
            <CheckCircle2 className="text-emerald-500 mt-0.5" size={22} />
            <div className="flex-1">
              <div className="flex items-center justify-between gap-3">
                <h3 className="text-base font-bold">Welcome back</h3>
                <button
                  onClick={() => setShowWelcome(false)}
                  className="text-muted-foreground hover:text-foreground"
                  aria-label="Close"
                >
                  <X size={18} />
                </button>
              </div>
              <p className="text-muted-foreground text-sm mt-1">
                Hi <span className="font-bold text-primary">{userName}</span>, ready to study?
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Navbar */}
      <nav className="sticky top-0 z-50 border-b border-border bg-background/75 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
          <button
            className="flex items-center gap-3 group"
            onClick={() => navigate("/")}
          >
            <div className="bg-gradient-to-br from-primary to-sky-600 p-2 rounded-xl shadow-sm shadow-primary/10 group-hover:scale-[1.02] transition">
              <BrainCircuit className="text-primary-foreground w-6 h-6" />
            </div>
            <span className="text-xl sm:text-2xl font-black tracking-tight">
              StudyMate<span className="text-primary">.ai</span>
            </span>
          </button>

          <div className="flex items-center gap-3">
            <ThemeToggle className="hidden sm:inline-flex" />

            {!isLoggedIn ? (
              <button
                onClick={() => navigate("/login")}
                className="sm-btn px-6 py-2.5 rounded-full"
              >
                Login
              </button>
            ) : (
              <div className="relative">
                <button
                  onClick={() => setProfileOpen((p) => !p)}
                  className="flex items-center gap-3 px-3 py-2 rounded-full border border-border bg-card/50 hover:bg-card transition"
                >
                  <div className="w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold text-sm">
                    {userName.charAt(0).toUpperCase()}
                  </div>
                  <span className="text-sm font-bold hidden sm:inline">
                    {userName}
                  </span>
                </button>

                {profileOpen && (
                  <div className="absolute right-0 mt-3 w-56 sm-card p-2">
                    <div className="px-3 py-2">
                      <p className="text-xs text-muted-foreground">Signed in as</p>
                      <p className="text-sm font-bold truncate">{userName}</p>
                    </div>
                    <div className="h-px bg-border my-1" />
                    <button
                      onClick={() => {
                        setProfileOpen(false);
                        navigate("/profile");
                      }}
                      className="w-full text-left px-3 py-2 rounded-xl hover:bg-muted transition text-sm font-semibold"
                    >
                      My Chats
                    </button>
                    <button
                      onClick={() => {
                        setProfileOpen(false);
                        navigate("/upload");
                      }}
                      className="w-full text-left px-3 py-2 rounded-xl hover:bg-muted transition text-sm font-semibold"
                    >
                      Upload PDF
                    </button>
                    <button
                      onClick={logout}
                      className="w-full text-left px-3 py-2 rounded-xl hover:bg-muted transition text-sm font-semibold text-destructive flex items-center gap-2"
                    >
                      <LogOut size={16} /> Logout
                    </button>
                  </div>
                )}
              </div>
            )}

            <ThemeToggle className="sm:hidden" />
          </div>
        </div>
      </nav>

      {/* Hero */}
      <main className="max-w-7xl mx-auto px-6 pt-16 pb-12">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 items-center">
          <div>
            <div className="inline-flex items-center gap-2 rounded-full border border-border bg-card/40 px-4 py-2 text-sm font-semibold">
              <Zap size={16} className="text-primary" />
              AI-first studying, built for focus
            </div>

            <h1 className="mt-6 text-4xl sm:text-5xl font-black tracking-tight leading-[1.05]">
              Understand PDFs faster, <span className="text-primary">one page</span>{" "}
              at a time.
            </h1>
            <p className="mt-5 text-muted-foreground text-lg leading-relaxed max-w-xl">
              StudyMate turns tough pages into clear explanations, helps you ask
              doubts, and generates quizzes + revision packs so you remember
              longer.
            </p>

            <div className="mt-8 flex flex-col sm:flex-row gap-3">
              <button
                onClick={() => navigate(isLoggedIn ? "/upload" : "/login")}
                className="sm-btn px-8 py-4 text-base"
              >
                Get Started <ChevronRight size={18} />
              </button>
              <button
                onClick={() => navigate(isLoggedIn ? "/upload" : "/login")}
                className="sm-btn-secondary px-8 py-4 text-base"
              >
                Try with a PDF
              </button>
            </div>

            <div className="mt-10 flex items-center gap-6 text-sm text-muted-foreground">
              <div className="flex items-center gap-2">
                <BarChart3 size={16} className="text-primary" />
                Page-by-page progress
              </div>
              <div className="flex items-center gap-2">
                <Sparkles size={16} className="text-amber-500" />
                AI explanations
              </div>
            </div>
          </div>

          <div className="sm-card p-8 lg:p-10">
            <h3 className="text-lg font-black">What you get</h3>
            <p className="text-muted-foreground text-sm mt-1">
              Designed to keep students calm, focused, and consistent.
            </p>

            <div className="mt-6 grid grid-cols-1 gap-4">
              {steps.map((s) => (
                <div
                  key={s.title}
                  className="rounded-2xl border border-border bg-card/40 p-5 flex items-start gap-4"
                >
                  <div className="w-11 h-11 rounded-2xl bg-muted flex items-center justify-center">
                    {s.icon}
                  </div>
                  <div>
                    <p className="font-bold">{s.title}</p>
                    <p className="text-muted-foreground text-sm mt-0.5">
                      {s.desc}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </main>

      {/* Steps */}
      <section className="border-y border-border bg-card/30">
        <div className="max-w-7xl mx-auto px-6 py-16">
          <h2 className="text-2xl sm:text-3xl font-black tracking-tight">
            A simple workflow that actually works
          </h2>
          <p className="text-muted-foreground mt-2 max-w-2xl">
            No messy dashboards. Just your PDF, page controls, and a calm chat for doubts.
          </p>

          <div className="mt-10 grid grid-cols-1 md:grid-cols-4 gap-4">
            {steps.map((step) => (
              <div key={step.title} className="rounded-3xl border border-border bg-card/50 p-6">
                <div className="w-12 h-12 rounded-2xl bg-muted flex items-center justify-center mb-4">
                  {step.icon}
                </div>
                <h3 className="font-bold text-lg">{step.title}</h3>
                <p className="text-muted-foreground text-sm mt-2">{step.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <footer className="max-w-7xl mx-auto px-6 py-12 text-center">
        <p className="text-muted-foreground text-sm font-medium">
          StudyMate.ai © 2026 • Built for students, designed for focus.
        </p>
      </footer>
    </div>
  );
}
