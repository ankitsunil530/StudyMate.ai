import { useState, useEffect, useRef } from "react";
import { useParams, useNavigate, useLocation } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import {
  BrainCircuit,
  ChevronLeft,
  ChevronRight,
  Send,
  Minimize2,
  Maximize2,
  LogOut,
  Loader,
  Play,
  HelpCircle,
  BookOpen,
  FileText,
} from "lucide-react";
import { API_BASE_URL } from "../config/api";
import ThemeToggle from "../components/ThemeToggle";

export default function Study() {
  const { pdf_id } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [pdfName, setPdfName] = useState("Loading...");
  const [explanation, setExplanation] = useState("");
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [pdfCollapsed, setPdfCollapsed] = useState(false);
  const [language, setLanguage] = useState("hinglish");
  const [explanationLoading, setExplanationLoading] = useState(false);
  const [hasExplanation, setHasExplanation] = useState(false);
  const [showQuizModal, setShowQuizModal] = useState(false);
  const [selectedPages, setSelectedPages] = useState([]);
  const [quiz, setQuiz] = useState(null);
  const [quizLoading, setQuizLoading] = useState(false);
  const [showRevisionModal, setShowRevisionModal] = useState(false);
  const [selectedRevisionPages, setSelectedRevisionPages] = useState([]);
  const [revisionPack, setRevisionPack] = useState(null);
  const [revisionLoading, setRevisionLoading] = useState(false);
  const [pdfPageImage, setPdfPageImage] = useState("");
  const [conversationId, setConversationId] = useState(null);
  const conversationIdRef = useRef(null);
  const [chatSaveError, setChatSaveError] = useState("");

  // Load language from session
  useEffect(() => {
    const savedLanguage = sessionStorage.getItem("explanationLanguage");
    if (savedLanguage) {
      setLanguage(savedLanguage);
    }
  }, []);

  // Read conversation id from URL (for "continue chat")
  useEffect(() => {
    const id = new URLSearchParams(location.search).get("conversationId");
    setConversationId(id || null);
  }, [location.search]);

  useEffect(() => {
    conversationIdRef.current = conversationId;
  }, [conversationId]);

  // Fetch PDF info on mount
  useEffect(() => {
    if (!pdf_id) return;
    const fetchPdfInfo = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/pdf/${pdf_id}`);
        let data;
try {
  data = await response.json();
} catch {
  data = {};
}
        if (response.ok) {
          setPdfName(data.fileName || "Unknown PDF");
          setTotalPages(data.totalPages || 1);
        }
      } catch (err) {
        console.error("Error fetching PDF info:", err);
      }
    };
    fetchPdfInfo();
  }, [pdf_id]);

  // Load saved chat messages (if continuing a conversation)
  useEffect(() => {
    if (!conversationId) return;
    const token = localStorage.getItem("userToken");
    if (!token) return;

    const loadConversation = async () => {
      try {
        const res = await fetch(
          `${API_BASE_URL}/api/conversations/${conversationId}`,
          { headers: { Authorization: `Bearer ${token}` } }
        );
        const data = await res.json().catch(() => ({}));
        if (!res.ok) return;
        const conv = data.conversation;
        if (!conv) return;
        const mapped = (conv.messages || []).map((m, idx) => ({
          id: `${idx}-${m.createdAt || ""}`,
          role: m.role || "model",
          text: m.text || "",
        }));
        setMessages(mapped);
      } catch {
        // ignore (chat can still work without history)
      }
    };

    loadConversation();
  }, [conversationId, pdf_id]);

  // Reset explanation state when page changes
  useEffect(() => {
    setHasExplanation(false);
    setExplanation("");
    setPdfPageImage("");

    if (!conversationIdRef.current) {
      setMessages([]);
    }
  }, [currentPage]);

  // Fetch PDF page image when page changes
  useEffect(() => {
    if (!pdf_id) return;
    const fetchPdfPageImage = async () => {
      try {
        const response = await fetch(
          `${API_BASE_URL}/pdf/${pdf_id}/page/${currentPage}/image`
        );
        let data;
try {
  data = await response.json();
} catch {
  data = {};
}
        if (response.ok && data.image) {
          setPdfPageImage(data.image);
        }
      } catch (err) {
        console.error("Error fetching PDF page image:", err);
      }
    };
    fetchPdfPageImage();
  }, [pdf_id, currentPage]);

  // Handle Explain button click
  const handleExplain = async () => {
    if (!pdf_id) return;
    setExplanationLoading(true);
    setHasExplanation(false);
    try {
      const response = await fetch(`${API_BASE_URL}/parse-page`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          pdf_id,
          page_no: currentPage,
          language,
        }),
      });

      let data;
      const text = await response.text();

      try {
        data = JSON.parse(text);
      } catch {
        console.error("Non-JSON response:", text);
        throw new Error("Server error (non-JSON response)");
      }

      if (response.ok) {
        setExplanation(data.explanation || "No explanation available");
        setHasExplanation(true);
        // Reset chat only for non-saved chats
        if (!conversationId) setMessages([]);
      } else {
        setExplanation("Failed to load explanation. Please try again.");
      }
    } catch (err) {
      console.error("Error fetching explanation:", err);
      setExplanation("Failed to load explanation. Please try again.");
    } finally {
      setExplanationLoading(false);
    }
  };

  // Handle doubt submission
  const handleSendDoubt = async (e) => {
    e.preventDefault();
    if (!inputValue.trim()) return;
    setChatSaveError("");

    const userMessage = {
      id: Date.now(),
      role: "user",
      text: inputValue,
    };

    setMessages((prev) => [...prev.slice(-50), userMessage]);
    const queryText = inputValue;
    setInputValue("");
    setIsLoading(true);

    try {
      const token = localStorage.getItem("userToken");
      let activeConversationId = conversationId;

      // Auto-create a saved conversation (so it appears in Profile)
      if (token && !activeConversationId) {
        const createRes = await fetch(`${API_BASE_URL}/api/conversations`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({ pdf_id, page_no: currentPage }),
        });
        const createData = await createRes.json().catch(() => ({}));
        if (createRes.status === 401) {
          localStorage.removeItem("userToken");
        }
        if (!createRes.ok) {
          // User expects chats to be saved; don't silently fall back.
          setChatSaveError(
            createRes.status === 401
              ? "Login expired. Please login again to save chats."
              : createData.error || "Failed to start a saved chat. Try again."
          );
          setIsLoading(false);
          return;
        }
        if (createData.conversation?.id) {
          activeConversationId = createData.conversation.id;
          setConversationId(activeConversationId);
          navigate(`/study/${pdf_id}?conversationId=${activeConversationId}`, {
            replace: true,
          });
        }
      }

      // If logged in, store messages in conversations; else fallback to old endpoint
      const response = await fetch(
        token && activeConversationId
          ? `${API_BASE_URL}/api/conversations/${activeConversationId}/messages`
          : `${API_BASE_URL}/ask-doubt`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(token && activeConversationId
              ? { Authorization: `Bearer ${token}` }
              : {}),
          },
          body: JSON.stringify({
            pdf_id,
            page_no: currentPage,
            query: queryText,
            language,
          }),
        }
      );

      const data = await response.json().catch(() => ({}));

      if (response.status === 401) {
        localStorage.removeItem("userToken");
        setChatSaveError("Login expired. Please login again to save chats.");
      }

      const aiMessage = {
        id: Date.now() + 1,
        role: "model",
        text: response.ok
          ? data.answer || "Unable to generate answer."
          : `Error: ${data.error || "Failed to get response"}`,
      };
      setMessages((prevMessages) => [...prevMessages, aiMessage]);
    } catch (err) {
      console.error("Error asking doubt:", err);
      const errorMessage = {
        id: Date.now() + 1,
        role: "model",
        text: "Connection error. Please try again.",
      };
      setMessages((prevMessages) => [...prevMessages, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle quiz generation
  const handleGenerateQuiz = async () => {
    if (selectedPages.length === 0) {
      alert("Please select at least one page");
      return;
    }
    setQuizLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/generate-quiz`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          pdf_id,
          page_numbers: selectedPages,
          language,
          num_questions: 5,
        }),
      });

      let data;
try {
  data = await response.json();
} catch {
  data = {};
}
      if (response.ok && data.quiz) {
        setQuiz(data.quiz);
      } else {
        alert(data.error || "Failed to generate quiz");
      }
    } catch (err) {
      console.error("Error generating quiz:", err);
      alert("Failed to generate quiz. Please try again.");
    } finally {
      setQuizLoading(false);
    }
  };

  // Handle revision pack generation
  const handleGenerateRevisionPack = async () => {
    if (selectedRevisionPages.length === 0) {
      alert("Please select at least one page");
      return;
    }
    setRevisionLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/generate-revision-pack`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          pdf_id,
          page_numbers: selectedRevisionPages,
          language,
        }),
      });

      let data;
try {
  data = await response.json();
} catch {
  data = {};
}
      if (response.ok && data.revision_pack) {
        setRevisionPack(data.revision_pack);
      } else {
        alert(data.error || "Failed to generate revision pack");
      }
    } catch (err) {
      console.error("Error generating revision pack:", err);
      alert("Failed to generate revision pack. Please try again.");
    } finally {
      setRevisionLoading(false);
    }
  };

  const togglePageSelection = (pageNum) => {
    setSelectedPages((prev) =>
      prev.includes(pageNum)
        ? prev.filter((p) => p !== pageNum)
        : [...prev, pageNum]
    );
  };

  const toggleRevisionPageSelection = (pageNum) => {
    setSelectedRevisionPages((prev) =>
      prev.includes(pageNum)
        ? prev.filter((p) => p !== pageNum)
        : [...prev, pageNum]
    );
  };

  const chatEnabled = hasExplanation || messages.length > 0 || !!conversationId;

  return (
    <div className="study-theme min-h-screen lg:h-screen lg:overflow-hidden flex flex-col">
      {/* Top Navbar */}
      <nav className="study-navbar shrink-0 w-full z-40 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-3 sm:px-6 py-3 lg:h-16 flex flex-col lg:flex-row lg:items-center justify-between gap-3">
          <div className="flex items-center gap-3 min-w-0 w-full lg:w-auto">
            <BrainCircuit className="study-accent shrink-0" size={24} />
            <div className="min-w-0">
              <p className="text-xs study-text-muted font-medium">STUDYING</p>
              <p className="text-sm font-bold truncate max-w-[72vw] sm:max-w-[420px]">
                {pdfName}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 sm:gap-3 flex-wrap w-full lg:w-auto justify-between lg:justify-end">
            <button
              onClick={() => setShowQuizModal(true)}
              className="flex items-center gap-2 px-3 sm:px-4 py-2 rounded-xl border border-border bg-card/50 hover:bg-card transition-colors text-sm font-semibold"
            >
              <BookOpen size={18} /> Quiz
            </button>
            <button
              onClick={() => setShowRevisionModal(true)}
              className="flex items-center gap-2 px-3 sm:px-4 py-2 rounded-xl border border-border bg-card/50 hover:bg-card transition-colors text-sm font-semibold"
            >
              <FileText size={18} /> <span className="hidden sm:inline">Revision</span>
            </button>
            <div className="text-center px-3 sm:px-4 py-2 rounded-lg study-card">
              <p className="text-xs study-text-muted">Page</p>
              <p className="text-base sm:text-lg font-bold study-accent whitespace-nowrap">
                {currentPage} / {totalPages}
              </p>
            </div>
            <ThemeToggle />
            <button
              onClick={() => navigate("/profile")}
              className="hidden md:inline-flex items-center gap-2 px-4 py-2 rounded-xl border border-border bg-card/40 hover:bg-card transition text-sm font-bold"
              title="Open saved chats"
            >
              Chats
            </button>
            <button
              onClick={() => navigate("/")}
              className="flex items-center gap-2 px-3 sm:px-4 py-2 rounded-xl transition-colors text-muted-foreground hover:text-foreground text-sm font-semibold"
            >
              <LogOut size={18} /> <span className="hidden sm:inline">Exit</span>
            </button>
          </div>
        </div>
      </nav>

      {/* Quiz Modal */}
      {showQuizModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-3 sm:p-6">
          <div className="study-panel rounded-2xl p-4 sm:p-6 max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold study-accent">Generate Quiz</h2>
              <button
                onClick={() => {
                  setShowQuizModal(false);
                  setQuiz(null);
                  setSelectedPages([]);
                }}
                className="text-muted-foreground hover:text-foreground"
              >
                ✕
              </button>
            </div>

            {!quiz ? (
              <div>
                <p className="text-muted-foreground mb-4">
                  Select pages you've studied to generate a quiz:
                </p>
                <div className="grid grid-cols-5 sm:grid-cols-10 gap-2 mb-6">
                  {Array.from({ length: totalPages }, (_, i) => i + 1).map(
                    (pageNum) => (
                      <button
                        key={pageNum}
                        onClick={() => togglePageSelection(pageNum)}
                        className={`p-2 rounded-lg border-2 transition-colors ${
                          selectedPages.includes(pageNum)
                            ? "bg-primary border-primary text-primary-foreground"
                            : "bg-card/40 border-border text-muted-foreground hover:border-primary"
                        }`}
                      >
                        {pageNum}
                      </button>
                    )
                  )}
                </div>
                <button
                  onClick={handleGenerateQuiz}
                  disabled={selectedPages.length === 0 || quizLoading}
                  className="w-full bg-primary hover:opacity-95 disabled:opacity-50 text-primary-foreground font-bold py-3 rounded-xl transition-all flex items-center justify-center gap-2"
                >
                  {quizLoading ? (
                    <>
                      <Loader className="animate-spin" size={20} />
                      Generating Quiz...
                    </>
                  ) : (
                    <>
                      <BookOpen size={20} />
                      Generate Quiz ({selectedPages.length} pages)
                    </>
                  )}
                </button>
              </div>
            ) : (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-xl font-bold">
                    Quiz ({quiz.questions?.length || 0} questions)
                  </h3>
                  <button
                    onClick={() => setQuiz(null)}
                    className="text-primary hover:opacity-90 font-semibold"
                  >
                    Generate New Quiz
                  </button>
                </div>
                {quiz.questions?.map((q, idx) => (
                  <div
                    key={idx}
                    className="bg-card/40 border border-border rounded-xl p-4"
                  >
                    <p className="font-bold mb-3 text-foreground">
                      {idx + 1}. {q.question}
                    </p>
                    <div className="space-y-2 mb-3">
                      {Object.entries(q.options || {}).map(([key, value]) => (
                        <div
                          key={key}
                          className={`p-2 rounded-lg ${
                            key === q.correct_answer
                              ? "bg-emerald-500/15 border-2 border-emerald-500/50"
                              : "bg-card/30 border-2 border-border"
                          }`}
                        >
                          <span className="font-bold mr-2">{key}.</span>
                          {value}
                          {key === q.correct_answer && (
                            <span className="ml-2 text-green-400">✓ Correct</span>
                          )}
                        </div>
                      ))}
                    </div>
                    {q.explanation && (
                      <p className="text-sm text-muted-foreground italic">
                        {q.explanation}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Revision Pack Modal */}
      {showRevisionModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-3 sm:p-6">
          <div className="study-panel rounded-2xl p-4 sm:p-6 max-w-5xl w-full max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold study-accent">Revision Pack</h2>
              <button
                onClick={() => {
                  setShowRevisionModal(false);
                  setRevisionPack(null);
                  setSelectedRevisionPages([]);
                }}
                className="text-muted-foreground hover:text-foreground"
              >
                ✕
              </button>
            </div>

            {!revisionPack ? (
              <div>
                <p className="text-muted-foreground mb-4">
                  Select pages to generate quick notes + flashcards:
                </p>
                <div className="grid grid-cols-5 sm:grid-cols-10 gap-2 mb-6">
                  {Array.from({ length: totalPages }, (_, i) => i + 1).map(
                    (pageNum) => (
                      <button
                        key={pageNum}
                        onClick={() => toggleRevisionPageSelection(pageNum)}
                        className={`p-2 rounded-lg border-2 transition-colors ${
                          selectedRevisionPages.includes(pageNum)
                            ? "bg-primary border-primary text-primary-foreground"
                            : "bg-card/40 border-border text-muted-foreground hover:border-primary"
                        }`}
                      >
                        {pageNum}
                      </button>
                    )
                  )}
                </div>
                <button
                  onClick={handleGenerateRevisionPack}
                  disabled={selectedRevisionPages.length === 0 || revisionLoading}
                  className="w-full bg-primary hover:opacity-95 disabled:opacity-50 text-primary-foreground font-bold py-3 rounded-xl transition-all flex items-center justify-center gap-2"
                >
                  {revisionLoading ? (
                    <>
                      <Loader className="animate-spin" size={20} />
                      Generating Revision Pack...
                    </>
                  ) : (
                    <>
                      <FileText size={20} />
                      Generate ({selectedRevisionPages.length} pages)
                    </>
                  )}
                </button>
                <p className="text-xs text-muted-foreground mt-3">
                  Tip: First run "Explain" on those pages so the text is parsed.
                </p>
              </div>
            ) : (
              <div className="space-y-8">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <h3 className="text-xl font-bold text-foreground">
                      {revisionPack.title || "Revision Pack"}
                    </h3>
                    <p className="text-xs text-muted-foreground mt-1">
                      Pages: {(revisionPack.pageNumbers || []).join(", ")} | Language:{" "}
                      {revisionPack.language}
                    </p>
                  </div>
                  <button
                    onClick={() => setRevisionPack(null)}
                    className="text-primary hover:opacity-90 shrink-0 font-semibold"
                  >
                    Generate New
                  </button>
                </div>

                {revisionPack.pack?.error ? (
                  <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4">
                    <p className="font-bold text-red-300 mb-2">
                      Failed to parse AI output
                    </p>
                    <pre className="whitespace-pre-wrap text-xs text-muted-foreground">
                      {revisionPack.pack.raw_response}
                    </pre>
                  </div>
                ) : (
                  <>
                    {revisionPack.pack?.notes_markdown && (
                      <div className="bg-card/40 border border-border rounded-2xl p-5">
                        <p className="font-bold text-foreground mb-3">Notes</p>
                        <div className="text-sm leading-relaxed markdown-content">
                          <ReactMarkdown>{revisionPack.pack.notes_markdown}</ReactMarkdown>
                        </div>
                      </div>
                    )}

                    {!!revisionPack.pack?.key_terms?.length && (
                      <div className="bg-card/40 border border-border rounded-2xl p-5">
                        <p className="font-bold text-foreground mb-3">Key Terms</p>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                          {revisionPack.pack.key_terms.map((t, idx) => (
                            <div
                              key={idx}
                              className="bg-card/30 border border-border rounded-xl p-4"
                            >
                              <p className="font-bold text-foreground">{t.term}</p>
                              <p className="text-sm text-muted-foreground mt-1">{t.meaning}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {!!revisionPack.pack?.flashcards?.length && (
                      <div className="bg-card/40 border border-border rounded-2xl p-5">
                        <p className="font-bold text-foreground mb-3">Flashcards</p>
                        <div className="space-y-3">
                          {revisionPack.pack.flashcards.map((c, idx) => (
                            <div
                              key={idx}
                              className="bg-card/30 border border-border rounded-xl p-4"
                            >
                              <p className="text-sm text-muted-foreground">
                                <span className="font-bold text-foreground">Q:</span>{" "}
                                {c.front}
                              </p>
                              <p className="text-sm text-muted-foreground mt-2">
                                <span className="font-bold text-primary">A:</span>{" "}
                                {c.back}
                              </p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {!!revisionPack.pack?.exam_questions?.length && (
                      <div className="bg-card/40 border border-border rounded-2xl p-5">
                        <p className="font-bold text-foreground mb-3">Exam Questions</p>
                        <div className="space-y-4">
                          {revisionPack.pack.exam_questions.map((q, idx) => (
                            <div
                              key={idx}
                              className="bg-card/30 border border-border rounded-xl p-4"
                            >
                              <p className="font-bold text-foreground mb-2">
                                {idx + 1}. {q.question}
                              </p>
                              {q.answer_outline && (
                                <div className="text-sm text-muted-foreground markdown-content">
                                  <ReactMarkdown>{q.answer_outline}</ReactMarkdown>
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {!!revisionPack.pack?.common_mistakes?.length && (
                      <div className="bg-card/40 border border-border rounded-2xl p-5">
                        <p className="font-bold text-foreground mb-3">Common Mistakes</p>
                        <ul className="list-disc pl-5 text-sm text-muted-foreground space-y-1">
                          {revisionPack.pack.common_mistakes.map((m, idx) => (
                            <li key={idx}>{m}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      <main className="flex-1 flex flex-col lg:flex-row overflow-visible lg:overflow-hidden gap-3 lg:gap-6 px-3 sm:px-6 pb-4 lg:pb-0">
        {/* LEFT PANEL - PDF Preview (full height, no scroll) */}
        <div
          className={`flex flex-col study-panel rounded-2xl transition-all duration-300 shrink-0 ${
            pdfCollapsed ? "hidden lg:flex lg:w-12" : "w-full lg:w-[35%] h-[42vh] lg:h-auto"
          }`}
        >
          {/* PDF Header */}
          {!pdfCollapsed && (
            <div className="border-b p-3 sm:p-4 study-divider shrink-0">
              <h3 className="font-bold text-sm mb-2">PDF Preview</h3>
              <p className="text-xs study-text-muted">
                Page content will appear here after explanation
              </p>
            </div>
          )}

          {/* PDF Content Area */}
          {!pdfCollapsed && (
            <div className="flex-1 overflow-hidden p-2 sm:p-4 flex items-center justify-center min-h-0">
              {pdfPageImage ? (
                <div className="study-card rounded-xl p-4 flex items-center justify-center w-full h-full">
                  <img
                    src={pdfPageImage}
                    alt={`Page ${currentPage}`}
                    className="max-w-full max-h-full object-contain rounded-lg shadow-lg"
                  />
                </div>
              ) : (
                <div className="study-card rounded-xl p-4 text-xs leading-relaxed flex items-center justify-center w-full h-full">
                  <p className="study-text-muted">Loading PDF page...</p>
                </div>
              )}
            </div>
          )}

          {/* Page Navigation Controls */}
          <div className="border-t p-2 sm:p-3 flex items-center justify-between gap-2 study-divider shrink-0">
            <button
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              disabled={currentPage === 1}
              className="p-2 disabled:opacity-50 rounded-lg transition-colors border border-border bg-card/30 hover:bg-card/60"
            >
              <ChevronLeft size={18} />
            </button>

            {!pdfCollapsed && (
              <span className="text-xs font-bold text-center flex-1">
                Page {currentPage}
              </span>
            )}

            <button
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              disabled={currentPage >= totalPages}
              className="p-2 disabled:opacity-50 rounded-lg transition-colors border border-border bg-card/30 hover:bg-card/60"
            >
              <ChevronRight size={18} />
            </button>

            <button
              onClick={() => setPdfCollapsed(!pdfCollapsed)}
              className="p-2 rounded-lg transition-colors border border-border bg-card/30 hover:bg-card/60"
            >
              {pdfCollapsed ? <Maximize2 size={16} /> : <Minimize2 size={16} />}
            </button>
          </div>
        </div>

        {/* RIGHT PANEL - ChatGPT style */}
        <div className="flex-1 flex flex-col overflow-hidden rounded-2xl study-panel min-h-[68vh] lg:min-h-0">
          {/* Scrollable content (explanation + messages) */}
          <div className="flex-1 overflow-y-auto p-3 sm:p-6 space-y-4">
            {/* Explanation Card */}
            <div className="study-explanation-panel rounded-2xl p-4 sm:p-6 mb-6">
              <div className="flex items-center justify-between gap-3 mb-4">
                <h2 className="text-lg font-bold study-accent">Page Explanation</h2>
                {!hasExplanation && !explanationLoading && (
                  <button
                    onClick={handleExplain}
                    disabled={explanationLoading}
                    className="flex items-center gap-2 px-4 py-2 rounded-xl transition-all font-bold study-btn"
                  >
                    <Play size={16} /> Explain
                  </button>
                )}
                {explanationLoading && (
                  <Loader className="animate-spin study-accent" size={18} />
                )}
              </div>

              {explanationLoading ? (
                <div className="flex items-center justify-center py-8">
                  <div className="text-center">
                    <Loader className="animate-spin mx-auto mb-2 text-primary" />
                    <p className="text-sm text-muted-foreground">
                      AI is explaining this page...
                    </p>
                  </div>
                </div>
              ) : hasExplanation ? (
                <div className="text-sm leading-relaxed markdown-content">
                  <ReactMarkdown>{explanation || "No explanation available"}</ReactMarkdown>
                </div>
              ) : (
                <div className="flex items-center justify-center py-8 text-center">
                  <div>
                    <HelpCircle className="mx-auto mb-3 text-muted-foreground/60" size={48} />
                    <p className="text-muted-foreground font-medium mb-2">
                      Click "Explain" to get AI explanation
                    </p>
                    <p className="text-xs text-muted-foreground/80">
                      Get detailed explanation with real-world examples
                    </p>
                  </div>
                </div>
              )}
            </div>

            {/* Chat Messages */}
            {messages.length === 0 && !isLoading ? (
                <div className="flex items-center justify-center h-full text-center">
                <div>
                  <p className="font-medium mb-2 text-muted-foreground">
                    {chatEnabled
                      ? "Ask your doubts about this page"
                      : "Get explanation first, then ask doubts"}
                  </p>
                  <p className="text-xs text-muted-foreground/80">
                    The AI will answer based on the page content
                  </p>
                </div>
              </div>
            ) : (
              messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex ${
                    msg.role === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  <div
                    className={`max-w-[86vw] sm:max-w-xs lg:max-w-2xl px-4 sm:px-5 py-3 sm:py-4 rounded-2xl text-[14.5px] ${
                      msg.role === "user"
                        ? "study-user-message rounded-br-none"
                        : "study-ai-message rounded-bl-none markdown-content"
                    }`}
                  >
                    {msg.role === "user" ? (
                      msg.text
                    ) : (
                      <ReactMarkdown>{msg.text}</ReactMarkdown>
                    )}
                  </div>
                </div>
              ))
            )}
            {isLoading && (
              <div className="flex justify-start">
                <div className="px-4 py-3 rounded-2xl rounded-bl-none bg-card/50 border border-border">
                  <div className="flex gap-2">
                    <div className="w-2 h-2 rounded-full animate-bounce bg-muted-foreground/70" />
                    <div className="w-2 h-2 rounded-full animate-bounce delay-100 bg-muted-foreground/70" />
                    <div className="w-2 h-2 rounded-full animate-bounce delay-200 bg-muted-foreground/70" />
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Input Area */}
          <div className="shrink-0 border-t p-3 sm:p-4 study-divider bg-card/30">
            {chatSaveError && (
              <div className="mb-3 rounded-2xl border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive font-semibold">
                {chatSaveError}
              </div>
            )}
            <form onSubmit={handleSendDoubt} className="flex gap-2 sm:gap-3">
              <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                placeholder={
                  chatEnabled
                    ? "Ask your doubt..."
                    : "Get explanation first..."
                }
                disabled={isLoading || !chatEnabled}
                className="min-w-0 flex-1 rounded-2xl px-4 py-3 text-sm outline-none transition disabled:opacity-50 bg-[hsl(var(--chat-input))] text-[hsl(var(--chat-input-foreground))] border border-[hsl(var(--chat-input-border))] placeholder:text-muted-foreground/70 focus:ring-2 focus:ring-ring/30"
              />
              <button
                type="submit"
                disabled={!inputValue.trim() || isLoading || !chatEnabled}
                className="px-4 py-3 rounded-2xl transition flex items-center justify-center font-bold disabled:opacity-50 hover:opacity-95 bg-primary text-primary-foreground"
              >
                <Send size={18} />
              </button>
            </form>
          </div>
        </div>
      </main>
    </div>
  );
}
