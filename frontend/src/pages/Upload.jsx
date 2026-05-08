import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { BrainCircuit, Upload as UploadIcon, ArrowLeft, Loader } from "lucide-react";
import ThemeToggle from "../components/ThemeToggle";
import { API_BASE_URL } from "../config/api";

export default function Upload() {
  const [file, setFile] = useState(null);
  const [language, setLanguage] = useState("hinglish");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.type === "application/pdf") {
      setFile(selectedFile);
      setError("");
    } else {
      setError("Please select a valid PDF file");
      setFile(null);
    }
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Please select a PDF file");
      return;
    }

    setIsLoading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", file);
      const token = localStorage.getItem("userToken");

      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: "POST",
        body: formData,
        headers: token ? { Authorization: `Bearer ${token}` } : undefined,
      });

      const data = await response.json();

      if (response.ok && data.pdf_id) {
        sessionStorage.setItem("explanationLanguage", language);
        navigate(`/study/${data.pdf_id}`);
      } else {
        setError(data.error || "Upload failed. Please try again.");
      }
    } catch (err) {
      console.error("Upload error:", err);
      setError("Upload failed. Please check your connection.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground p-3 sm:p-6 relative overflow-hidden">
      <div className="absolute bottom-[-20%] left-[-10%] w-[55%] h-[55%] bg-primary/10 blur-[120px] rounded-full animate-pulse" />
      <div className="absolute top-[-10%] right-[-5%] w-[45%] h-[45%] bg-sky-600/10 blur-[110px] rounded-full" />

      <div className="max-w-2xl mx-auto relative z-10">
        <div className="flex items-center justify-between gap-3 mb-8 sm:mb-10">
          <button
            onClick={() => navigate("/")}
            className="flex items-center gap-2 text-sm sm:text-base text-muted-foreground hover:text-primary transition-colors"
          >
            <ArrowLeft size={20} /> Back to Home
          </button>
          <ThemeToggle />
        </div>

        <div className="text-center mb-8 sm:mb-10">
          <div className="inline-flex items-center gap-3 group mb-6">
            <div className="bg-gradient-to-br from-primary to-sky-600 p-2 rounded-xl">
              <BrainCircuit className="text-primary-foreground w-6 h-6" />
            </div>
            <span className="text-2xl font-black">
              StudyMate<span className="text-primary">.ai</span>
            </span>
          </div>
          <h1 className="text-3xl sm:text-4xl font-black mb-2">Upload Your PDF</h1>
          <p className="text-sm sm:text-base text-muted-foreground">
            Share your book, notes, or research paper. AI will break it down for you.
          </p>
        </div>

        <div className="sm-card p-4 sm:p-8 lg:p-12">
          <form onSubmit={handleUpload} className="space-y-6 sm:space-y-8">
            <div
              className="border-2 border-dashed border-border rounded-2xl p-6 sm:p-12 text-center hover:border-primary transition-colors cursor-pointer bg-card/40"
              onDragOver={(e) => {
                e.preventDefault();
                e.currentTarget.classList.add("border-primary");
              }}
              onDragLeave={(e) => {
                e.currentTarget.classList.remove("border-primary");
              }}
              onDrop={(e) => {
                e.preventDefault();
                e.currentTarget.classList.remove("border-primary");
                const droppedFile = e.dataTransfer.files[0];
                if (droppedFile && droppedFile.type === "application/pdf") {
                  setFile(droppedFile);
                  setError("");
                }
              }}
            >
              <input
                type="file"
                accept=".pdf"
                onChange={handleFileChange}
                className="hidden"
                id="pdfInput"
                disabled={isLoading}
              />
              <label htmlFor="pdfInput" className="cursor-pointer">
                <UploadIcon className="w-12 h-12 sm:w-16 sm:h-16 mx-auto mb-4 text-primary" />
                <p className="text-base sm:text-lg font-bold mb-2 break-words">
                  {file ? file.name : "Drop your PDF here"}
                </p>
                <p className="text-muted-foreground text-sm">
                  or click to browse from your computer
                </p>
              </label>
            </div>

            {file && (
              <div className="bg-primary/10 border border-primary/20 rounded-2xl p-4 flex items-center justify-between gap-3">
                <div className="flex items-center gap-3 min-w-0">
                  <div className="w-12 h-12 bg-primary/15 rounded-lg flex items-center justify-center">
                    <UploadIcon className="text-primary" size={24} />
                  </div>
                  <div className="min-w-0">
                    <p className="font-bold truncate">{file.name}</p>
                    <p className="text-sm text-muted-foreground">
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => setFile(null)}
                  className="text-muted-foreground hover:text-destructive transition-colors"
                  aria-label="Remove selected file"
                >
                  ×
                </button>
              </div>
            )}

            <div className="space-y-3">
              <label className="text-sm font-bold text-muted-foreground block">
                Explanation Language
              </label>
              <select
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
                disabled={isLoading}
                className="sm-input bg-card/40"
              >
                <option value="hinglish">Hinglish (Hindi + English)</option>
                <option value="hindi">Hindi</option>
                <option value="english">English</option>
              </select>
            </div>

            {error && (
              <div className="bg-destructive/10 border border-destructive/20 rounded-xl p-4 text-destructive text-sm">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={!file || isLoading}
              className="sm-btn w-full py-4 shadow-sm shadow-primary/10"
            >
              {isLoading ? (
                <>
                  <Loader className="animate-spin" size={20} />
                  Processing Your PDF...
                </>
              ) : (
                <>
                  <UploadIcon size={20} /> Upload & Start Learning
                </>
              )}
            </button>
          </form>

          <p className="text-center text-muted-foreground text-xs mt-8 font-medium">
            Your PDFs are processed securely. No data is stored permanently.
          </p>
        </div>
      </div>
    </div>
  );
}
