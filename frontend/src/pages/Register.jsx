import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { BrainCircuit, User, Mail, Lock, Eye, EyeOff, AlertCircle } from "lucide-react";
import ThemeToggle from "../components/ThemeToggle";
import { registerUser } from "../services/api";

export default function Register() {
  const [formData, setFormData] = useState({ name: "", email: "", password: "" });
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const handleRegister = async (e) => {
    e.preventDefault();
    setError("");
    setIsLoading(true);

    try {
      const response = await registerUser(formData);
      const data = await response.json();

      if (response.ok) {
        navigate("/login", { state: { email: formData.email } });
      } else {
        setError(data.error || "Registration failed");
      }
    } catch (err) {
      console.error("Register error:", err);
      setError("Registration failed. Try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground flex items-center justify-center p-3 sm:p-6 relative overflow-hidden">
      <div className="absolute bottom-[-20%] right-[-10%] w-[50%] h-[50%] bg-primary/10 blur-[120px] rounded-full animate-pulse" />

      <div className="w-full max-w-[450px] relative z-10">
        <div className="text-center mb-6 sm:mb-8 relative">
          <div className="absolute right-0 top-0">
            <ThemeToggle />
          </div>

          <Link to="/" className="inline-flex items-center gap-3 group">
            <div className="bg-gradient-to-br from-primary to-sky-600 p-2 rounded-xl">
              <BrainCircuit className="text-primary-foreground w-7 h-7" />
            </div>
            <span className="text-2xl font-black">
              StudyMate<span className="text-primary">.ai</span>
            </span>
          </Link>
          <h2 className="text-2xl font-bold mt-6">Create Account</h2>
          <p className="text-muted-foreground text-sm mt-1">
            Start your AI-powered study session today.
          </p>
        </div>

        <div className="sm-card p-5 sm:p-10">
          {error && (
            <div className="mb-6 flex items-center gap-3 bg-destructive/10 border border-destructive/20 p-4 rounded-2xl">
              <AlertCircle size={20} className="text-destructive" />
              <p className="text-destructive text-sm font-medium">{error}</p>
            </div>
          )}

          <form onSubmit={handleRegister} className="space-y-5">
            <div className="space-y-2">
              <label className="text-xs uppercase tracking-widest font-bold text-muted-foreground ml-1">
                Full Name
              </label>
              <div className="relative">
                <User className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                <input
                  type="text"
                  required
                  placeholder="Rajkumar Singh"
                  className="sm-input py-4 pl-12 pr-4"
                  value={formData.name}
                  onChange={(e) =>
                    setFormData((prev) => ({ ...prev, name: e.target.value }))
                  }
                  disabled={isLoading}
                />
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-xs uppercase tracking-widest font-bold text-muted-foreground ml-1">
                Email Address
              </label>
              <div className="relative">
                <Mail className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                <input
                  type="email"
                  required
                  placeholder="raj@example.com"
                  className="sm-input py-4 pl-12 pr-4"
                  value={formData.email}
                  onChange={(e) =>
                    setFormData((prev) => ({ ...prev, email: e.target.value }))
                  }
                  disabled={isLoading}
                />
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-xs uppercase tracking-widest font-bold text-muted-foreground ml-1">
                Password
              </label>
              <div className="relative">
                <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                <input
                  type={showPassword ? "text" : "password"}
                  required
                  placeholder="••••••••"
                  className="sm-input py-4 pl-12 pr-12"
                  value={formData.password}
                  onChange={(e) =>
                    setFormData((prev) => ({
                      ...prev,
                      password: e.target.value,
                    }))
                  }
                  disabled={isLoading}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((s) => !s)}
                  className="absolute right-4 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-primary"
                  disabled={isLoading}
                  aria-label={showPassword ? "Hide password" : "Show password"}
                >
                  {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
                </button>
              </div>
            </div>

            <button
              disabled={isLoading}
              type="submit"
              className="sm-btn w-full py-4 shadow-sm shadow-primary/10"
            >
              {isLoading ? (
                <div className="w-6 h-6 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              ) : (
                "Create Account"
              )}
            </button>
          </form>

          <p className="text-center mt-8 text-muted-foreground text-sm font-medium">
            Already have an account?{" "}
            <Link to="/login" className="text-primary font-bold hover:underline">
              Sign In
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}

