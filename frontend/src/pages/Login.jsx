import { useState, useEffect } from "react";
import { Link, useNavigate, useLocation } from "react-router-dom";
import { BrainCircuit, Mail, Lock, Eye, EyeOff, AlertCircle } from "lucide-react";
import ThemeToggle from "../components/ThemeToggle";
import { loginUser } from "../services/api";

export default function Login() {
  const [formData, setFormData] = useState({ email: "", password: "" });
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    if (location.state?.email) {
      setFormData((prev) => ({ ...prev, email: location.state.email }));
    }
  }, [location.state?.email]);

  const handleLogin = async (e) => {
    e.preventDefault();
    setError("");
    setIsLoading(true);

    try {
      const response = await loginUser(formData);
      let data;
try {
  data = await response.json();
} catch {
  data = {};
}

      if (response.ok) {
        localStorage.setItem("userToken", data.token);
        localStorage.setItem("userName", data.user.name);
        navigate("/");
      } else {
        setError(data.error || "Login failed");
      }
    } catch (err) {
      console.error("Login error:", err);
      setError("Login failed. Try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground flex items-center justify-center p-6 relative overflow-hidden">
      <div className="absolute bottom-[-20%] right-[-10%] w-[50%] h-[50%] bg-primary/10 blur-[120px] rounded-full animate-pulse" />

      <div className="w-full max-w-[450px] relative z-10">
        <div className="text-center mb-8 relative">
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
          <h2 className="text-2xl font-bold mt-6">Welcome Back</h2>
          <p className="text-muted-foreground text-sm mt-1">
            Sign in to continue your learning journey.
          </p>
        </div>

        <div className="sm-card p-10">
          {error && (
            <div className="mb-6 flex items-center gap-3 bg-destructive/10 border border-destructive/20 p-4 rounded-2xl">
              <AlertCircle size={20} className="text-destructive" />
              <p className="text-destructive text-sm font-medium">{error}</p>
            </div>
          )}

          <form onSubmit={handleLogin} className="space-y-5">
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
                "Sign In"
              )}
            </button>
          </form>

          <p className="text-center mt-8 text-muted-foreground text-sm font-medium">
            Don't have an account?{" "}
            <Link to="/register" className="text-primary font-bold hover:underline">
              Create one
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}

