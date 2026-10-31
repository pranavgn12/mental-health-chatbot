import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import "../styles/Register.css";

const API_BASE = (
    import.meta.env.VITE_API_URL || "http://localhost:8000"
).replace(/\/+$/, "");

// Change this if your FastAPI route uses a different path.
const REGISTER_URL = `${API_BASE}/auth/register`;

export default function Register() {
    const navigate = useNavigate();

    const [form, setForm] = useState({
        username: "",
        email: "",
        password: "",
        confirmPassword: "",
    });

    const [showPassword, setShowPassword] = useState(false);
    const [error, setError] = useState("");
    const [loading, setLoading] = useState(false);

    function handleChange(event) {
        const { name, value } = event.target;

        setForm((prev) => ({
            ...prev,
            [name]: value,
        }));

        setError("");
    }

    async function handleSubmit(event) {
        event.preventDefault();
        setError("");

        const username = form.username.trim();
        const email = form.email.trim();

        if (username.length < 3 || username.length > 50) {
            setError("Username must be between 3 and 50 characters.");
            return;
        }

        if (form.password.length < 8) {
            setError("Password must contain at least 8 characters.");
            return;
        }

        if (form.password !== form.confirmPassword) {
            setError("Passwords do not match.");
            return;
        }

        setLoading(true);

        try {
            const response = await fetch(REGISTER_URL, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    username,
                    email,
                    password: form.password,
                }),
            });

            const data = await response.json().catch(() => ({}));

            if (!response.ok) {
                const detail = data.detail;

                const message = Array.isArray(detail)
                    ? detail.map((item) => item.msg).join(", ")
                    : detail || data.message || "Registration failed.";

                throw new Error(message);
            }

            navigate("/login", {
                state: {
                    message: "Account created successfully. Please log in.",
                },
            });
        } catch (err) {
            setError(
                err instanceof TypeError
                    ? "Unable to connect to the server. Check that the backend is running."
                    : err.message || "Something went wrong."
            );
        } finally {
            setLoading(false);
        }
    }

    return (
        <main className="auth-page">
            <section className="auth-card">
                <div className="auth-heading">
                    <div className="auth-logo" aria-hidden="true">
                        +
                    </div>

                    <h1>Create your account</h1>
                    <p>Get started by creating your account.</p>
                </div>

                <form className="auth-form" onSubmit={handleSubmit}>
                    <label className="auth-field">
                        <span>Username</span>
                        <input
                            type="text"
                            name="username"
                            placeholder="Choose a username"
                            value={form.username}
                            onChange={handleChange}
                            minLength={3}
                            maxLength={50}
                            autoComplete="username"
                            required
                        />
                    </label>

                    <label className="auth-field">
                        <span>Email address</span>
                        <input
                            type="email"
                            name="email"
                            placeholder="you@example.com"
                            value={form.email}
                            onChange={handleChange}
                            maxLength={100}
                            autoComplete="email"
                            required
                        />
                    </label>

                    <label className="auth-field">
                        <span>Password</span>
                        <input
                            type={showPassword ? "text" : "password"}
                            name="password"
                            placeholder="At least 8 characters"
                            value={form.password}
                            onChange={handleChange}
                            minLength={8}
                            autoComplete="new-password"
                            required
                        />
                    </label>

                    <label className="auth-field">
                        <span>Confirm password</span>
                        <input
                            type={showPassword ? "text" : "password"}
                            name="confirmPassword"
                            placeholder="Enter your password again"
                            value={form.confirmPassword}
                            onChange={handleChange}
                            minLength={8}
                            autoComplete="new-password"
                            required
                        />
                    </label>

                    <label className="auth-show-password">
                        <input
                            type="checkbox"
                            checked={showPassword}
                            onChange={(event) =>
                                setShowPassword(event.target.checked)
                            }
                        />
                        <span>Show passwords</span>
                    </label>

                    {error && (
                        <div className="auth-error" role="alert">
                            {error}
                        </div>
                    )}

                    <button
                        className="auth-submit"
                        type="submit"
                        disabled={loading}
                    >
                        {loading ? "Creating account..." : "Create account"}
                    </button>
                </form>

                <p className="auth-footer">
                    Already have an account?{" "}
                    <Link to="/login">Log in</Link>
                </p>
            </section>
        </main>
    );
}
