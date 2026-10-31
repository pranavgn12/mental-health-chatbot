import { useState, useEffect } from "react";
import { useNavigate, Link } from "react-router-dom";

import { login } from "../services/auth";

import "../styles/login.css";

export default function Login() {

    const navigate = useNavigate();

    const [email, setEmail] = useState("");

    const [password, setPassword] = useState("");

    const [error, setError] = useState("");

    useEffect(() => {

        const token = localStorage.getItem("token");

        if (token) {

            navigate("/chat");

        }

    }, [navigate]);

    async function handleLogin(e) {

        e.preventDefault();

        setError("");

        try {

            const data = await login(
                email,
                password
            );

            localStorage.setItem(
                "token",
                data.access_token
            );

            localStorage.setItem(
                "isLoggedIn",
                "true"
            );

            navigate("/chat");

        }

        catch (err) {

            setError("Invalid email or password");

            console.error(err);

        }

    }

    return (

        <div className="login-page">

            <form
                className="login-card"
                onSubmit={handleLogin}
            >

                <h1>Mental Health AI</h1>

                <p className="subtitle">

                    Sign in to continue

                </p>

                <input

                    type="email"

                    placeholder="Email"

                    value={email}

                    onChange={(e) =>
                        setEmail(e.target.value)
                    }

                    required

                />

                <input

                    type="password"

                    placeholder="Password"

                    value={password}

                    onChange={(e) =>
                        setPassword(e.target.value)
                    }

                    required

                />

                {

                    error &&

                    <p className="error">

                        {error}

                    </p>

                }

                <button
                    type="submit"
                >

                    Login

                </button>

                <p className="register-link">

                    Don't have an account?{" "}

                    <Link to="/register">

                        Register

                    </Link>

                </p>

            </form>

        </div>

    );

}