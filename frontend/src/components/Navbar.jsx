import { useEffect, useState } from "react";
import { PanelLeft } from "lucide-react";

const API_BASE = (
    import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");

export default function Navbar({ toggleSidebar }) {
    const [model, setModel] = useState("Loading model...");

    useEffect(() => {
        let active = true;

        async function fetchModel() {
            try {
                const response = await fetch(`${API_BASE}/model`, {
                    cache: "no-store",
                });

                if (!response.ok) {
                    throw new Error("Could not fetch model");
                }

                const data = await response.json();

                if (active) {
                    setModel(data.model || "Unknown model");
                }
            } catch {
                if (active) {
                    setModel("Model unavailable");
                }
            }
        }

        fetchModel();

        return () => {
            active = false;
        };
    }, []);

    const modelLabel = model.split("/").pop();

    return (
        <div className="navbar">
            <div className="nav-left">
                <button
                    className="icon-btn"
                    onClick={toggleSidebar}
                >
                    <PanelLeft size={20} />
                </button>

                <h2>AI Companion</h2>
            </div>

            <div className="nav-right">
                <span
                    className="model-name"
                    title={model}
                >
                    {modelLabel}
                </span>

                <button
                    className="icon-btn"
                    onClick={() => {
                        localStorage.removeItem("token");
                        localStorage.removeItem("isLoggedIn");
                        window.location.href = "/";
                    }}
                >
                    Logout
                </button>
            </div>
        </div>
    );
}
