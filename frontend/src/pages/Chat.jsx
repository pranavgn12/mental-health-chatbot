import { useCallback, useEffect, useMemo, useState } from "react";

import Sidebar from "../components/Sidebar";
import Navbar from "../components/Navbar";
import ChatWindow from "../components/ChatWindow";
import MessageInput from "../components/MessageInput";

import {
    X,
    BookOpen,
    HeartPulse,
    Brain,
    UserRound,
    LoaderCircle,
    RefreshCw,
    Clock3,
} from "lucide-react";

import "../styles/chat.css";

// These endpoints must be provided by the FastAPI backend.
// Set VITE_API_BASE_URL in frontend/.env if your API is not at localhost:8000.
const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

const MEMORY_TABS = [
    { id: "profile", label: "Profile", icon: UserRound },
    { id: "episodes", label: "Episodes", icon: BookOpen },
    { id: "moods", label: "Moods", icon: HeartPulse },
    { id: "reflections", label: "Reflections", icon: Brain },
];

const RESOURCE_CONFIG = {
    episodes: { path: "/memory/episodes", label: "episode" },
    moods: { path: "/memory/moods", label: "mood record" },
    reflections: { path: "/memory/reflections", label: "reflection" },
};

function formatDate(value) {
    if (!value) return "Date unavailable";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "Date unavailable";
    return new Intl.DateTimeFormat(undefined, {
        dateStyle: "medium",
        timeStyle: "short",
    }).format(date);
}

function unwrapList(payload) {
    if (Array.isArray(payload)) return payload;
    if (Array.isArray(payload?.items)) return payload.items;
    if (Array.isArray(payload?.data)) return payload.data;
    return [];
}

async function requestJson(path, options = {}) {
    const token = localStorage.getItem("access_token") || localStorage.getItem("token") || localStorage.getItem("accessToken");
    const response = await fetch(`${API_BASE}${path}`, {
        ...options,
        credentials: "include",
        headers: {
            Accept: "application/json",
            ...(token ? { Authorization: `Bearer ${token}` } : {}),
            ...(options.body ? { "Content-Type": "application/json" } : {}),
            ...options.headers,
        },
    });

    if (!response.ok) {
        let message = `Request failed (${response.status})`;
        try {
            const body = await response.json();
            if (body?.detail) message = body.detail;
        } catch {
            // Keep the HTTP status message when the body isn't JSON.
        }
        throw new Error(message);
    }

    if (response.status === 204) return null;
    return response.json();
}

const PROFILE_THEME_RULES = [
    {
        id: "anxious",
        label: "Anxious",
        category: "Emotional pattern",
        tone: "amber",
        pattern: /\b(anxious|anxiety|worried|worry|worries|stress|stressed|nervous|overwhelmed|uncertain|uncertainty)\b/i,
    },
    {
        id: "happy",
        label: "Happy",
        category: "Emotional pattern",
        tone: "green",
        pattern: /\b(happy|happiness|joy|joyful|excited|excitement|proud|enjoy|enjoyed|gratitude|grateful|positive)\b/i,
    },
    {
        id: "sad",
        label: "Sad",
        category: "Emotional pattern",
        tone: "blue",
        pattern: /\b(sad|sadness|down|upset|disappointed|grief)\b/i,
    },
    {
        id: "frustrated",
        label: "Frustrated",
        category: "Emotional pattern",
        tone: "amber",
        pattern: /\b(angry|anger|frustrated|frustration|irritated)\b/i,
    },
    {
        id: "hopeful",
        label: "Hopeful",
        category: "Emotional pattern",
        tone: "green",
        pattern: /\b(hopeful|hope|optimistic|motivated|motivation|determined)\b/i,
    },
    {
        id: "lonely",
        label: "Lonely",
        category: "Emotional pattern",
        tone: "blue",
        pattern: /\b(lonely|loneliness|alone|isolated|isolation|disconnected)\b/i,
    },
    {
        id: "relationships",
        label: "Relationships",
        category: "Recurring theme",
        tone: "blue",
        pattern: /\b(relationship|relationships|friend|friends|family|connection|closeness|understood|support|belonging)\b/i,
    },
    {
        id: "growth",
        label: "Growth",
        category: "Recurring theme",
        tone: "purple",
        pattern: /\b(growth|learn|learning|improve|improving|progress|goal|goals|ambition|discipline|skill|skills)\b/i,
    },
    {
        id: "studies",
        label: "Studies",
        category: "Recurring theme",
        tone: "blue",
        pattern: /\b(study|studies|student|exam|college|academic|education|course|university)\b/i,
    },
    {
        id: "work",
        label: "Work & career",
        category: "Recurring theme",
        tone: "purple",
        pattern: /\b(work|job|career|internship|placement|profession|professional)\b/i,
    },
    {
        id: "wellbeing",
        label: "Well-being",
        category: "Recurring theme",
        tone: "green",
        pattern: /\b(sleep|exercise|health|routine|rest|self-care|wellbeing|well-being)\b/i,
    },
];

function shortenProfileText(value, maxLength = 116) {
    const text = String(value || "").replace(/\s+/g, " ").trim();
    if (text.length <= maxLength) return text;
    return `${text.slice(0, maxLength - 1).trimEnd()}…`;
}

function summarizeThemeContext(statement, pattern, maxLength = 116) {
    const text = String(statement || "").replace(/\s+/g, " ").trim();
    if (text.length <= maxLength) return text;

    const matchIndex = text.search(pattern);
    if (matchIndex < 0) return shortenProfileText(text, maxLength);

    let start = Math.max(0, matchIndex - 42);
    let end = Math.min(text.length, start + maxLength - 3);
    if (end - start < maxLength - 3) start = Math.max(0, end - (maxLength - 3));

    const excerpt = `${start > 0 ? "…" : ""}${text.slice(start, end).trim()}${end < text.length ? "…" : ""}`;
    return excerpt;
}

function profileStatements(memoryText) {
    return String(memoryText || "")
        .replace(/\r/g, "")
        .split(/\n+/)
        .map((line) => line
            .replace(/^\s*#{1,6}\s*/, "")
            .replace(/^\s*(?:[-*•]|\d+[.)])\s+/, "")
            .replace(/\*\*/g, "")
            .trim())
        .filter(Boolean)
        .flatMap((line) => line.match(/[^.!?]+(?:[.!?]+|$)/g) || [line])
        .map((statement) => statement.trim())
        .filter((statement) => statement.length >= 4);
}

function profileToneFor(text, fallback = "slate") {
    const value = String(text || "");
    const rule = PROFILE_THEME_RULES.find((item) => item.pattern.test(value));
    return rule?.tone || fallback;
}

function titleFromInsight(statement, index) {
    const cleaned = String(statement || "")
        .replace(/^(?:the user|user|they|he|she|i|you)\s+/i, "")
        .replace(/^[,;:\s-]+/, "");
    const meaningfulWords = (cleaned.match(/[A-Za-z][A-Za-z'-]*/g) || [])
        .filter((word) => !/^(?:is|are|was|were|has|have|had|the|a|an|and|or|to|of|in|on|with|for|that|this|often|sometimes|tends)$/i.test(word));
    if (!meaningfulWords.length) return `Memory insight ${index + 1}`;
    return meaningfulWords.slice(0, 3)
        .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
        .join(" ");
}

function structuredProfileThemes(memoryText) {
    try {
        const parsed = JSON.parse(memoryText);
        let rows = Array.isArray(parsed)
            ? parsed
            : [parsed?.themes, parsed?.insights, parsed?.patterns].find(Array.isArray);

        if (!rows && parsed && typeof parsed === "object") {
            rows = Object.entries(parsed).flatMap(([key, value]) => {
                if (typeof value === "string" && value.trim()) {
                    return [{ title: key, summary: value, details: value }];
                }
                if (Array.isArray(value)) {
                    return value.map((entry) => typeof entry === "string"
                        ? { title: key, summary: entry, details: entry }
                        : { title: key, ...(entry || {}) });
                }
                return [];
            });
        }

        if (!Array.isArray(rows)) return [];
        return rows.map((item, index) => {
            const title = item && typeof item === "object"
                ? (item.label || item.keyword || item.title || item.name || item.category)
                : "";
            const details = item && typeof item === "object"
                ? (item.summary || item.description || item.insight || item.text || item.value || "")
                : item;
            if (!String(title || details || "").trim()) return null;
            const fullText = String(details || title).trim();
            return {
                id: `structured-${index}-${String(title || "insight").toLowerCase().replace(/\W+/g, "-")}`,
                title: String(title || titleFromInsight(fullText, index)).replace(/[_-]+/g, " "),
                category: item?.category || "Remembered insight",
                tone: profileToneFor(`${title || ""} ${fullText}`, ["amber", "green", "blue", "purple"][index % 4]),
                summary: shortenProfileText(fullText),
                details: fullText,
            };
        }).filter(Boolean);
    } catch {
        return [];
    }
}

function extractProfileThemes(memoryText) {
    if (typeof memoryText !== "string" || !memoryText.trim()) return [];

    const structured = structuredProfileThemes(memoryText.trim());
    if (structured.length) return structured.slice(0, 4);

    const statements = profileStatements(memoryText);
    const themes = [];
    const usedStatements = new Set();

    PROFILE_THEME_RULES.forEach((rule) => {
        const statementIndex = statements.findIndex((statement) => {
            if (!rule.pattern.test(statement)) return false;
            // Avoid turning an explicit denial into a positive remembered label.
            const matchIndex = statement.search(rule.pattern);
            const prefix = statement.slice(Math.max(0, matchIndex - 40), matchIndex);
            return !/\b(?:not|never|no longer|isn't|is not|aren't|are not|don't|doesn't|didn't|without)\b[^.!?]{0,35}$/i.test(prefix);
        });
        if (statementIndex < 0) return;

        const details = statements[statementIndex];
        usedStatements.add(statementIndex);
        themes.push({
            id: rule.id,
            title: rule.label,
            category: rule.category,
            tone: rule.tone,
            summary: summarizeThemeContext(details, rule.pattern),
            details,
            frequency: (String(memoryText).match(new RegExp(rule.pattern.source, "gi")) || []).length,
            position: String(memoryText).toLowerCase().indexOf(details.toLowerCase()),
        });
    });

    themes.sort((a, b) => b.frequency - a.frequency || a.position - b.position);

    // When the memory contains useful information but none of the known topic
    // words, show short excerpts instead of inventing emotional labels.
    if (themes.length < 4) {
        statements.forEach((statement, index) => {
            if (themes.length >= 4 || usedStatements.has(index)) return;
            themes.push({
                id: `insight-${index}`,
                title: titleFromInsight(statement, index),
                category: "Remembered insight",
                tone: profileToneFor(statement, ["amber", "green", "blue", "purple"][themes.length % 4]),
                summary: shortenProfileText(statement),
                details: statement,
                position: String(memoryText).toLowerCase().indexOf(statement.toLowerCase()),
            });
            usedStatements.add(index);
        });
    }

    return themes.slice(0, 4).map(({ position, frequency, ...theme }) => theme);
}

function MemoryItem({ item, tab, onDelete, deleting }) {
    let title = "";
    let description = "";
    let timestamp = item.created_at || item.updated_at;
    let id;

    if (tab === "episodes") {
        id = item.episode_id ?? item.id;
        title = item.title || "Untitled episode";
        description = item.summary || "No summary has been saved for this episode yet.";
        if (item.status) description += `\nStatus: ${item.status}`;
    } else if (tab === "moods") {
        id = item.id;
        title = item.mood || "Mood not specified";
        description = item.episode_title
            ? `Episode: ${item.episode_title}`
            : "Recorded mood";
        timestamp = item.created_at;
    } else {
        id = item.id;
        title = "Reflection";
        description = item.reflection || "No reflection text available.";
        timestamp = item.created_at;
    }

    return (
        <article className="memory-item" tabIndex={0}>
            <div className="memory-item-body">
                <h4 className="memory-item-title">{title}</h4>
                <p className="memory-item-description">{description}</p>
                <span className="memory-item-date">
                    <Clock3 size={13} aria-hidden="true" />
                    {formatDate(timestamp)}
                </span>
            </div>
            <button
                type="button"
                className="memory-item-delete"
                title={`Delete ${RESOURCE_CONFIG[tab].label}`}
                aria-label={`Delete ${title}`}
                disabled={deleting}
                onClick={() => onDelete(item, id, title)}
            >
                {deleting ? <LoaderCircle size={17} className="memory-spinner" /> : <X size={18} />}
            </button>
        </article>
    );
}

function UserMemoryPanel({ onClose }) {
    const [activeTab, setActiveTab] = useState("profile");
    const [records, setRecords] = useState({ episodes: [], moods: [], reflections: [] });
    const [profileData, setProfileData] = useState(null);
    const [selectedProfileThemeId, setSelectedProfileThemeId] = useState("");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const [deletingKey, setDeletingKey] = useState("");
    const [notice, setNotice] = useState("");

    const loadTabData = useCallback(async (tab, { quiet = false } = {}) => {
        if (!RESOURCE_CONFIG[tab]) return;
        if (!quiet) setLoading(true);
        setError("");
        try {
            const payload = await requestJson(RESOURCE_CONFIG[tab].path);
            setRecords((previous) => ({ ...previous, [tab]: unwrapList(payload) }));
        } catch (err) {
            setError(err.message || "Could not load this section.");
        } finally {
            if (!quiet) setLoading(false);
        }
    }, []);

    const loadProfile = useCallback(async ({ quiet = false } = {}) => {
        if (!quiet) setLoading(true);
        setError("");
        try {
            const payload = await requestJson("/memory/profile");
            setProfileData({
                memory: typeof payload?.memory === "string" ? payload.memory : "",
                updated_at: payload?.updated_at || null,
            });
            setSelectedProfileThemeId("");
        } catch (err) {
            setError(err.message || "Could not load your long-term memory.");
        } finally {
            if (!quiet) setLoading(false);
        }
    }, []);

    useEffect(() => {
        if (activeTab === "profile") loadProfile();
        else loadTabData(activeTab);
    }, [activeTab, loadProfile, loadTabData]);

    useEffect(() => {
        function handleKeyDown(event) {
            if (event.key === "Escape") onClose();
        }
        window.addEventListener("keydown", handleKeyDown);
        const previousOverflow = document.body.style.overflow;
        document.body.style.overflow = "hidden";
        return () => {
            window.removeEventListener("keydown", handleKeyDown);
            document.body.style.overflow = previousOverflow;
        };
    }, [onClose]);

    const handleDelete = async (tab, item, id, title) => {
        if (id === undefined || id === null || id === "") {
            setError("This record has no identifier, so it cannot be deleted.");
            return;
        }
        const confirmLabel = title ? `“${title}”` : `this ${RESOURCE_CONFIG[tab].label}`;
        if (!window.confirm(`Delete ${confirmLabel}? This cannot be undone.`)) return;

        const key = `${tab}:${id}`;
        setDeletingKey(key);
        setError("");
        setNotice("");
        try {
            // Episode endpoints use the stable episode_id; mood/reflection endpoints use numeric id.
            const encodedId = encodeURIComponent(String(id));
            await requestJson(`${RESOURCE_CONFIG[tab].path}/${encodedId}`, { method: "DELETE" });
            setRecords((previous) => ({
                ...previous,
                [tab]: previous[tab].filter((row) => {
                    const rowId = tab === "episodes" ? (row.episode_id ?? row.id) : row.id;
                    return String(rowId) !== String(id);
                }),
            }));
            setNotice("Deleted successfully.");
        } catch (err) {
            setError(err.message || "Could not delete this record.");
        } finally {
            setDeletingKey("");
        }
    };

    const activeTabDetails = MEMORY_TABS.find((tab) => tab.id === activeTab) || MEMORY_TABS[0];
    const ActiveIcon = activeTabDetails.icon;
    const currentRecords = RESOURCE_CONFIG[activeTab] ? records[activeTab] : [];
    const profileThemes = useMemo(() => extractProfileThemes(profileData?.memory || ""), [profileData?.memory]);
    const selectedProfileTheme = profileThemes.find((theme) => theme.id === selectedProfileThemeId) || null;

    return (
        <div
            className="memory-modal-backdrop"
            onMouseDown={(event) => {
                if (event.target === event.currentTarget) onClose();
            }}
        >
            <section className="memory-modal" role="dialog" aria-modal="true" aria-labelledby="memory-modal-title">
                <header className="memory-modal-header">
                    <div className="memory-user-heading">
                        <div className="memory-user-avatar"><UserRound size={23} /></div>
                        <div>
                            <h2 id="memory-modal-title">Your Memory</h2>
                            <p>Manage your profile, episodes, moods, and insights</p>
                        </div>
                    </div>
                    <button type="button" className="memory-modal-close" onClick={onClose} aria-label="Close memory panel" title="Close">
                        <X size={21} />
                    </button>
                </header>

                <nav className="memory-tabs" aria-label="Memory sections">
		    {MEMORY_TABS.map((tab) => {
			const Icon = tab.icon;
			const selected = activeTab === tab.id;

			return (
			    <button
				key={tab.id}
				type="button"
				className={`memory-tab ${selected ? "active" : ""}`}
				aria-current={selected ? "page" : undefined}
				onClick={() => {
				    setActiveTab(tab.id);
				    setError("");
				    setNotice("");
				}}
			    >
				<Icon size={17} />
				<span>{tab.label}</span>
			    </button>
			);
		    })}
		</nav>

                <div className="memory-modal-content">
                    <div className="memory-section-toolbar">
                        <div className="memory-section-heading">
                            <ActiveIcon size={21} />
                            <h3>{activeTabDetails.label}</h3>
                            {RESOURCE_CONFIG[activeTab] && (
                                <span className="memory-record-count">{currentRecords.length}</span>
                            )}
                        </div>
                        {(activeTab === "profile" || RESOURCE_CONFIG[activeTab]) && (
                            <button
                                type="button"
                                className="memory-refresh-button"
                                onClick={() => activeTab === "profile" ? loadProfile() : loadTabData(activeTab)}
                                disabled={loading}
                                title={activeTab === "profile" ? "Refresh long-term memory" : "Refresh records"}
                            >
                                <RefreshCw size={15} /> Refresh
                            </button>
                        )}
                    </div>

                    {error && (
                        <div className="memory-message memory-error" role="alert">
                            <span>{error}</span>
                            <button
                                type="button"
                                onClick={() => activeTab === "profile" ? loadProfile() : loadTabData(activeTab)}
                            >
                                Try again
                            </button>
                        </div>
                    )}
                    {notice && <div className="memory-message memory-success" role="status">{notice}</div>}

                    {activeTab === "profile" ? (
                        (loading || (!profileData && !error)) ? (
                            <div className="memory-status-state" role="status">
                                <LoaderCircle size={24} className="memory-spinner" />
                                <span>Loading your long-term memory…</span>
                            </div>
                        ) : error && !profileData ? null : !profileData?.memory?.trim() ? (
                            <div className="memory-profile-empty">
                                <div className="memory-profile-empty-icon"><Brain size={25} /></div>
                                <h4>No long-term memory yet</h4>
                                <p>As the chatbot saves lasting insights from your conversations, they will appear here as connected chat bubbles.</p>
                            </div>
                        ) : (
                            <section className="memory-profile" aria-label="Long-term memory map">
                                <div className="memory-profile-intro">
                                    <div>
                                        <span className="memory-profile-eyebrow">LONG-TERM MEMORY</span>
                                        <h4>Patterns and themes from your conversations</h4>
                                        <p>Choose a bubble to read the saved memory behind it. These are remembered themes, not diagnoses or fixed labels.</p>
                                    </div>
                                    {profileData.updated_at && (
                                        <span className="memory-profile-updated">Updated {formatDate(profileData.updated_at)}</span>
                                    )}
                                </div>

                                <div className="memory-profile-map">
                                    <svg className="memory-profile-lines" viewBox="0 0 1000 520" preserveAspectRatio="none" aria-hidden="true">
                                        {profileThemes.length > 0 && <path d="M455 238 C425 205 402 165 376 132" />}
                                        {profileThemes.length > 1 && <path d="M545 238 C575 205 598 165 624 132" />}
                                        {profileThemes.length > 2 && <path d="M455 286 C425 316 402 349 376 386" />}
                                        {profileThemes.length > 3 && <path d="M545 286 C575 316 598 349 624 386" />}
                                        {profileThemes.map((theme, index) => {
                                            const endpoints = [
                                                { x: 376, y: 132 },
                                                { x: 624, y: 132 },
                                                { x: 376, y: 386 },
                                                { x: 624, y: 386 },
                                            ];
                                            const point = endpoints[index];
                                            return point ? <circle key={`${theme.id}-dot`} cx={point.x} cy={point.y} r="4" /> : null;
                                        })}
                                    </svg>

                                    <div className="memory-profile-person" role="img" aria-label="You, at the center of your remembered themes">
                                        <div className="memory-profile-person-avatar">
                                            <svg viewBox="0 0 64 64" aria-hidden="true">
                                                <circle cx="32" cy="20" r="10" />
                                                <path d="M13 55 C14 42 21 34 32 34 C43 34 50 42 51 55" />
                                            </svg>
                                        </div>
                                        <span>You</span>
                                    </div>

                                    {profileThemes.map((theme, index) => (
                                        <button
                                            key={theme.id}
                                            type="button"
                                            className={`memory-profile-bubble memory-profile-bubble--${index + 1} memory-profile-tone-${theme.tone || "slate"}${selectedProfileThemeId === theme.id ? " is-selected" : ""}`}
                                            onClick={() => setSelectedProfileThemeId(theme.id)}
                                            aria-pressed={selectedProfileThemeId === theme.id}
                                            title={`Explore remembered theme: ${theme.title}`}
                                        >
                                            <span className="memory-profile-bubble-kicker">{theme.category}</span>
                                            <span className="memory-profile-bubble-title">{theme.title}</span>
                                            <span className="memory-profile-bubble-summary">{theme.summary}</span>
                                        </button>
                                    ))}
                                </div>

                                {selectedProfileTheme ? (
                                    <article className="memory-profile-detail" aria-live="polite">
                                        <div className="memory-profile-detail-heading">
                                            <div>
                                                <span className="memory-profile-eyebrow">SAVED MEMORY</span>
                                                <h4>{selectedProfileTheme.title}</h4>
                                            </div>
                                            <button
                                                type="button"
                                                className="memory-profile-detail-close"
                                                onClick={() => setSelectedProfileThemeId("")}
                                                aria-label="Close selected memory"
                                                title="Close details"
                                            >
                                                <X size={16} />
                                            </button>
                                        </div>
                                        <p>{selectedProfileTheme.details}</p>
                                        <span className="memory-profile-detail-note">Shown from your saved long-term memory.</span>
                                    </article>
                                ) : (
                                    <div className="memory-profile-hint">
                                        <span className="memory-profile-hint-dot" /> Select a bubble to explore the related memory
                                    </div>
                                )}
                                {profileThemes.length === 0 && (
                                    <p className="memory-profile-no-themes">Your memory is saved, but no short themes could be extracted from its format yet. The saved text is preserved.</p>
                                )}
                            </section>
                        )
                    ) : loading ? (
                        <div className="memory-status-state" role="status">
                            <LoaderCircle size={24} className="memory-spinner" />
                            <span>Loading {activeTabDetails.label.toLowerCase()}…</span>
                        </div>
                    ) : currentRecords.length === 0 ? (
                        <div className="memory-empty-state">
                            <ActiveIcon size={30} />
                            <h4>No {activeTabDetails.label.toLowerCase()} yet</h4>
                            <p>Saved {activeTabDetails.label.toLowerCase()} will appear here when available.</p>
                        </div>
                    ) : (
                        <div className="memory-items-list">
                            {currentRecords.map((item) => {
                                const id = activeTab === "episodes" ? (item.episode_id ?? item.id) : item.id;
                                return (
                                    <MemoryItem
                                        key={`${activeTab}:${id}`}
                                        item={item}
                                        tab={activeTab}
                                        deleting={deletingKey === `${activeTab}:${id}`}
                                        onDelete={(row, rowId, title) => handleDelete(activeTab, row, rowId, title)}
                                    />
                                );
                            })}
                        </div>
                    )}
                </div>

                <footer className="memory-modal-footer">
                    <span>Your personal memory</span>
                    <span>Explore and manage what has been saved</span>
                </footer>
            </section>
        </div>
    );
}

export default function Chat() {
    const [sidebarOpen, setSidebarOpen] = useState(true);
    const [memoryPanelOpen, setMemoryPanelOpen] = useState(false);

    return (
        <div className="chat-page">
            <Sidebar open={sidebarOpen} onOpenUserPanel={() => setMemoryPanelOpen(true)} />
            <div className="main-content">
                <Navbar toggleSidebar={() => setSidebarOpen((previous) => !previous)} />
                <ChatWindow />
                <MessageInput />
            </div>
            {memoryPanelOpen && <UserMemoryPanel onClose={() => setMemoryPanelOpen(false)} />}
        </div>
    );
}

