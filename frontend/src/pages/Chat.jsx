
import { useCallback, useEffect, useState } from "react";

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
    FileText,
    GitBranch,
    LoaderCircle,
    RefreshCw,
    Clock3,
} from "lucide-react";

import "../styles/chat.css";

// =========================================================
// API CONFIGURATION
// =========================================================

const API_BASE = (
    import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");

const MEMORY_TABS = [
    { id: "profile", label: "Profile", icon: UserRound },
    { id: "episodes", label: "Episodes", icon: BookOpen },
    { id: "moods", label: "Moods", icon: HeartPulse },
    { id: "reflections", label: "Reflections", icon: Brain },
    { id: "long-term-memory", label: "Long Term Memory", icon: FileText },
    { id: "patterns", label: "Patterns", icon: GitBranch },
];

const RESOURCE_CONFIG = {
    episodes: { path: "/memory/episodes", label: "episode" },
    moods: { path: "/memory/moods", label: "mood record" },
    reflections: { path: "/memory/reflections", label: "reflection" },
};

const PROFILE_TONES = new Set(["amber", "green", "blue", "purple"]);

const PATTERN_CATEGORIES = [
    "trigger",
    "coping",
    "mood",
    "behavior",
    "general",
];


// =========================================================
// GENERAL HELPERS
// =========================================================

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
    const token =
        localStorage.getItem("access_token") ||
        localStorage.getItem("token") ||
        localStorage.getItem("accessToken");

    const response = await fetch(`${API_BASE}${path}`, {
        ...options,
        cache: "no-store",
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
            // Keep the HTTP status message when the response is not JSON.
        }

        throw new Error(message);
    }

    if (response.status === 204) return null;

    return response.json();
}


// =========================================================
// PROFILE THEMES FROM THE BACKEND
// =========================================================

/**
 * The backend Profile generator sends structured themes in
 * GET /memory/profile as payload.themes.
 *
 * Do not derive themes from payload.memory here. The LLM has
 * already generated, validated and saved these themes.
 */
function normalizeProfileThemes(value) {
    let rows = value;

    // Be tolerant if an API/client returns a JSON string.
    if (typeof rows === "string") {
        try {
            rows = JSON.parse(rows);
        } catch {
            rows = [];
        }
    }

    if (!Array.isArray(rows) && Array.isArray(rows?.themes)) {
        rows = rows.themes;
    }

    if (!Array.isArray(rows)) return [];

    const seen = new Set();

    return rows
        .map((item, index) => {
            if (!item || typeof item !== "object") return null;

            const title = String(
                item.title || item.label || ""
            ).trim();

            const details = String(
                item.details ||
                item.description ||
                item.summary ||
                ""
            ).trim();

            if (!title || !details) return null;

            const category = String(
                item.category || "Life theme"
            ).trim();

            const summary = String(
                item.summary || details
            ).trim();

            const toneValue = String(
                item.tone || "blue"
            ).toLowerCase();

            const tone = PROFILE_TONES.has(toneValue)
                ? toneValue
                : "blue";

            // The backend generates stable IDs. This fallback is only
            // for older or incomplete records that do not contain an ID.
            const id = String(
                item.id ||
                `profile-theme-${index}-${title
                    .toLowerCase()
                    .replace(/[^a-z0-9]+/g, "-")}`
            );

            if (seen.has(id)) return null;

            seen.add(id);

            return {
                id,
                category,
                title,
                summary,
                details,
                tone,
            };
        })
        .filter(Boolean);
}


// =========================================================
// EPISODE / MOOD / REFLECTION LIST ITEM
// =========================================================

function MemoryItem({ item, tab, onDelete, deleting }) {
    let title = "";
    let description = "";
    let timestamp = item.created_at || item.updated_at;
    let id;

    if (tab === "episodes") {
        id = item.episode_id ?? item.id;

        title = item.title || "Untitled episode";

        description =
            item.summary ||
            "No summary has been saved for this episode yet.";

        if (item.status) {
            description += `\nStatus: ${item.status}`;
        }
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

        description =
            item.reflection || "No reflection text available.";

        timestamp = item.created_at;
    }

    return (
        <article className="memory-item" tabIndex={0}>
            <div className="memory-item-body">
                <h4 className="memory-item-title">
                    {title}
                </h4>

                <p className="memory-item-description">
                    {description}
                </p>

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
                {deleting ? (
                    <LoaderCircle
                        size={17}
                        className="memory-spinner"
                    />
                ) : (
                    <X size={18} />
                )}
            </button>
        </article>
    );
}


// =========================================================
// USER MEMORY PANEL
// =========================================================

function UserMemoryPanel({ onClose }) {
    const [activeTab, setActiveTab] = useState("profile");

    const [records, setRecords] = useState({
        episodes: [],
        moods: [],
        reflections: [],
    });

    const [profileData, setProfileData] = useState(null);

    // Detected-pattern data and editor state. These must be declared
    // before loadPatterns and the JSX that references them.
    const [patterns, setPatterns] = useState([]);
    const [editingPatternId, setEditingPatternId] = useState("");
    const [savingPatternId, setSavingPatternId] = useState("");
    const [patternDraft, setPatternDraft] = useState({
        category: "general",
        description: "",
        status: "ACTIVE",
    });

    const [memoryDraft, setMemoryDraft] = useState("");
    const [editingMemory, setEditingMemory] = useState(false);
    const [savingMemory, setSavingMemory] = useState(false);

    const [selectedProfileThemeId, setSelectedProfileThemeId] =
        useState("");

    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const [deletingKey, setDeletingKey] = useState("");
    const [notice, setNotice] = useState("");


    // -----------------------------------------------------
    // LOAD EPISODES / MOODS / REFLECTIONS
    // -----------------------------------------------------

    const loadTabData = useCallback(
        async (tab, { quiet = false } = {}) => {
            if (!RESOURCE_CONFIG[tab]) return;

            if (!quiet) setLoading(true);

            setError("");

            try {
                const payload = await requestJson(
                    RESOURCE_CONFIG[tab].path
                );

                setRecords(previous => ({
                    ...previous,
                    [tab]: unwrapList(payload),
                }));
            } catch (err) {
                setError(
                    err.message || "Could not load this section."
                );
            } finally {
                if (!quiet) setLoading(false);
            }
        },
        []
    );


    // -----------------------------------------------------
    // LOAD DETECTED PATTERNS
    // -----------------------------------------------------

    const loadPatterns = useCallback(
        async ({ quiet = false } = {}) => {
            if (!quiet) setLoading(true);

            setError("");

            try {
                const payload = await requestJson("/memory/patterns");
                setPatterns(unwrapList(payload));
            } catch (err) {
                setError(
                    err.message || "Could not load detected patterns."
                );
            } finally {
                if (!quiet) setLoading(false);
            }
        },
        []
    );


    // -----------------------------------------------------
    // LOAD LONG-TERM MEMORY AND LLM-GENERATED THEMES
    // -----------------------------------------------------

    const loadProfile = useCallback(
        async ({ quiet = false } = {}) => {
            if (!quiet) setLoading(true);

            setError("");

            try {
                const payload = await requestJson("/memory/profile");

                const freshMemory =
                    typeof payload?.memory === "string"
                        ? payload.memory
                        : "";

                // Themes are generated by the backend LLM and stored in
                // long_term_memory.profile_themes. Do not parse memory text.
                const freshThemes = normalizeProfileThemes(
                    payload?.themes
                );

                setProfileData({
                    memory: freshMemory,
                    themes: freshThemes,
                    updated_at: payload?.updated_at || null,
                });

                // Keep the selected bubble if its stable ID still exists.
                setSelectedProfileThemeId(currentId =>
                    freshThemes.some(
                        theme => theme.id === currentId
                    )
                        ? currentId
                        : ""
                );

                console.log("Profile memory loaded.");

                console.log(
                    "LLM-generated Profile themes:",
                    freshThemes
                );

            } catch (err) {
                setError(
                    err.message ||
                    "Could not load your long-term memory."
                );
            } finally {
                if (!quiet) setLoading(false);
            }
        },
        []
    );


    // -----------------------------------------------------
    // LOAD DATA WHEN TAB CHANGES
    // -----------------------------------------------------

    useEffect(() => {
        if (
            activeTab === "profile" ||
            activeTab === "long-term-memory"
        ) {
            loadProfile();
        } else if (activeTab === "patterns") {
            loadPatterns();
        } else {
            loadTabData(activeTab);
        }
    }, [activeTab, loadProfile, loadPatterns, loadTabData]);


    // Keep the editor synchronized with fetched memory unless the
    // user is actively editing and has unsaved changes.
    useEffect(() => {
        if (
            activeTab === "long-term-memory" &&
            !editingMemory
        ) {
            setMemoryDraft(profileData?.memory || "");
        }
    }, [
        activeTab,
        profileData?.memory,
        editingMemory
    ]);


    // -----------------------------------------------------
    // REFRESH PROFILE AFTER BACKEND MEMORY UPDATE EVENT
    // -----------------------------------------------------

    useEffect(() => {
        const handleMemoryUpdated = () => {
            console.log(
                "Long-term memory update event received."
            );

            // Refresh even if another Memory tab is currently selected.
            // This keeps Profile ready when the user switches back to it.
            loadProfile({ quiet: true });
        };

        window.addEventListener(
            "long-term-memory-updated",
            handleMemoryUpdated
        );

        return () => {
            window.removeEventListener(
                "long-term-memory-updated",
                handleMemoryUpdated
            );
        };
    }, [loadProfile]);


    // -----------------------------------------------------
    // ESCAPE KEY AND MODAL SCROLL
    // -----------------------------------------------------

    useEffect(() => {
        function handleKeyDown(event) {
            if (event.key === "Escape") {
                onClose();
            }
        }

        window.addEventListener("keydown", handleKeyDown);

        const previousOverflow = document.body.style.overflow;

        document.body.style.overflow = "hidden";

        return () => {
            window.removeEventListener(
                "keydown",
                handleKeyDown
            );

            document.body.style.overflow = previousOverflow;
        };
    }, [onClose]);


    // -----------------------------------------------------
    // DELETE MEMORY RECORD
    // -----------------------------------------------------

    const handleDelete = async (tab, item, id, title) => {
        if (
            id === undefined ||
            id === null ||
            id === ""
        ) {
            setError(
                "This record has no identifier, so it cannot be deleted."
            );

            return;
        }

        const confirmLabel = title
            ? `“${title}”`
            : `this ${RESOURCE_CONFIG[tab].label}`;

        if (
            !window.confirm(
                `Delete ${confirmLabel}? This cannot be undone.`
            )
        ) {
            return;
        }

        const key = `${tab}:${id}`;

        setDeletingKey(key);
        setError("");
        setNotice("");

        try {
            const encodedId = encodeURIComponent(String(id));

            await requestJson(
                `${RESOURCE_CONFIG[tab].path}/${encodedId}`,
                { method: "DELETE" }
            );

            setRecords(previous => ({
                ...previous,

                [tab]: previous[tab].filter(row => {
                    const rowId = tab === "episodes"
                        ? (row.episode_id ?? row.id)
                        : row.id;

                    return String(rowId) !== String(id);
                }),
            }));

            setNotice("Deleted successfully.");

        } catch (err) {
            setError(
                err.message || "Could not delete this record."
            );
        } finally {
            setDeletingKey("");
        }
    };


    // -----------------------------------------------------
    // SAVE EDITED LONG-TERM MEMORY
    // -----------------------------------------------------

    const handleSaveLongTermMemory = async () => {
        setSavingMemory(true);
        setError("");
        setNotice("");

        try {
            const payload = await requestJson(
                "/memory/profile",
                {
                    method: "PUT",
                    body: JSON.stringify({
                        memory: memoryDraft
                    })
                }
            );

            const savedMemory =
                typeof payload?.memory === "string"
                    ? payload.memory
                    : memoryDraft;

            setProfileData(previous => ({
                memory: savedMemory,
                themes: normalizeProfileThemes(
                    payload?.themes ?? previous?.themes
                ),
                updated_at:
                    payload?.updated_at ||
                    previous?.updated_at ||
                    null
            }));

            setMemoryDraft(savedMemory);
            setEditingMemory(false);
            setNotice("Long-term memory saved successfully.");

        } catch (err) {
            setError(
                err.message ||
                "Could not save long-term memory."
            );
        } finally {
            setSavingMemory(false);
        }
    };


    // -----------------------------------------------------
    // EDIT AND SAVE DETECTED PATTERNS
    // -----------------------------------------------------

    const startEditingPattern = pattern => {
        setEditingPatternId(pattern.pattern_id);
        setPatternDraft({
            category: PATTERN_CATEGORIES.includes(pattern.category)
                ? pattern.category
                : "general",
            description: pattern.description || "",
            status: String(pattern.status || "ACTIVE").toUpperCase(),
        });
        setError("");
        setNotice("");
    };

    const cancelEditingPattern = () => {
        setEditingPatternId("");
        setPatternDraft({
            category: "general",
            description: "",
            status: "ACTIVE",
        });
        setError("");
    };

    const handleSavePattern = async patternId => {
        if (!patternDraft.description.trim()) {
            setError("Pattern description cannot be empty.");
            return;
        }

        setSavingPatternId(patternId);
        setError("");
        setNotice("");

        try {
            const payload = await requestJson(
                `/memory/patterns/${encodeURIComponent(patternId)}`,
                {
                    method: "PUT",
                    body: JSON.stringify({
                        category: patternDraft.category,
                        description: patternDraft.description.trim(),
                        status: patternDraft.status,
                    }),
                }
            );

            setPatterns(previous => previous.map(pattern =>
                pattern.pattern_id === patternId
                    ? { ...pattern, ...payload }
                    : pattern
            ));

            setEditingPatternId("");
            setNotice("Pattern updated successfully.");

        } catch (err) {
            setError(
                err.message || "Could not update this pattern."
            );
        } finally {
            setSavingPatternId("");
        }
    };


    // -----------------------------------------------------
    // DERIVED DATA
    // -----------------------------------------------------

    const activeTabDetails =
        MEMORY_TABS.find(tab => tab.id === activeTab) ||
        MEMORY_TABS[0];

    const ActiveIcon = activeTabDetails.icon;

    const currentRecords = RESOURCE_CONFIG[activeTab]
        ? records[activeTab]
        : [];

    // These themes are generated and saved by the backend LLM.
    // Every saved theme is displayed as a node around the central profile icon.
    const profileThemes = profileData?.themes || [];
    const profileRows = Math.max(1, Math.ceil(profileThemes.length / 2));
    const profileMapHeight = Math.max(420, profileRows * 150);

    const selectedProfileTheme = profileThemes.find(
        theme => theme.id === selectedProfileThemeId
    ) || null;

    const hasProfileContent = Boolean(
        profileData?.memory?.trim() || profileThemes.length
    );


    // =====================================================
    // RENDER
    // =====================================================

    return (
        <div
            className="memory-modal-backdrop"
            onMouseDown={event => {
                if (event.target === event.currentTarget) {
                    onClose();
                }
            }}
        >
            <section
                className="memory-modal"
                role="dialog"
                aria-modal="true"
                aria-labelledby="memory-modal-title"
            >

                {/* HEADER */}

                <header className="memory-modal-header">
                    <div className="memory-user-heading">
                        <div className="memory-user-avatar">
                            <UserRound size={23} />
                        </div>

                        <div>
                            <h2 id="memory-modal-title">
                                Your Memory
                            </h2>

                            <p>
                                Manage your profile, episodes, moods,
                                reflections, and long-term memory
                            </p>
                        </div>
                    </div>

                    <button
                        type="button"
                        className="memory-modal-close"
                        onClick={onClose}
                        aria-label="Close memory panel"
                        title="Close"
                    >
                        <X size={21} />
                    </button>
                </header>


                {/* NAVIGATION TABS */}

                <nav
                    className="memory-tabs"
                    aria-label="Memory sections"
                >
                    {MEMORY_TABS.map(tab => {
                        const Icon = tab.icon;
                        const selected = activeTab === tab.id;

                        return (
                            <button
                                key={tab.id}
                                type="button"
                                className={`memory-tab ${
                                    selected ? "active" : ""
                                }`}
                                aria-current={
                                    selected ? "page" : undefined
                                }
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


                {/* MAIN CONTENT */}

                <div className="memory-modal-content">

                    <div className="memory-section-toolbar">
                        <div className="memory-section-heading">
                            <ActiveIcon size={21} />

                            <h3>{activeTabDetails.label}</h3>

                            {(RESOURCE_CONFIG[activeTab] || activeTab === "patterns") && (
                                <span className="memory-record-count">
                                    {activeTab === "patterns"
                                        ? patterns.length
                                        : currentRecords.length}
                                </span>
                            )}
                        </div>

                        {(
                            activeTab === "profile" ||
                            activeTab === "long-term-memory" ||
                            activeTab === "patterns" ||
                            RESOURCE_CONFIG[activeTab]
                        ) && (
                            <button
                                type="button"
                                className="memory-refresh-button"
                                onClick={() => {
                                    if (
                                        activeTab === "profile" ||
                                        activeTab === "long-term-memory"
                                    ) {
                                        loadProfile();
                                    } else if (activeTab === "patterns") {
                                        loadPatterns();
                                    } else {
                                        loadTabData(activeTab);
                                    }
                                }}
                                disabled={loading || savingMemory || Boolean(savingPatternId)}
                                title={
                                    activeTab === "profile" ||
                                    activeTab === "long-term-memory"
                                        ? "Refresh long-term memory"
                                        : "Refresh records"
                                }
                            >
                                <RefreshCw size={15} />
                                Refresh
                            </button>
                        )}
                    </div>


                    {/* ERROR AND SUCCESS MESSAGES */}

                    {error && (
                        <div
                            className="memory-message memory-error"
                            role="alert"
                        >
                            <span>{error}</span>

                            <button
                                type="button"
                                onClick={() => {
                                    if (
                                        activeTab === "profile" ||
                                        activeTab === "long-term-memory"
                                    ) {
                                        loadProfile();
                                    } else if (activeTab === "patterns") {
                                        loadPatterns();
                                    } else {
                                        loadTabData(activeTab);
                                    }
                                }}
                            >
                                Try again
                            </button>
                        </div>
                    )}

                    {notice && (
                        <div
                            className="memory-message memory-success"
                            role="status"
                        >
                            {notice}
                        </div>
                    )}


                    {/* =================================================
                        PROFILE TAB
                    ================================================= */}

                    {activeTab === "profile" ? (

                        (loading || (!profileData && !error)) ? (

                            <div
                                className="memory-status-state"
                                role="status"
                            >
                                <LoaderCircle
                                    size={24}
                                    className="memory-spinner"
                                />

                                <span>
                                    Loading your long-term memory…
                                </span>
                            </div>

                        ) : error && !profileData ? (

                            null

                        ) : !hasProfileContent ? (

                            <div className="memory-profile-empty">
                                <div className="memory-profile-empty-icon">
                                    <Brain size={25} />
                                </div>

                                <h4>No long-term memory yet</h4>

                                <p>
                                    As the chatbot saves lasting insights
                                    from your conversations, your generated
                                    Profile themes will appear here.
                                </p>
                            </div>

                        ) : (

                            <section
                                className="memory-profile"
                                aria-label="Long-term memory profile"
                            >

                                {/* PROFILE INTRO */}

                                <div className="memory-profile-intro">
                                    <div>
                                        <span className="memory-profile-eyebrow">
                                            LONG-TERM MEMORY
                                        </span>

                                        <h4>
                                            Your remembered themes
                                        </h4>

                                        <p>
                                            These themes were generated
                                            from your consolidated
                                            long-term memory. Select a
                                            bubble to read the supporting
                                            insight. They are not diagnoses
                                            or fixed labels.
                                        </p>

                                        <p>
                                            {profileThemes.length} remembered{" "}
                                            {profileThemes.length === 1
                                                ? "theme"
                                                : "themes"}
                                        </p>
                                    </div>

                                    {profileData?.updated_at && (
                                        <span className="memory-profile-updated">
                                            Updated{" "}
                                            {formatDate(
                                                profileData.updated_at
                                            )}
                                        </span>
                                    )}
                                </div>


                                {/* ALL-THEME PROFILE CONSTELLATION */}

                                {profileThemes.length > 0 && (
                                    <div
                                        className="memory-profile-map"
                                        style={{ height: `${profileMapHeight}px` }}
                                    >
                                        {/* A connector is drawn from the profile icon to every theme. */}
                                        <svg
                                            className="memory-profile-lines"
                                            viewBox="0 0 100 100"
                                            preserveAspectRatio="none"
                                            aria-hidden="true"
                                        >
                                            {profileThemes.map((theme, index) => {
                                                const row = Math.floor(index / 2);
                                                const y = ((row + 0.5) / profileRows) * 100;
                                                const endX = index % 2 === 0 ? 40 : 60;

                                                return (
                                                    <g key={`connector-${theme.id}`}>
                                                        <path d={`M 50 50 L ${endX} ${y}`} />
                                                        <circle cx={endX} cy={y} r="0.55" />
                                                    </g>
                                                );
                                            })}
                                        </svg>

                                        {/* CENTRAL PROFILE ICON */}
                                        <div
                                            className="memory-profile-person"
                                            role="img"
                                            aria-label="You, at the center of your remembered themes"
                                        >
                                            <div className="memory-profile-person-avatar">
                                                <svg
                                                    viewBox="0 0 64 64"
                                                    aria-hidden="true"
                                                >
                                                    <circle
                                                        cx="32"
                                                        cy="20"
                                                        r="10"
                                                    />
                                                    <path d="M13 55 C14 42 21 34 32 34 C43 34 50 42 51 55" />
                                                </svg>
                                            </div>
                                            <span>You</span>
                                        </div>

                                        {/* EVERY PROFILE THEME IS A CONNECTED BUBBLE */}
                                        <div className="memory-profile-bubbles">
                                            {profileThemes.map((theme, index) => {
                                                const row = Math.floor(index / 2);
                                                const top = `${((row + 0.5) / profileRows) * 100}%`;
                                                const horizontalPosition = index % 2 === 0
                                                    ? { left: "2%" }
                                                    : { right: "2%" };

                                                return (
                                                    <button
                                                        key={theme.id}
                                                        type="button"
                                                        className={`memory-profile-bubble memory-profile-tone-${
                                                            theme.tone || "blue"
                                                        }${
                                                            selectedProfileThemeId === theme.id
                                                                ? " is-selected"
                                                                : ""
                                                        }`}
                                                        style={{
                                                            top,
                                                            ...horizontalPosition,
                                                        }}
                                                        onClick={() =>
                                                            setSelectedProfileThemeId(theme.id)
                                                        }
                                                        aria-pressed={
                                                            selectedProfileThemeId === theme.id
                                                        }
                                                        title={`Explore remembered theme: ${theme.title}`}
                                                    >
                                                        <span className="memory-profile-bubble-kicker">
                                                            {theme.category}
                                                        </span>
                                                        <span className="memory-profile-bubble-title">
                                                            {theme.title}
                                                        </span>
                                                        <span className="memory-profile-bubble-summary">
                                                            {theme.summary}
                                                        </span>
                                                    </button>
                                                );
                                            })}
                                        </div>
                                    </div>
                                )}


                                {/* SELECTED THEME DETAILS */}

                                {selectedProfileTheme ? (

                                    <article
                                        className="memory-profile-detail"
                                        aria-live="polite"
                                    >
                                        <div className="memory-profile-detail-heading">
                                            <div>
                                                <span className="memory-profile-eyebrow">
                                                    SAVED MEMORY
                                                </span>

                                                <h4>
                                                    {selectedProfileTheme.title}
                                                </h4>
                                            </div>

                                            <button
                                                type="button"
                                                className="memory-profile-detail-close"
                                                onClick={() =>
                                                    setSelectedProfileThemeId("")
                                                }
                                                aria-label="Close selected memory"
                                                title="Close details"
                                            >
                                                <X size={16} />
                                            </button>
                                        </div>

                                        <p>
                                            {selectedProfileTheme.details}
                                        </p>

                                        <span className="memory-profile-detail-note">
                                            Shown from your saved
                                            LLM-generated Profile themes.
                                        </span>
                                    </article>

                                ) : profileThemes.length > 0 ? (

                                    <div className="memory-profile-hint">
                                        <span className="memory-profile-hint-dot" />

                                        Select a bubble to explore the
                                        related memory
                                    </div>

                                ) : (

                                    <p className="memory-profile-no-themes">
                                        Your long-term memory is saved,
                                        but no Profile themes have been
                                        generated yet. Close an episode
                                        successfully and refresh this panel.
                                    </p>
                                )}

                            </section>
                        )

                    ) : activeTab === "long-term-memory" ? (

                        (loading && !profileData) ? (

                            <div
                                className="memory-status-state"
                                role="status"
                            >
                                <LoaderCircle
                                    size={24}
                                    className="memory-spinner"
                                />

                                <span>
                                    Loading long-term memory…
                                </span>
                            </div>

                        ) : error && !profileData ? (

                            null

                        ) : (

                            <section className="memory-ltm-panel">

                                <div className="memory-ltm-heading">
                                    <div>
                                        <span className="memory-profile-eyebrow">
                                            CONSOLIDATED MEMORY
                                        </span>

                                        <h4>
                                            What the assistant remembers
                                        </h4>

                                        <p>
                                            This is your consolidated
                                            long-term memory. You can
                                            review and edit its contents.
                                        </p>
                                    </div>

                                    {profileData?.updated_at && (
                                        <span className="memory-profile-updated">
                                            Updated{" "}
                                            {formatDate(profileData.updated_at)}
                                        </span>
                                    )}
                                </div>

                                {editingMemory ? (

                                    <textarea
                                        className="memory-ltm-textarea"
                                        value={memoryDraft}
                                        onChange={event =>
                                            setMemoryDraft(event.target.value)
                                        }
                                        placeholder="Your long-term memory will appear here..."
                                        disabled={savingMemory}
                                        spellCheck={true}
                                    />

                                ) : (

                                    <div className="memory-ltm-text">
                                        {profileData?.memory?.trim()
                                            ? profileData.memory
                                            : "No long-term memory has been saved yet."}
                                    </div>

                                )}

                                <div className="memory-ltm-actions">
                                    {editingMemory ? (
                                        <>
                                            <button
                                                type="button"
                                                className="memory-ltm-cancel"
                                                disabled={savingMemory}
                                                onClick={() => {
                                                    setMemoryDraft(
                                                        profileData?.memory || ""
                                                    );
                                                    setEditingMemory(false);
                                                    setError("");
                                                }}
                                            >
                                                Cancel
                                            </button>

                                            <button
                                                type="button"
                                                className="memory-ltm-done"
                                                disabled={savingMemory}
                                                onClick={handleSaveLongTermMemory}
                                            >
                                                {savingMemory
                                                    ? "Saving..."
                                                    : "Done"}
                                            </button>
                                        </>
                                    ) : (
                                        <button
                                            type="button"
                                            className="memory-ltm-edit"
                                            onClick={() => {
                                                setMemoryDraft(
                                                    profileData?.memory || ""
                                                );
                                                setEditingMemory(true);
                                                setNotice("");
                                            }}
                                        >
                                            Edit Memory
                                        </button>
                                    )}
                                </div>

                            </section>

                        )

                    ) : activeTab === "patterns" ? (

                        loading && patterns.length === 0 ? (

                            <div className="memory-status-state" role="status">
                                <LoaderCircle
                                    size={24}
                                    className="memory-spinner"
                                />
                                <span>Loading detected patterns…</span>
                            </div>

                        ) : patterns.length === 0 ? (

                            <div className="memory-empty-state">
                                <GitBranch size={30} />
                                <h4>No recurring patterns yet</h4>
                                <p>
                                    Patterns will appear here when the detector
                                    finds recurring themes supported by evidence
                                    across your episodes.
                                </p>
                            </div>

                        ) : (

                            <section className="memory-patterns-panel">
                                <div className="memory-patterns-intro">
                                    <div>
                                        <span className="memory-profile-eyebrow">
                                            DETECTED PATTERNS
                                        </span>
                                        <h4>Your recurring patterns</h4>
                                        <p>
                                            Review and edit the descriptions,
                                            categories, and statuses. Confidence
                                            and source evidence remain read-only
                                            because they are produced by the detector.
                                        </p>
                                    </div>
                                </div>

                                <div className="memory-patterns-list">
                                    {patterns.map(pattern => {
                                        const patternId = pattern.pattern_id;
                                        const isEditing = editingPatternId === patternId;
                                        const evidence = Array.isArray(pattern.evidence)
                                            ? pattern.evidence
                                            : [];
                                        const confidence = Number(pattern.confidence);
                                        const confidenceLabel = Number.isFinite(confidence)
                                            ? `${Math.round(confidence * 100)}%`
                                            : "Unavailable";
                                        const status = String(
                                            pattern.status || "ACTIVE"
                                        ).toUpperCase();

                                        return (
                                            <article
                                                className="memory-pattern-card"
                                                key={patternId}
                                            >
                                                <div className="memory-pattern-topline">
                                                    <div className="memory-pattern-meta">
                                                        <span className="memory-pattern-category">
                                                            {pattern.category || "general"}
                                                        </span>
                                                        <span className={`memory-pattern-status is-${status.toLowerCase()}`}>
                                                            {status}
                                                        </span>
                                                    </div>

                                                    <span className="memory-pattern-confidence">
                                                        Confidence {confidenceLabel}
                                                    </span>
                                                </div>

                                                {isEditing ? (
                                                    <div className="memory-pattern-editor">
                                                        <label className="memory-pattern-field">
                                                            <span>Pattern description</span>
                                                            <textarea
                                                                value={patternDraft.description}
                                                                onChange={event =>
                                                                    setPatternDraft(previous => ({
                                                                        ...previous,
                                                                        description: event.target.value,
                                                                    }))
                                                                }
                                                                rows={4}
                                                                disabled={savingPatternId === patternId}
                                                            />
                                                        </label>

                                                        <div className="memory-pattern-fields-row">
                                                            <label className="memory-pattern-field">
                                                                <span>Category</span>
                                                                <select
                                                                    value={patternDraft.category}
                                                                    onChange={event =>
                                                                        setPatternDraft(previous => ({
                                                                            ...previous,
                                                                            category: event.target.value,
                                                                        }))
                                                                    }
                                                                    disabled={savingPatternId === patternId}
                                                                >
                                                                    {PATTERN_CATEGORIES.map(category => (
                                                                        <option key={category} value={category}>
                                                                            {category}
                                                                        </option>
                                                                    ))}
                                                                </select>
                                                            </label>

                                                            <label className="memory-pattern-field">
                                                                <span>Status</span>
                                                                <select
                                                                    value={patternDraft.status}
                                                                    onChange={event =>
                                                                        setPatternDraft(previous => ({
                                                                            ...previous,
                                                                            status: event.target.value,
                                                                        }))
                                                                    }
                                                                    disabled={savingPatternId === patternId}
                                                                >
                                                                    <option value="ACTIVE">Active</option>
                                                                    <option value="RESOLVED">Resolved</option>
                                                                </select>
                                                            </label>
                                                        </div>

                                                        <div className="memory-pattern-actions">
                                                            <button
                                                                type="button"
                                                                className="memory-pattern-cancel"
                                                                onClick={cancelEditingPattern}
                                                                disabled={savingPatternId === patternId}
                                                            >
                                                                Cancel
                                                            </button>
                                                            <button
                                                                type="button"
                                                                className="memory-pattern-save"
                                                                onClick={() => handleSavePattern(patternId)}
                                                                disabled={savingPatternId === patternId}
                                                            >
                                                                {savingPatternId === patternId
                                                                    ? "Saving…"
                                                                    : "Done"}
                                                            </button>
                                                        </div>
                                                    </div>
                                                ) : (
                                                    <>
                                                        <p className="memory-pattern-description">
                                                            {pattern.description}
                                                        </p>

                                                        <div className="memory-pattern-dates">
                                                            <span>
                                                                First seen: {formatDate(pattern.first_seen_at)}
                                                            </span>
                                                            <span>
                                                                Last seen: {formatDate(pattern.last_seen_at)}
                                                            </span>
                                                        </div>

                                                        <div className="memory-pattern-actions">
                                                            <button
                                                                type="button"
                                                                className="memory-pattern-edit"
                                                                onClick={() => startEditingPattern(pattern)}
                                                            >
                                                                Edit pattern
                                                            </button>
                                                        </div>
                                                    </>
                                                )}

                                                <details className="memory-pattern-evidence">
                                                    <summary>
                                                        Evidence ({evidence.length})
                                                    </summary>
                                                    {evidence.length === 0 ? (
                                                        <p>No evidence was returned for this pattern.</p>
                                                    ) : (
                                                        <div className="memory-pattern-evidence-list">
                                                            {evidence.map((entry, index) => (
                                                                <div
                                                                    className="memory-pattern-evidence-item"
                                                                    key={`${entry?.episode_id || "episode"}-${index}`}
                                                                >
                                                                    <span>
                                                                        {entry?.episode_id || "Episode"}
                                                                    </span>
                                                                    <p>
                                                                        {entry?.detail || "No evidence detail available."}
                                                                    </p>
                                                                </div>
                                                            ))}
                                                        </div>
                                                    )}
                                                </details>
                                            </article>
                                        );
                                    })}
                                </div>
                            </section>

                        )

                    ) : (

                        /* =================================================
                           EPISODES / MOODS / REFLECTIONS TABS
                        ================================================= */

                        loading ? (

                            <div
                                className="memory-status-state"
                                role="status"
                            >
                                <LoaderCircle
                                    size={24}
                                    className="memory-spinner"
                                />

                                <span>
                                    Loading{" "}
                                    {activeTabDetails.label.toLowerCase()}…
                                </span>
                            </div>

                        ) : currentRecords.length === 0 ? (

                            <div className="memory-empty-state">
                                <ActiveIcon size={30} />

                                <h4>
                                    No{" "}
                                    {activeTabDetails.label.toLowerCase()} yet
                                </h4>

                                <p>
                                    Saved{" "}
                                    {activeTabDetails.label.toLowerCase()} will
                                    appear here when available.
                                </p>
                            </div>

                        ) : (

                            <div className="memory-items-list">
                                {currentRecords.map(item => {
                                    const id = activeTab === "episodes"
                                        ? (item.episode_id ?? item.id)
                                        : item.id;

                                    return (
                                        <MemoryItem
                                            key={`${activeTab}:${id}`}
                                            item={item}
                                            tab={activeTab}
                                            deleting={
                                                deletingKey ===
                                                `${activeTab}:${id}`
                                            }
                                            onDelete={(
                                                row,
                                                rowId,
                                                title
                                            ) =>
                                                handleDelete(
                                                    activeTab,
                                                    row,
                                                    rowId,
                                                    title
                                                )
                                            }
                                        />
                                    );
                                })}
                            </div>
                        )
                    )}

                </div>


                {/* FOOTER */}

                <footer className="memory-modal-footer">
                    <span>Your personal memory</span>

                    <span>
                        Explore and manage what has been saved
                    </span>
                </footer>

            </section>
        </div>
    );
}


// =========================================================
// MAIN CHAT PAGE
// =========================================================

export default function Chat() {
    const [sidebarOpen, setSidebarOpen] = useState(true);
    const [memoryPanelOpen, setMemoryPanelOpen] = useState(false);

    return (
        <div className="chat-page">

            <Sidebar
                open={sidebarOpen}
                onOpenUserPanel={() => setMemoryPanelOpen(true)}
            />

            <div className="main-content">

                <Navbar
                    toggleSidebar={() =>
                        setSidebarOpen(previous => !previous)
                    }
                />

                <ChatWindow />

                <MessageInput />

            </div>

            {memoryPanelOpen && (
                <UserMemoryPanel
                    onClose={() => setMemoryPanelOpen(false)}
                />
            )}

        </div>
    );
}
