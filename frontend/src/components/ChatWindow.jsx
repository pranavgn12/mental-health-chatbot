

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkBreaks from "remark-breaks";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { ExternalLink, FileText, Globe, X } from "lucide-react";
import "katex/dist/katex.min.css";

import useChat from "../hooks/useChat";


function normalizeMath(content) {
    let text = String(content ?? "").replace(/\r\n?/g, "\n");

    function normalizeFormula(formula) {
        // Repair accidentally doubled backslashes before common
        // LaTeX commands, e.g. \\boxed -> \boxed.
        return formula
            .trim()
            .replace(
                /\\{2,}(?=(?:boxed|text|frac|sqrt|sum|cdot|times|alpha|beta|theta|pi|mathrm|mathbf)\b)/g,
                "\\"
            );
    }

    // Convert \[ ... \] into display math.
    text = text.replace(
        /\\\[([\s\S]*?)\\\]/g,
        (_, formula) =>
            `\n\n$$\n${normalizeFormula(formula)}\n$$\n\n`
    );

    // Convert \( ... \) into inline math.
    text = text.replace(
        /\\\(([\s\S]*?)\\\)/g,
        (_, formula) => `$${normalizeFormula(formula)}$`
    );

    // Convert standalone square-bracket LaTeX blocks.
    text = text.replace(
        /^[ \t]*\[[ \t]*\n([\s\S]*?)\n[ \t]*\][ \t]*$/gm,
        (match, formula) => {
            if (!/\\[a-zA-Z]+/.test(formula)) {
                return match;
            }

            return `\n\n$$\n${normalizeFormula(formula)}\n$$\n\n`;
        }
    );

    // Normalize multiline display math blocks.
    text = text.replace(
        /^[ \t]*\$\$[ \t]*\n([\s\S]*?)\n[ \t]*\$\$[ \t]*$/gm,
        (_, formula) =>
            `\n\n$$\n${normalizeFormula(formula)}\n$$\n\n`
    );

    // Normalize single-line display math, e.g. $$\boxed{28}$$.
    text = text.replace(
        /^[ \t]*\$\$[ \t]*(.+?)[ \t]*\$\$[ \t]*$/gm,
        (_, formula) =>
            `\n\n$$\n${normalizeFormula(formula)}\n$$\n\n`
    );

    return text.replace(/\n{3,}/g, "\n\n");
}


function normalizeReferences(value) {
    if (!Array.isArray(value)) return [];

    return value.filter(item =>
        item && typeof item === "object" &&
        (item.source_type === "web" || item.source_type === "local")
    );
}

function getSafeWebUrl(value) {
    if (typeof value !== "string") return null;

    try {
        const url = new URL(value);
        return url.protocol === "https:" || url.protocol === "http:"
            ? url.href
            : null;
    } catch {
        return null;
    }
}

function getDomain(value) {
    const safeUrl = getSafeWebUrl(value);
    if (!safeUrl) return "";

    try {
        return new URL(safeUrl).hostname.replace(/^www\./, "");
    } catch {
        return "";
    }
}

export default function ChatWindow() {
    console.log("CHATWINDOW COMPONENT RENDERED");
    
    const { chatList, selectedChatId, messages } = useChat();
    const messagesEndRef = useRef(null);
    const [activeReferences, setActiveReferences] = useState(null);

    const selectedChat = chatList.find(
        chat => chat.id === selectedChatId
    );

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({
            behavior: "auto",
            block: "end"
        });
    }, [messages]);

    useEffect(() => {
        if (!activeReferences) return undefined;

        function handleKeyDown(event) {
            if (event.key === "Escape") {
                setActiveReferences(null);
            }
        }

        document.addEventListener("keydown", handleKeyDown);
        return () => document.removeEventListener("keydown", handleKeyDown);
    }, [activeReferences]);

    return (
        <div className="chat-window">
            <div className="chat-header">
                <h2>
                    {selectedChat
                        ? selectedChat.title
                        : "Mental Health AI"}
                </h2>
            </div>

            <div className="messages">
                {messages.length === 0 ? (
                    <div className="empty-chat">
                        <h2>👋 Welcome</h2>
                        <p>Start a conversation.</p>
                    </div>
                ) : (
                    messages.map(message => {
                        const isWaiting =
                            message.role === "assistant" &&
                            !message.content?.trim();

                        const references = normalizeReferences(message.references);
			
			if (message.role === "assistant") {
			    console.log("REFERENCES DEBUG:", {
				messageId: message.id,
				rawReferences: message.references,
				normalizedReferences: references,
				messageKeys: Object.keys(message)
			    });
			}
                        return (
                            <div
                                key={message.id}
                                className={`message-row ${message.role}`}
                            >
                                <div className="message-content-group">
                                    <div className="bubble">
                                        {isWaiting ? (
                                            <div
                                                className="typing-indicator"
                                                role="status"
                                                aria-label="Generating response"
                                            >
                                                <span />
                                                <span />
                                                <span />
                                            </div>
                                        ) : (
                                            <div className="markdown-content">
                                                <ReactMarkdown
                                                    remarkPlugins={[
                                                        remarkGfm,
                                                        remarkMath,
                                                        remarkBreaks
                                                    ]}
                                                    rehypePlugins={[
                                                        rehypeKatex
                                                    ]}
                                                >
                                                    {normalizeMath(message.content)}
                                                </ReactMarkdown>
                                            </div>
                                        )}
                                    </div>

                                    {message.role === "assistant" && references.length > 0 && (
                                        <button
                                            type="button"
                                            className="message-references-trigger"
                                            onClick={() => setActiveReferences(references)}
                                            aria-label={`Show ${references.length} references`}
                                        >
                                            References ({references.length})
                                        </button>
                                    )}
                                </div>
                            </div>
                        );
                    })
                )}

                <div ref={messagesEndRef} />
            </div>

            {activeReferences && (
                <div
                    className="references-modal-backdrop"
                    role="presentation"
                    onMouseDown={event => {
                        if (event.target === event.currentTarget) {
                            setActiveReferences(null);
                        }
                    }}
                >
                    <section
                        className="references-modal"
                        role="dialog"
                        aria-modal="true"
                        aria-labelledby="references-modal-title"
                    >
                        <div className="references-modal-header">
                            <div>
                                <h2 id="references-modal-title">References</h2>
                                <p>{activeReferences.length} sources used for this response</p>
                            </div>
                            <button
                                type="button"
                                className="references-modal-close"
                                onClick={() => setActiveReferences(null)}
                                aria-label="Close references"
                            >
                                <X size={19} />
                            </button>
                        </div>

                        <div className="references-modal-list">
                            {activeReferences.map((reference, index) => {
                                const isWeb = reference.source_type === "web";
                                const safeUrl = isWeb ? getSafeWebUrl(reference.url) : null;
                                const title = reference.title || reference.source || (isWeb ? "Website source" : "Local knowledge source");
                                const domain = getDomain(reference.url);

                                return (
                                    <article className="reference-item" key={`${reference.source_type}-${reference.url || reference.source || title}-${index}`}>
                                        <div className="reference-item-icon">
                                            {isWeb ? <Globe size={17} /> : <FileText size={17} />}
                                        </div>
                                        <div className="reference-item-body">
                                            <div className="reference-item-kind">
                                                {isWeb ? "WEB SOURCE" : "LOCAL KNOWLEDGE BASE"}
                                            </div>
                                            {safeUrl ? (
                                                <a
                                                    className="reference-item-title"
                                                    href={safeUrl}
                                                    target="_blank"
                                                    rel="noopener noreferrer"
                                                >
                                                    {title}
                                                    <ExternalLink size={13} />
                                                </a>
                                            ) : (
                                                <div className="reference-item-title">{title}</div>
                                            )}
                                            <div className="reference-item-detail">
                                                {isWeb
                                                    ? (domain || "Website")
                                                    : (reference.topic || "Curated local reference")
                                                }
                                                {reference.published_date ? ` · ${reference.published_date}` : ""}
                                            </div>
                                        </div>
                                    </article>
                                );
                            })}
                        </div>
                    </section>
                </div>
            )}
        </div>
    );
}
