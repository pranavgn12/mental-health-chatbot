
import { useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkBreaks from "remark-breaks";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
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


export default function ChatWindow() {
    const { chatList, selectedChatId, messages } = useChat();
    const messagesEndRef = useRef(null);

    const selectedChat = chatList.find(
        chat => chat.id === selectedChatId
    );

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({
            behavior: "auto",
            block: "end"
        });
    }, [messages]);

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

                        return (
                            <div
                                key={message.id}
                                className={`message-row ${message.role}`}
                            >
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
                            </div>
                        );
                    })
                )}

                <div ref={messagesEndRef} />
            </div>
        </div>
    );
}

