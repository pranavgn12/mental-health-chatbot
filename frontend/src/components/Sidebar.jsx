import { useEffect } from "react";

import {
    Plus,
    MessageSquare,
    X,
    UserRound,
} from "lucide-react";

import useChat from "../hooks/useChat";

export default function Sidebar({
    open,
    onOpenUserPanel,
}) {
    const {
        chatList,
        selectedChatId,
        loadChats,
        openChat,
        newChat,
        removeChat,
    } = useChat();

    useEffect(() => {
        loadChats();
    }, []);

    return (
        <div className={`sidebar ${open ? "open" : "closed"}`}>
            {/* New conversation */}
            <button
                type="button"
                className="new-chat"
                onClick={newChat}
                title="New Chat"
            >
                <Plus size={18} />
                <span>New Chat</span>
            </button>

            {/* Conversation list */}
            <div className="chat-list">
                {chatList.length === 0 ? (
                    <div className="empty-chat-list">
                        No conversations
                    </div>
                ) : (
                    chatList.map((chat) => (
                        <div
                            key={chat.id}
                            className={`chat-item ${
                                selectedChatId === chat.id
                                    ? "active"
                                    : ""
                            }`}
                            onClick={() => openChat(chat.id)}
                        >
                            <MessageSquare size={18} />

                            <span className="chat-title">
                                {chat.title}
                            </span>

                            <button
                                type="button"
                                className="delete-chat-btn"
                                title="Delete conversation"
                                aria-label={`Delete ${chat.title}`}
                                onClick={(event) => {
                                    event.stopPropagation();
                                    removeChat(chat.id);
                                }}
                            >
                                <X size={16} />
                            </button>
                        </div>
                    ))
                )}
            </div>

            {/* User button at the bottom */}
            <div className="sidebar-footer">
                <button
                    type="button"
                    className="sidebar-user-button"
                    onClick={onOpenUserPanel}
                    title="Open your personal memory"
                    aria-label="Open your personal memory"
                >
                    <span className="sidebar-user-avatar">
                        <UserRound size={20} />
                    </span>

                    <span className="sidebar-user-details">
                        <span className="sidebar-user-name">
                            Your Account
                        </span>

                        <span className="sidebar-user-subtitle">
                            Episodes · Moods · Reflections
                        </span>
                    </span>
                </button>
            </div>
        </div>
    );
}
