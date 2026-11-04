
import { useContext, useRef, useState } from "react";

import { AppContext } from "../context/AppContext";

import {
    getChats,
    createChat,
    deleteChat as deleteChatRequest
} from "../services/chat";

import {
    getMessages,
    sendMessageStream
} from "../services/messages";

export default function useChat() {
    const {
        chatList,
        setChatList,
        selectedChatId,
        setSelectedChatId,
        messages,
        setMessages
    } = useContext(AppContext);

    const [isSending, setIsSending] = useState(false);
    const isSendingRef = useRef(false);

    // Track the currently selected conversation for async operations.
    const selectedChatIdRef = useRef(selectedChatId);
    selectedChatIdRef.current = selectedChatId;

    // =====================================================
    // LOAD CHATS
    // =====================================================

    async function loadChats() {
        try {
            const chats = await getChats();

            setChatList(chats);

            if (
                chats.length > 0 &&
                selectedChatIdRef.current === null
            ) {
                await openChat(chats[0].id);
            }

        } catch (err) {
            console.error("Failed to load chats:", err);
        }
    }

    // =====================================================
    // OPEN CHAT
    // =====================================================

    async function openChat(chatId) {
        try {
            selectedChatIdRef.current = chatId;
            setSelectedChatId(chatId);

            const history = await getMessages(chatId);

            // Do not replace messages if the user switched chats
            // while this request was in progress.
            if (selectedChatIdRef.current === chatId) {
                setMessages(history);
            }

        } catch (err) {
            console.error("Failed to open chat:", err);
        }
    }

    // =====================================================
    // CREATE NEW CHAT
    // =====================================================

    async function newChat() {
        try {
            const chat = await createChat("New Conversation");

            setChatList(prev => [
                chat,
                ...prev
            ]);

            selectedChatIdRef.current = chat.id;
            setSelectedChatId(chat.id);
            setMessages([]);

        } catch (err) {
            console.error("Failed to create chat:", err);
        }
    }

    // =====================================================
    // DELETE CHAT
    // =====================================================

    async function removeChat(chatId) {
        try {
            // Delete on server before changing local state.
            await deleteChatRequest(chatId);

            const remainingChats = chatList.filter(
                chat => chat.id !== chatId
            );

            setChatList(remainingChats);

            // If another conversation is selected,
            // leave it open.
            if (selectedChatIdRef.current !== chatId) {
                return;
            }

            if (remainingChats.length > 0) {
                await openChat(remainingChats[0].id);
            } else {
                selectedChatIdRef.current = null;
                setSelectedChatId(null);
                setMessages([]);
            }

        } catch (err) {
            console.error("Failed to delete conversation:", err);
            throw err;
        }
    }

    // =====================================================
    // SEND MESSAGE
    // =====================================================

    async function send(text) {
        const messageText = text.trim();
        const chatId = selectedChatIdRef.current;

        if (!chatId || !messageText) {
            return;
        }

        // Prevent overlapping requests.
        if (isSendingRef.current) {
            return;
        }

        isSendingRef.current = true;
        setIsSending(true);

        const userTempId =
            `temp-user-${Date.now()}-${Math.random()}`;

        const assistantTempId =
            `temp-assistant-${Date.now()}-${Math.random()}`;

        // Optimistically display user message and assistant bubble.
        setMessages(prev => [
            ...prev,
            {
                id: userTempId,
                role: "user",
                content: messageText
            },
            {
                id: assistantTempId,
                role: "assistant",
                content: ""
            }
        ]);

        let completed = false;
        let streamError = false;

        try {
            await sendMessageStream(
                chatId,
                messageText,
                event => {

                    // -------------------------------------
                    // STREAM CHUNK
                    // -------------------------------------

                    if (event.type === "chunk") {
                        const chunk = event.content || "";

                        if (!chunk) {
                            return;
                        }

                        setMessages(prev => {
                            if (
                                selectedChatIdRef.current !== chatId
                            ) {
                                return prev;
                            }

                            return prev.map(message =>
                                message.id === assistantTempId
                                    ? {
                                        ...message,
                                        content:
                                            message.content + chunk
                                    }
                                    : message
                            );
                        });
                    }

                    // -------------------------------------
                    // STREAM COMPLETED
                    // -------------------------------------

                    else if (event.type === "done") {
                        completed = true;

                        const completedReferences =
                            event.assistant_message?.references ||
                            event.rag?.sources ||
                            [];

                        setMessages(prev => {
                            if (selectedChatIdRef.current !== chatId) {
                                return prev;
                            }

                            return prev.map(message =>
                                message.id === assistantTempId
                                    ? { ...message, references: completedReferences }
                                    : message
                            );
                        });

                        // Update conversation title.
                        if (event.conversation_title) {
                            setChatList(prev =>
                                prev.map(chat =>
                                    chat.id === chatId
                                        ? {
                                            ...chat,
                                            title: event.conversation_title
                                        }
                                        : chat
                                )
                            );
                        }

                        // ---------------------------------
                        // PROFILE MEMORY UPDATE EVENT
                        // ---------------------------------

                        // The backend sends this flag only after
                        // long-term memory has been saved and
                        // the episode has been closed.
                        if (
                            event.long_term_memory_updated === true
                        ) {
                            console.log(
                                "Long-term memory updated:",
                                event.closed_episode_id
                            );

                            window.dispatchEvent(
                                new CustomEvent(
                                    "long-term-memory-updated",
                                    {
                                        detail: {
                                            closedEpisodeId:
                                                event.closed_episode_id || null
                                        }
                                    }
                                )
                            );
                        }
                    }

                    // -------------------------------------
                    // STREAM ERROR
                    // -------------------------------------

                    else if (event.type === "error") {
                        streamError = true;

                        console.error(
                            "Streaming error:",
                            event.message
                        );

                        setMessages(prev => {
                            if (
                                selectedChatIdRef.current !== chatId
                            ) {
                                return prev;
                            }

                            return prev.map(message =>
                                message.id === assistantTempId
                                    ? {
                                        ...message,
                                        content: message.content
                                            ? message.content +
                                                "\n\n[Response interrupted. Please try again.]"
                                            : "The response could not be completed. Please try again."
                                    }
                                    : message
                            );
                        });
                    }
                }
            );

            // ---------------------------------------------
            // REFRESH CHAT HISTORY
            // ---------------------------------------------

            if (completed && !streamError) {
                const history = await getMessages(chatId);

                if (selectedChatIdRef.current === chatId) {
                    setMessages(history);
                }

            } else if (!streamError) {
                throw new Error(
                    "The response stream ended before completion."
                );
            }

        } catch (err) {
            console.error("Failed to send message:", err);

            setMessages(prev => {
                if (selectedChatIdRef.current !== chatId) {
                    return prev;
                }

                return prev.map(message =>
                    message.id === assistantTempId &&
                    !message.content
                        ? {
                            ...message,
                            content:
                                "The response could not be loaded. Please try again."
                        }
                        : message
                );
            });

        } finally {
            isSendingRef.current = false;
            setIsSending(false);
        }
    }

    // =====================================================
    // RETURN
    // =====================================================

    return {
        chatList,
        selectedChatId,
        messages,
        isSending,

        loadChats,
        openChat,
        newChat,
        send,
        removeChat
    };
}

