
import api from "./api";


export async function getMessages(conversationId) {
    const response = await api.get(
        `/messages/${conversationId}`
    );

    return response.data;
}


export async function sendMessage(
    conversation_id,
    content
) {
    const response = await api.post(
        "/messages/send",
        {
            conversation_id,
            content
        }
    );

    return response.data;
}


/**
 * Send a message and read the assistant response incrementally.
 *
 * onEvent receives events such as:
 *   { type: "chunk", content: "Hello" }
 *   { type: "done", assistant_message: {...} }
 *   { type: "error", message: "..." }
 */
export async function sendMessageStream(
    conversation_id,
    content,
    onEvent
) {
    const token = localStorage.getItem("token");

    const baseURL = api.defaults.baseURL.replace(/\/$/, "");

    const response = await fetch(
        `${baseURL}/messages/send-stream`,
        {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                ...(token
                    ? {
                        Authorization: `Bearer ${token}`
                    }
                    : {})
            },
            body: JSON.stringify({
                conversation_id,
                content
            })
        }
    );

    if (!response.ok) {
        let detail = `Request failed with status ${response.status}`;

        try {
            const data = await response.json();
            detail = data.detail || detail;
        } catch {
            // Keep the default error message if the body is not JSON.
        }

        throw new Error(detail);
    }

    if (!response.body) {
        throw new Error(
            "The browser does not support reading this response stream."
        );
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    let buffer = "";

    function processEventBlock(block) {
        const dataLines = block
            .split(/\r?\n/)
            .filter(line => line.startsWith("data:"))
            .map(line => line.slice(5).replace(/^ /, ""));

        if (dataLines.length === 0) {
            return;
        }

        const data = dataLines.join("\n");

        try {
            const event = JSON.parse(data);

            if (typeof onEvent === "function") {
                onEvent(event);
            }
        } catch (error) {
            console.error(
                "Failed to parse streaming event:",
                error,
                data
            );

            throw error;
        }
    }

    function processAvailableEvents() {
        while (true) {
            const boundary = /\r?\n\r?\n/.exec(buffer);

            if (!boundary) {
                break;
            }

            const block = buffer.slice(0, boundary.index);

            buffer = buffer.slice(
                boundary.index + boundary[0].length
            );

            if (block.trim()) {
                processEventBlock(block);
            }
        }
    }

    try {
        while (true) {
            const { value, done } = await reader.read();

            if (done) {
                break;
            }

            buffer += decoder.decode(value, {
                stream: true
            });

            // Process complete events without waiting for the full reply.
            processAvailableEvents();
        }

        // Flush any bytes still held by the decoder.
        buffer += decoder.decode();

        // Process the final event if it wasn't terminated by a blank line.
        if (buffer.trim()) {
            processEventBlock(buffer);
        }
    } finally {
        reader.releaseLock();
    }
}

