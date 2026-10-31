
import { useState, useRef, useEffect } from "react";
import {
    Paperclip,
    Mic,
    MicOff,
    SendHorizontal,
    LoaderCircle,
    X,
} from "lucide-react";

import useChat from "../hooks/useChat";
import api from "../services/api";

const MAX_FILE_SIZE = 10 * 1024 * 1024;

export default function MessageInput() {
    const [text, setText] = useState("");
    const [uploading, setUploading] = useState(false);
    const [uploadError, setUploadError] = useState("");
    const [uploadedFile, setUploadedFile] = useState("");
    const [listening, setListening] = useState(false);
    const [voiceError, setVoiceError] = useState("");

    const textareaRef = useRef(null);
    const fileInputRef = useRef(null);
    const recognitionRef = useRef(null);

    const textRef = useRef("");
    const listeningRef = useRef(false);
    const dictationBaseRef = useRef("");
    const finalTranscriptRef = useRef("");

    const { send, isSending } = useChat();

    // Keep the latest input value accessible to speech callbacks.
    useEffect(() => {
        textRef.current = text;
    }, [text]);

    // Automatically resize the textarea.
    useEffect(() => {
        const textarea = textareaRef.current;
        if (!textarea) return;

        textarea.style.height = "auto";

        const maxHeight = 180;
        const newHeight = Math.min(textarea.scrollHeight, maxHeight);

        textarea.style.height = `${newHeight}px`;
        textarea.style.overflowY =
            textarea.scrollHeight > maxHeight ? "auto" : "hidden";
    }, [text]);

    // Stop speech recognition when this component unmounts.
    useEffect(() => {
        return () => {
            listeningRef.current = false;

            if (recognitionRef.current) {
                recognitionRef.current.onresult = null;
                recognitionRef.current.onend = null;
                recognitionRef.current.onerror = null;

                try {
                    recognitionRef.current.abort();
                } catch {
                    // Recognition may already have stopped.
                }

                recognitionRef.current = null;
            }
        };
    }, []);

    function stopDictation() {
        listeningRef.current = false;
        setListening(false);

        try {
            recognitionRef.current?.stop();
        } catch {
            // Recognition may already have stopped.
        }
    }

    function toggleDictation() {
        if (listeningRef.current) {
            stopDictation();
            return;
        }

        const SpeechRecognition =
            window.SpeechRecognition ||
            window.webkitSpeechRecognition;

        if (!SpeechRecognition) {
            setVoiceError(
                "Voice dictation is not supported by this browser. Try Chrome."
            );
            return;
        }

        setVoiceError("");

        // Capture the text present before this dictation session.
        dictationBaseRef.current = textRef.current.trimEnd();
        finalTranscriptRef.current = "";

        const recognition = new SpeechRecognition();

        recognition.lang = "en-IN";
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.maxAlternatives = 1;

        recognition.onstart = () => {
            listeningRef.current = true;
            setListening(true);
        };

        recognition.onresult = (event) => {
            // Reconstruct the complete transcript for this session.
            // Do not append it to the previous textarea value.
            let transcript = "";

            for (let i = 0; i < event.results.length; i++) {
                transcript += event.results[i][0].transcript;
            }

            finalTranscriptRef.current = transcript.trim();

            const base = dictationBaseRef.current;
            const updatedText = [base, finalTranscriptRef.current]
                .filter(Boolean)
                .join(base ? " " : "");

            textRef.current = updatedText;
            setText(updatedText);
        };

        recognition.onerror = (event) => {
            if (event.error === "not-allowed") {
                setVoiceError(
                    "Microphone access denied. Allow microphone access in your browser settings."
                );
            } else if (event.error === "audio-capture") {
                setVoiceError("No microphone was detected.");
            } else if (
                event.error !== "no-speech" &&
                event.error !== "aborted"
            ) {
                setVoiceError(`Dictation error: ${event.error}`);
            }

            listeningRef.current = false;
            setListening(false);
        };

        recognition.onend = () => {
            listeningRef.current = false;
            setListening(false);

            if (recognitionRef.current === recognition) {
                recognitionRef.current = null;
            }
        };

        recognitionRef.current = recognition;

        try {
            recognition.start();
        } catch (error) {
            console.error("Could not start dictation:", error);

            setVoiceError("Could not start the microphone. Try again.");
            listeningRef.current = false;
            setListening(false);
            recognitionRef.current = null;
        }
    }

    async function handleSend() {
        const message = text.trim();

        if (!message || isSending || uploading) return;

        if (listeningRef.current) {
            stopDictation();
        }

        setText("");
        textRef.current = "";
        setUploadError("");
        setUploadedFile("");
        setVoiceError("");

        try {
            await send(message);
        } catch (error) {
            console.error("Failed to send message:", error);
            textRef.current = message;
            setText(message);
        }
    }

    async function handleFileChange(event) {
        const file = event.target.files?.[0];

        // Allow selecting the same file again later.
        event.target.value = "";

        if (!file) return;

        setUploadError("");
        setUploadedFile("");

        if (!/\.(pdf|docx|txt)$/i.test(file.name)) {
            setUploadError(
                "Unsupported file. Choose a PDF, DOCX, or TXT file."
            );
            return;
        }

        if (file.size > MAX_FILE_SIZE) {
            setUploadError("The maximum file size is 10 MB.");
            return;
        }

        if (file.size === 0) {
            setUploadError("The selected file is empty.");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        setUploading(true);

        try {
            const response = await api.post(
                "/documents/extract",
                formData
            );

            const extractedText = response.data?.text;

            if (
                typeof extractedText !== "string" ||
                !extractedText.trim()
            ) {
                throw new Error(
                    "No text was extracted from this document."
                );
            }

            const documentContent =
                `Document: ${file.name}\n\n${extractedText.trim()}`;

            setText((previous) => {
                const updated = previous.trim()
                    ? `${previous.trim()}\n\n${documentContent}`
                    : documentContent;

                textRef.current = updated;
                return updated;
            });

            setUploadedFile(file.name);
        } catch (error) {
            console.error("Document extraction failed:", error);

            setUploadError(
                error.response?.data?.detail ||
                error.message ||
                "Failed to extract document text."
            );
        } finally {
            setUploading(false);
        }
    }

    function handleKeyDown(event) {
        if (
            event.key === "Enter" &&
            !event.shiftKey &&
            !event.nativeEvent.isComposing
        ) {
            event.preventDefault();
            handleSend();
        }
    }

    return (
        <div className="message-input-wrapper">
            <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.docx,.txt"
                hidden
                onChange={handleFileChange}
            />

            {uploadedFile && (
                <div className="upload-status">
                    <Paperclip size={15} />

                    <span>Text extracted: {uploadedFile}</span>

                    <button
                        type="button"
                        aria-label="Dismiss upload status"
                        title="Dismiss"
                        onClick={() => setUploadedFile("")}
                    >
                        <X size={15} />
                    </button>
                </div>
            )}

            {uploadError && (
                <div className="upload-error" role="alert">
                    {uploadError}
                </div>
            )}

            {voiceError && (
                <div className="upload-error" role="alert">
                    <span>{voiceError}</span>

                    <button
                        type="button"
                        aria-label="Dismiss voice error"
                        onClick={() => setVoiceError("")}
                    >
                        <X size={15} />
                    </button>
                </div>
            )}

            <div className="message-input">
                <button
                    type="button"
                    aria-label="Attach document"
                    title="Attach PDF, DOCX, or TXT"
                    disabled={isSending || uploading}
                    onClick={() => fileInputRef.current?.click()}
                >
                    {uploading ? (
                        <LoaderCircle
                            size={20}
                            className="message-input-loader"
                        />
                    ) : (
                        <Paperclip size={20} />
                    )}
                </button>

                <textarea
                    ref={textareaRef}
                    rows={1}
                    placeholder={
                        listening
                            ? "Listening... speak now"
                            : uploading
                              ? "Extracting document text..."
                              : "Message Mental Health AI..."
                    }
                    value={text}
                    onChange={(event) => {
                        textRef.current = event.target.value;
                        setText(event.target.value);
                    }}
                    onKeyDown={handleKeyDown}
                    aria-label="Message"
                    disabled={uploading}
                />

                <button
                    type="button"
                    onClick={toggleDictation}
                    disabled={uploading || isSending}
                    aria-label={
                        listening
                            ? "Stop voice dictation"
                            : "Start voice dictation"
                    }
                    title={
                        listening
                            ? "Stop dictation"
                            : "Start voice dictation"
                    }
                    className={listening ? "mic-listening" : ""}
                >
                    {listening ? (
                        <MicOff size={20} />
                    ) : (
                        <Mic size={20} />
                    )}
                </button>

                <button
                    type="button"
                    onClick={handleSend}
                    disabled={!text.trim() || isSending || uploading}
                    aria-label={
                        isSending ? "Generating response" : "Send message"
                    }
                    title="Send message"
                >
                    {isSending ? (
                        <LoaderCircle
                            size={20}
                            className="message-input-loader"
                        />
                    ) : (
                        <SendHorizontal size={20} />
                    )}
                </button>
            </div>
        </div>
    );
}

