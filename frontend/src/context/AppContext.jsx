import { createContext, useState } from "react";

export const AppContext = createContext();

export function AppProvider({ children }) {

    // Logged-in user
    const [user, setUser] = useState(null);

    // JWT (optional, since you're using localStorage)
    const [token, setToken] = useState(
        localStorage.getItem("token")
    );

    // Sidebar
    const [chatList, setChatList] = useState([]);

    // Currently selected chat
    const [selectedChatId, setSelectedChatId] = useState(null);

    // Messages of selected chat
    const [messages, setMessages] = useState([]);

    // Current LLM
    const [model, setModel] = useState("Gemini");

    // System Prompt
    const [systemPrompt, setSystemPrompt] = useState("");

    // Multimodal Toggle
    const [multimodalEnabled, setMultimodalEnabled] = useState(true);

    return (

        <AppContext.Provider

            value={{

                user,
                setUser,

                token,
                setToken,

                chatList,
                setChatList,

                selectedChatId,
                setSelectedChatId,

                messages,
                setMessages,

                model,
                setModel,

                systemPrompt,
                setSystemPrompt,

                multimodalEnabled,
                setMultimodalEnabled

            }}

        >

            {children}

        </AppContext.Provider>

    );

}