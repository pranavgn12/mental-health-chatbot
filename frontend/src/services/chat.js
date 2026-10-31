
import api from "./api";

export async function getChats() {
    const response = await api.get("/chats");
    return response.data;
}

export async function createChat(title) {
    const response = await api.post("/chats/new", {
        title,
    });

    return response.data;
}

export async function deleteChat(chatId) {
    const response = await api.delete(`/chats/${chatId}`);
    return response.data;
}

