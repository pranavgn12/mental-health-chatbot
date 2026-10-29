
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel

from database import get_db
from models import Conversation, Message
from auth_utils import get_current_user

router = APIRouter()


class CreateChatRequest(BaseModel):
    title: str


@router.post("/new")
def create_chat(
    request: CreateChatRequest,
    db: Session = Depends(get_db),
    user=Depends(get_current_user)
):
    chat = Conversation(
        user_id=user["user_id"],
        title=request.title
    )

    db.add(chat)
    db.commit()
    db.refresh(chat)

    return {
        "id": chat.id,
        "user_id": chat.user_id,
        "title": chat.title
    }


@router.get("/")
def list_chats(
    db: Session = Depends(get_db),
    user=Depends(get_current_user)
):
    chats = db.query(Conversation).filter(
        Conversation.user_id == user["user_id"]
    ).all()

    return chats


@router.delete("/{chat_id}")
def delete_chat(
    chat_id: int,
    db: Session = Depends(get_db),
    user=Depends(get_current_user)
):
    # Only allow users to delete their own conversations.
    chat = db.query(Conversation).filter(
        Conversation.id == chat_id,
        Conversation.user_id == user["user_id"]
    ).first()

    if not chat:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found"
        )

    # Delete the conversation's messages first.
    db.query(Message).filter(
        Message.conversation_id == chat_id
    ).delete(synchronize_session=False)

    db.delete(chat)
    db.commit()

    return {
        "message": "Conversation deleted",
        "id": chat_id
    }

