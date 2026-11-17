from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db
from models import Conversation
from auth_utils import get_current_user

from pydantic import BaseModel

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