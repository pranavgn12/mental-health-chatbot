
import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel

from database import get_db, SessionLocal
from models import Message, Conversation
from auth_utils import get_current_user
from services.ai.orchestrator import AIOrchestrator

router = APIRouter()

logger = logging.getLogger(__name__)


class SendMessageRequest(BaseModel):
    conversation_id: int
    content: str


def format_sse(event: dict) -> str:
    """Format a dictionary as a Server-Sent Event."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("/send")
def send_message(
    request: SendMessageRequest,
    db: Session = Depends(get_db),
    user=Depends(get_current_user)
):
    """Existing non-streaming endpoint."""

    conversation = db.query(Conversation).filter(
        Conversation.id == request.conversation_id,
        Conversation.user_id == user["user_id"]
    ).first()

    if not conversation:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found"
        )

    # Save user message.
    user_message = Message(
        conversation_id=request.conversation_id,
        role="user",
        content=request.content
    )

    db.add(user_message)
    db.commit()
    db.refresh(user_message)

    # Load conversation history, including the new user message.
    history = db.query(Message).filter(
        Message.conversation_id == request.conversation_id
    ).order_by(Message.id).all()

    conversation_messages = [
        {
            "role": message.role,
            "content": message.content
        }
        for message in history
    ]

    # Generate the response.
    orchestrator = AIOrchestrator()

    result = orchestrator.chat(
        db=db,
        user_id=user["user_id"],
        conversation=conversation_messages,
        latest_message=request.content,
        current_title=conversation.title
    )

    reply = result["reply"]
    new_title = result["title"]
    episode = result["episode"]

    # Save the complete assistant response.
    assistant_message = Message(
        conversation_id=request.conversation_id,
        role="assistant",
        content=reply
    )

    db.add(assistant_message)

    if new_title:
        conversation.title = new_title

    db.commit()
    db.refresh(assistant_message)
    db.refresh(conversation)

    return {
        "user_message": {
            "id": user_message.id,
            "role": user_message.role,
            "content": user_message.content
        },
        "assistant_message": {
            "id": assistant_message.id,
            "role": assistant_message.role,
            "content": assistant_message.content
        },
        "conversation_title": conversation.title,
        "episode_id": episode.episode_id if episode else None
    }


@router.post("/send-stream")
def send_message_stream(
    request: SendMessageRequest,
    db: Session = Depends(get_db),
    user=Depends(get_current_user)
):
    """Stream an assistant response using Server-Sent Events."""

    # Validate conversation ownership before starting the stream.
    conversation = db.query(Conversation).filter(
        Conversation.id == request.conversation_id,
        Conversation.user_id == user["user_id"]
    ).first()

    if not conversation:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found"
        )

    # Save the user message before streaming.
    user_message = Message(
        conversation_id=request.conversation_id,
        role="user",
        content=request.content
    )

    db.add(user_message)
    db.commit()
    db.refresh(user_message)

    user_message_data = {
        "id": user_message.id,
        "role": user_message.role,
        "content": user_message.content
    }

    conversation_id = conversation.id
    current_title = conversation.title
    user_id = user["user_id"]

    # Load history, including the newly saved user message.
    history = db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.id).all()

    conversation_messages = [
        {
            "role": message.role,
            "content": message.content
        }
        for message in history
    ]

    def event_generator():
        # Use a dedicated session for the lifetime of the stream.
        # This avoids relying on the request-scoped session remaining open
        # while StreamingResponse is sending data.
        stream_db = SessionLocal()

        reply_parts = []
        completion_metadata = {}
        conversation_stream = None

        try:
            conversation_stream = stream_db.query(
                Conversation
            ).filter(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id
            ).first()

            if conversation_stream is None:
                yield format_sse({
                    "type": "error",
                    "message": "Conversation not found."
                })
                return

            orchestrator = AIOrchestrator()

            # The orchestrator yields chunks first and a done event
            # after its post-processing has completed.
            for event in orchestrator.chat_stream(
                db=stream_db,
                user_id=user_id,
                conversation=conversation_messages,
                latest_message=request.content,
                current_title=current_title
            ):
                if event["type"] == "chunk":
                    chunk_text = event.get("content", "")

                    if chunk_text:
                        reply_parts.append(chunk_text)

                        # Send immediately; do not wait for the full reply.
                        yield format_sse({
                            "type": "chunk",
                            "content": chunk_text
                        })

                elif event["type"] == "done":
                    completion_metadata = event

            # Combine all chunks into the complete assistant response.
            reply = "".join(reply_parts)

            # Save exactly one complete assistant message.
            assistant_message = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=reply
            )

            stream_db.add(assistant_message)

            # Update the title if the orchestrator generated one.
            new_title = completion_metadata.get("title")

            if new_title:
                conversation_stream.title = new_title

            # Persist the assistant message and title together.
            stream_db.commit()
            stream_db.refresh(assistant_message)
            stream_db.refresh(conversation_stream)

            # Tell the frontend that streaming and saving are complete.
            yield format_sse({
                "type": "done",
                "user_message": user_message_data,
                "assistant_message": {
                    "id": assistant_message.id,
                    "role": assistant_message.role,
                    "content": assistant_message.content
                },
                "conversation_title": conversation_stream.title,
                "episode_id": completion_metadata.get("episode_id")
            })

        except Exception:
            stream_db.rollback()
            logger.exception("Error during streaming response")

            # The HTTP response has already started, so report the
            # error as an SSE event instead of raising HTTPException.
            yield format_sse({
                "type": "error",
                "message": (
                    "The response could not be completed. "
                    "Please try again."
                )
            })

        finally:
            stream_db.close()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/{conversation_id}")
def get_messages(
    conversation_id: int,
    db: Session = Depends(get_db),
    user=Depends(get_current_user)
):
    """Retrieve the messages in a conversation."""

    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == user["user_id"]
    ).first()

    if not conversation:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found"
        )

    messages = db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.id).all()

    return messages

