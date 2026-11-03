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


# =========================================================
# NON-STREAMING MESSAGE ENDPOINT
# =========================================================

@router.post("/send")
def send_message(
    request: SendMessageRequest,
    db: Session = Depends(get_db),
    user=Depends(get_current_user)
):
    """Send a message and return the complete assistant response."""

    # 1. Validate conversation ownership.
    conversation = db.query(Conversation).filter(
        Conversation.id == request.conversation_id,
        Conversation.user_id == user["user_id"]
    ).first()

    if not conversation:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found"
        )

    try:

        # 2. Save user message.
        user_message = Message(
            conversation_id=request.conversation_id,
            role="user",
            content=request.content
        )

        db.add(user_message)
        db.commit()
        db.refresh(user_message)

        # 3. Load conversation history.
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

        # 4. Run orchestrator.
        orchestrator = AIOrchestrator()

        result = orchestrator.chat(
            db=db,
            user_id=user["user_id"],
            conversation=conversation_messages,
            latest_message=request.content,
            current_title=conversation.title,
            user_message_id=user_message.id
        )

        reply = result["reply"]
        new_title = result.get("title")

        # 5. Read episode and memory-update metadata.
        episode_id = result.get("episode_id")

        closed_episode_id = result.get(
            "closed_episode_id"
        )

        long_term_memory_updated = result.get(
            "long_term_memory_updated",
            False
        )

        # 6. Save assistant response.
        assistant_message = Message(
            conversation_id=request.conversation_id,
            role="assistant",
            content=reply,
            episode_id=(closed_episode_id or episode_id)
        )

        db.add(assistant_message)

        if new_title:
            conversation.title = new_title

        db.commit()

        db.refresh(assistant_message)
        db.refresh(conversation)

        # 7. Return response and memory-update status.
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

            "episode_id": episode_id,

            "closed_episode_id": closed_episode_id,

            "long_term_memory_updated": long_term_memory_updated,

            "rag": result.get("rag") or {}

        }

    except Exception:

        db.rollback()

        logger.exception(
            "Error while processing non-streaming message."
        )

        raise HTTPException(
            status_code=500,
            detail="Could not process the message."
        )


# =========================================================
# STREAMING MESSAGE ENDPOINT
# =========================================================

@router.post("/send-stream")
def send_message_stream(
    request: SendMessageRequest,
    db: Session = Depends(get_db),
    user=Depends(get_current_user)
):
    """Stream assistant response using Server-Sent Events."""

    # 1. Validate conversation ownership.
    conversation = db.query(Conversation).filter(
        Conversation.id == request.conversation_id,
        Conversation.user_id == user["user_id"]
    ).first()

    if not conversation:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found"
        )

    # 2. Save user message before starting the stream.
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

    # Store request information before the generator executes.
    conversation_id = conversation.id
    current_title = conversation.title
    user_id = user["user_id"]

    # 3. Load conversation history.
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

    # =====================================================
    # SSE EVENT GENERATOR
    # =====================================================

    def event_generator():

        stream_db = SessionLocal()

        reply_parts = []
        completion_metadata = {}

        done_received = False

        try:

            # 4. Get conversation using the streaming session.
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

            # 5. Start orchestrator streaming.
            for event in orchestrator.chat_stream(
                db=stream_db,
                user_id=user_id,
                conversation=conversation_messages,
                latest_message=request.content,
                current_title=current_title,
                user_message_id=user_message_data["id"]
            ):

                event_type = event.get("type")

                # -----------------------------------------
                # STREAM CHUNK
                # -----------------------------------------

                if event_type == "chunk":

                    chunk_text = event.get("content", "")

                    if chunk_text:

                        reply_parts.append(chunk_text)

                        yield format_sse({
                            "type": "chunk",
                            "content": chunk_text
                        })

                # -----------------------------------------
                # ORCHESTRATOR COMPLETION
                # -----------------------------------------

                elif event_type == "done":

                    completion_metadata = event

                    done_received = True

            # 6. Verify orchestrator completed successfully.
            if not done_received:

                raise RuntimeError(
                    "Orchestrator did not send a completion event."
                )

            # 7. Combine streamed chunks.
            reply = "".join(reply_parts)

            # 8. Save one complete assistant message.
            assistant_message = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=reply,
                episode_id=(completion_metadata.get("closed_episode_id") or completion_metadata.get("episode_id"))
            )

            stream_db.add(assistant_message)

            # 9. Update conversation title.
            new_title = completion_metadata.get("title")

            if new_title:
                conversation_stream.title = new_title

            # 10. Commit assistant message and title.
            stream_db.commit()

            stream_db.refresh(assistant_message)
            stream_db.refresh(conversation_stream)

            # 11. Extract episode information.
            episode_id = completion_metadata.get(
                "episode_id"
            )

            closed_episode_id = completion_metadata.get(
                "closed_episode_id"
            )

            long_term_memory_updated = completion_metadata.get(
                "long_term_memory_updated",
                False
            )

            # 12. Notify frontend that everything is complete.
            yield format_sse({

                "type": "done",

                "user_message": user_message_data,

                "assistant_message": {
                    "id": assistant_message.id,
                    "role": assistant_message.role,
                    "content": assistant_message.content
                },

                "conversation_title": conversation_stream.title,

                "episode_id": episode_id,

                "closed_episode_id": closed_episode_id,

                "long_term_memory_updated": long_term_memory_updated,

                "rag": completion_metadata.get("rag") or {}

            })

        except Exception:

            stream_db.rollback()

            logger.exception(
                "Error during streaming response."
            )

            yield format_sse({
                "type": "error",
                "message": (
                    "The response could not be completed. "
                    "Please try again."
                )
            })

        finally:

            stream_db.close()

    # 13. Return streaming response.
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )


# =========================================================
# GET CONVERSATION MESSAGES
# =========================================================

@router.get("/{conversation_id}")
def get_messages(
    conversation_id: int,
    db: Session = Depends(get_db),
    user=Depends(get_current_user)
):
    """Retrieve messages belonging to a conversation."""

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





####################################################################################################