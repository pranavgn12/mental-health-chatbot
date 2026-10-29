from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    ForeignKey
)

from database import Base

from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from datetime import datetime



class User(Base):

    __tablename__ = "users"

    id = Column(
        Integer,
        primary_key=True
    )

    username = Column(
        String(50),
        unique=True
    )

    email = Column(
        String(100),
        unique=True
    )

    password_hash = Column(
        String(255)
    )


class Conversation(Base):

    __tablename__ = "conversations"

    id = Column(
        Integer,
        primary_key=True
    )

    user_id = Column(
        Integer,
        ForeignKey("users.id")
    )

    title = Column(String(255))

    system_prompt = Column(Text)


class Message(Base):

    __tablename__ = "messages"

    id = Column(
        Integer,
        primary_key=True
    )

    conversation_id = Column(
        Integer,
        ForeignKey("conversations.id")
    )

    role = Column(
        String(20)
    )

    content = Column(Text)



class Episode(Base):

    __tablename__ = "episodes"

    id = Column(Integer, primary_key=True, index=True)

    episode_id = Column(String(20), unique=True, nullable=False)

    user_id = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=False
    )

    title = Column(String(255), nullable=False)

    summary = Column(Text, nullable=True)

    status = Column(String(20), default="ACTIVE")

    created_at = Column(DateTime, default=datetime.utcnow)

    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )

    closed_at = Column(DateTime, nullable=True)
    
    messages_since_last_mood = Column(
        Integer,
        default=0,
        nullable=False
    )

    messages_since_last_reflection = Column(
        Integer,
        default=0,
        nullable=False
    )

class EpisodeMood(Base):

    __tablename__ = "episode_moods"

    id = Column(Integer, primary_key=True, index=True)

    episode_id = Column(
        String(20),
        ForeignKey("episodes.episode_id"),
        nullable=False
    )

    mood = Column(
        String(100),
        nullable=False
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

class EpisodeReflection(Base):

    __tablename__ = "episode_reflections"

    id = Column(Integer, primary_key=True, index=True)
    
    episode_id = Column(
        String(20),
        ForeignKey("episodes.episode_id"),
        nullable=False
    )

    reflection = Column(
        Text,
        nullable=False
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

class LongTermMemory(Base):

    __tablename__ = "long_term_memory"

    id = Column(

        Integer,

        primary_key=True,

        autoincrement=True

    )

    user_id = Column(

        Integer,

        ForeignKey("users.id"),

        nullable=False,

        unique=True

    )

    memory = Column(

        Text,

        nullable=False

    )

    updated_at = Column(

        DateTime,

        default=datetime.utcnow,

        onupdate=datetime.utcnow

    )