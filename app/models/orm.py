"""SQLAlchemy ORM tables: users, items, interactions, recommendations."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class User(Base):
    __tablename__ = "users"

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    country: Mapped[str | None] = mapped_column(String(8), nullable=True)
    age_group: Mapped[str | None] = mapped_column(String(16), nullable=True)
    signup_date: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Item(Base):
    __tablename__ = "items"

    item_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    title: Mapped[str] = mapped_column(String(255))
    genres: Mapped[str] = mapped_column(String(255))  # pipe separated, e.g. "sci-fi|thriller"
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    release_year: Mapped[int | None] = mapped_column(Integer, nullable=True)


class Interaction(Base):
    __tablename__ = "interactions"
    __table_args__ = (
        UniqueConstraint("user_id", "item_id", name="uq_interaction_user_item"),
        Index("ix_interactions_user_ts", "user_id", "timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id"), index=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("items.item_id"), index=True)
    event_type: Mapped[str] = mapped_column(String(32))
    weight: Mapped[float] = mapped_column(Float)
    timestamp: Mapped[datetime] = mapped_column(DateTime)


class Recommendation(Base):
    """Output of the OFFLINE batch job; read by the ONLINE serving layer."""

    __tablename__ = "recommendations"
    __table_args__ = (Index("ix_recs_user_model", "user_id", "model_name", "model_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id"))
    item_id: Mapped[int] = mapped_column(ForeignKey("items.item_id"))
    model_name: Mapped[str] = mapped_column(String(32))
    model_version: Mapped[str] = mapped_column(String(64))
    rank: Mapped[int] = mapped_column(Integer)
    score: Mapped[float] = mapped_column(Float)
    reason_type: Mapped[str] = mapped_column(String(32))
    explanation: Mapped[str] = mapped_column(Text)
    generated_at: Mapped[datetime] = mapped_column(DateTime)
