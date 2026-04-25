import datetime as dt
import json
from pathlib import Path
from typing import AsyncGenerator, Iterable

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker


DATABASE_URL = "sqlite+aiosqlite:///./cognirad.db"

engine = create_async_engine(DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
Base = declarative_base()


CHANNEL_SEED_DATA = (
    {"id": 1, "frequency": 2.412},
    {"id": 2, "frequency": 2.437},
    {"id": 3, "frequency": 2.462},
    {"id": 4, "frequency": 5.180},
    {"id": 5, "frequency": 5.240},
)

CHANNEL_STATUS_FREE = "FREE"
CHANNEL_STATUS_BUSY = "BUSY"
CHANNEL_STATUS_CONGESTED = "CONGESTED"
CHANNEL_STATUS_JAMMED = "JAMMED"


class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cms = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    channel_id = Column(Integer, nullable=True, index=True)
    is_active = Column(Boolean, default=False, nullable=False)
    joined_at = Column(DateTime, nullable=True)


class Channel(Base):
    __tablename__ = "channels"

    id = Column(Integer, primary_key=True)
    frequency = Column(Float, nullable=False)
    status = Column(String, default=CHANNEL_STATUS_FREE, nullable=False)
    user_count = Column(Integer, default=0, nullable=False)
    is_jammed = Column(Boolean, default=False, nullable=False)
    confidence = Column(Float, default=0.0, nullable=False)


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    channel_id = Column(Integer, nullable=False, index=True)
    cms = Column(String, nullable=False, index=True)
    recipient_cms = Column(String, nullable=True, index=True)
    student_name = Column(String, nullable=False)
    content = Column(String, nullable=False)
    message_type = Column(String, default="DM", nullable=False)
    timestamp = Column(DateTime, default=dt.datetime.utcnow, nullable=False, index=True)
    delivered_at = Column(DateTime, nullable=True)


class Session(Base):
    __tablename__ = "sessions"

    token = Column(String, primary_key=True)
    cms = Column(String, nullable=False, index=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


def _normalize_student_rows(raw_data: object) -> list[dict[str, str]]:
    if isinstance(raw_data, dict):
        return [{"cms": str(cms), "name": str(name)} for cms, name in raw_data.items()]

    if isinstance(raw_data, list):
        normalized: list[dict[str, str]] = []
        for entry in raw_data:
            if not isinstance(entry, dict):
                continue

            cms = entry.get("cms", entry.get("id"))
            name = entry.get("name", entry.get("student_name"))
            if cms is None or name is None:
                continue

            normalized.append({"cms": str(cms), "name": str(name)})
        return normalized

    raise ValueError("students.json must be either a CMS-name object or a list of student objects.")


async def init_db(students_json_path: str | Path = "students.json") -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    await ensure_schema()
    await seed_channels()
    await load_students_from_json(students_json_path)


async def ensure_schema() -> None:
    """
    Apply lightweight SQLite-compatible schema migrations for local dev.
    """
    async with engine.begin() as conn:
        result = await conn.execute(text("PRAGMA table_info(messages)"))
        existing_columns = {row[1] for row in result.fetchall()}

        if "recipient_cms" not in existing_columns:
            await conn.execute(text("ALTER TABLE messages ADD COLUMN recipient_cms VARCHAR"))
        if "message_type" not in existing_columns:
            await conn.execute(
                text("ALTER TABLE messages ADD COLUMN message_type VARCHAR NOT NULL DEFAULT 'DM'")
            )
        if "delivered_at" not in existing_columns:
            await conn.execute(text("ALTER TABLE messages ADD COLUMN delivered_at DATETIME"))


async def seed_channels() -> None:
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Channel.id))
        existing_channel_ids = set(result.scalars().all())

        missing_channels = [
            Channel(id=channel["id"], frequency=channel["frequency"])
            for channel in CHANNEL_SEED_DATA
            if channel["id"] not in existing_channel_ids
        ]

        if not missing_channels:
            return

        session.add_all(missing_channels)
        await session.commit()


async def load_students_from_json(filepath: str | Path = "students.json") -> int:
    student_file = Path(filepath)
    if not student_file.is_absolute():
        student_file = Path(__file__).resolve().parent / student_file

    if not student_file.exists():
        return 0

    with student_file.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)

    rows = _normalize_student_rows(raw_data)
    if not rows:
        return 0

    inserted_count = 0
    async with AsyncSessionLocal() as session:
        for entry in rows:
            result = await session.execute(select(Student).where(Student.cms == entry["cms"]))
            existing = result.scalar_one_or_none()
            if existing:
                if existing.name != entry["name"]:
                    existing.name = entry["name"]
                continue

            session.add(Student(cms=entry["cms"], name=entry["name"]))
            inserted_count += 1

        await session.commit()

    return inserted_count


async def get_student_by_cms(cms: str) -> Student | None:
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Student).where(Student.cms == str(cms)))
        return result.scalar_one_or_none()


async def get_all_students() -> list[Student]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Student).order_by(Student.name.asc()))
        return list(result.scalars().all())


async def get_all_channels() -> list[Channel]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Channel).order_by(Channel.id.asc()))
        return list(result.scalars().all())


async def get_channel(channel_id: int) -> Channel | None:
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Channel).where(Channel.id == channel_id))
        return result.scalar_one_or_none()


async def get_all_students() -> list[Student]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Student).order_by(Student.name.asc()))
        return list(result.scalars().all())


async def get_students_on_channel(channel_id: int, active_only: bool = True) -> list[Student]:
    async with AsyncSessionLocal() as session:
        query = select(Student).where(Student.channel_id == channel_id)
        if active_only:
            query = query.where(Student.is_active.is_(True))

        result = await session.execute(query.order_by(Student.joined_at.desc(), Student.id.desc()))
        return list(result.scalars().all())


async def assign_student_to_channel(cms: str, channel_id: int) -> Student | None:
    async with AsyncSessionLocal() as session:
        student_result = await session.execute(select(Student).where(Student.cms == str(cms)))
        student = student_result.scalar_one_or_none()
        if student is None:
            return None

        channel_result = await session.execute(select(Channel).where(Channel.id == channel_id))
        new_channel = channel_result.scalar_one_or_none()
        if new_channel is None:
            return None

        old_channel = None
        old_channel_id = student.channel_id
        if old_channel_id is not None and old_channel_id != channel_id:
            old_channel_result = await session.execute(select(Channel).where(Channel.id == old_channel_id))
            old_channel = old_channel_result.scalar_one_or_none()

        if old_channel is not None:
            old_channel.user_count = max(0, old_channel.user_count - 1)

        if old_channel_id != channel_id or not student.is_active:
            new_channel.user_count += 1

        student.channel_id = channel_id
        student.is_active = True
        student.joined_at = dt.datetime.utcnow()

        await session.commit()
        await session.refresh(student)
        return student


async def remove_student_from_channel(cms: str) -> Student | None:
    async with AsyncSessionLocal() as session:
        student_result = await session.execute(select(Student).where(Student.cms == str(cms)))
        student = student_result.scalar_one_or_none()
        if student is None:
            return None

        if student.channel_id is not None:
            channel_result = await session.execute(select(Channel).where(Channel.id == student.channel_id))
            channel = channel_result.scalar_one_or_none()
            if channel is not None and student.is_active:
                channel.user_count = max(0, channel.user_count - 1)

        student.channel_id = None
        student.is_active = False
        student.joined_at = None

        await session.commit()
        await session.refresh(student)
        return student


async def move_student(cms: str, old_channel_id: int, new_channel_id: int) -> bool:
    async with AsyncSessionLocal() as session:
        student_result = await session.execute(select(Student).where(Student.cms == str(cms)))
        student = student_result.scalar_one_or_none()

        old_channel_result = await session.execute(select(Channel).where(Channel.id == old_channel_id))
        old_channel = old_channel_result.scalar_one_or_none()

        new_channel_result = await session.execute(select(Channel).where(Channel.id == new_channel_id))
        new_channel = new_channel_result.scalar_one_or_none()

        if student is None or old_channel is None or new_channel is None:
            return False

        if student.channel_id != old_channel_id:
            return False

        student.channel_id = new_channel_id
        student.is_active = True
        student.joined_at = dt.datetime.utcnow()
        old_channel.user_count = max(0, old_channel.user_count - 1)
        new_channel.user_count += 1

        await session.commit()
        return True


async def move_students(cms_values: Iterable[str], new_channel_id: int) -> list[str]:
    moved_students: list[str] = []
    for cms in cms_values:
        student = await get_student_by_cms(str(cms))
        if student is None or student.channel_id is None:
            continue

        moved = await move_student(str(cms), student.channel_id, new_channel_id)
        if moved:
            moved_students.append(str(cms))

    return moved_students


async def update_channel_status(
    channel_id: int,
    status: str,
    confidence: float,
    is_jammed: bool = False,
) -> Channel | None:
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Channel).where(Channel.id == channel_id))
        channel = result.scalar_one_or_none()
        if channel is None:
            return None

        channel.status = status
        channel.confidence = confidence
        channel.is_jammed = is_jammed

        await session.commit()
        await session.refresh(channel)
        return channel


async def refresh_channel_user_counts() -> None:
    async with AsyncSessionLocal() as session:
        channels_result = await session.execute(select(Channel))
        channels = list(channels_result.scalars().all())

        for channel in channels:
            count_result = await session.execute(
                select(Student).where(
                    Student.channel_id == channel.id,
                    Student.is_active.is_(True),
                )
            )
            channel.user_count = len(count_result.scalars().all())

        await session.commit()


async def save_message(
    channel_id: int,
    cms: str,
    student_name: str,
    content: str,
    *,
    recipient_cms: str | None = None,
    message_type: str = "DM",
    delivered_at: dt.datetime | None = None,
) -> Message:
    async with AsyncSessionLocal() as session:
        message = Message(
            channel_id=channel_id,
            cms=str(cms),
            recipient_cms=str(recipient_cms) if recipient_cms else None,
            student_name=student_name,
            content=content,
            message_type=message_type,
            delivered_at=delivered_at,
        )
        session.add(message)
        await session.commit()
        await session.refresh(message)
        return message


async def get_recent_messages(channel_id: int, limit: int = 30) -> list[Message]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Message)
            .where(Message.channel_id == channel_id)
            .order_by(Message.timestamp.desc(), Message.id.desc())
            .limit(limit)
        )
        messages = list(result.scalars().all())
        return list(reversed(messages))


async def get_recent_messages_across_channels(limit: int = 30) -> list[Message]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Message).order_by(Message.timestamp.desc(), Message.id.desc()).limit(limit)
        )
        messages = list(result.scalars().all())
        return list(reversed(messages))


async def create_session(token: str, cms: str, invalidate_existing: bool = False) -> Session:
    async with AsyncSessionLocal() as session:
        if invalidate_existing:
            await session.execute(delete(Session).where(Session.cms == str(cms)))

        existing = await session.execute(select(Session).where(Session.token == token))
        current_session = existing.scalar_one_or_none()

        if current_session is None:
            current_session = Session(token=token, cms=str(cms))
            session.add(current_session)
        else:
            current_session.cms = str(cms)
            current_session.created_at = dt.datetime.utcnow()

        await session.commit()
        await session.refresh(current_session)
        return current_session


async def get_cms_from_token(token: str) -> str | None:
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Session).where(Session.token == token))
        session_row = result.scalar_one_or_none()
        return session_row.cms if session_row else None


async def get_session(token: str) -> Session | None:
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Session).where(Session.token == token))
        return result.scalar_one_or_none()


async def delete_session(token: str) -> bool:
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Session).where(Session.token == token))
        session_row = result.scalar_one_or_none()
        if session_row is None:
            return False

        await session.delete(session_row)
        await session.commit()
        return True


async def delete_sessions_for_cms(cms: str) -> int:
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Session).where(Session.cms == str(cms)))
        sessions = list(result.scalars().all())
        for session_row in sessions:
            await session.delete(session_row)

        await session.commit()
        return len(sessions)
