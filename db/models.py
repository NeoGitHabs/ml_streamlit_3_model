from .database import Base
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime, timezone
from sqlalchemy import String, Integer, DateTime


class MnistModel(Base):
    __tablename__ = 'mnist'
    id: Mapped[int] = mapped_column(Integer, autoincrement=True, primary_key=True)
    image: Mapped[str] = mapped_column(String)
    label: Mapped[int] = mapped_column(Integer)
    created_date: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

class FashionModel(Base):
    __tablename__ = 'fashion'
    id: Mapped[int] = mapped_column(Integer, autoincrement=True, primary_key=True)
    image: Mapped[str] = mapped_column(String)
    label: Mapped[str] = mapped_column(String)
    created_date: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

class CifarModel(Base):
    __tablename__ = 'cifar_10'
    id: Mapped[int] = mapped_column(Integer, autoincrement=True, primary_key=True)
    image: Mapped[str] = mapped_column(String)
    label: Mapped[str] = mapped_column(String)
    created_date: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
