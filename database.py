from sqlalchemy import create_engine, Column, Integer, String, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


Base = declarative_base()


class Chat(Base):
    __tablename__ = 'chats'
    id = Column(Integer, primary_key=True, autoincrement=True)
    chat_id = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    messages = Column(JSON, nullable=False)
    # New field for storing file paths
    file_paths = Column(JSON, nullable=True)


engine = create_engine('sqlite:///chats.db')
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()
