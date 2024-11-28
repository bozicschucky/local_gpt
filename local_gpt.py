import os
import streamlit as st
from database import Chat, session
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv

from utils import process_pdf, process_image


# Load environment variables
load_dotenv(
    override=True,
)

# Directory to save uploaded files
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
llm_model = os.getenv("llm_model")



class StreamlitMessageHistory:
    def __init__(self, chat_id):
        self.chat_id = chat_id
        if 'chats' not in st.session_state:
            st.session_state.chats = {}
        if self.chat_id not in st.session_state.chats:
            st.session_state.chats[self.chat_id] = {
                "messages": [], "message_history": [], "file_paths": []}

    def add_message(self, message):
        if 'chats' not in st.session_state:
            st.session_state.chats = {}
        st.session_state.chats[self.chat_id]["message_history"].append(message)

    def add_messages(self, messages):
        for message in messages:
            self.add_message(message)

    def clear(self):
        if 'chats' not in st.session_state:
            st.session_state.chats = {}
        st.session_state.chats[self.chat_id]["message_history"] = []

    def get_messages(self):
        if 'chats' not in st.session_state:
            st.session_state.chats = {}
        if self.chat_id not in st.session_state.chats:
            st.session_state.chats[self.chat_id] = {
                "messages": [], "message_history": [], "file_paths": []}
        return st.session_state.chats[self.chat_id]["message_history"]

    @property
    def messages(self):
        return self.get_messages()


@st.cache_resource
def init_llm(document_content=""):
    llm = OllamaLLM(model=llm_model)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful AI assistant."),
        SystemMessage(content=document_content),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    chain = prompt | llm

    def get_session_history(chat_id):
        return StreamlitMessageHistory(chat_id)

    return lambda chat_id: RunnableWithMessageHistory(
        chain,
        lambda: get_session_history(chat_id),
        input_messages_key="input",
        history_messages_key="history"
    )

# Add this function to load and display saved files


# Add this function to load and initialize documents
def initialize_chat_with_documents(chat_id):
    """Initialize chat with saved documents and display them"""
    chat_data = st.session_state.chats[chat_id]
    document_content = []

    if chat_data.get("file_paths"):
        for file_path in chat_data["file_paths"]:
            if os.path.exists(file_path):
                if file_path.endswith('.pdf'):
                    result = process_pdf(file_path)
                    # Add PDF filename to chat if not already present
                    display_content = f"📄 PDF file: {
                        os.path.basename(file_path)}"
                    if not any(msg.get("content") == display_content for msg in chat_data["messages"]):
                        chat_data["messages"].append({
                            "role": "assistant",
                            "content": display_content,
                            "full_content": result["context"]
                        })
                    document_content.append(result["context"])

                elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    result = process_image(file_path)
                    if not any(msg.get("content") == result["context"] for msg in chat_data["messages"]):
                        chat_data["messages"].append({
                            "role": "assistant",
                            "content": result["context"]
                        })
                    document_content.append(result["context"])

    # Initialize conversation with combined document content
    return init_llm("\n".join(document_content))(chat_id)


# Initialize session state
if 'chats' not in st.session_state:
    st.session_state.chats = {}
    # Load chats from the database
    for chat in session.query(Chat).all():
        st.session_state.chats[chat.chat_id] = {
            "messages": chat.messages,
            "message_history": [HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]) for msg in chat.messages],
            "name": chat.name,
            "file_paths": chat.file_paths or []
        }

# Update the session state initialization
if 'current_chat' not in st.session_state:
    chat_ids = list(st.session_state.chats.keys())
    st.session_state.current_chat = chat_ids[0] if chat_ids else None
    if st.session_state.current_chat:
        st.session_state.conversation = initialize_chat_with_documents(st.session_state.current_chat)

# Initialize conversation if not already done
if st.session_state.current_chat is None:
    st.session_state.current_chat = "chat_1"
    st.session_state.chats[st.session_state.current_chat] = {
        "messages": [], "message_history": [], "file_paths": [], "name": "Chat 1"}
    st.session_state.conversation = init_llm()(st.session_state.current_chat)


# Add this function before the sidebar code
def handle_chat_selection():
    if "selected_chat" not in st.session_state:
        return

    if st.session_state.selected_chat != st.session_state.current_chat:
        st.session_state.current_chat = st.session_state.selected_chat
        st.session_state.conversation = initialize_chat_with_documents(
            st.session_state.current_chat)


# Sidebar for managing chats
with st.sidebar:
    # Create two columns for the buttons
    col1, col2 = st.columns(2)

    # Add Chat button in left column
    with col1:
        if st.button("Add Chat"):
            new_chat_id = f"chat_{len(st.session_state.chats) + 1}"
            st.session_state.chats[new_chat_id] = {
                "messages": [], "message_history": [], "name": f"Chat {len(st.session_state.chats) + 1}", "file_paths": []}
            st.session_state.current_chat = new_chat_id
            st.session_state.conversation = init_llm()(new_chat_id)

    # Delete Chat button in right column
    with col2:
        if st.button("Delete Chat"):
            if st.session_state.current_chat:
                chat_id_to_delete = st.session_state.current_chat
                del st.session_state.chats[chat_id_to_delete]
                session.query(Chat).filter(
                    Chat.chat_id == chat_id_to_delete).delete()
                session.commit()
                chat_ids = list(st.session_state.chats.keys())
                st.session_state.current_chat = chat_ids[0] if chat_ids else None
                if st.session_state.current_chat:
                    st.session_state.conversation = init_llm()(st.session_state.current_chat)

    # Chat selection radio buttons below
    chat_ids = list(st.session_state.chats.keys())
    selected_chat = st.radio(
        "Select Chat",
        chat_ids,
        index=chat_ids.index(st.session_state.current_chat),
        key="selected_chat",
        on_change=handle_chat_selection,
        format_func=lambda x: st.session_state.chats[x].get("name", x)
    )


st.subheader(f"Chat with Ollama Model ({llm_model})")


# Display chat messages
# if st.session_state.current_chat:
#     for message in st.session_state.chats[st.session_state.current_chat]["messages"]:
#         with st.chat_message(message["role"]):
#             if "<img src=" in str(message["content"]):
#                 st.markdown(message["content"], unsafe_allow_html=True)
#             else:
#                 st.write(message["content"])

if st.session_state.current_chat:
    for message in st.session_state.chats[st.session_state.current_chat]["messages"]:
        with st.chat_message(message["role"]):
            content = message["content"]
            if isinstance(content, str) and "<img src=" in content:
                st.markdown(content, unsafe_allow_html=True)
            else:
                st.write(content)

# Chat input
if prompt := st.chat_input():
    human_message = HumanMessage(content=prompt)
    chat = st.session_state.chats[st.session_state.current_chat]
    chat["message_history"].append(human_message)
    chat["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        response = st.session_state.conversation.invoke(
            {"input": prompt}
        )
        ai_message = AIMessage(content=str(response))
        chat["message_history"].append(ai_message)
        chat["messages"].append({"role": "assistant", "content": response})
        st.write(response)

    # Update chat name if it's the first user message (after assistant responds)
    user_messages = [msg for msg in chat["messages"] if msg["role"] == "user"]
    if len(user_messages) == 1:
        new_name = ' '.join(prompt.split()[:5])
        chat["name"] = new_name

    # Save chat to the database
    db_chat = session.query(Chat).filter(
        Chat.chat_id == st.session_state.current_chat).first()
    if db_chat:
        db_chat.messages = chat["messages"]
        db_chat.name = chat["name"]
        db_chat.file_paths = chat["file_paths"]
    else:
        new_chat = Chat(chat_id=st.session_state.current_chat,
                        name=chat["name"], messages=chat["messages"], file_paths=chat["file_paths"])
        session.add(new_chat)
    session.commit()

# File upload
uploaded_file = st.file_uploader("Upload a file (PDF or Image)", type=[
                                 'pdf', 'png', 'jpg', 'jpeg'])

# Update file upload handler
if uploaded_file:
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    chat = st.session_state.chats[st.session_state.current_chat]
    chat.setdefault("file_paths", [])
    chat.setdefault("name", "Default_Chat_Name")
    chat["file_paths"].append(file_path)

    if uploaded_file.type == "application/pdf":
        result = process_pdf(uploaded_file)
        # Store full content for LLM context
        pdf_content = result["context"]
        # Show only filename in chat
        display_content = f"📄 PDF uploaded: {uploaded_file.name}"
        st.write(display_content)

        # Add display message to chat history
        chat["messages"].append({
            "role": "assistant",
            "content": display_content,
            "full_content": pdf_content  # Store full content but don't display
        })

        # Update LLM context with full content
        st.session_state.conversation = init_llm(
            pdf_content)(st.session_state.current_chat)

    else:
        result = process_image(uploaded_file)
        content = result["context"]
        st.markdown(content, unsafe_allow_html=True)

        # Add image to chat history
        chat["messages"].append({
            "role": "assistant",
            "content": content
        })

        # Update LLM context
        st.session_state.conversation = init_llm(
            content)(st.session_state.current_chat)

    # Save chat to the database
    db_chat = session.query(Chat).filter(
        Chat.chat_id == st.session_state.current_chat).first()
    if db_chat:
        db_chat.messages = chat["messages"]
        db_chat.name = chat["name"]
        db_chat.file_paths = chat["file_paths"]
    else:
        new_chat = Chat(chat_id=st.session_state.current_chat,
                        name=chat["name"], messages=chat["messages"], file_paths=chat["file_paths"])
        session.add(new_chat)
    session.commit()
