
import streamlit as st

from ingestion.loader import load_clinic_data
from ingestion.validator import validate_clinic_data
from ingestion.normalizer import normalize_clinic_data
from ingestion.document_builder import build_documents

from embeddings.embedder import Embedder

from retrieval.faiss_store import FAISSStore
from retrieval.vector_retriever import VectorRetriever
from retrieval.structured_retriever import StructuredRetriever
from retrieval.hybrid_retriever import HybridRetriever

from conversation.entity_resolver import ContextResolver

from orchestration.pipeline import RAGPipeline

from ui.components import (
    apply_clinic_theme,
    render_clinic_header,
    render_welcome,
    render_user_message,
    render_assistant_message,
    render_doctor_card,
    render_footer,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sunrise Medical Center",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ============================================================
# APPLY UI THEME
# ============================================================

apply_clinic_theme()


# ============================================================
# CONSTANTS
# ============================================================

DATA_PATH = "data/clinic_data.json"

CLINIC_NAME = "Sunrise Medical Center"


# ============================================================
# BUILD RAG PIPELINE
# ============================================================

@st.cache_resource
def build_pipeline():
    """
    Build and cache the complete RAG pipeline.

    The pipeline is built only once during the
    Streamlit session instead of rebuilding it
    after every user message.
    """

    data = load_clinic_data(DATA_PATH)

    validate_clinic_data(data)

    normalized_data = normalize_clinic_data(data)

    documents = build_documents(normalized_data)

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    embedder = Embedder()

    embeddings = embedder.embed_documents(
        documents
    )

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    dimension = embeddings.shape[1]

    store = FAISSStore(
        dimension=dimension
    )

    store.add(
        embeddings,
        documents,
    )

    # --------------------------------------------------------
    # Retrievers
    # --------------------------------------------------------

    vector_retriever = VectorRetriever(
        store=store,
        embedder=embedder,
    )

    structured_retriever = StructuredRetriever(
        documents=documents,
    )

    hybrid_retriever = HybridRetriever(
        structured_retriever=structured_retriever,
        vector_retriever=vector_retriever,
    )

    # --------------------------------------------------------
    # Conversation context
    # --------------------------------------------------------

    context_resolver = ContextResolver()

    # --------------------------------------------------------
    # Final pipeline
    # --------------------------------------------------------

    return RAGPipeline(
        retriever=hybrid_retriever,
        context_resolver=context_resolver,
    )


# ============================================================
# SESSION STATE
# ============================================================

def initialize_session():
    """
    Initialize Streamlit session state.
    """

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None


# ============================================================
# RESET CONVERSATION
# ============================================================

def reset_conversation():
    """
    Clear the visible chat and the pipeline's
    conversation context.
    """

    st.session_state.messages = []

    if st.session_state.pipeline is not None:
        st.session_state.pipeline.reset_conversation()


# ============================================================
# HEADER
# ============================================================

initialize_session()

render_clinic_header(
    clinic_name=CLINIC_NAME
)


# ============================================================
# NEW CONVERSATION BUTTON
# ============================================================

button_col1, button_col2 = st.columns(
    [5, 1]
)

with button_col2:

    if st.button(
        "↻ New",
        use_container_width=True,
    ):
        reset_conversation()
        st.rerun()


# ============================================================
# WELCOME SCREEN
# ============================================================

if not st.session_state.messages:
    render_welcome()


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    role = message["role"]
    content = message["content"]

    if role == "user":

        render_user_message(
            content
        )

    else:

        render_assistant_message(
            content
        )

        # Optional doctor information
        doctor_data = message.get(
            "doctor_data"
        )

        if doctor_data:

            render_doctor_card(
                doctor=doctor_data.get(
                    "doctor"
                ),
                specialization=doctor_data.get(
                    "specialization"
                ),
                clinic=doctor_data.get(
                    "clinic"
                ),
                availability=doctor_data.get(
                    "availability"
                ),
            )


# ============================================================
# CHAT INPUT
# ============================================================

user_query = st.chat_input(
    "Ask about doctors, availability, or clinic services..."
)


# ============================================================
# PROCESS USER QUESTION
# ============================================================

if user_query:

    # --------------------------------------------------------
    # Display user message immediately
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query,
        }
    )

    render_user_message(
        user_query
    )

    # --------------------------------------------------------
    # Build pipeline if necessary
    # --------------------------------------------------------

    if st.session_state.pipeline is None:

        with st.spinner(
            "Connecting to the clinic..."
        ):

            st.session_state.pipeline = (
                build_pipeline()
            )

    pipeline = st.session_state.pipeline

    # --------------------------------------------------------
    # Run RAG pipeline
    # --------------------------------------------------------

    with st.spinner(
        "Checking clinic information..."
    ):

        result = pipeline.run(
            query=user_query,
            top_k=3,
        )

    # --------------------------------------------------------
    # Assistant response
    # --------------------------------------------------------

    answer = result.answer

    render_assistant_message(
        answer
    )

    # --------------------------------------------------------
    # Extract doctor information
    # --------------------------------------------------------

    doctor_data = None

    if result.results:

        best_document = (
            result.results[0].document
        )

        metadata = best_document.metadata

        retrieved_doctor = metadata.get(
            "doctor_name"
        )

        retrieved_specialization = (
            metadata.get(
                "specialization"
            )
        )

        retrieved_clinic = metadata.get(
            "clinic_name"
        )

        retrieved_availability = (
            metadata.get(
                "availability"
            )
        )

        # Only show the card when we actually
        # have a doctor entity.

        if retrieved_doctor:

            doctor_data = {
                "doctor": retrieved_doctor,
                "specialization": (
                    retrieved_specialization
                ),
                "clinic": retrieved_clinic,
                "availability": (
                    retrieved_availability
                ),
            }

            render_doctor_card(
                doctor=retrieved_doctor,
                specialization=(
                    retrieved_specialization
                ),
                clinic=retrieved_clinic,
                availability=(
                    retrieved_availability
                ),
            )

    # --------------------------------------------------------
    # Save assistant response
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "doctor_data": doctor_data,
        }
    )


# ============================================================
# FOOTER
# ============================================================

render_footer()
