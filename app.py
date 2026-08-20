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

from conversation.entity_resolver import EntityCatalog

from orchestration.factory import (
    SharedRAGResources,
    build_session_pipeline,
)
from generation.provider_errors import ProviderError, ProviderFailureKind

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
    page_title="ClinicCare Assistant",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ============================================================
# APPLY THEME
# ============================================================

apply_clinic_theme()


# ============================================================
# CONSTANTS
# ============================================================

DATA_PATH = "data/clinic_data.json"

TEMPORARY_SERVICE_MESSAGE = (
    "I'm unable to complete that request right now. "
    "Please try again."
)

CONFIGURATION_SERVICE_MESSAGE = (
    "I'm unable to complete that request because the response service "
    "is not available. Please try again later."
)


def safe_application_failure_message(error):
    """Map infrastructure failures without exposing exception details."""

    if isinstance(error, ProviderError) and error.kind in {
        ProviderFailureKind.CONFIGURATION,
        ProviderFailureKind.AUTHENTICATION,
        ProviderFailureKind.PERMISSION,
    }:
        return CONFIGURATION_SERVICE_MESSAGE
    return TEMPORARY_SERVICE_MESSAGE


# ============================================================
# BUILD SHARED RESOURCES AND SESSION PIPELINE
# ============================================================

@st.cache_resource
def build_shared_resources():
    """
    Build expensive resources that are safe to share across sessions.
    """

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

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
    # Entity catalog
    # --------------------------------------------------------

    entity_catalog = EntityCatalog.from_clinic_data(
        normalized_data
    )

    # --------------------------------------------------------
    # Shared resources
    # --------------------------------------------------------

    return SharedRAGResources(
        retriever=hybrid_retriever,
        entity_catalog=entity_catalog,
    )


def build_pipeline():
    """Build a pipeline with conversation state owned by this session."""

    return build_session_pipeline(
        build_shared_resources()
    )


# ============================================================
# SESSION STATE
# ============================================================

def initialize_session():

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None


# ============================================================
# RESET CONVERSATION
# ============================================================

def reset_conversation():

    st.session_state.messages = []

    if st.session_state.pipeline is not None:
        st.session_state.pipeline.reset_conversation()


# ============================================================
# INITIALIZE
# ============================================================

initialize_session()


# ============================================================
# HEADER
# ============================================================

render_clinic_header()


# ============================================================
# NEW CONVERSATION
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
# CHAT HISTORY
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
    "Ask about doctors, availability, clinic services..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if user_query:

    # --------------------------------------------------------
    # Save and display user message
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
    # Build pipeline
    # --------------------------------------------------------

    result = None
    answer = None

    if st.session_state.pipeline is None:

        try:
            with st.spinner(
                "Connecting to the clinic..."
            ):
                pipeline_factory = st.session_state.get(
                    "_pipeline_factory",
                    build_pipeline,
                )
                st.session_state.pipeline = pipeline_factory()
                if st.session_state.pipeline is None:
                    raise RuntimeError("Pipeline factory returned no pipeline.")
        except Exception as error:
            answer = safe_application_failure_message(error)

    pipeline = st.session_state.pipeline

    # --------------------------------------------------------
    # Run pipeline with a final infrastructure boundary
    # --------------------------------------------------------

    if pipeline is not None and answer is None:
        try:
            with st.spinner(
                "Checking clinic information..."
            ):
                result = pipeline.run(
                    query=user_query,
                    top_k=5,
                )
            answer = result.answer
        except Exception as error:
            answer = safe_application_failure_message(error)

    render_assistant_message(answer)

    # --------------------------------------------------------
    # Doctor information
    # --------------------------------------------------------

    doctor_data = None

    if result is not None and result.evidence:

        best_document = (
            result.evidence[0].document
        )

        metadata = best_document.metadata

        retrieved_doctor = metadata.get(
            "doctor_name"
        )

        retrieved_specialization = metadata.get(
            "specialization"
        )

        retrieved_clinic = metadata.get(
            "clinic_name"
        )

        retrieved_availability = metadata.get(
            "availability"
        )

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
