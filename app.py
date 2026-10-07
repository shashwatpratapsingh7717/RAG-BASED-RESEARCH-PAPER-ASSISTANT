import json
import streamlit as st
from src.config import (
    APP_TITLE,
    LLM_MODEL,
    TOP_K_DEFAULT,
    UPLOAD_DIR
)
from src.pdf_processor import save_uploaded_pdfs
from src.rag_pipeline import RAGPipeline
from src.evaluator import evaluate_retrieval

# STREAMLIT CONFIGURATION
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🔭",
    layout="wide"
)
# TITLE
st.title("🔭 AI Research Paper Assistant")
st.caption(
    "Retrieval-Augmented Generation for "
    "Research Paper Question Answering"
)
# PIPELINE
@st.cache_resource
def load_pipeline():
    return RAGPipeline()
try:
    pipeline = load_pipeline()
except Exception as error:
    st.error(
        f"Pipeline initialization failed:\n{error}"
    )
    st.stop()

# SIDEBAR
with st.sidebar:
    st.header("📚 Knowledge Base")

# PDF UPLOAD
    uploaded_files = st.file_uploader(
        "Upload Research Papers",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more research paper PDFs."
    )

    if uploaded_files:
        st.write(
            f"Selected: **{len(uploaded_files)} paper(s)**"
        )
        for file in uploaded_files:
            st.caption(
                f"📄 {file.name}"
            )
# BUILD KNOWLEDGE BASE
    build_button = st.button(
        "🚀 Build / Update Knowledge Base",
        type="primary",
        use_container_width=True,
        disabled=not bool(uploaded_files)
    )

    if build_button:
        try:
# Save uploaded PDFs
            with st.spinner(
                "Saving uploaded papers..."
            ):
                save_uploaded_pdfs(
                    uploaded_files,
                    UPLOAD_DIR
                )

# Build FAISS knowledge base
            with st.spinner(
                "Building knowledge base..."
            ):
                stats = pipeline.build_from_folder()
            st.success(
                "✅ Knowledge Base Ready!"
            )
            st.session_state["kb_cleared"] = False
            st.json(stats)

# Refresh page so metrics update
            st.rerun()
        except Exception as error:
            st.error(
                f"Build failed:\n{error}"
            )
# CLEAR KNOWLEDGE BASE
    st.divider()
    st.subheader("🗑️ Knowledge Base Management")
    clear_button = st.button(
        "🗑️ Clear Knowledge Base",
        use_container_width=True
    )

    if clear_button:
        st.session_state[
            "confirm_clear_kb"
        ] = True
# CONFIRM CLEAR
    if st.session_state.get(
        "confirm_clear_kb",
        False
    ):
        st.warning(
            "⚠️ This will permanently delete:"
        )
        st.markdown(
            """
            - All uploaded PDF files
            - FAISS vector index
            - Vector metadata
            - Current in-memory knowledge base
            """
        )
        st.error(
            "This action cannot be undone."
        )
        confirm_col1, confirm_col2 = st.columns(2)
# CONFIRM
        with confirm_col1:
            confirm_clear = st.button(
                "Yes, Clear",
                type="primary",
                use_container_width=True,
                key="confirm_clear_button"
            )
            if confirm_clear:
                try:
                    with st.spinner(
                        "Clearing knowledge base..."
                    ):
                        pipeline.clear_knowledge_base()
# Reset Streamlit cached RAGPipeline so the next run starts fresh.
                        st.cache_resource.clear()
                    st.session_state[
                        "confirm_clear_kb"
                    ] = False
                    st.session_state["kb_cleared"] = True
                    st.success(
                        "✅ Knowledge Base cleared successfully!"
                    )
# Refresh page
                    st.rerun()
                except Exception as error:
                    st.error(
                        f"Clear failed:\n{error}"
                    )

# CANCEL
        with confirm_col2:
            cancel_clear = st.button(
                "Cancel",
                use_container_width=True,
                key="cancel_clear_button"
            )
            if cancel_clear:
                st.session_state[
                    "confirm_clear_kb"
                ] = False
                st.rerun()
# SETTINGS
    st.divider()
    st.subheader("⚙️ Settings")
    top_k = st.slider(
        "Top-K Retrieved Chunks",
        min_value=1,
        max_value=10,
        value=TOP_K_DEFAULT
    )
    st.caption(
        "Embedding: `all-MiniLM-L6-v2`"
    )
    st.caption(

        f"LLM: `{LLM_MODEL}`"
    )

    st.caption(
        "Vector DB: `FAISS`"
    )

# STATS
stats = pipeline.get_stats()

# After Clear Knowledge Base, show the guaranteed empty state.
if st.session_state.get("kb_cleared", False):
    stats = {
        **stats,
        "documents": 0,
        "chunks": 0,
        "embedding_dim": 0,
        "ready": False,
    }
col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Documents",
    stats["documents"]
)

col2.metric(
    "Chunks",
    stats["chunks"]
)

col3.metric(
    "Embedding Dimension",
    stats["embedding_dim"]
)

col4.metric(
    "Status",
    "Ready"
    if stats["ready"]
    else "Not Ready"
)

# TABS
tab_ask, tab_summary, tab_compare, tab_eval = st.tabs(
    [
        "💬 Ask",
        "📄 Summarize",
        "⚖️ Compare",
        "📊 Evaluation",
    ]
)
# ASK TAB
with tab_ask:
    st.subheader(
        "Ask Questions"
    )

    question = st.text_area(
        "Your question",
        placeholder=(
            "What is the main contribution "
            "of this paper?"
        )
    )
    ask_button = st.button(
        "Ask",
        type="primary"
    )

    if ask_button:
        if not question.strip():
            st.warning(
                "Please enter a question."
            )
        elif not stats["ready"]:
            st.warning(
                "Build the knowledge base first."
            )
        else:
            try:
                with st.spinner(
                    "Searching research papers..."
                ):
                    result = pipeline.ask(
                        question,
                        top_k=top_k
                    )

                st.markdown(
                    "## Answer"
                )

                st.write(
                    result["answer"]
                )
                st.caption(
                    f"Retrieval latency: "
                    f"{result['retrieval_latency']:.3f}s | "
                    f"Total latency: "
                    f"{result['latency_seconds']:.3f}s"
                )

                st.markdown(
                    "## Sources"
                )

                for i, source in enumerate(
                    result["sources"],
                    start=1
                ):
                    st.markdown(
                        f"""
**{i}. {source['source']}**
Page: `{source['page']}`  
Similarity: `{source['score']:.4f}`
"""                 )
                    with st.expander(
                        "View retrieved text"
                    ):
                        st.write(
                            source["text"]
                        )

            except Exception as error:
                st.error(
                    f"Query failed:\n{error}"
                )
# SUMMARY TAB
with tab_summary:
    st.subheader(
        "📄 Summarize Research Paper"
    )

    documents = pipeline.list_documents()
    if not documents:
        st.info(
            "Build the knowledge base first."
        )
    else:
        selected_document = st.selectbox(
            "Select paper",
            documents
        )
        summary_button = st.button(
            "Generate Summary"
        )
        if summary_button:
            try:
                with st.spinner(
                    "Generating summary..."
                ):
                    summary = (
                        pipeline.summarize_document(
                            selected_document
                        )
                    )
                st.markdown(
                    summary
                )
            except Exception as error:
                st.error(
                    f"Summary failed:\n{error}"
                )



# COMPARE TAB
with tab_compare:
    st.subheader(
        "⚖️ Compare Research Papers"
    )
    documents = pipeline.list_documents()
    if len(documents) < 2:
        st.info(
            "Upload at least two papers."
        )

    else:
        col_a, col_b = st.columns(2)
        with col_a:
            paper_a = st.selectbox(
                "Paper A",
                documents,
                key="paper_a"
            )
        with col_b:
            paper_b = st.selectbox(
                "Paper B",
                documents,
                index=1,
                key="paper_b"
            )
        comparison_question = st.text_area(
            "What do you want to compare?",
            value=(
                "Compare their problem, methodology, "
                "datasets, results, limitations, "
                "and future work."
            )
        )
        compare_button = st.button(
            "Compare Papers"
        )

        if compare_button:
            if paper_a == paper_b:
                st.warning(
                    "Choose two different papers."
                )

            else:
                try:
                    with st.spinner(
                        "Comparing papers..."
                    ):

                        result = (
                            pipeline.compare_documents(
                                paper_a,
                                paper_b,
                                comparison_question
                            )
                        )
                    st.markdown(
                        "## Comparison"
                    )
                    st.write(
                        result["answer"]
                    )
                    st.markdown(
                        "## Sources"
                    )

                    for source in result["sources"]:
                        st.write(
                            f"- {source['source']} "
                            f"(Page {source['page']})"
                        )

                except Exception as error:
                    st.error(
                        f"Comparison failed:\n{error}"
                    )
# EVALUATION TAB
with tab_eval:
    st.subheader(
        "📊 Retrieval Evaluation"
    )

    st.write(
        "Evaluation uses evaluation/questions.json."
    )
    try:
        with open(
            "evaluation/questions.json",
            "r",

            encoding="utf-8-sig"
        ) as file:
            questions = json.load(file)

        st.write(
            f"Questions: **{len(questions)}**"
        )

        evaluate_button = st.button(
            "Run Retrieval Evaluation"
        )
        if evaluate_button:
            if not stats["ready"]:
                st.warning(
                    "Build the knowledge base "
                    "before running evaluation."
                )
            else:
                with st.spinner(
                    "Evaluating retrieval..."
                ):
                    metrics = evaluate_retrieval(
                        pipeline,
                        questions,
                        top_k_values=(3, 5)
                    )

                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "Hit@3",
                    f"{metrics['hit_at_3'] * 100:.1f}%"
                )

                col2.metric(
                    "Hit@5",
                    f"{metrics['hit_at_5'] * 100:.1f}%"
                )
                col3.metric(
                    "Avg Retrieval Latency",
                    f"{metrics['avg_retrieval_latency']:.3f}s"
                )

                st.dataframe(
                    metrics["details"],
                    use_container_width=True
                )
    except FileNotFoundError:
        st.warning(
            "evaluation/questions.json not found."
        )
    except Exception as error:
        st.error(
            f"Evaluation failed:\n{error}"
        )


