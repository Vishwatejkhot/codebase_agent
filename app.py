

import streamlit as st
from pathlib import Path
from src.agent import setup_from_url, load_existing
from src.cloner import list_cloned_repos
from src.vectorstore import list_indexed_repos


st.set_page_config(
    page_title="🤖 Codebase Q&A Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>
.main-header {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.chat-user {
    background: #1e1e2e;
    border-left: 4px solid #6366f1;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.chat-agent {
    background: #1a1a2e;
    border-left: 4px solid #10b981;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.source-badge {
    background: #374151;
    padding: 0.2rem 0.6rem;
    border-radius: 1rem;
    font-size: 0.75rem;
    margin-right: 0.4rem;
}
</style>
""", unsafe_allow_html=True)


if "agent"       not in st.session_state: st.session_state.agent       = None
if "repo_name"   not in st.session_state: st.session_state.repo_name   = None
if "chat_history"not in st.session_state: st.session_state.chat_history = []
if "ready"       not in st.session_state: st.session_state.ready       = False


with st.sidebar:
    st.markdown("## ⚙️ Repository Setup")
    st.divider()

    # --- Option A: New GitHub URL ---
    st.markdown("### 🔗 New Repository")
    github_url = st.text_input(
        "GitHub URL",
        placeholder="https://github.com/user/repo",
        label_visibility="collapsed",
    )

    col1, col2 = st.columns(2)
    force_clone = col1.checkbox("Re-clone", value=False)
    force_index = col2.checkbox("Re-index", value=False)

    if st.button("🚀 Load & Index", use_container_width=True, type="primary"):
        if not github_url.strip():
            st.error("Please enter a GitHub URL.")
        else:
            with st.spinner("Cloning & indexing... (this may take 1–2 min)"):
                try:
                    agent, repo_name = setup_from_url(
                        github_url.strip(),
                        force_reclone=force_clone,
                        force_reindex=force_index,
                    )
                    st.session_state.agent        = agent
                    st.session_state.repo_name    = repo_name
                    st.session_state.chat_history = []
                    st.session_state.ready        = True
                    st.success(f"✅ Ready: {repo_name}")
                except Exception as e:
                    st.error(f"❌ {e}")

    st.divider()

    
    indexed = list_indexed_repos()
    if indexed:
        st.markdown("### 📚 Load Existing")
        selected = st.selectbox("Indexed repos", options=indexed, label_visibility="collapsed")

        if st.button("📂 Load", use_container_width=True):
            with st.spinner(f"Loading {selected}..."):
                try:
                    agent, repo_name = load_existing(selected)
                    st.session_state.agent        = agent
                    st.session_state.repo_name    = repo_name
                    st.session_state.chat_history = []
                    st.session_state.ready        = True
                    st.success(f"✅ Loaded: {selected}")
                except Exception as e:
                    st.error(f"❌ {e}")

    st.divider()

    # --- Quick Actions ---
    if st.session_state.ready:
        st.markdown("### ⚡ Quick Actions")
        quick_actions = [
            "Give me an overview of this codebase",
            "What are the main entry points?",
            "Explain the authentication flow",
            "Find any potential security issues",
            "What could be refactored?",
            "Generate docs for the main module",
        ]
        for action in quick_actions:
            if st.button(action, use_container_width=True, key=action):
                st.session_state.pending_question = action
                st.rerun()

        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


st.markdown('<div class="main-header">🤖 Codebase Q&A Agent</div>', unsafe_allow_html=True)
st.caption("Powered by Groq + LangChain + FAISS")

if not st.session_state.ready:
    st.info("👈 Enter a GitHub URL in the sidebar and click **Load & Index** to get started.")

    st.markdown("### 💡 What you can ask:")
    examples = {
        "🔍 Understand": [
            "Explain how the authentication system works",
            "What does the `process_payment()` function do?",
            "Give me an overview of the entire codebase",
        ],
        "🔧 Improve": [
            "How can I refactor the database layer?",
            "Suggest improvements for the API module",
            "What design patterns could be applied here?",
        ],
        "🐛 Debug": [
            "Are there any security vulnerabilities?",
            "Find potential null pointer issues",
            "Check error handling in the main module",
        ],
        "📝 Document": [
            "Generate a docstring for the User class",
            "Write README documentation for this module",
            "Document all public functions in auth.py",
        ],
    }

    cols = st.columns(4)
    for col, (category, items) in zip(cols, examples.items()):
        with col:
            st.markdown(f"**{category}**")
            for item in items:
                st.markdown(f"• {item}")
else:
    # Active repo header
    st.markdown(f"**Analyzing:** `{st.session_state.repo_name}`")
    st.divider()

    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message["content"])

    # Handle quick action
    pending = st.session_state.pop("pending_question", None)

    # Chat input
    user_input = st.chat_input("Ask anything about the codebase...") or pending

    if user_input:
        # Show user message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Run agent
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🤔 Analyzing codebase..."):
                try:
                    result = st.session_state.agent.invoke({"input": user_input})
                    answer = result["output"]
                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    err_msg = f"❌ Error: {e}"
                    st.error(err_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": err_msg})
