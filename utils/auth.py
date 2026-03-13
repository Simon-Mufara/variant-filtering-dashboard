"""Authentication gate — simple password protection via Streamlit secrets.

Usage in app.py:
    from utils.auth import require_auth
    require_auth()   # Call at the very top before rendering anything

Setup (Streamlit Cloud):
    In Settings → Secrets, add:
        APP_PASSWORD = "your-password-here"

Local development:
    Create .streamlit/secrets.toml:
        APP_PASSWORD = "your-password-here"

If APP_PASSWORD is not set in secrets, auth is disabled (useful for development).
"""
from __future__ import annotations
import streamlit as st


def require_auth() -> None:
    """Block the app until the user provides the correct password.

    Stores auth state in st.session_state["authenticated"] so the user
    is not prompted again on every rerun.
    """
    # If no password is configured, skip auth entirely
    password = st.secrets.get("APP_PASSWORD", "") if hasattr(st, "secrets") else ""
    if not password:
        return  # auth disabled

    # Already authenticated this session
    if st.session_state.get("authenticated"):
        return

    # Show login screen
    st.markdown("""
    <style>
        .login-box { max-width: 400px; margin: 6rem auto; padding: 2rem;
                     background: white; border-radius: 12px;
                     box-shadow: 0 4px 24px rgba(0,0,0,.12); }
        .login-box h2 { color: #1e40af; margin-bottom: 1rem; text-align: center; }
    </style>
    <div class="login-box"><h2>🧬 Variant Analysis Suite</h2></div>
    """, unsafe_allow_html=True)

    with st.form("login_form"):
        st.subheader("🔐 Access Required")
        pwd = st.text_input("Password", type="password", autocomplete="current-password")
        submitted = st.form_submit_button("Sign In", type="primary")

    if submitted:
        if pwd == password:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password. Please try again.")
    st.stop()
