"""Authentication and role-based access for Streamlit.

Backed by a persistent SQLite user store in data/app_users.db.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Mapping, Optional

import streamlit as st

from utils.user_management import UserStore


VALID_ROLES = {"admin", "org_admin", "team_member", "team", "individual"}


@dataclass(frozen=True)
class AuthContext:
    """Authenticated user identity and role."""

    user_id: int
    username: str
    role: str
    display_name: str
    organization_name: str
    team_name: str


def _secret(key: str, default: str = "") -> str:
    if not hasattr(st, "secrets"):
        return default
    value = st.secrets.get(key, default)
    return value if isinstance(value, str) else default


def _normalize_role(role: Any) -> str:
    val = str(role or "individual").strip().lower()
    if val == "team":
        return "team_member"
    return val if val in VALID_ROLES else "individual"


def _coerce_users(raw_users: Mapping[str, Any]) -> Dict[str, Dict[str, str]]:
    """Backward-compatible helper used by tests and migration scripts."""
    users: Dict[str, Dict[str, str]] = {}
    for username, info in raw_users.items():
        if not isinstance(info, Mapping):
            continue
        password = str(info.get("password", ""))
        if not password:
            continue
        role = _normalize_role(info.get("role", "individual"))
        display_name = str(info.get("display_name") or username)
        users[str(username)] = {
            "password": password,
            "role": role,
            "display_name": display_name,
        }
    return users


@st.cache_resource(show_spinner=False)
def get_user_store() -> UserStore:
    """Return cached user store and bootstrap admin account if configured."""
    repo_root = os.path.dirname(os.path.dirname(__file__))
    default_db = os.path.join(repo_root, "data", "app_users.db")
    db_path = _secret("APP_USERS_DB", default_db)

    store = UserStore(db_path)
    admin_username = _secret("APP_ADMIN_USERNAME", "admin")
    admin_password = _secret("APP_ADMIN_PASSWORD", "")
    # Backward compatibility with legacy APP_PASSWORD.
    if not admin_password:
        admin_password = _secret("APP_PASSWORD", "")

    if admin_password:
        store.ensure_admin(
            username=admin_username,
            password=admin_password,
            full_name=_secret("APP_ADMIN_NAME", "Platform Administrator"),
        )
    return store


def _get_timeout_minutes() -> int:
    if not hasattr(st, "secrets"):
        return 480
    try:
        timeout = int(st.secrets.get("AUTH_SESSION_TIMEOUT_MIN", 480))
    except Exception:
        timeout = 480
    return max(5, timeout)


def _is_session_expired() -> bool:
    logged_at = st.session_state.get("auth_logged_at")
    if not logged_at:
        return False
    if not isinstance(logged_at, datetime):
        return True
    expiry = logged_at + timedelta(minutes=_get_timeout_minutes())
    return datetime.now(timezone.utc) > expiry


def sign_out() -> None:
    """Clear auth session state and rerun app."""
    for key in [
        "authenticated",
        "auth_user",
        "auth_logged_at",
    ]:
        st.session_state.pop(key, None)
    st.rerun()


def get_auth_context() -> AuthContext:
    """Return current auth context without enforcing login UI."""
    user = st.session_state.get("auth_user", {}) or {}
    return AuthContext(
        user_id=int(user.get("id", 0)),
        username=str(user.get("username", "anonymous")),
        role=_normalize_role(user.get("role", "individual")),
        display_name=str(user.get("full_name", "Guest")),
        organization_name=str(user.get("organization_name") or "Independent"),
        team_name=str(user.get("team_name") or "N/A"),
    )


def can_access_mode(role: str, mode_name: str) -> bool:
    """Role-based access checks for top-level app modes."""
    role = _normalize_role(role)
    allowed = {
        "individual": {
            "🔬 Single VCF",
            "📦 Batch Pipeline",
        },
        "team_member": {
            "🔬 Single VCF",
            "⚖️ Multi-VCF Compare",
            "👨‍👩‍👧 Trio Analysis",
            "🧫 Somatic (Tumor/Normal)",
            "📦 Batch Pipeline",
        },
        "org_admin": {
            "🔬 Single VCF",
            "⚖️ Multi-VCF Compare",
            "👨‍👩‍👧 Trio Analysis",
            "🧫 Somatic (Tumor/Normal)",
            "📦 Batch Pipeline",
        },
        "admin": {
            "🔬 Single VCF",
            "⚖️ Multi-VCF Compare",
            "👨‍👩‍👧 Trio Analysis",
            "🧫 Somatic (Tumor/Normal)",
            "📦 Batch Pipeline",
            "🛠️ Admin Console",
        },
    }
    return mode_name in allowed.get(role, set())


def available_modes(role: str) -> list[str]:
    all_modes = [
        "🔬 Single VCF",
        "⚖️ Multi-VCF Compare",
        "👨‍👩‍👧 Trio Analysis",
        "🧫 Somatic (Tumor/Normal)",
        "📦 Batch Pipeline",
        "🛠️ Admin Console",
    ]
    return [m for m in all_modes if can_access_mode(role, m)]


def require_auth() -> AuthContext:
    """Block the app until a valid user session exists."""
    store = get_user_store()

    if st.session_state.get("authenticated") and not _is_session_expired():
        return get_auth_context()

    if _is_session_expired():
        sign_out()
        st.warning("Your session expired. Please sign in again.")

    st.markdown(
        """
        <style>
            .login-box {
                max-width: 460px; margin: 5rem auto 1rem; padding: 2rem;
                background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
                border: 1px solid #e2e8f0; border-radius: 12px;
                box-shadow: 0 4px 24px rgba(0,0,0,.12);
            }
            .login-box h2 { color: #0f172a; margin-bottom: .4rem; text-align: center; }
            .login-box p { color: #475569; text-align: center; margin: 0; }
        </style>
        <div class="login-box">
          <h2>Variant Analysis Suite</h2>
          <p>Secure access for organisations, teams, and individual researchers.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("login_form"):
        st.subheader("Access Required")
        username = st.text_input("Username", autocomplete="username")
        pwd = st.text_input("Password", type="password", autocomplete="current-password")
        submitted = st.form_submit_button("Sign In", type="primary")

    if submitted:
        user = store.authenticate(username=username, password=pwd)
        if user:
            st.session_state["authenticated"] = True
            st.session_state["auth_user"] = user
            st.session_state["auth_logged_at"] = datetime.now(timezone.utc)
            st.rerun()

        st.error("Invalid username or password.")

    if not _secret("APP_ADMIN_PASSWORD") and not _secret("APP_PASSWORD"):
        st.info("Set APP_ADMIN_PASSWORD in Streamlit secrets to bootstrap the first admin account.")
    st.stop()


def create_user_account(
    username: str,
    full_name: str,
    password: str,
    role: str,
    organization_id: Optional[int],
    team_id: Optional[int],
) -> Dict[str, Any]:
    """Admin helper for account provisioning."""
    return get_user_store().create_user(
        username=username,
        full_name=full_name,
        password=password,
        role=role,
        organization_id=organization_id,
        team_id=team_id,
    )


def create_organization(name: str) -> int:
    return get_user_store().create_organization(name)


def create_team(organization_id: int, name: str) -> int:
    return get_user_store().create_team(organization_id, name)


def list_users() -> list[Dict[str, Any]]:
    return get_user_store().list_users()


def list_organizations() -> list[Dict[str, Any]]:
    return get_user_store().list_organizations()


def list_teams(organization_id: Optional[int] = None) -> list[Dict[str, Any]]:
    return get_user_store().list_teams(organization_id)


def set_user_active(user_id: int, active: bool) -> None:
    get_user_store().set_user_active(user_id, active)


def render_user_status(auth_ctx: AuthContext) -> None:
    """Render current signed-in user status in the sidebar."""
    role_label = {
        "admin": "Platform Admin",
        "org_admin": "Organisation Admin",
        "team_member": "Team Member",
        "individual": "Individual",
    }.get(auth_ctx.role, auth_ctx.role)

    st.markdown(
        f"""
        <div style="padding:.55rem .7rem; border:1px solid #1e293b; border-radius:10px;
                    background:rgba(15,23,42,.45); margin-bottom:.45rem;">
          <div style="font-size:.72rem; color:#94a3b8;">Signed in</div>
          <div style="color:#e2e8f0; font-weight:700;">{auth_ctx.display_name}</div>
          <div style="color:#93c5fd; font-size:.75rem; margin-top:.2rem;">{role_label}</div>
          <div style="color:#64748b; font-size:.67rem; margin-top:.25rem;">
            {auth_ctx.organization_name} · {auth_ctx.team_name}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Sign Out", use_container_width=True):
        sign_out()
