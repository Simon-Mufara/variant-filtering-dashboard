import os

from utils.user_management import UserStore


def test_create_and_authenticate_user(tmp_path):
    db_path = os.path.join(tmp_path, "users.db")
    store = UserStore(db_path)

    org_id = store.create_organization("School of Health Sciences")
    team_id = store.create_team(org_id, "Cancer Genomics Lab")

    created = store.create_user(
        username="team.lead",
        password="StrongPass123",
        full_name="Team Lead",
        role="team_member",
        organization_id=org_id,
        team_id=team_id,
    )

    assert created["username"] == "team.lead"
    assert created["role"] == "team_member"

    user = store.authenticate("team.lead", "StrongPass123")
    assert user is not None
    assert user["organization_name"] == "School of Health Sciences"
    assert user["team_name"] == "Cancer Genomics Lab"


def test_admin_bootstrap(tmp_path):
    db_path = os.path.join(tmp_path, "users.db")
    store = UserStore(db_path)

    admin = store.ensure_admin(username="admin", password="AdminPass123")
    assert admin["role"] == "admin"

    authed = store.authenticate("admin", "AdminPass123")
    assert authed is not None
    assert authed["role"] == "admin"


def test_deactivate_user_blocks_login(tmp_path):
    db_path = os.path.join(tmp_path, "users.db")
    store = UserStore(db_path)

    user = store.create_user(
        username="researcher1",
        password="Research123",
        full_name="Research User",
        role="individual",
        organization_id=None,
        team_id=None,
    )

    assert store.authenticate("researcher1", "Research123") is not None
    store.set_user_active(user["id"], False)
    assert store.authenticate("researcher1", "Research123") is None
