from utils.auth import available_modes, can_access_mode, _coerce_users


def test_available_modes_for_individual():
    modes = available_modes("individual")
    assert modes == ["🔬 Single VCF"]


def test_available_modes_for_team():
    modes = available_modes("team")
    assert "🔬 Single VCF" in modes
    assert "⚖️ Multi-VCF Compare" in modes
    assert "📦 Batch Pipeline" not in modes
    assert "🛠️ Admin Console" not in modes


def test_available_modes_for_admin():
    modes = available_modes("admin")
    assert "📦 Batch Pipeline" in modes
    assert "🛠️ Admin Console" in modes


def test_invalid_role_defaults_to_individual_access():
    assert can_access_mode("unknown-role", "🔬 Single VCF")
    assert not can_access_mode("unknown-role", "🛠️ Admin Console")


def test_coerce_users_discards_invalid_records():
    raw = {
        "admin": {"password": "secret", "role": "admin", "display_name": "Admin"},
        "broken": {"role": "team"},
        "not_mapping": "abc",
    }
    users = _coerce_users(raw)

    assert "admin" in users
    assert "broken" not in users
    assert "not_mapping" not in users
    assert users["admin"]["role"] == "admin"
