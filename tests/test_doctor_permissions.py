from pathlib import Path

from dgx_monarch.cli.doctor_permissions import config_permissions_row
from dgx_monarch.config import ClusterConfig


def test_config_permission_row_names_the_safe_repair(tmp_path: Path) -> None:
    path = tmp_path / "cluster.toml"
    path.write_text("[cluster]\n")
    path.chmod(0o644)
    row = config_permissions_row(ClusterConfig(source=str(path)))
    assert row == {
        "status": "WARN",
        "name": "config permissions",
        "detail": "mode 0644 exposes cluster metadata; run dgxm doctor --repair",
    }
    assert str(tmp_path) not in str(row)


def test_config_permission_row_accepts_private_regular_file(tmp_path: Path) -> None:
    path = tmp_path / "cluster.toml"
    path.write_text("[cluster]\n")
    path.chmod(0o600)
    assert config_permissions_row(ClusterConfig(source=str(path)))["status"] == "ok"


def test_config_permission_row_refuses_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.toml"
    target.write_text("[cluster]\n")
    link = tmp_path / "cluster.toml"
    link.symlink_to(target)
    row = config_permissions_row(ClusterConfig(source=str(link)))
    assert row["status"] == "WARN"
    assert "automatic repair is unavailable" in row["detail"]
