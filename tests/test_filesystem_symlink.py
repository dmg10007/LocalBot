"""Symlink-escape regression test for the filesystem sandbox guard."""
import os

import pytest

from localbot.tools import filesystem


@pytest.fixture
def tmp_sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(filesystem.cfg, "sandbox_root", str(tmp_path))
    return tmp_path


def test_symlink_pointing_outside_sandbox_blocked(tmp_sandbox, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside")
    secret = outside / "secret.txt"
    secret.write_text("top secret")

    link = tmp_sandbox / "escape"
    os.symlink(outside, link)  # symlink *inside* the jail → target outside

    with pytest.raises(PermissionError):
        filesystem._safe_resolve("escape/secret.txt")
