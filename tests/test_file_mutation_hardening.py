@pytest.mark.asyncio
async def test_list_dir_shows_project_dot_directories_but_not_git(tmp_path: Path):
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github" / "workflows.yml").write_text("name: ci\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("internal\n", encoding="utf-8")

    result = await ListDir(tmp_path).invoke({"path": ".", "depth": 2})

    assert result.ok
    normalized = result.output.replace("\\", "/")
    assert ".github/" in normalized
    assert "workflows.yml" in normalized
    assert ".git/" not in normalized


@pytest.mark.asyncio
async def test_write_file_preserves_existing_mode(tmp_path: Path):
    target = tmp_path / "script.sh"
    target.write_text("echo old\n", encoding="utf-8")
    target.chmod(0o750)

    result = await WriteFile(tmp_path).invoke({"path": "script.sh", "content": "echo new\n"})

    assert result.ok
    assert target.read_text(encoding="utf-8") == "echo new\n"
    if os.name != "nt":
        assert target.stat().st_mode & 0o777 == 0o750