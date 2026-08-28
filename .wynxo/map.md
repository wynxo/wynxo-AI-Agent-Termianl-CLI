# Project map

102 source files.
Each line is a file and the names it defines, so you can open the right one
instead of searching for it.

.github/check_pure_python.py       closure, +2
get.sh                             say
install.py                         Style, +11
install.sh
rm.sh                              say
scripts/fake_ollama.py             plan_for, +3
tests/conftest.py                  pytest_configure, +1
tests/test_agent.py                FakeOllama, +11
tests/test_agent_hardening.py      test_stream_test_output_awaits_async_callback, +1
tests/test_ascii_terminal.py       ascii_ui, +11
tests/test_asciiart.py             grid, +4
tests/test_bootstrap.py            test_windows_default_uses_classic_renderer, +2
tests/test_chat_layout.py          TestTheTranscript, +11
tests/test_checkpoints.py          TestUndo, +2
tests/test_code_streaming.py       TestReadingHalfWrittenJson, +3
tests/test_config.py               TestNormaliseUrl, +8
tests/test_context_guard.py        big, +5
tests/test_doctor.py               Server, +7
tests/test_fullscreen.py           FakeTTY, +8
tests/test_install.py              TestPrompts, +7
tests/test_keys.py                 TestKeyNames, +2
tests/test_live_ui.py              TestCodeStreamer, +1
tests/test_logo.py                 TestFittingTheTerminal, +6
tests/test_memory.py               memory, +7
tests/test_mentions.py             project, +5
tests/test_model_picker.py         transport, +6
tests/test_parallel_tools.py       agent, +4
tests/test_parsing.py              TestThinking, +7
tests/test_pet.py                  TestFaces, +5
tests/test_platforms.py            termux, +7
tests/test_projectmap.py           project, +6
tests/test_provider_retry.py       client, +5
tests/test_provider_timeout.py     FakeResponse, +2
tests/test_queue_commands.py       TestPending, +4
tests/test_regex_cost.py           test_the_sweep_actually_found_the_patterns, +1
tests/test_rendering.py            TestAnsiIsNotMangled, +7
tests/test_repo_scope_stream.py    TestRepoTargets, +4
tests/test_scope.py                repo, +9
tests/test_secrets.py              TestWhichFilesAreCredentials, +7
tests/test_session_compaction.py   make_session, +2
tests/test_session_recovery.py     sessions, +5
tests/test_shell_streaming.py      run, +6
tests/test_speech_duo.py           TestSpeakable, +11
tests/test_status.py               render, +4
tests/test_stdin.py                run_cli, +3
tests/test_terminal_handoff.py     FakeWatcher, +4
tests/test_testing.py              project, +6
tests/test_theme_log_select.py     TestTheme, +11
tests/test_thinking_panel.py       plain, +5
tests/test_tools.py                TestReadFile, +11
tests/test_uninstall.py            git, +8
uninstall.py                       Style, +11
uninstall.sh
wynxo/__init__.py
wynxo/__main__.py
wynxo/_agent_hardening.py
wynxo/agent.py                     is_small_talk, +4
wynxo/asciiart.py                  ramp_for, +7
wynxo/bootstrap.py                 main
wynxo/checkpoints.py               Snapshot, +1
wynxo/cli.py                       resolve_command, +9
wynxo/coerce.py                    as_text, +3
wynxo/config.py                    config_dir, +8
wynxo/discovery.py                 Found, +6
wynxo/doctor.py                    Status, +3
wynxo/duo.py                       DuoConfig, +1
wynxo/effort.py                    EffortPolicy, +2
wynxo/fullscreen.py                supported, +2
wynxo/journal.py                   Journal, +2
wynxo/kawaii_effects.py            sparkle_text, +11
wynxo/keys.py                      key_name, +2
wynxo/logo.py                      available, +6
wynxo/memory.py                    MemoryFile, +1
wynxo/mentions.py                  find, +2
wynxo/parsing.py                   ToolCall, +7
wynxo/permissions.py               Decision, +4
wynxo/pet.py                       Mood, +2
wynxo/platforms.py                 is_termux, +11
wynxo/projectmap.py                Entry, +8
wynxo/prompts.py                   git_context, +2
wynxo/provider.py                  ProviderError, +5
wynxo/queue.py                     Pending
wynxo/repo.py                      Target, +5
wynxo/schema.py                    ValidationError, +2
wynxo/scope.py                     Scope, +4
wynxo/secrets.py                   is_secret_file, +4
wynxo/select.py                    Choice, +3
wynxo/session.py                   estimate_tokens, +3
wynxo/speech.py                    Engine, +6
wynxo/status.py                    enable_windows_vt, +2
wynxo/testing.py                   Runner, +3
wynxo/theme.py                     Palette, +2
wynxo/tools/__init__.py            Registry, +1
wynxo/tools/base.py                ToolResult, +1
wynxo/tools/files.py               Decoded, +11
wynxo/tools/memory_tool.py         RememberInput, +1
wynxo/tools/search.py              GlobInput, +3
wynxo/tools/shell.py               hard_refusal, +2
wynxo/tools/todo.py                TodoItem, +2
wynxo/tui.py                       Transcript, +3
wynxo/ui.py                        apply_palette, +10
wynxo/wizard.py                    describe_model, +6