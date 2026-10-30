"""The shell forms a tutorial block may hold (tests/docs/shell.py): everything outside the allow-list
is refused by name, quoting its line."""
from __future__ import annotations

import pytest

from tests.docs import shell


def test_tokens_keep_quotes_and_read_operators_and_file_descriptors():
    toks, open_quote = shell.tokens("a 'b c' \"d $x\" 2>&1 >> f |& g # h && i")
    assert [t.text for t in toks] == ["a", "'b c'", '"d $x"', ">&", "1", ">>", "f", "|&", "g"]
    assert [t.op for t in toks] == [False, False, False, True, False, True, False, True, False]
    assert not open_quote and shell.tokens('echo "open')[1]


def test_logical_lines_join_continuations_and_skip_heredoc_bodies():
    text = ("qvla train \\\n  a.toml\n"
            "echo \"one\ntwo\"\n"
            "grep x f |\n  tee y\n"
            "# a comment\n\n"
            "cat > f <<-'EOF'\n\tqvla x && y\n\tEOF\n"
            "ls\n")
    assert [(n, cur) for n, cur, _, _ in shell.logical_lines(text)] == [
        (1, "qvla train   a.toml"), (3, 'echo "one\ntwo"'), (5, "grep x f |\n  tee y"), (9, "cat > f <<-'EOF'"),
        (12, "ls")]


def test_commands_are_the_words_of_each_pipeline_command_without_redirections():
    assert shell.commands("A=1 qvla x > log 2>&1 | tee -a y\n") == ["A=1 qvla x", "tee -a y"]


@pytest.mark.parametrize("text, why", [
    ("a; b\n", "a `;` list"),
    ("a && b\n", "an `&&` list"),
    ("a || true\n", "an `||` list"),                       # also as the last command: `||` swallows
    ("for f in a; do false; echo x; done || true\n", "a `;` list"),
    ("{ false; echo x; } || true\n", "a `;` list"),
    ("done || true\n", "an `||` list"),
    ("a || continue\n", "an `||` list"),
    ("a || exit 0\n", "an `||` list"),
    ("if a\n", "a control structure or group (`if`)"),
    ("while a\n", "a control structure or group (`while`)"),
    ("time ! a\n", "a control structure or group (`time`)"),
    ("{ a\n", "a control structure or group (`{`)"),
    ("[[ -n a ]]\n", "a control structure or group (`[[`)"),
    ("(cd d)\n", "a subshell, group or substitution"),
    ("f() { a\n", "a subshell, group or substitution"),
    ("diff a <(b)\n", "a subshell, group or substitution"),
    ("trap 'exit 0' EXIT\n", "`trap` changes the block's shell"),
    ("eval 'set +e'\n", "`eval` changes the block's shell"),
    ("set +e\n", "`set` changes the block's shell"),
    ("shopt -u inherit_errexit\n", "`shopt` changes the block's shell"),
    ("exec true\n", "`exec` changes the block's shell"),
    ("exit 0\n", "`exit` changes the block's shell"),
    ('W="$PWD" exit 0\n', "`exit` changes the block's shell"),
    ("wait\n", "`wait` changes the block's shell"),
    ("source env.sh\n", "`source` changes the block's shell"),
    ("bash -c 'false; true'\n", "a nested shell (`bash`)"),
    ("env -i /bin/sh run.sh\n", "a nested shell (`/bin/sh`)"),
    ("cat > f <<EOF\nx\nEOF\n", "a heredoc"),
    ("bash <<EOF\nfalse\nEOF\n", "a heredoc"),
    ("server &\n", "a background job (`&`)"),
    ('echo "$(false)"\n', "a command substitution hides its status"),
    ("echo `false`\n", "a command substitution hides its status"),
    ("x=$(false)\n", "a command substitution hides its status"),
    ("export X=$(false)\n", "a command substitution hides its status"),
    ("export -n X\n", "`export` takes NAME or NAME=value only"),
    ("! a\nb\n", "`!` stands only before the block's last command"),
    ("a | ! b\n", "`!` stands only before the block's last command"),
    ("echo 'open\n", "a quote is left open"),
    # the command word is literal, so the name checked is the name bash runs
    ('"set" +e\n', "the command word `\"set\"` is not a literal word"),
    ("\\set +e\n", "the command word `\\set` is not a literal word"),
    ("$x +e\n", "the command word `$x` is not a literal word"),
    ("${x} +e\n", "the command word `${x}` is not a literal word"),
    ('"exit" 0\n', "the command word `\"exit\"` is not a literal word"),
    ('"bash" -c true\n', "the command word `\"bash\"` is not a literal word"),
    ("/bin/ba?h -c true\n", "the command word `/bin/ba?h` is not a literal word"),
    ('env "bash" -c true\n', 'a nested shell (`"bash"`)'),
    ("xargs /bin/ba?h -c true\n", "a nested shell (`/bin/ba?h`)"),
    ("setsid -f server\n", "`setsid` changes the block's shell"),
    ("find . -name x -exec rm {} \\;\n", "`find -exec ... ;` ignores the status"),
    ("find . -execdir rm {} ';'\n", "`find -exec ... ;` ignores the status"),
    ("ba?h -c true\n", "the command word `ba?h` is not a literal word"),
    ("xargs b\\ash -c true\n", "a nested shell (`b\\ash`)"),
    ("env -S 'sh -c \"false; true\"'\n", "`env -S` splits a string into a command"),
    ("script -qc 'false; true' /dev/null\n", "`script` changes the block's shell"),
])
def test_every_form_outside_the_allow_list_is_refused_by_name(text, why):
    (found,) = shell.refused(text)[:1]
    assert why in found[2] and found[0] == 1, found


@pytest.mark.parametrize("text", [
    "a\n",
    "# only a comment\na --flag 'x; y && z' \"$HOME/b || c\" \\# d\n",
    "a > out 2>&1\nb >> out 2> err < in\nc &> both\nd >&2\n",
    "a | b |& c\n",
    "qvla train \\\n  --x y\n",
    "grep x f |\n  tee y\n",
    "X=1\nexport Y=2 Z\nW=\"$PWD/srv\" qvla x\n",
    "cat <<< 'a string'\n",
    "[ -s f ]\ntest -d d\n",
    "a\n! b\n",                                          # `!` before the last command
    "echo {a,b} ${X:-d}\n",
    "find . -name x -exec ls {} +\n",                  # `+`: find fails when the command does
    # globs that match any name are arguments, never a shell: the planned pages' fetches and copies
    "qvla data get corpus --include 'index/*' --include 'signals/v1/approach_aware/chunk/*' "
    "--include 'episodes/*/meta/episodes/*' --include 'episodes/*/meta/tasks.parquet' --dest corpus\n",
    "qvla data get base-weights --include pi05_droid\n",
    "qvla data get corpus --include 'index/*' --include 'signals/v1/approach_aware/chunk/*' "
    "--include 'norm_stats/*' --include 'episodes/*'\n",
    "cp data/qvla-v1-bench/assets/textures/polyhaven/downloaded_256/* "
    "third_party/axis-render/assets/table_textures/polyhaven/downloaded_256/\n",
    "cp a/* b/\nls *\nqvla data get corpus --include \"index/*\" --include '[ab]*'\n",
    "grep -lw shell notes.txt\n",
])
def test_what_the_allow_list_holds_passes(text):
    assert shell.refused(text) == []
