"""Hold every shell block of a tutorial to the few forms a strict bash checks completely.

A block runs under `bash -e -u -o pipefail` with `inherit_errexit`. Bash leaves many forms unchecked
by design: a negated command, every command of an `&&` or `||` list but the last, the whole body of
a compound command on the left of one, a background job, a command substitution in an argument, a
nested shell, a trap. Rather than chase them, a block may hold only:
- comments and blank lines, and lines continued by a trailing backslash or `|`;
- variable assignments, `export NAME=value` included;
- simple commands and pipelines, with redirections (a here-string `<<<` too);
- `!` before the block's last command only: its status is the block's.

The command word of every command (the first word after its assignments and `!`) is a literal
word, so the name checked is the name bash runs. `refused` names every line holding anything else
and quotes it: a `;`, `&&` or `||` list, a control structure, a group or a subshell, `trap`, `eval`,
`set`, `exec`, `exit`, `return`, `wait` and the other commands of REFUSED (`setsid` among them: it
leaves the block's process group), `find -exec ... ;` (find ignores that command's status), a
nested shell, a heredoc, a background `&`, and command or process substitution, which also rules out
a substitution inside an assignment. A server that a later block talks to runs in a block of its
own marked `background=1` (tests/docs/runner.py), never behind `&`.

Core tier: stdlib only.
"""
from __future__ import annotations

import dataclasses
import fnmatch
import re

_OPS = ("&&", "||", ";;", "|&", "<<<", "<<-", "&>>", "<<", ">>", "&>", ">&", "<&", "<>", ">|",
        ";", "&", "|", "(", ")", ">", "<")
PIPES = ("|", "|&")
REDIRECTS = ("<<<", "&>>", ">>", "&>", ">&", "<&", "<>", ">|", ">", "<")
LISTS = {";": "a `;` list", ";;": "a `;` list", "&&": "an `&&` list", "||": "an `||` list",
         "&": "a background job (`&`): a server runs in a `background=1` block",
         "(": "a subshell, group or substitution", ")": "a subshell, group or substitution",
         "<<": "a heredoc", "<<-": "a heredoc"}
RESERVED = ("if", "then", "else", "elif", "fi", "for", "while", "until", "do", "done", "case", "esac",
            "select", "function", "time", "coproc", "{", "}", "[[", "]]")
REFUSED = ("trap", "eval", "set", "exec", "exit", "return", "wait", "shopt", "source", ".", "builtin",
           "command", "declare", "local", "readonly", "typeset", "alias", "enable", "setsid", "script")
SHELLS = ("bash", "sh", "dash", "zsh", "ksh", "fish")
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_LITERAL = re.compile(r"[A-Za-z0-9_./+-]+|\[")       # a command word bash takes as it is written
_ASSIGN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\[[^\]]*\])?\+?=")


@dataclasses.dataclass(frozen=True)
class Token:
    text: str              # as written, quotes included
    op: bool               # an operator (`|`, `>`, `&&`, ...) rather than a word
    quoted: bool = False   # holds a quote or an escape
    substitutes: bool = False   # holds `$(`, `$((` or a backtick outside single quotes


def tokens(line: str) -> tuple[list[Token], bool]:
    """(the tokens of one logical line up to an unquoted comment, whether it ends inside a quote)."""
    out: list[Token] = []
    buf, quoted, subst, i, n = [], False, False, 0, len(line)

    def flush() -> None:
        nonlocal buf, quoted, subst
        if buf:
            out.append(Token("".join(buf), False, quoted, subst))
        buf, quoted, subst = [], False, False

    while i < n:
        c = line[i]
        if c in " \t\n":
            flush()
            i += 1
        elif c == "#" and not buf:
            break
        elif c == "\\":
            buf.append(line[i:i + 2])
            quoted, i = True, i + 2
        elif c in "'\"":
            j = i + 1
            while j < n and line[j] != c:
                if c == '"' and line[j] == "\\":
                    j += 1
                elif c == '"' and (line[j] == "`" or line.startswith("$(", j)):
                    subst = True
                j += 1
            if j >= n:
                buf.append(line[i:])
                flush()
                return out, True
            buf.append(line[i:j + 1])
            quoted, i = True, j + 1
        elif c == "`" or line.startswith("$(", i):
            subst = True
            buf.append(c)
            i += 1
        elif c in ";&|()<>":
            if c in "<>" and buf and "".join(buf).isdigit():   # a file descriptor: 2>, 0<
                buf = []
            flush()
            op = next(o for o in _OPS if line.startswith(o, i))
            out.append(Token(op, True))
            i += len(op)
        else:
            buf.append(c)
            i += 1
    flush()
    return out, False


def _continued(line: str, toks: list[Token], open_quote: bool) -> bool:
    backslashes = len(line) - len(line.rstrip("\\"))
    return open_quote or backslashes % 2 == 1 or bool(toks and toks[-1].op and toks[-1].text in PIPES)


def logical_lines(text: str) -> list[tuple[int, str, list[Token], bool]]:
    """(1-based line of the block where it starts, the line as written with its continuations joined,
    its tokens, whether a quote is left open) of every line of a shell block that holds a command; blank
    lines, comments and heredoc bodies hold none."""
    raw, out, i = text.splitlines(), [], 0
    while i < len(raw):
        start, cur = i + 1, raw[i]
        i += 1
        toks, open_quote = tokens(cur)
        while i < len(raw) and _continued(cur, toks, open_quote):
            backslash = not open_quote and (len(cur) - len(cur.rstrip("\\"))) % 2 == 1
            cur = cur[:-1] + raw[i] if backslash else f"{cur}\n{raw[i]}"
            i += 1
            toks, open_quote = tokens(cur)
        for k, t in enumerate(toks):                 # a heredoc's body is data, not commands
            if t.op and t.text in ("<<", "<<-") and k + 1 < len(toks) and not toks[k + 1].op:
                delimiter = re.sub(r"[\"'\\]", "", toks[k + 1].text)
                while i < len(raw) and raw[i].strip() != delimiter:
                    i += 1
                i += 1
        if toks:
            out.append((start, cur, toks, open_quote))
    return out


def stages(toks: list[Token]) -> list[list[Token]]:
    """The words of each command of a pipeline, without its redirections and their targets."""
    out, cur, skip = [], [], False
    for t in toks:
        if t.op and t.text in PIPES:
            out.append(cur)
            cur = []
        elif t.op:
            skip = t.text in REDIRECTS
        elif skip:
            skip = False
        else:
            cur.append(t)
    return out + [cur]


def _why(toks: list[Token], last: bool, open_quote: bool) -> str | None:
    if open_quote:
        return "a quote is left open"
    for t in toks:
        if t.op and t.text in LISTS:
            return LISTS[t.text]
        if t.substitutes:
            return "a command substitution hides its status"
    for k, words in enumerate(stages(toks)):
        lead = next((j for j, w in enumerate(words) if not _ASSIGN.match(w.text)), len(words))
        names = words[lead:]
        if not names:
            continue                                  # assignments only
        name = names[0].text
        if name == "!":
            if k or not last:
                return "`!` stands only before the block's last command: bash -e does not stop on it"
            names = names[1:]
            name = names[0].text if names else ""
        if name in RESERVED:
            return f"a control structure or group (`{name}`)"
        if name in REFUSED:
            return f"`{name}` changes the block's shell or what it checks"
        if names and (names[0].quoted or not _LITERAL.fullmatch(name)):
            return f"the command word `{name}` is not a literal word (no quotes, escapes, `$` or globs)"
        if name == "export" and not all(_ASSIGN.match(w.text) or _NAME.fullmatch(w.text) for w in names[1:]):
            return "`export` takes NAME or NAME=value only"
        shell = next((w.text for w in names if _names_a_shell(w)), None)
        if shell:
            return f"a nested shell (`{shell}`) runs commands this block does not check"
        if name == "env" and any(w.text in ("-S", "--split-string") or w.text.startswith(("-S", "--split-string="))
                                 for w in names[1:]):
            return "`env -S` splits a string into a command this block does not check"
        if name == "find" and _finds_with_semicolon(names):
            return "`find -exec ... ;` ignores the status of the command it runs: end it with `+`"
    return None


def _names_a_shell(w: Token) -> bool:
    """A word that names a shell program: literally, or as an unquoted glob that matches a shell's
    name and not just any name (`ba?h` does, `*` and `pi05_droid/*` do not)."""
    base = re.sub(r"[\"'\\]", "", w.text).rsplit("/", 1)[-1]
    if w.quoted or not any(c in base for c in "*?["):
        return base in SHELLS
    return any(fnmatch.fnmatchcase(s, base) for s in SHELLS) and not fnmatch.fnmatchcase("zzzz", base)


def _finds_with_semicolon(names: list[Token]) -> bool:
    plain = [re.sub(r"[\"'\\]", "", w.text) for w in names]
    return any(w in ("-exec", "-execdir", "-ok", "-okdir") and ";" in plain[k + 1:] for k, w in enumerate(plain))


def refused(text: str) -> list[tuple[int, str, str]]:
    """(line of the block, the line as written, why) of every line of a shell block that holds a form
    the module docstring does not allow."""
    lines = logical_lines(text)
    out = []
    for k, (line, cur, toks, open_quote) in enumerate(lines):
        why = _why(toks, k == len(lines) - 1, open_quote)
        if why:
            out.append((line, " ".join(cur.split()), why))
    return out


def commands(text: str) -> list[str]:
    """Each command of a shell block as written (its words, without redirections), one per pipeline
    stage."""
    return [" ".join(w.text for w in words)
            for _, _, toks, _ in logical_lines(text) for words in stages(toks) if words]
