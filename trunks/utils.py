import subprocess
import click
from pathlib import Path
from contextlib import contextmanager


def run(*git_action: str, cwd=None):
    # print(" ".join(git_action))
    result = subprocess.run(["git", *git_action], check=True, capture_output=True, cwd=cwd)
    # print(result.stdout.decode())
    return result.stdout.decode("utf-8")


def get_local_branches() -> list[str]:
    result = run("for-each-ref", "--format=%(refname:short)", "refs/heads/")
    return result.strip().split("\n")


def get_root() -> Path:
    result = run("rev-parse", "--show-toplevel")
    return Path(result.strip())


def get_head() -> str:
    with open(get_root() / ".git" / "HEAD") as f:
        head = f.read().strip()
    if head.startswith("ref: "):
        ref = head.split()[1]
        assert ref.startswith("refs/heads/")
        return ref[len("refs/heads/"):]
    return head


def checkout(ref):
    run("checkout", ref)


def have_diverged(ref_a, ref_b):
    rev_list_1 = run("rev-list", "--format=oneline", f"{ref_a}..{ref_b}")
    rev_list_2 = run("rev-list", "--format=oneline", f"{ref_a}...{ref_b}")
    return rev_list_1 != rev_list_2


def get_current_branch():
    run("rev-parse", "--abbrev-ref", "HEAD")


class Pause(Exception):
    pass


@contextmanager
def preserve_state():
    result = run("status", "--porcelain")
    stash = bool(result.strip())
    pause = False
    if stash:
        run("stash", "-u")
    try:
        head = get_head()
        try:
            yield
        except Pause:
            pause = True
    finally:
        if not pause:
            checkout(head)
            if stash:
                run("stash", "pop")
        else:
            click.echo(f"git checkout {head}")
            click.echo("git stash pop")
