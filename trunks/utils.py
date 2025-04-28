import subprocess
import click
import uuid
from pathlib import Path
from contextlib import contextmanager


class CalledProcessError(subprocess.CalledProcessError):
    pass


def run(*git_action: str, cwd=None):
    # print(" ".join(git_action))
    # try:
    result = subprocess.run(
        ["git", *git_action], check=True, capture_output=True, cwd=cwd
    )
    # except subprocess.CalledProcessError as exc:
    #     raise Exception(f"subprocess exited with error: {exc.stderr}.")
    # print(result.stdout.decode())
    return result.stdout.decode("utf-8")


def get_local_branches() -> list[str]:
    result = run("for-each-ref", "--format=%(refname:short)", "refs/heads/")
    return result.strip().split("\n")


def get_repository_root() -> Path:
    result = run("rev-parse", "--show-toplevel")
    return Path(result.strip())


def get_head() -> str:
    with open(get_repository_root() / ".git" / "HEAD") as f:
        head = f.read().strip()
    if head.startswith("ref: "):
        ref = head.split()[1]
        assert ref.startswith("refs/heads/")
        return ref[len("refs/heads/"):]
    return head


def object_exists(rev):
    try:
        run("rev-parse", "--verify", rev)
    except subprocess.CalledProcessError as exc:
        if exc.returncode == 128:
            return False
        raise
    return True


def checkout(*args):
    run("checkout", *args)


def have_diverged(rev_a, rev_b):
    rev_list_1 = run("rev-list", "--format=oneline", f"{rev_a}..{rev_b}")
    rev_list_2 = run("rev-list", "--format=oneline", f"{rev_a}...{rev_b}")
    return rev_list_1 != rev_list_2


def get_current_branch():
    run("rev-parse", "--abbrev-ref", "HEAD")


class Pause(Exception):
    pass


@contextmanager
def temporary_branch():
    head = get_head()
    name = uuid.uuid4().hex
    run("checkout", "-b", name)
    try:
        yield
    finally:
        checkout(head)
        run("branch", "-D", name)


@contextmanager
def preserve_state(auto_stash=False):
    result = run("status", "-u", "no", "--porcelain")
    work_tree_clean = not bool(result.strip())
    stash = False
    if auto_stash and not work_tree_clean:
        stash = True
    elif not work_tree_clean and not auto_stash:
        # Git interactive rebase message:
        # error: cannot rebase: You have unstaged changes.
        # error: Please commit or stash them.
        raise click.ClickException("work tree not clean.")
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
