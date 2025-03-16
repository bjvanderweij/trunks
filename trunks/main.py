"""Tool for managing a stacked diffs workflow.

Concepts:

Upstream
Local
Difference between local and upstream.

Configuration files:

    global ~/.trunksrc
    repo .trunksrc

Configuration options:

    upstream
    local
    branch-template
"""

import functools
import typing
import subprocess
import tempfile
import re
from hashlib import md5
from graphlib import TopologicalSorter

import graphviz
import click
from trunks import utils

UPSTREAM = "origin/develop"
LOCAL = "bastiaan-develop"
BRANCH_TEMPLATE = "feature/{}"
EDITOR = "vim"
AUTO_STASH = True


INSTRUCTIONS = """

# Edit branch-creation plan.
#
# Commands:
# b<label> <commit> = use commit in labeled branch
# b<label>@b<target-label> <commit> = use commit in labeled branch from target-label
# s <commit> = do not use commit
#
# If you delete a line, the commit will not be used (equivalent to "s")
# If you remove everything, nothing will be changed
#
"""


def no_hot_branch(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if utils.get_current_branch() in _get_hot_branches():
            raise click.ClickException(
                "please switch to a branch not managed by trunks before "
                "continuing"
            )
        return f(*args, **kwargs)

    return wrapper


def undiverged_trunks(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if utils.have_diverged(UPSTREAM, LOCAL):
            raise click.ClickException(
                "your trunks have diverged, rebase local on your upstream"
            )
        return f(*args, **kwargs)

    return wrapper


def no_unstaged_changes(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        result = utils.run("status", "--porcelain")
        if bool(result.strip()):
            raise click.ClickException(
                "you have unstaged changes, please commit or stash them"
            )
        return f(*args, **kwargs)

    return wrapper


class TrunksDiverged(Exception):
    """Raise if local and upstream have diverged."""

    pass


class PlanError(Exception):
    pass


class ParsingError(Exception):
    pass


class CherryPickFailed(Exception):
    pass


class Commit(typing.NamedTuple):
    sha: str
    message: str

    @classmethod
    def from_oneline(cls, oneline: str):
        """Parse from 'oneline' format of git rev-list."""
        sha, *message_words = oneline.split()
        return cls(sha, " ".join(message_words))

    @property
    def short_str(self):
        return f"{self.sha[:8]} {self.message}"

    @property
    def branch_name(self):
        uniqueish = md5(self.message.encode()).hexdigest()[:8]
        readable = "-".join(self.message.lower().split()[:4])
        return BRANCH_TEMPLATE.format(f"{readable}-{uniqueish}")


class Branch(typing.NamedTuple):
    commits: list[Commit]
    target: typing.Optional["Branch"]

    @property
    def name(self):
        return self.commits[-1].branch_name

    @property
    def full_name(self):
        return f"refs/heads/{self.name}"

    def get_force_push_command(
        self, remote: str, gitlab_merge_request: bool = False
    ):
        command = [
            "push",
            "--force",
            f"--set-upstream {remote}",
            f"{self.full_name}:{self.fullname}",
        ]
        if gitlab_merge_request:
            command += ["--push-option", "merge_request.create"]
            if self.target is not None:
                command += [
                    "--push-option",
                    "merge_request.target={self.target.name}",
                ]
        return command

    def exists(self):
        return self.name in utils.get_local_branches()


def validate_tree(tree: dict[str, Branch]):
    """Validate a branch DAG.

    Actually, it's not a DAG, it's a tree

    All commits in the branch must be consecutive.
    Following a branch, new branches can only point to that branch, and not
    branches before it? This one might be optional but it feels necessary.

    There should be no duplicate commit messages
    """
    ...


def build_tree_from_local_commits(update_commits=False) -> dict[str, Branch]:
    """Take the leading commits and rebuild the tree."""
    commits = get_local_commits()
    commits_by_message = {c.message: c for c in commits}
    local_branches = utils.get_local_branches()
    tree: dict[str, Branch] = {}
    for i, commit in enumerate(commits):
        if commit.branch_name in local_branches:
            preceding_commits = get_last_n_commits(commit.branch_name, i)
            assert preceding_commits[-1].branch_name == commit.branch_name, (
                "inconsistent state: trunks-managed branch name "
                f"{commit.branch_name} does not match branch name expected "
                "based on its last commit"
            )
            target_branch = None
            branch_commits: list[Commit] = []
            for preceding_commit in reversed(preceding_commits):
                branch_name = preceding_commit.branch_name
                if branch_name in tree and branch_name != commit.branch_name:
                    target_branch = tree[preceding_commit.branch_name]
                    break
                elif preceding_commit.message in commits_by_message:
                    if update_commits:
                        branch_commits.insert(
                            0, commits_by_message[preceding_commit.message]
                        )
                    else:
                        branch_commits.insert(0, preceding_commit)
                else:
                    break
            branch = Branch(branch_commits, target_branch)
            tree[commit.branch_name] = branch
    return tree


def create_or_update_branches(tree: dict[str, Branch]):
    """Sort the tree in topological order.

    - stash everything (git stash -u)
    - record HEAD position

    Start at the roots of the tree (Branch with None target). For each:

    1 check out origin/develop
    2 check if the feature branch exists already, if so, force-delete it
    3 create a feature branch for the commits (using the name of the first
      commit)
    4 cherry-pick the commits
    5 if the cherry-pick fails (as may happen with non-consecutive commits)
        - abort, revert to HEAD, delete the branch, pop stash
        - allowing fixing this would break the requirement that merge requests
          are subsets of the local develop
    5 if successful
        - (optionally) force push to origin (maybe interactive question?)
        - print a message that the branch was created successfully

    When reaching a non-root, do pretty much the same:

    - check out the target branch (it must already exist due to topological
      sorting)
    - do steps 2 to 5
    """
    dag: dict[str, list[str]] = {}
    for name, branch in tree.items():
        if name not in dag:
            dag[name] = []
        if branch.target is not None:
            dag[name].append(branch.target.name)
    ts = TopologicalSorter(dag)
    for branch_name in ts.static_order():
        branch = tree[branch_name]
        target = UPSTREAM
        if branch.target is not None:
            target = branch.target.name
        utils.run("checkout", target)
        action = "created"
        if branch.exists():
            action = "updated"
            utils.run("branch", "-D", branch.name)
        utils.run("checkout", "-b", branch.name)
        try:
            utils.run("cherry-pick", *[c.sha for c in branch.commits])
        except subprocess.CalledProcessError:
            raise CherryPickFailed(
                f"Cherry-pick failed at branch {branch.name}."
            )
        click.echo(f"{action} {branch_name} (target {target}) with commits:")
        for commit in reversed(branch.commits):
            click.echo(f"\t{commit.short_str}")


class _BranchCommand(typing.NamedTuple):
    label: str
    target_label: None | str
    commit_sha: str


class _CommitList(typing.NamedTuple):
    label: str
    target_label: None | str
    commits: list[Commit]


def iterate_plan(plan: str):
    """Iterate through lines, skipping empty lines or comments."""
    for line in plan.split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        yield line


def _tokenize_plan(plan: str) -> typing.Iterable[_BranchCommand]:
    for line in iterate_plan(plan):
        try:
            command, sha, *_ = line.split()
        except ValueError:
            raise ParsingError(
                "each line should contain at least a command and a commit SHA"
            )
        if command.startswith("b"):
            m = re.match(r"(b[0-9]*)(@(b[0-9]*))?$", command)
            if not m:
                raise ParsingError(f"unrecognized command: {command}")
            label, _, target = m.groups()
            yield _BranchCommand(label, target, sha)
        elif command != "s":
            raise ParsingError(f"unrecognized command: {command}")


def _make_commit_lists(
    branch_commands: typing.Iterable[_BranchCommand],
) -> list[_CommitList]:
    """Build lists of contiguous commits belonging to a branch."""
    branches: list[_CommitList] = []
    local_commits = iter(get_local_commits())
    for bc in branch_commands:
        if len(branches) == 0 or bc.label != branches[-1].label:
            branches.append(_CommitList(bc.label, None, []))
        try:
            commit = next(
                c for c in local_commits if c.sha.startswith(bc.commit_sha)
            )
        except StopIteration:
            raise PlanError(
                "commits are not ordered correctly or unrecognized commit"
            )
        branches[-1].commits.append(commit)
        if bc.target_label is not None:
            if branches[-1].target_label is None:
                branch = branches.pop(-1)
                branches.append(branch._replace(target_label=bc.target_label))
            elif branches[-1].target_label != bc.target_label:
                raise PlanError(
                    f"multiple targets specified for {branches[-1].label}"
                )
    return branches


def _build_tree(
    candidate_branches: typing.Iterable[_CommitList],
) -> dict[str, Branch]:
    """Parse branching plan and return a branch DAG.

    This enforces constraints on the DAG that can be built:

    - branches only point to either
        - the target of the last branch
        - one of the set of immediately preceding branches with the same target
    - because the order of commits must be preserved, commits in a branch
      must appear in the same order as the local commits
    """
    branches: dict[str, Branch] = {}
    last_target_label = None
    valid_target_labels: set[None | str] = {None}
    for cb in candidate_branches:
        if cb.label in branches:
            raise PlanError("commits in branch must be contiguous")
        if cb.target_label not in valid_target_labels:
            raise PlanError(
                f"invalid target for {cb.label}: {cb.target_label}"
            )
        target_branch = None
        if cb.target_label is not None:
            target_branch = branches[cb.target_label]
        if cb.target_label != last_target_label:
            last_target_label = cb.target_label
            valid_target_labels = {last_target_label}
        valid_target_labels.add(cb.label)
        branches[cb.label] = Branch(cb.commits, target_branch)
    return {b.name: b for b in branches.values()}


def parse_plan(plan: str) -> dict[str, Branch]:
    tokens = _tokenize_plan(plan)
    commit_lists = _make_commit_lists(tokens)
    return _build_tree(commit_lists)


def generate_plan(tree: dict[str, Branch] | None = None) -> str:
    """Generate merging plan.

    Use the commit messages of the local commits to derive branch names
    and check if these branches exist. If they do, figure out what their
    target branch is by walking backwards in the git log until a commit
    is o

    The plan is on the tree built from the current branches and contains
    commits whose sha might differ from the one in the local commits,
    but the commits in the plan correspond to those in the local commits.
    """
    if tree is None:
        tree = build_tree_from_local_commits()
    current_branch = None
    branch_index = len(tree)
    index_map: dict[str, int] = {}
    commands = []
    local_commits = get_local_commits()
    for commit in reversed(local_commits):
        new_current_branch = tree.get(commit.branch_name, None)
        if new_current_branch is not None:
            if (
                current_branch is not None
                and new_current_branch.name != current_branch.name
            ):
                branch_index -= 1
                current_branch = new_current_branch
            if current_branch is None:
                current_branch = new_current_branch
        command = "s"
        if current_branch is not None:
            if commit.message in [c.message for c in current_branch.commits]:
                index_map[current_branch.name] = branch_index
                command = f"b{branch_index}"
                if current_branch.target is not None:
                    command += f"@b{current_branch.target.name}"
        commands.append(command)
    lines = []
    for command, commit in zip(commands, reversed(local_commits)):
        m = re.match(r"(b[0-9]+)@b(.*)$", command)
        if m is not None:
            for branch in tree.values():
                if m.group(2) == branch.name:
                    command = m.group(1) + "@b" + str(index_map[branch.name])
        lines.append(f"{command} {commit.sha[:8]} {commit.message}")
    return "\n".join(reversed(lines))


def _get_hot_branches() -> set[str]:
    commits = get_local_commits()
    local_branches = utils.get_local_branches()
    return set(local_branches) & set(c.branch_name for c in commits)


def prune_local_branches(tree):
    hot_branches = _get_hot_branches()
    branches_to_prune = hot_branches - set(tree.keys())
    for branch_name in branches_to_prune:
        click.echo(f"pruning {branch_name}")
        utils.run("branch", "-D", branch_name)


def get_commits(rev_a, rev_b):
    """Return commits between rev_a and rev_b in chronological order."""
    rev_list_output = utils.run(
        "rev-list",
        "--no-merges",
        "--format=oneline",
        f"{rev_a}..{rev_b}",
        "--",
    )
    rev_list = reversed(rev_list_output.strip().split("\n"))
    return [Commit.from_oneline(line) for line in rev_list if line]


def get_last_n_commits(rev, n) -> list[Commit]:
    """Return the n commits leading up to rev, including rev."""
    return get_commits(f"{rev}~{n + 1}", rev)


def get_local_commits() -> list[Commit]:
    """Return all commits between upstream and local."""
    commits = get_commits(UPSTREAM, LOCAL)
    if len(commits) != len(set(c.message for c in commits)):
        raise click.ClickException("Local commit messages must be unique.")
    return commits


def edit_interactively(contents: str) -> str:
    with tempfile.NamedTemporaryFile("w") as text_file:
        text_file.write(contents)
        text_file.seek(0)
        subprocess.run([EDITOR, text_file.name])
        with open(text_file.name, "r") as text_file_read:
            return text_file_read.read()


@click.group()
def cli_group():
    pass


def cli():
    try:
        cli_group()
    except subprocess.CalledProcessError as exc:
        click.echo("subprocess failed:")
        click.echo(exc)
        click.echo("captured output:")
        click.echo(exc.output)
        click.echo(exc.stderr)


def update_interactively(tree):
    with utils.preserve_state(auto_stash=AUTO_STASH):
        try:
            create_or_update_branches(tree)
        except CherryPickFailed as exc:
            pause = click.confirm("{exc}. Pause here and investigate?")
            if pause:
                click.echo("OK. To reset your workspace, run:")
                click.echo("git cherry-pick --abort")
                raise utils.Pause()
            else:
                utils.run("cherry-pick", "--abort")
                utils.run("checkout", "-")
                utils.run("branch", "-D", exc.branch.name)
            raise click.ClickException(f"{exc}")
    prune_local_branches(tree)


def update_options(f):
    click.argument(
        "remote",
        nargs=1,
        default="origin",
        type=str,
    )(f)
    click.option(
        "-p",
        "--push",
        is_flag=True,
        type=bool,
        help="Force-push updated branches to remote",
    )(f)
    click.option(
        "-n",
        "--no-prune",
        is_flag=True,
        type=bool,
        help="Do not prune trunks branches that are not in the plan anymore",
    )(f)
    click.option(
        "-m",
        "--gitlab-merge-request",
        is_flag=True,
        type=bool,
        help="Use Gitlab-specific push-options to create a merge request",
    )(f)
    return f


@cli_group.command
@update_options
@no_hot_branch
def update(remote, local, no_prune, gitlab_merge_request):
    """Create local branches based on the plan."""
    tree = build_tree_from_local_commits(update_commits=True)
    _update(tree, remote, local, no_prune, gitlab_merge_request)


def _update(tree, remote, local, no_prune, gitlab_merge_request):
    update_interactively(tree, prune=not no_prune)
    if not local:
        for branch in tree.values():
            push_command = branch.get_force_push_command(
                remote, gitlab_merge_request=gitlab_merge_request
            )
            utils.run(*push_command)


@cli_group.command
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    type=bool,
    help="Print relevant trunks configuration",
)
@click.option(
    "-V",
    "--visual",
    is_flag=True,
    type=bool,
    help="Render a visual representation of the plan",
)
def plan(verbose, visual):
    """Display the current plan."""
    if verbose:
        click.echo(f"upstream: {UPSTREAM}")
        click.echo(f"local: {LOCAL}\n")
    if visual:
        tree = build_tree_from_local_commits()
        dot = graphviz.Digraph()
        dot.node(UPSTREAM)
        for branch in tree.values():
            dot.node(branch.name)
            target = UPSTREAM if branch.target is None else branch.target.name
            dot.edge(branch.name, target)
        with tempfile.NamedTemporaryFile() as f:
            dot.render(format="png", filename=f.name, view=True)
    else:
        click.echo(generate_plan())


@cli_group.command()
@update_options
@no_hot_branch
@undiverged_trunks
def edit(remote, local, no_prune, gitlab_merge_request):
    """Edit the plan interactively and update local branches."""
    plan = generate_plan()
    new_plan = edit_interactively(plan + INSTRUCTIONS)
    new_plan = "\n".join(iterate_plan(new_plan))
    if not new_plan.strip() or new_plan.strip() == plan.strip():
        click.echo("Nothing to do.")
        return
    try:
        tree = parse_plan(new_plan)
    except PlanError as exc:
        click.echo("\n".join(iterate_plan(new_plan)))
        raise click.ClickException(f"invalid plan: {exc}")
    except ParsingError as exc:
        click.echo("\n".join(iterate_plan(new_plan)))
        raise click.ClickException(f"failed to parse plan: {exc}")
    validate_tree(tree)
    _update(tree, remote, local, no_prune, gitlab_merge_request)


@cli_group.command()
@click.option(
    "-c",
    "--commits",
    is_flag=True,
    type=bool,
    help="Print commits in each branch",
)
@click.option(
    "-t",
    "--show-targets",
    is_flag=True,
    type=bool,
    help="Print target of each branch",
)
def branches(commits: bool, show_targets: bool):
    """List active branches.

    For all commits between origin/develop and the tip of local develop
    find, feature branches with a branch name derived from that commit.
    """
    tree = build_tree_from_local_commits()
    for branch in tree.values():
        if show_targets:
            target = UPSTREAM
            if branch.target is not None:
                target = branch.target.name
            click.echo(f"{branch.name} --> {target}")
        else:
            click.echo(branch.name)
        if commits:
            for commit in reversed(branch.commits):
                click.echo(f"\t{commit.sha[:8]} {commit.message}")
