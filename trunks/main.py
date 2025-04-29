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
import sys
from hashlib import md5
from graphlib import TopologicalSorter

import graphviz
import click
from trunks import utils

UPSTREAM = "origin/develop"
LOCAL = "bastiaan-develop"
# UPSTREAM = "remote"
# LOCAL = "local"
BRANCH_TEMPLATE = "feature/{}"
EDITOR = "vim"
AUTO_STASH = False


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
    def __init__(self, *args, branch, **kwargs):
        self.branch = branch


class Commit(typing.NamedTuple):
    sha: str
    message: str

    @classmethod
    def from_oneline(cls, oneline: str):
        """Parse from 'oneline' format of git rev-list."""
        sha, *message_words = oneline.split()
        return cls(sha, " ".join(message_words))

    @property
    def short_message(self):
        return self.message.split('\n')[0]

    @property
    def short_str(self):
        return f"{self.sha[:8]} {self.short_message}"

    @property
    def branch_name(self):
        uniqueish = md5(self.message.encode()).hexdigest()[:8]
        words = re.findall(r"\w+", self.message.lower())
        readable = '-'.join(words)
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

    @property
    def target_name(self):
        if self.target is None:
            return UPSTREAM
        else:
            return self.target.name

    def exists(self):
        return utils.object_exists(self.full_name)

    def delete(self):
        utils.run("branch", "-D", self.name)

    def create(self):
        utils.checkout("-b", self.name)

    @property
    def create_instructions(self) -> str:
        return (
            f"git checkout {self.target_name}\n"
            f"git checkout -b temporary-investigation-branch\n"
            f"git cherry-pick {' '.join([c.sha for c in self.commits])}"
        )

    def cherry_pick(self):
        try:
            utils.run("cherry-pick", *[c.sha for c in self.commits])
        except subprocess.CalledProcessError:
            raise CherryPickFailed(
                f"Cherry-pick failed at branch {self.name}.", branch=self
            )

    def get_force_push_command(
        self, remote: str, gitlab_merge_request: bool = False
    ):
        command = [
            "push",
            "--force",
            "--set-upstream",
            remote,
            f"{self.full_name}:{self.full_name}",
        ]
        if gitlab_merge_request:
            command += ["--push-option", "merge_request.create"]
            if self.target is not None:
                command += [
                    "--push-option",
                    f"merge_request.target={self.target.name}",
                ]
        return command

    def __str__(self):
        return (
            "Branch {self.name} with commits:"
            "\n".join(f"\t{c.short_message}" for c in self.commits)
        )


def get_trunks_branches():
    local_branches = utils.get_local_branches()
    commits = get_local_commits()
    branch_names = [c.branch_name for c in commits]
    return [b for b in local_branches if b in branch_names]


def make_simple_tree(stack) -> dict[str, Branch]:
    """Use local commits create a tree of stacked branches."""
    commits = get_local_commits()
    tree: dict[str, Branch] = {}
    target_branch = None
    for commit in commits:
        branch = Branch([commit], target_branch)
        tree[commit.branch_name] = branch
        if stack:
            target_branch = branch
    return tree


def reconstruct_tree(use_branch_commits=False) -> dict[str, Branch]:
    """Use local commits to reconstruct the plan."""
    commits = get_local_commits()
    commits_by_message = {c.message: c for c in commits}
    local_branches = utils.get_local_branches()
    tree: dict[str, Branch] = {}
    for i, commit in enumerate(commits):
        if commit.branch_name in local_branches:
            preceding_commits = get_last_n_commits(commit.branch_name, i)
            if preceding_commits[-1].branch_name != commit.branch_name:
                raise click.ClickException(
                    "Invalid state: trunks-managed branch name "
                    f"{commit.branch_name} does not match branch name "
                    "expected based on its last commit.\n\nRun\n\ngit branch "
                    "-D {commit.branch_name}\n\nto remove the offending "
                    "branch. Or run\n\ntrunks reset\n\nif you'd like to start "
                    "with a clean slate."
                )
            target_branch = None
            branch_commits: list[Commit] = []
            for preceding_commit in reversed(preceding_commits):
                branch_name = preceding_commit.branch_name
                if branch_name in tree and branch_name != commit.branch_name:
                    target_branch = tree[preceding_commit.branch_name]
                    break
                elif preceding_commit.message in commits_by_message:
                    if use_branch_commits:
                        branch_commits.insert(0, preceding_commit)
                    else:
                        branch_commits.insert(
                            0, commits_by_message[preceding_commit.message]
                        )
                else:
                    break
            branch = Branch(branch_commits, target_branch)
            tree[commit.branch_name] = branch
    return tree


def create_or_update_branches(tree: dict[str, Branch]):
    """Create feature branches based on the plan in tree.

    Start at the roots of the tree and for each branch in the topologically
    sorted branches, checkout its target (UPSTREAM if None), delete the branch
    if it already exists, create the branch, cherry-pick its commits.

    Return a dictionary that maps each branch name to a boolean that is True
    only if the branch already existed and was re-created.
    """
    dag: dict[str, list[str]] = {}
    for name, branch in tree.items():
        if name not in dag:
            dag[name] = []
        if branch.target is not None:
            dag[name].append(branch.target.name)
    ts = TopologicalSorter(dag)
    updated = {}
    for branch_name in ts.static_order():
        branch = tree[branch_name]
        utils.checkout(branch.target_name)
        with utils.temporary_branch():
            try:
                branch.cherry_pick()
            except CherryPickFailed:
                try:
                    utils.run("cherry-pick", "--abort")
                except subprocess.CalledProcessError:
                    click.echo("Failed to abort cherry-pick.")
                raise
            if branch.exists():
                branch.delete()
                updated[branch_name] = True
            branch.create()
            updated[branch_name] = False
    return updated


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


def generate_plan(tree: dict[str, Branch]) -> str:
    """Generate merging plan.

    Use the commit messages of the local commits to derive branch names
    and check if these branches exist. If they do, figure out what their
    target branch is by walking backwards in the git log until a commit
    is o

    The plan is on the tree built from the current branches and contains
    commits whose sha might differ from the one in the local commits,
    but the commits in the plan correspond to those in the local commits.
    """
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
        lines.append(f"{command} {commit.sha[:8]} {commit.short_message}")
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
    if not utils.object_exists(UPSTREAM):
        raise click.ClickException(f"Upstream {UPSTREAM} does not exist")
    if not utils.object_exists(LOCAL):
        raise click.ClickException(f"Local {LOCAL} does not exist")


def cli():
    try:
        cli_group()
    except subprocess.CalledProcessError as exc:
        click.echo("subprocess failed:")
        click.echo(exc)
        click.echo("captured output:")
        click.echo(exc.output)
        click.echo(exc.stderr)
        raise




def update_options(f):
    click.option(
        "-n",
        "--no-prune",
        is_flag=True,
        type=bool,
        help="Do not prune trunks branches that are not in the plan anymore",
    )(f)
    return f


@cli_group.command
@click.argument(
    "remote",
    nargs=1,
    default="origin",
    type=str,
)
@click.option(
    "-s",
    "--soft",
    is_flag=True,
    type=bool,
    help="Print the push commands instead of executing them.",
)
@click.option(
    "-i",
    "--interactive",
    is_flag=True,
    type=bool,
    help="Choose which branches to push.",
)
@click.option(
    "-m",
    "--gitlab-merge-request",
    is_flag=True,
    type=bool,
    help="Use Gitlab-specific push-options to create a merge request",
)
def push(remote, soft, interactive, gitlab_merge_request):
    tree = reconstruct_tree()
    for branch in tree.values():
        push_command = branch.get_force_push_command(
            remote, gitlab_merge_request=gitlab_merge_request
        )
        do_it = True
        if interactive:
            do_it = click.confirm(
                f"Push {branch.name} to {remote}?", default=True
            )
        if soft:
            click.echo(f"git {' '.join(push_command)}")
        elif do_it:
            click.echo(f"Pushing {branch.name}.")
            utils.run(*push_command)
    if not soft:
        click.echo("Done.")


@cli_group.command
@update_options
@no_hot_branch
def update(no_prune):
    """Infer the plan from local branches and update the commits in the
    branches.
    """
    tree = reconstruct_tree()
    try:
        _update(tree, no_prune)
    except CherryPickFailed as exc:
        raise click.ClickException(
            f"{exc}\n\n"
            f"To investigate, run:\n\n{exc.branch.create_instructions}."
        )


def _update(tree, no_prune):
    with utils.preserve_state(auto_stash=AUTO_STASH):
        create_or_update_branches(tree)
        click.echo(
            "Branches updated. "
            "Run `trunks push` to push them to a remote."
        )
    if not no_prune:
        prune_local_branches(tree)


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
def show(verbose, visual):
    """Display the current plan."""
    if verbose:
        click.echo(f"upstream: {UPSTREAM}")
        click.echo(f"local: {LOCAL}\n")
    tree = reconstruct_tree()
    if visual:
        dot = graphviz.Digraph()
        dot.node(UPSTREAM)
        for branch in tree.values():
            dot.node(branch.name)
            target = UPSTREAM if branch.target is None else branch.target.name
            dot.edge(branch.name, target)
        with tempfile.NamedTemporaryFile() as f:
            dot.render(format="png", filename=f.name, view=True)
    else:
        click.echo(generate_plan(tree))


@cli_group.command()
@click.option(
    "-g",
    "--generator",
    type=click.Choice(["detect", "stacked", "independent", "reset"]),
    default="detect",
    help="Plan generation strategy."
)
@click.option(
    "-e",
    "--edit",
    is_flag=True,
    type=bool,
    help="Edit the plan before executing it."
)
@update_options
@no_hot_branch
@undiverged_trunks
def plan(generator, edit, no_prune):
    """Create a plan and update local branches."""
    old_tree = reconstruct_tree(use_branch_commits=True)
    if generator == "stacked":
        tree = make_simple_tree(stack=True)
    elif generator == "independent":
        tree = make_simple_tree(stack=False)
    elif generator == "reset":
        tree = {}
    elif generator == "detect":
        tree = reconstruct_tree()
    else:
        assert False, "You, the programmer, missed a case."
    plan = generate_plan(tree)
    if edit:
        new_plan = edit_interactively(plan + INSTRUCTIONS)
        new_plan = "\n".join(iterate_plan(new_plan))
        if not new_plan.strip():
            click.echo("Aborting.")
            return
    else:
        new_plan = plan
    click.echo(f"The plan:\n\n{new_plan}\n")
    try:
        tree = parse_plan(new_plan)
        if tree == old_tree:
            click.echo("No updates required.")
            return
        update = click.confirm("Update branches accordingly (N discards the plan)?", default=True)
        if update:
            _update(tree, no_prune)
    except (PlanError, ParsingError) as exc:
        raise click.ClickException(f"{exc}")
    except CherryPickFailed as exc:
        raise click.ClickException(
            f"{exc} To reproduce the failed cherry-pick, run the following "
            f"commands:\n\n{exc.branch.create_instructions}"
        )


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
    branches = get_trunks_branches()
    if show_targets or commits:
        tree = reconstruct_tree()
    for branch_name in branches:
        if show_targets:
            branch = tree[branch_name]
            target = UPSTREAM
            if branch.target is not None:
                target = branch.target.name
            click.echo(f"{branch_name} --> {target}")
        else:
            click.echo(branch_name)
        if commits:
            branch = tree[branch_name]
            for commit in reversed(branch.commits):
                click.echo(f"\t{commit.sha[:8]} {commit.message}")


@cli_group.command()
@click.option(
    '--yes',
    is_flag=True,
)
def reset(yes):
    """Remove trunks-managed branches."""
    branches = get_trunks_branches()
    if len(branches) == 0:
        click.echo("No active trunks branches found")
        return
    if not yes:
        click.echo("This will delete the following branches:")
        for branch_name in branches:
            click.echo(branch_name)
        confirmed = click.confirm("Continue?")
    if confirmed or yes:
        for branch_name in branches:
            utils.run("branch", "-D", branch_name)
