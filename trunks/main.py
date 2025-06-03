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


# UPSTREAM = "origin/main"
# LOCAL = "main"
UPSTREAM = "origin/develop"
LOCAL = "bastiaan-develop"
# UPSTREAM = "remote"
# LOCAL = "local"
BRANCH_TEMPLATE = "feature/{}"
EDITOR = "vim"
USE_FIRST_COMMIT_FOR_BRANCH_NAME = False
BRANCH_ANCHOR = "first"  # first/last


INSTRUCTIONS = """

# Edit branch-creation plan.
#
# Commands:
# b<label> <commit> = use commit in labeled branch
# b<label>@b<target-label> <commit> = use commit in labeled branch off branch with target-label
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
            click.echo(
                f"Hint: Use `git pull --rebase {UPSTREAM}` to pull "
                "upstream changes into your local branch."
            )
            raise click.ClickException("Your trunks have diverged.")
        return f(*args, **kwargs)

    return wrapper


def clean_work_tree(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        result = utils.run("status", "--untracked-files=no", "--porcelain")
        if bool(result.strip()):
            raise click.ClickException("Work tree not clean.")
        return f(*args, **kwargs)

    return wrapper


def local_and_upstream_exist(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if not utils.object_exists(UPSTREAM):
            raise click.ClickException(f"Upstream {UPSTREAM} does not exist")
        if not utils.object_exists(LOCAL):
            raise click.ClickException(f"Local {LOCAL} does not exist")
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
    def __init__(self, *args, change_set, **kwargs):
        self.change_set = change_set


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


class ChangeSet(typing.NamedTuple):
    commits: list[Commit]
    target: typing.Optional["ChangeSet"]

    @property
    def branch_name(self):
        if BRANCH_ANCHOR == "first":
            return self.commits[0].branch_name
        else:
            return self.commits[-1].branch_name

    @property
    def full_branch_name(self):
        return f"refs/heads/{self.branch_name}"

    @property
    def target_branch_name(self) -> str:
        if self.target is None:
            return UPSTREAM
        else:
            return self.target.branch_name

    def branch_exists(self):
        return utils.object_exists(self.full_branch_name)

    def delete_branch(self):
        utils.run("branch", "-D", self.branch_name)

    def create_branch(self):
        utils.checkout("-b", self.branch_name)

    @property
    def create_instructions(self) -> str:
        return (
            f"git checkout {self.target_branch_name}\n"
            f"git checkout -b temporary-investigation-branch\n"
            f"git cherry-pick {' '.join([c.sha for c in self.commits])}"
        )

    def cherry_pick(self):
        try:
            utils.run("cherry-pick", *[c.sha for c in self.commits])
        except subprocess.CalledProcessError:
            raise CherryPickFailed(
                f"Cherry-pick failed at branch {self.branch_name}.",
                change_set=self
            )

    def get_force_push_command(
        self, remote: str, gitlab_merge_request: bool = False
    ):
        command = [
            "push",
            "--force",
            "--set-upstream",
            remote,
            f"{self.full_branch_name}:{self.full_branch_name}",
        ]
        if gitlab_merge_request:
            command += ["--push-option", "merge_request.create"]
            if self.target is not None:
                command += [
                    "--push-option",
                    f"merge_request.target={self.target_branch_name}",
                ]
        return command

    def __str__(self):
        return (
            "Branch {self.branch_name} with commits:"
            "\n".join(f"\t{c.short_message}" for c in self.commits)
        )


def get_trunks_branches():
    local_branches = utils.get_local_branches()
    commits = get_local_commits()
    branch_names = [c.branch_name for c in commits]
    return [b for b in local_branches if b in branch_names]


def make_simple_tree(stack) -> dict[str, ChangeSet]:
    """Use local commits create a tree of stacked branches."""
    commits = get_local_commits()
    tree: dict[str, ChangeSet] = {}
    target = None
    for commit in commits:
        change_set = ChangeSet([commit], target)
        tree[change_set.branch_name] = change_set
        if stack:
            target = change_set
    return tree


def infer_change_set_last_commit(commit, i, commits_by_message, tree, root) -> ChangeSet:
    candidate_commits = get_last_n_commits(commit.branch_name, i + 1)
    if candidate_commits[-1].branch_name != commit.branch_name:
        raise click.ClickException(
            "Invalid state: trunks-managed branch name "
            f"{commit.branch_name} does not match branch name "
            "expected based on its last commit.\n\nRun\n\ngit branch "
            "-D {commit.branch_name}\n\nto remove the offending "
            "branch. Or run\n\ntrunks reset\n\nif you'd like to start "
            "with a clean slate."
        )
    target = None
    commits: list[Commit] = []
    # Find the first commit of the branch by iterating through
    # preceding commits in reverse order
    for cc in reversed(candidate_commits):
        branch_name = cc.branch_name
        # When finding a commit whose branch name corresponds to
        # a branch in the tree and it isn't the current commits branch
        # name assume we've found the target branch and stop
        if branch_name in tree and branch_name != commit.branch_name:
            target = tree[cc.branch_name]
            break
        # if we find a commit that is in the commits by message
        elif cc.message in commits_by_message:
            commits.insert(
                0, commits_by_message[cc.message]
            )
        # either we've reached the bottom of the tree, in which case
        # the preceding commits should be the same as the tip of
        # remote. If not, a commit has been renamed
        # No biggie, just print a warning
        else:
            if cc.message != root.message:
                click.echo(
                    f"warning: unknown commit message encountered: {cc.short_message}"
                )
            break
    return ChangeSet(commits, target)


def infer_change_set_first_commit(commit, i, commits_by_message, tree, root) -> ChangeSet:
    n_local = len(commits_by_message)
    candidate_commits = get_last_n_commits(commit.branch_name, n_local - i + 1)
    target = None
    branch_commits: list[Commit] = []
    start_index = [c.branch_name for c in candidate_commits].index(
        commit.branch_name
    )
    for change_set in tree.values():
        if change_set.commits[-1].message == candidate_commits[start_index - 1].message:
            target = change_set
            break
    for cc in candidate_commits[start_index:]:
        if cc.message in commits_by_message:
            branch_commits.append(commits_by_message[cc.message])
        else:
            click.echo(f"warning: unknown commit message encountered: {cc.message}")
            break
    return ChangeSet(branch_commits, target)


def reconstruct_tree() -> dict[str, ChangeSet]:
    """Use local commits to reconstruct the plan.

    Assumes that the commits in local branches have the same commit messages
    as commits in local commits.

    Algorithm when branch name is derived from last commit:

    get local commits in chronological order
    get local branches
    for commit in local commits
    if branch name exists as local branch
    get n preceding commits where n is index of commit in local commits
    iterate in reverse chronological order until you encounter either
    a commit corresponding to another branch already created or an unknown
    commit.

    when name is derived from first commit:

    get local commits in chronological order
    get local branches
    for commit in local commits
    if branch name exists as local branch
    get commits in branch from tip (n_local_commits - index_of_current_commit)
        until hitting epynomous commit. Then check if previous commit is known
        as final commit of branch
        If so, set corresponding branch as target
        If not, preceding commit must be
    record the final commit in the branch
    """
    commits = get_local_commits()
    commits_by_message = {c.message: c for c in commits}
    local_branches = utils.get_local_branches()
    root = get_last_upstream_commit()
    tree: dict[str, ChangeSet] = {}
    for i, commit in enumerate(commits):
        if commit.branch_name in local_branches:
            if BRANCH_ANCHOR == "first":
                change_set = infer_change_set_first_commit(commit, i, commits_by_message, tree, root)
            else:
                change_set = infer_change_set_last_commit(commit, i, commits_by_message, tree, root)
            tree[change_set.branch_name] = change_set
    return tree


def get_last_upstream_commit() -> Commit:
    return get_commits(UPSTREAM)[0]


def create_or_update_branches(tree: dict[str, ChangeSet]):
    """Create feature branches based on the plan in tree.

    Start at the roots of the tree and for each branch in the topologically
    sorted branches, checkout its target (UPSTREAM if None), delete the branch
    if it already exists, create the branch, cherry-pick its commits.

    Return a dictionary that maps each branch name to a boolean that is True
    only if the branch already existed and was re-created.
    """
    dag: dict[str, list[str]] = {}
    for name, change_set in tree.items():
        if name not in dag:
            dag[name] = []
        if change_set.target is not None:
            dag[name].append(change_set.target.branch_name)
    ts = TopologicalSorter(dag)
    updated = {}
    for branch_name in ts.static_order():
        change_set = tree[branch_name]
        utils.checkout(change_set.target_branch_name)
        with utils.temporary_branch():
            try:
                change_set.cherry_pick()
            except CherryPickFailed:
                try:
                    utils.run("cherry-pick", "--abort")
                except subprocess.CalledProcessError:
                    click.echo("Failed to abort cherry-pick.", err=True)
                raise
            if change_set.branch_exists():
                change_set.delete_branch()
                updated[branch_name] = True
            change_set.create_branch()
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
) -> dict[str, ChangeSet]:
    """Parse branching plan and return a branch DAG.

    Enforce the following constraints on the DAG:

    - branches point to either
        - the target of the last branch
        - one of the set of immediately preceding branches with the same target
    - commits in a branch appear in the same order as the local commits
    """
    change_sets: dict[str, ChangeSet] = {}
    last_target_label = None
    valid_target_labels: set[None | str] = {None}
    for cb in candidate_branches:
        if cb.label in change_sets:
            raise PlanError("commits in branch must be contiguous")
        if cb.target_label not in valid_target_labels:
            raise PlanError(
                f"invalid target for {cb.label}: {cb.target_label}"
            )
        target_branch = None
        if cb.target_label is not None:
            target_branch = change_sets[cb.target_label]
        if cb.target_label != last_target_label:
            last_target_label = cb.target_label
            valid_target_labels = {last_target_label}
        valid_target_labels.add(cb.label)
        change_sets[cb.label] = ChangeSet(cb.commits, target_branch)
    return {b.branch_name: b for b in change_sets.values()}


def parse_plan(plan: str) -> dict[str, ChangeSet]:
    tokens = _tokenize_plan(plan)
    commit_lists = _make_commit_lists(tokens)
    return _build_tree(commit_lists)


def render_plan(tree: dict[str, ChangeSet]) -> str:
    local_commits = get_local_commits()
    sorted_change_sets = list(sorted(tree.values(), key=lambda cs: local_commits.index(cs.commits[0])))
    lines = []
    for commit in local_commits:
        command = "s"
        change_set = None
        try:
            cs_i, change_set = next((i, cs) for i, cs in enumerate(sorted_change_sets) if commit in cs.commits)
        except StopIteration:
            pass
        if change_set is not None:
            command = f"b{cs_i}"
            if change_set.target is not None:
                target_i = sorted_change_sets.index(change_set.target)
                command += f"@b{target_i}"
        lines.append(f"{command} {commit.short_str}")
    return "\n".join(lines)


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


def get_commits_between(rev_a, rev_b) -> list[Commit]:
    """Return commits from rev_a up to and including rev_b."""
    return get_commits(f"{rev_a}..{rev_b}")


def get_commits(commits: str, number: None | int = None) -> list[Commit]:
    """Return commits chronological order."""
    args = [
        "rev-list",
        "--no-merges",
        "--format=oneline",
        commits,
    ]
    if number is not None:
        args += ["--max-count", str(number)]
    rev_list_output = utils.run(*args, "--")
    rev_list = reversed(rev_list_output.strip().split("\n"))
    return [Commit.from_oneline(line) for line in rev_list if line]


def get_last_n_commits(rev, n) -> list[Commit]:
    """Return at most n commits leading up to rev, including rev."""
    return get_commits(rev, number=n)


@local_and_upstream_exist
@undiverged_trunks
def get_local_commits() -> list[Commit]:
    """Return all commits between upstream and local."""
    commits = get_commits_between(UPSTREAM, LOCAL)
    if len(commits) != len(set(c.message for c in commits)):
        raise click.ClickException(
            "Duplicate commit messages found in local commits."
        )
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
        click.echo(f"Subprocess failed:\n{exc}\n", err=True)
        click.echo(f"Captured output:\n{exc.output}\n{exc.stderr}\n", err=True)
        raise


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
    for change_set in tree.values():
        push_command = change_set.get_force_push_command(
            remote, gitlab_merge_request=gitlab_merge_request
        )
        do_it = True
        if interactive:
            do_it = click.confirm(
                f"Push {change_set.branch_name} to {remote}?", default=True
            )
        if soft:
            click.echo(f"git {' '.join(push_command)}")
        elif do_it:
            click.echo(f"Pushing {change_set.branch_name}.")
            utils.run(*push_command)
    if not soft:
        click.echo("Done.")


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
        for change_set in tree.values():
            dot.node(change_set.branch_name)
            target = UPSTREAM if change_set.target is None else change_set.target.branch_name
            dot.edge(change_set.branch_name, target)
        with tempfile.NamedTemporaryFile() as f:
            dot.render(format="png", filename=f.name, view=True)
    else:
        click.echo(render_plan(tree))

@cli_group.command()
@click.argument(
    "strategy",
    type=click.Choice(["detect", "stacked", "flat", "empty"]),
    default="detect",
)
@click.option(
    "-e",
    "--edit",
    is_flag=True,
    type=bool,
    help="Edit the plan before executing it."
)
@click.option(
    "-n",
    "--no-prune",
    is_flag=True,
    type=bool,
    help="Do not prune trunks branches that are not in the plan anymore",
)
@click.option(
    '-y',
    '--yes',
    is_flag=True,
    help="Do not ask for confirmation to apply plan."
)
@no_hot_branch
@clean_work_tree
def plan(strategy, edit, no_prune, yes):
    """Create a plan and update local branches.

    The optional argument specifies the type of plan to generate. Available
    types are:

    \b
    detect (default): use the last-applied plan
    stacked: package each commit in a branch and make each branch depend on the
             previous branch.
    flat: package each commit in a separate independent branch
    empty: generate an empty plan

    """
    if strategy == "stacked":
        tree = make_simple_tree(stack=True)
    elif strategy == "flat":
        tree = make_simple_tree(stack=False)
    elif strategy == "empty":
        tree = {}
    elif strategy == "detect":
        tree = reconstruct_tree()
    else:
        assert False, "You, programmer, missed a case."
    plan = render_plan(tree)
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
        if not yes:
            yes = click.confirm(
                "Update branches accordingly ('n' discards the plan)?",
                default=True
            )
        if yes:
            with utils.return_to_head():
                create_or_update_branches(tree)
            click.echo(
                "Branches updated. Run `trunks push` to push them to a remote."
            )
            if not no_prune:
                prune_local_branches(tree)
    except (PlanError, ParsingError) as exc:
        raise click.ClickException(f"{exc}")
    except CherryPickFailed as exc:
        raise click.ClickException(
            f"{exc} To reproduce the failed cherry-pick, run the following "
            f"commands:\n\n{exc.change_set.create_instructions}"
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
            change_set = tree[branch_name]
            target = UPSTREAM
            if change_set.target is not None:
                target = change_set.target_branch_name
            click.echo(f"{branch_name} --> {target}")
        else:
            click.echo(branch_name)
        if commits:
            change_set = tree[branch_name]
            for commit in reversed(change_set.commits):
                click.echo(f"\t{commit.sha[:8]} {commit.message}")


@cli_group.command()
@click.option(
    '-y',
    '--yes',
    is_flag=True,
    help="Do not ask for confirmation."
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
