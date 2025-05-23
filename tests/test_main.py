import tempfile
import os
import subprocess
from unittest.mock import patch
from pathlib import Path
from functools import partial

import pytest
from trunks import utils
from trunks.main import (
    ChangeSet, Commit, parse_plan, ParsingError, PlanError, UPSTREAM, LOCAL,
    reconstruct_tree, create_or_update_branches, get_local_commits, render_plan,
    make_simple_tree
)


@pytest.fixture()
def git_repository():
    with tempfile.TemporaryDirectory() as temp_dir:
        new_run = partial(utils.run, cwd=temp_dir)
        with patch("trunks.utils.run", new_run):
            utils.run("init", cwd=temp_dir)
            yield Path(temp_dir)


@pytest.fixture()
def trunks(git_repository):
    def run(cmd):
        subprocess.run(
            ["python", "-m" "trunks.main", *cmd],
            cwd=git_repository,
            env=dict(PYTHONPATH=os.getcwd()),
        )
    return run


@pytest.fixture()
def remote_trunk(git_repository):
    utils.run(*(f"checkout -b {UPSTREAM}".split()), cwd=git_repository)
    utils.run(*("checkout -".split()), cwd=git_repository)


@pytest.fixture()
def local_trunk(git_repository):
    utils.run(*(f"checkout -b {LOCAL}".split()), cwd=git_repository)
    utils.run(*("checkout -".split()), cwd=git_repository)


@pytest.fixture()
def commit_a(git_repository):
    with open(git_repository / "a", "w") as f:
        f.write("a")
    utils.run(*("add a".split()), cwd=git_repository)
    utils.run(*("commit -m a".split()), cwd=git_repository)
    return utils.run(*("rev-parse HEAD".split()))


@pytest.fixture()
def commit_b(git_repository):
    with open(git_repository / "b", "w") as f:
        f.write("")
    utils.run(*("add .".split()), cwd=git_repository)
    utils.run(*("commit -m b".split()), cwd=git_repository)
    return utils.run(*("rev-parse HEAD".split()))


@pytest.fixture()
def commit_c(git_repository):
    with open(git_repository / "c", "w") as f:
        f.write("")
    utils.run(*("add .".split()), cwd=git_repository)
    utils.run(*("commit -m c".split()), cwd=git_repository)
    return utils.run(*("rev-parse HEAD".split()))


@pytest.fixture()
def local_commits():
    commits = [Commit("0", "a"), Commit("1", "b"), Commit("2", "c"), Commit("3", "d")]
    with patch("trunks.main.get_local_commits", return_value=commits):
        yield commits


@pytest.fixture
def commit(git_repository):
    def _commit(files, message):
        for path, contents in files.items():
            with open(git_repository / path, "w") as f:
                f.write(contents)
            utils.run("add", path)
        utils.run("commit", "-m", message)
        return utils.run("rev-parse", "HEAD")
    return _commit


@pytest.fixture
def create_branch(git_repository):
    def _create_branch(name):
        utils.run("checkout", "-b", name, cwd=git_repository)
        utils.run("checkout", "-", cwd=git_repository)
    return _create_branch



def test_parse_plan__syntax_errors(local_commits):
    with pytest.raises(ParsingError):
        parse_plan("s 0 a\na 1 b\ns 2 v") == {}
    with pytest.raises(ParsingError):
        parse_plan("b@ 0 a") == {}
    with pytest.raises(ParsingError):
        parse_plan("s 0 a\nb\ns 2 v") == {}


def test_parse_plan__illegal_plans(local_commits):
    with pytest.raises(PlanError, match="unrecognized"):
        # Unrecognized commit
        parse_plan("s 0 a\nb1 a\ns 2 v") == {}
    with pytest.raises(PlanError, match="unrecognized"):
        # Out of order commits
        parse_plan("b 1 a\nb 0 foo") == {}
    with pytest.raises(PlanError, match="must be contiguous"):
        # Non contiguous commits in branch
        parse_plan("b 0 a\nb1@b 1 foo\nb  2 v") == {}
    with pytest.raises(PlanError, match="invalid target"):
        # Incorrect target
        parse_plan("b@b1 0 a\nb1 1 foo") == {}
    with pytest.raises(PlanError, match="multiple targets"):
        # Conflicting targets
        parse_plan("b 0 a\nb1 1\nb2@b 2 v\nb2@b1 3") == {}
    with pytest.raises(PlanError, match="invalid target"):
        # "Crossing" branches
        parse_plan("b 0 a\nb1@b 1 foo\nb2 2 v") == {}


def test_parse_plan__legal_plans(local_commits):
    a, b, c, d = local_commits
    # Equivalent plans
    change_set = ChangeSet([c], None)
    tree = {change_set.branch_name: change_set}
    v0 = parse_plan("s 0 a\ns 1 b\nb0 2 v")
    v1 = parse_plan("s 0 a\nb0 2 v")
    v2 = parse_plan("b0 2 v")
    v3 = parse_plan("b 2 v")
    v4 = parse_plan("b 2")
    assert v0 == v1 == v2 == v3 == v4 == tree
    # Empty plans
    assert parse_plan("") == {}
    assert parse_plan("s 0 a\ns 1 b\ns 2 v") == {}
    # Optional target specifications
    b0 = ChangeSet([a], None)
    b1 = ChangeSet([b, c], b0)
    tree = {cs.branch_name: cs for cs in [b0, b1]}
    variant_1 = parse_plan("b 0 a\nb1@b 1 b\nb1 2 v")
    variant_2 = parse_plan("b 0 a\nb1 1 b\nb1@b 2 v")
    variant_3 = parse_plan("b 0 a\nb1@b 1 b\nb1@b 2 v")
    assert variant_1 == variant_2 == variant_3
    assert variant_1 == tree
    b = ChangeSet([a, c], None)
    tree = {b.branch_name: b}
    assert parse_plan("b 0 a\ns 1 foo\nb 2 v") == tree



def test_render_plan():
    pass


@pytest.fixture
def independent_commits(commit, create_branch):
    commit(dict(x="x"), "0")
    create_branch(UPSTREAM)
    commit(dict(a="a"), "1")
    commit(dict(b="b"), "2")
    commit(dict(c="c"), "3")
    commit(dict(d="d"), "4")
    create_branch(LOCAL)
    return get_local_commits()


@pytest.fixture
def serially_dependent_commits(commit, create_branch):
    commit(dict(a="a"), "0")
    create_branch(UPSTREAM)
    commit(dict(a="b"), "1")
    commit(dict(a="c"), "2")
    commit(dict(a="d"), "3")
    commit(dict(a="e"), "4")
    create_branch(LOCAL)
    return get_local_commits()


@pytest.fixture
def dag_commits(commit, create_branch):
    commit(dict(a="a"), "0")
    create_branch(UPSTREAM)
    commit(dict(a="b"), "1")
    commit(dict(a="c"), "2")
    commit(dict(b="a"), "3")
    commit(dict(a="d"), "4")
    create_branch(LOCAL)
    return get_local_commits()


def test_reconstruct_tree__branch_anchor_first(dag_commits):
    c1, c2, c3, c4 = dag_commits
    plan = (
        f"b0 {c1.short_str}\n"
        f"b0 {c2.short_str}\n"
        f"b1 {c3.short_str}\n"
        f"b2@b0 {c4.short_str}"
    )
    with patch("trunks.main.BRANCH_ANCHOR", "first"):
        tree = parse_plan(plan)
        create_or_update_branches(tree)
        reconstructed_tree = reconstruct_tree()
    b = ChangeSet(commits=[c1, c2], target=None)
    b1 = ChangeSet(commits=[c3], target=None)
    b2 = ChangeSet(commits=[c4], target=b)
    assert reconstructed_tree == {
        c1.branch_name: b,
        c3.branch_name: b1,
        c4.branch_name: b2,
    }


def test_reconstruct_tree__branch_anchor_last(dag_commits):
    c1, c2, c3, c4 = dag_commits
    plan = (
        f"b0 {c1.short_str}\n"
        f"b0 {c2.short_str}\n"
        f"b1 {c3.short_str}\n"
        f"b2@b0 {c4.short_str}"
    )
    with patch("trunks.main.BRANCH_ANCHOR", "last"):
        tree = parse_plan(plan)
        create_or_update_branches(tree)
        reconstructed_tree = reconstruct_tree()
    b = ChangeSet(commits=[c1, c2], target=None)
    b1 = ChangeSet(commits=[c3], target=None)
    b2 = ChangeSet(commits=[c4], target=b)
    assert reconstructed_tree == {
        c2.branch_name: b,
        c3.branch_name: b1,
        c4.branch_name: b2,
    }


@pytest.mark.parametrize("branch_anchor", ["first", "last"])
def test_reconstruct_tree(dag_commits, branch_anchor):
    c1, c2, c3, c4 = dag_commits
    plan = (
        f"b0 {c1.short_str}\n"
        f"b0 {c2.short_str}\n"
        f"b1 {c3.short_str}\n"
        f"b2@b0 {c4.short_str}"
    )
    with patch("trunks.main.BRANCH_ANCHOR", branch_anchor):
        tree = parse_plan(plan)
        create_or_update_branches(tree)
        reconstructed_tree = reconstruct_tree()
        reconstructed_plan = render_plan(reconstructed_tree)
        assert reconstructed_plan == plan
        b0 = ChangeSet(commits=[c1, c2], target=None)
        b1 = ChangeSet(commits=[c3], target=None)
        b2 = ChangeSet(commits=[c4], target=b0)
        assert reconstructed_tree == {
            b0.branch_name: b0,
            b1.branch_name: b1,
            b2.branch_name: b2,
        }


@pytest.mark.parametrize("branch_anchor", ["first", "last"])
def test_reconstruct_tree_stacked(serially_dependent_commits, branch_anchor):
    c1, c2, c3, c4 = serially_dependent_commits
    with patch("trunks.main.BRANCH_ANCHOR", branch_anchor):
        tree = make_simple_tree(stack=True)
        create_or_update_branches(tree)
        reconstructed_tree = reconstruct_tree()
        reconstructed_plan = render_plan(reconstructed_tree)
        plan = (
            f"b0 {c1.short_str}\n"
            f"b1@b0 {c2.short_str}\n"
            f"b2@b1 {c3.short_str}\n"
            f"b3@b2 {c4.short_str}"
        )
        assert reconstructed_plan == plan
        b0 = ChangeSet(commits=[c1], target=None)
        b1 = ChangeSet(commits=[c2], target=b0)
        b2 = ChangeSet(commits=[c3], target=b1)
        b3 = ChangeSet(commits=[c4], target=b2)
        assert reconstructed_tree == {
            b0.branch_name: b0,
            b1.branch_name: b1,
            b2.branch_name: b2,
            b3.branch_name: b3,
        }


@pytest.mark.parametrize("branch_anchor", ["first", "last"])
def test_reconstruct_tree_independent(independent_commits, branch_anchor):
    c1, c2, c3, c4 = independent_commits
    with patch("trunks.main.BRANCH_ANCHOR", branch_anchor):
        tree = make_simple_tree(stack=False)
        create_or_update_branches(tree)
        reconstructed_tree = reconstruct_tree()
        reconstructed_plan = render_plan(reconstructed_tree)
        plan = (
            f"b0 {c1.short_str}\n"
            f"b1 {c2.short_str}\n"
            f"b2 {c3.short_str}\n"
            f"b3 {c4.short_str}"
        )
        assert reconstructed_plan == plan
        b0 = ChangeSet(commits=[c1], target=None)
        b1 = ChangeSet(commits=[c2], target=None)
        b2 = ChangeSet(commits=[c3], target=None)
        b3 = ChangeSet(commits=[c4], target=None)
        assert reconstructed_tree == {
            b0.branch_name: b0,
            b1.branch_name: b1,
            b2.branch_name: b2,
            b3.branch_name: b3,
        }


def test_reconstruct_tree_branch_label_first(commit, create_branch):
    commit(dict(a="a"), "0")
    create_branch(UPSTREAM)
    commit(dict(a="aa"), "1")
    commit(dict(a="ab"), "2")
    commit(dict(b="a"), "3")
    commit(dict(a="bb"), "4")
    create_branch(LOCAL)
    c1, c2, c3, c4 = get_local_commits()
    plan = f"""
    b {c1.sha} {c1.short_message}
    b {c2.sha} {c2.short_message}
    b1 {c3.sha} {c3.short_message}
    b2@b {c4.sha} {c4.short_message}
    """
    tree = parse_plan(plan)
    create_or_update_branches(tree)
    reconstructed_tree = reconstruct_tree()
    b = ChangeSet(commits=[c1, c2], target=None)
    b1 = ChangeSet(commits=[c3], target=None)
    b2 = ChangeSet(commits=[c4], target=b)
    assert reconstructed_tree == {
        c1.branch_name: b,
        c3.branch_name: b1,
        c4.branch_name: b2,
    }


# def test_reconstruct_tree():
#     c0, c1, c2, c3, c4 = [Commit("0", "0"), Commit("1", "1"), Commit("2", "2"), Commit("3", "3"), Commit("4", "4")]
#     local_commits = [c0, c1, c2, c3, c4]
#     local_branches = [c1.branch_name, c4.branch_name, c5.branch_name]
#     last_
#     with patch.multiple(
#         "trunks.main", 
#         get_local_commits=Mock(return_value=local_commits),
#         get_local_branches=Mock(return_value=local_branches),
#         get_last_upstream_commit=[c0],
#     ):
#         tree = reconstruct_tree()
#     assert tree == {}


def test_build_empty_tree(commit_b, remote_trunk, commit_a, commit_c, local_trunk, git_repository):
    tree = reconstruct_tree()
    assert tree == {}


def test_build_empty_tree_empty_repo(git_repository):
    tree = reconstruct_tree()
    assert tree == {}


def test_empty_tree__git(create_branch, commit, git_repository):
    commit(dict(a="a"), "a")
    create_branch(LOCAL)
    commit(dict(b="b"), "b")
    create_branch(UPSTREAM)
    # r = utils.run("log", cwd=git_repository)
    # print(r)
    tree = reconstruct_tree()
    assert tree == {}
