import tempfile
import subprocess
from unittest.mock import patch
from pathlib import Path

import pytest
from trunks import utils
from trunks.main import (
    Branch, Commit, parse_plan, ParsingError, PlanError, REMOTE_TRUNK, LOCAL_TRUNK,
    build_tree_from_local_commits
)


@pytest.fixture()
def git_repository():
    with tempfile.TemporaryDirectory() as temp_dir:
        utils.run("init", cwd=temp_dir)
        yield Path(temp_dir)


@pytest.fixture()
def remote_trunk(git_repository):
    utils.run(*(f"checkout -b {REMOTE_TRUNK}".split()), cwd=git_repository)
    utils.run(*("checkout -".split()), cwd=git_repository)


@pytest.fixture()
def local_trunk(git_repository):
    utils.run(*(f"checkout -b {LOCAL_TRUNK}".split()), cwd=git_repository)
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
            utils.run("add", path, cwd=git_repository)
        utils.run("commit", "-m", message, cwd=git_repository)
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
    branch = Branch([c], None)
    tree = {branch.name: branch}
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
    b0 = Branch([a], None)
    b1 = Branch([b, c], b0)
    tree = {b0.name: b0, b1.name: b1}
    variant_1 = parse_plan("b 0 a\nb1@b 1 b\nb1 2 v")
    variant_2 = parse_plan("b 0 a\nb1 1 b\nb1@b 2 v")
    variant_3 = parse_plan("b 0 a\nb1@b 1 b\nb1@b 2 v")
    assert variant_1 == variant_2 == variant_3
    assert variant_1 == tree


def test_generate_plan():
    pass


def test_create_or_update_branches():
    pass


def test_build_empty_tree(commit_b, remote_trunk, commit_a, commit_c, local_trunk, git_repository):
    tree = build_tree_from_local_commits()
    assert tree == {}


def test_build_empty_tree_empty_repo(git_repository):
    tree = build_tree_from_local_commits()
    assert tree == {}


def test_a(create_branch, commit, git_repository):
    commit(dict(a="a"), "a")
    create_branch(LOCAL_TRUNK)
    commit(dict(b="b"), "b")
    create_branch(REMOTE_TRUNK)
    r = utils.run("log", cwd=git_repository)
    print(r)
    tree = build_tree_from_local_commits()
    assert tree == {}
