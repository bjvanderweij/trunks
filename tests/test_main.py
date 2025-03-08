import tempfile
from unittest.mock import patch
from pathlib import Path

import pytest
from trunks import utils, main


@pytest.fixture()
def git_repository():
    with tempfile.TemporaryDirectory() as temp_dir:
        utils.run("init", cwd=temp_dir)
        yield Path(temp_dir)


@pytest.fixture()
def commit_a(git_repository):
    with open(git_repository / "a", "w"):
        pass
    utils.run(*("git add .".split()))
    utils.run(*("git commit -m a".split()))


@pytest.fixture()
def commit_b(git_repository):
    with open(git_repository / "b", "w"):
        pass
    utils.run(*("git add .".split()))
    utils.run(*("git commit -m b".split()))


@pytest.fixture()
def commit_c(git_repository):
    with open(git_repository / "c", "w"):
        pass
    utils.run(*("git add .".split()))
    utils.run(*("git commit -m c".split()))


@pytest.fixture()
def local_commits():
    commits = [main.Commit("0", "a"), main.Commit("1", "b"), main.Commit("2", "c")]
    with patch("trunks.main.get_local_commits", return_value=commits):
        yield commits


def test_parse_plan__unrecognized_command(local_commits):
    with pytest.raises(main.ParsingError):
        main.parse_plan("s 0 a\na 1 b\ns 2 v") == {}
    with pytest.raises(main.ParsingError):
        main.parse_plan("s 0 a\nb\ns 2 v") == {}


def test_parse_plan__illegal_plans(local_commits):
    with pytest.raises(main.PlanError):
        # Unrecognized commit
        main.parse_plan("s 0 a\nb1 a foo\ns 2 v") == {}
    with pytest.raises(main.PlanError):
        # Non contiguous commits in branch
        main.parse_plan("b 0 a\nb1@b 1 foo\nb  2 v") == {}
    with pytest.raises(main.PlanError):
        # "Crossing" branches
        main.parse_plan("b 0 a\nb1@b 1 foo\nb2 2 v") == {}
    with pytest.raises(main.PlanError):
        # Out of order commits
        main.parse_plan("b 1 a\nb 0 foo") == {}


def test_parse_plan__legal_plans(local_commits):
    assert main.parse_plan("s 0 a\ns 1 b\ns 2 v") == {}
    assert list(main.parse_plan("b1 0 a\ns 1 b\ns 2 v").values()) == [main.Branch([local_commits[0]], None)]
    # These things should be fine:
    # Missing label
    main.parse_plan("b 0 a\nb1 1 b\nb2 2 v")
    # Missing initial commits
    main.parse_plan("b 1 b\nb1 2 v")
    # Missing middle commit
    main.parse_plan("b 0 a\nb2 2 v")
    # Missing final commit
    main.parse_plan("b 0 a\nb2 1 v")


def test_generate_plan():
    pass


def test_create_or_update_branches():
    pass


def test_build_tree_from_local_commits():
    pass
