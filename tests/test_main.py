from unittest.mock import patch

import pytest
from trunks import main


@pytest.fixture()
def local_commits():
    commits = [main.Commit("0", "a"), main.Commit("1", "b"), main.Commit("2", "c")]
    with patch("trunks.main.get_local_commits", return_value=commits):
        yield commits


def test_parse_plan__unrecognized_command(local_commits):
    with pytest.raises(main.ParsingError):
        main.parse_plan("s 0 a\na 1 b\ns 2 v") == {}


def test_parse_plan__unrecognized_commit(local_commits):
    main.parse_plan("s 0 a\nb1 1 foo\ns 2 v") == {}


def test_parse_plan(local_commits):
    assert main.parse_plan("s 0 a\ns 1 b\ns 2 v") == {}
    assert list(main.parse_plan("b1 0 a\ns 1 b\ns 2 v").values()) == [main.Branch([local_commits[0]], None)]
