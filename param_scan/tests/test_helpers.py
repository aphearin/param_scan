"""
"""
import os
from ..helpers import get_parallel_outbase_pattern, get_rank_outname


_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
DUMMY_FNAME = os.path.join(_THIS_DRNAME, "dummy.ext")


def test_get_parallel_outbase_pattern():
    outbase_pat = get_parallel_outbase_pattern(DUMMY_FNAME)
    assert outbase_pat == "dummy.*.*.ext"


def test_get_rank_outname():
    rank_outname = get_rank_outname(DUMMY_FNAME, 0, 0)
    assert rank_outname == os.path.join(_THIS_DRNAME, "dummy.0.0.ext")

    rank_outname = get_rank_outname(DUMMY_FNAME, 50, 34440)
    assert rank_outname == os.path.join(_THIS_DRNAME, "dummy.50.34440.ext")
