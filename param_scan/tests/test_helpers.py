"""
"""
import os
import numpy as np
from ..helpers import get_parallel_outbase_pattern, get_mpi_rank_outname
from ..helpers import get_equal_sized_data_chunks


_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
DUMMY_FNAME = os.path.join(_THIS_DRNAME, "dummy.ext")
SEED = 0


def test_get_parallel_outbase_pattern():
    outbase_pat = get_parallel_outbase_pattern(DUMMY_FNAME)
    assert outbase_pat == "dummy.*.*.ext"


def test_get_rank_outname():
    rank_outname = get_mpi_rank_outname(DUMMY_FNAME, 0, 0)
    assert rank_outname == os.path.join(_THIS_DRNAME, "dummy.0.0.ext")

    rank_outname = get_mpi_rank_outname(DUMMY_FNAME, 50, 34440)
    assert rank_outname == os.path.join(_THIS_DRNAME, "dummy.50.34440.ext")


def test_get_equal_sized_data_chunks():
    n_tests = 50_000
    rng = np.random.RandomState(SEED)
    rng2 = np.random.RandomState(SEED + 1)
    rng3 = np.random.RandomState(SEED + 2)
    nranks_arr = np.array(10 ** rng.uniform(0.3, 3.3, n_tests)).astype("i8")
    n_scan_tot_arr = nranks_arr * np.array(10 ** rng3.uniform(0, 7, n_tests))
    n_scan_tot_arr = n_scan_tot_arr.astype("i8")
    n_cube_max_arr = np.array(10 ** rng2.uniform(2, 5, n_tests)).astype("i8")

    pat = "(n_scan_tot, n_ranks, n_cube_max) = ({0}, {1}, {2})"
    gen = zip(n_scan_tot_arr, nranks_arr, n_cube_max_arr)

    for n_scan_tot, n_ranks, n_cube_max in gen:
        msg = pat.format(n_ranks, n_cube_max, n_scan_tot)
        n_cubes, n_per_cube = get_equal_sized_data_chunks(
            n_scan_tot, n_ranks, n_cube_max
        )
        assert 0 < n_per_cube <= n_cube_max, msg
        num_computed = n_ranks * n_cubes * n_per_cube
        assert n_scan_tot / 2 < num_computed <= n_scan_tot, msg
