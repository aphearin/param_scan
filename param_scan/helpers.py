"""
"""
import os
import numpy as np
from glob import glob
from subprocess import check_output, CalledProcessError


def get_parallel_outbase_pattern(fn):
    s = os.path.basename(fn).split(".")
    s.insert(-1, "*")
    s.insert(-1, "*")
    rank_outbase_pat = ".".join(s)
    return rank_outbase_pat


def get_mpi_rank_outname(fn, rank, batch):
    bnpat = get_parallel_outbase_pattern(fn)
    i = bnpat.find("*")
    j = bnpat.find("*", i + 1)
    bnpat_seq = [s for s in bnpat]
    bnpat_seq.insert(i, str(rank))
    bnpat_seq.pop(i + 1)
    bnpat_seq.insert(j, str(batch))
    bnpat_seq.pop(j + 1)
    bn = "".join(bnpat_seq)
    drn = os.path.dirname(fn)
    rank_outname = os.path.join(drn, bn)
    return rank_outname


def get_equal_sized_data_chunks(n_tot, n_ranks, n_cube_max):
    """"""
    n_per_rank = max(1, n_tot // n_ranks)
    n_cubes, remainder = divmod(n_per_rank, n_cube_max)
    if n_cubes == 0:
        n_cubes = 1
        n_per_cube = remainder
    else:
        n_per_cube = n_cube_max
    return n_cubes, n_per_cube


def write_param_chunk(outname, param_chunk, loss_arr):
    n_chunk, n_params = param_chunk.shape
    n_loss = loss_arr.size
    msg = (
        "For outname = {0}, "
        "mismatch in number of loss evaluations = {1} vs param_chunk shape = {2}"
    )
    assert n_loss == n_chunk, msg.format(outname, n_loss, param_chunk.shape)
    output_data = np.zeros((n_chunk, n_params + 1))
    output_data[:, :-1] = param_chunk
    output_data[:, -1] = loss_arr
    np.save(outname, output_data)


def cleanup_and_collate(outname):
    drn = os.path.dirname(outname)
    _bnpat = get_parallel_outbase_pattern(outname)
    i = _bnpat.find("*")
    j = _bnpat.find("*", i + 1)
    bpat = _bnpat[: j + 1]
    fnpat = os.path.join(drn, bpat)
    rank_fnames = glob(fnpat)
    collector = []
    for rank_fname in rank_fnames:
        collector.append(np.load(rank_fname))
    results = np.concatenate(collector)
    np.save(outname, results)

    for fn in rank_fnames:
        command = "rm " + fn
        try:
            check_output(command, shell=True)
        except CalledProcessError:
            pass
