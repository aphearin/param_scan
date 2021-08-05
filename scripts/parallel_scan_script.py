"""
"""
import os
import argparse
from mpi4py import MPI
import numpy as np
from glob import glob
from param_scan.latin_hypercube import latin_hypercube


def get_equal_sized_cubes(n_tot, n_ranks, n_cube_max):
    n_per_rank = max(1, n_tot // n_ranks)
    n_cubes, remainder = divmod(n_per_rank, n_cube_max)
    if n_cubes == 0:
        n_cubes = 1
        n_per_cube = remainder
    else:
        n_per_cube = n_cube_max
    return n_cubes, n_per_cube


def get_param_bounds():
    raise NotImplementedError()


def get_header():
    raise NotImplementedError()


def get_outline(params, loss, rank):
    raise NotImplementedError()


def compute_loss(params, data):
    raise NotImplementedError()


def cleanup(outname):
    drn = os.path.dirname(outname)
    bpat = _get_outbase_pattern(outname)
    fnpat = os.path.join(drn, bpat)
    rank_fnames = glob(fnpat)


PARAM_BOUNDS = get_param_bounds()
HEADER = get_header()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("outname", help="Name of the output file")
    parser.add_argument("n_tot", help="Total number of points in the param scan")
    parser.add_argument(
        "-n_max_lh",
        help="Maximum number of points in each Latin Hypercube",
        type=int,
        default=5000,
    )
    args = parser.parse_args()
    outname = args.outname
    n_tot = args.n_tot
    n_max_lh = args.n_max_lh

    comm = MPI.COMM_WORLD
    rank, nranks = comm.Get_rank(), comm.Get_size()

    n_cubes_per_rank, n_per_cube = get_equal_sized_cubes(n_tot, nranks, n_max_lh)
    total_cubes = n_cubes_per_rank * nranks
    total_seeds = np.arange(total_cubes).astype("i8")
    seeds_per_rank = np.array_split(total_seeds, nranks)[rank]

    rank_outname = _get_rank_outname(outname, rank)
    with open(rank_outname, "w") as fout:
        fout.write(HEADER)

        for seed in seeds_per_rank:
            cube_params = latin_hypercube(PARAM_BOUNDS, n_per_cube, seed=seed)
            for params in cube_params:
                loss = compute_loss(params, loss_data)
                outline = get_outline(params, loss, rank)
                fout.write(outline)
    if rank == 0:
        cleanup(outname)
