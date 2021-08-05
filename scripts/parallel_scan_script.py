"""
"""
import argparse
from mpi4py import MPI
import numpy as np
from param_scan.latin_hypercube import latin_hypercube
from param_scan.helpers import get_equal_sized_data_chunks, cleanup_and_collate
from param_scan.helpers import get_mpi_rank_outname, write_param_chunk


def get_param_bounds():
    return [(-5, 5), (5, 10)]


def get_param_names():
    return ("param_a", "param_b")


def compute_loss(params, data):
    return -1.0


def get_loss_data():
    return None


PARAM_BOUNDS = get_param_bounds()

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

    n_cubes_per_rank, n_per_chunk = get_equal_sized_data_chunks(n_tot, nranks, n_max_lh)
    total_cubes = n_cubes_per_rank * nranks
    total_seeds = np.arange(total_cubes).astype("i8")
    seeds_per_rank = np.array_split(total_seeds, nranks)[rank]

    for ichunk, seed in enumerate(seeds_per_rank):
        param_chunk = latin_hypercube(PARAM_BOUNDS, n_per_chunk, seed=seed)
        loss_data = get_loss_data()
        rank_outname = get_mpi_rank_outname(outname, rank, ichunk)
        loss_collector = []
        for params in param_chunk:
            loss = compute_loss(params, loss_data)
            loss_collector.append(loss)
        write_param_chunk(rank_outname, param_chunk, np.array(loss_collector))

    if rank == 0:
        cleanup_and_collate(outname)
