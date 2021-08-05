"""
"""
import os


def get_parallel_outbase_pattern(fn):
    s = os.path.basename(fn).split(".")
    s.insert(-1, "*")
    s.insert(-1, "*")
    rank_outbase_pat = ".".join(s)
    return rank_outbase_pat


def get_rank_outname(fn, rank, batch):
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
