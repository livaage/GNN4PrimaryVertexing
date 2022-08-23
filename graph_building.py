import argparse

import pandas as pd
import numpy as np
import time
import multiprocessing as mp
from functools import partial
import os


def build_graph(event_number):

    # select one event
    event = trk[trk["entry"] == event_number]

    event["unique_trk_id"] = range(len(event))

    # 85% of samples are not primary vertex, repeat true vertices to rebalance
    pv = event[event["is_pv"] == 1]
    not_pv = event[event["is_pv"] == 0]

    # balanced_df = pd.concat([pd.concat([pv] * 6), not_pv])
    balanced_df = pd.concat([pv] + [not_pv])

    # fully connected edges
    combos = balanced_df[["entry", "unique_trk_id", "trk_z0"]].merge(
        balanced_df[["unique_trk_id", "entry", "trk_z0"]],
        on="entry",
        suffixes=("_1", "_2"),
    )

    # remove node connection to itself
    combos = combos[combos["unique_trk_id_1"] != combos["unique_trk_id_2"]]

    if radius is not None:
        # remove nodes too far in z0 space
        combos["abs_diff"] = np.abs(combos["trk_z0_1"] - combos["trk_z0_2"])
        combos = combos[combos["abs_diff"] < radius]

    # edge indices in correct format
    connections = np.stack(
        (combos["unique_trk_id_1"].values, combos["unique_trk_id_2"].values)
    )
    X = balanced_df[["trk_pt", "trk_eta", "trk_phi", "trk_z0"]].values

    np.savez(
        outdir + "graph_" + str(event_number),
        **dict(x=X, edge_index=connections, y=balanced_df["is_pv"].values),
    )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r,", "--radius", type=float, default=None)
    parser.add_argument("--outdir", type=str, default="built_graphs/")
    parser.add_argument("--nevents", type=int, default=None)
    parser.add_argument("--nworkers", type=int, default=4)
    args = parser.parse_args()

    # mc = pd.read_pickle('mc_25k.pkl')
    # trk = pd.read_csv('untracked/trk_processed_25k.csv')
    trk = pd.read_pickle("/media/lucas/QS/l1_nnt/trk_processed_25k.pkl")
    trk = trk.reset_index()
    if args.nevents is not None:
        trk = trk.query(f"entry<{args.nevents}").copy()

    n_events = trk["entry"].nunique()

    outdir = args.outdir

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    events = list(range(n_events))
    n_workers = args.nworkers
    radius = args.radius

    t0 = time.time()
    with mp.Pool(processes=n_workers) as pool:
        process_func = partial(build_graph)
        pool.map(process_func, events)
    t1 = time.time()
    print("Finished in", t1 - t0, "seconds")
