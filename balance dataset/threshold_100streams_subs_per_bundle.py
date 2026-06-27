import glob, os
import json
 
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

# input: gliodef "*bundle.csv"
# algorithm: for each subject -> open file, 
# check each bundle if its streamlines > 100 for all bundles
# then with each sub, for the bundles that pass the threshold 
# append it to the dictionary of bundle_idx
# where keys() are bundles, items() are paths to trk of subjects that passed
# output: a json file of bundle_idx

rootdir = '/nilab-nexus/datasets/GLIODEF'  # the data is on nilab-nexus
tracto_sub = [1112,1117,1126,1140,1160,1161,1167,1173,1174,1178,1181,1182,1188,1189,1190,1193,1204,1207,1213,1248]
bundles = ["IFOF_L", "PYT_L", "ILF_L", "FAT_L", "AF_L"]

# all bundles
def check_bundles(path,bundles,threshold):
    df = pd.read_csv(path, usecols = bundles)
    passing = [b for b in bundles if df[b].sum() > threshold]
    return path,passing
    # for each subject, return the path & the bundle that passed the threshold

threshold = 100
if __name__ == "__main__":
    all_files = []
    for sub in tracto_sub:
        all_files += glob.glob(os.path.join(rootdir,f"sub-{sub}", "tractography", "*bundle.csv"))

    bundle_idx = {b: [] for b in bundles}
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        # for every file in all_files, ex.submit() hands one task to the pool 
        # & returns a Future object as a placeholder
        # all 12540 tasks get queued up near-instantly & distributed into processes
        futures = {ex.submit(check_bundles,f,bundles, threshold): f for f in all_files}
        # -> {future: original_filepath}

        # as_completed(futures) yields each future the moment its underlying task finishes.
        for fut in as_completed(futures):
            path,passing = fut.result()
            for b in passing: # only the bundle that passes
                bundle_idx[b].append(path)
# for each subject, it goes through 5 bundles, returns path and passing where
# path is path of the sb, passing is the name of bundles that are relevant

# then each subject is a process, for the subject, with the bundles in passing
# then the path of the subject is appended to that bundle


    with open("bundle_idx.json", "w") as f:
        json.dump(bundle_idx,f,indent=2)
# ex.submit(func,*args) schedules a single call to func(*args) schedules a single call to func(*args)
# submit() creates a Future (empty placeholder, task not necessarily started) 
# → task runs in a worker process → Future gets filled in with either a return value or an exception 
# → as_completed() notices it's filled in and yields it to you → you call .result() to unwrap whatever's inside.
