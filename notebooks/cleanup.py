import pandas as pd

"""
# Noise dataset
read the original noise dataset (without x,y,z,dx,dy & dz)
"""

noise = pd.read_csv("data/noise.csv")

"""
add a column containing the corresponding id in the positions array. The
formula was provided to us by Roel.
"""

noise["pos_idx"] = 31 * (noise["dom id"] - 1) + noise["pmt id"]

"""
Next convert numpy array to pandas dataframe.
NOTE run notebooks/parse_detx.py and have `positions` in memory
"""

pos = pd.DataFrame(positions)
pos['pos_idx'] = pos.index

"""
Finally, combine the two, cleanup and save to disk.
"""

df = pd.merge(noise, pos, on='pos_idx')
df = df.drop(columns=['pos_idx'])
df = df.rename(columns={'dom id': 'dom_id', 'pmt id': 'pmt_id',
    'time-over-threshold': 'tot', 'x': 'pos_x', 'y' : 'pos_y', 'z': 'pos_z',
    'dx': 'dir_x', 'dy': 'dir_y', 'dz': 'dir_z'})

"""
# Events dataset
NOTE One Jun 09, 2020 Roel provided a new, cleanup up events dataset.
We do not care for `mc_info` for the time being
"""

mc_info = pd.read_hdf("data/events.h5", key="/data/mc_info")
mc_hits = pd.read_hdf("data/events.h5", key="/data/mc_hits")

"""
align the column names of `mc_hits` with that of noise
"""

mc_hits = mc_hits.rename(columns={'h.dom_id': 'dom_id', 'h.pmt_id': 'pmt_id',
    'h.pos.x': 'pos_x', 'h.pos.y': 'pos_y', 'h.pos.z': 'pos_z', 'h.dir.x':
    'dir_x', 'h.dir.y': 'dir_y', 'h.dir.z': 'dir_z', 'h.tot': 'tot', 'h.t':
    'time'})

"""
add labels
"""

df["label"] = 0
mc_hits["label"] = 1


"""
and save to disk.
"""

mc_hits.to_csv("data/hits.csv", index=False)
df.to_csv("data/noise.csv", index=False)
