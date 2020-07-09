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
add labels
"""

df["label"] = 0


"""
and save to disk.
"""

df.to_csv("data/noise.csv", index=False)
