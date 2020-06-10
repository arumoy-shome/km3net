import k40gen
import re
import numpy as np


def parse_detx(filename, pmt2index):
    line_expr = re.compile(r"\s*(\d+)\s+(-?\d+\.\d+\s*){7}")
    float_expr = re.compile(r"-?\d+\.\d+")

    position_dt = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                            ('dx', 'f4'), ('dy', 'f4'), ('dz', 'f4')])
    positions = np.zeros(115 * 18 * 31, dtype=position_dt)
    with open(filename) as det_file:
        for line in det_file:
            line = line.strip()
            m = line_expr.match(line)
            if m:
                idx = pmt2index(int(m.group(1)))
                positions[idx] = tuple(float(e) for e in float_expr.findall(line[m.end(1):])[:6])
    return positions


positions = parse_detx("data/noise.detx", lambda pmt: pmt - 1)


# gens = k40gen.Generators(21341, 1245, [7000., 700., 70., 0.])
# times = k40gen.generate_k40(0, int(1e8), gens, "reference", False)
