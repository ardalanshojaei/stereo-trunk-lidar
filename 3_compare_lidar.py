import numpy as np
import cv2
from pathlib import Path

OUT = Path('results')
FR = Path('frames')
bins = [0, 5, 10, 15, 30]

with open(OUT / 'trunks.csv') as f:
    names = f.readline().lstrip('# ').strip().split(',')
T = dict(zip(names, np.loadtxt(OUT / 'trunks.csv', delimiter=',').T))
cal = np.load(FR / '0048' / 'calib.npz')
f_x, c_x, c_y, B = float(cal['f_x']), float(cal['c_x']), float(cal['c_y']), float(cal['B'])
fB = f_x * B

def circle_fit(X, Z):
    D, E, F = np.linalg.lstsq(np.c_[X, Z, np.ones_like(X)], -(X ** 2 + Z ** 2), rcond=None)[0]
    p = np.array([-D / 2, -E / 2, np.sqrt(D ** 2 / 4 + E ** 2 / 4 - F)])
    for _ in range(10):
        rho = np.hypot(X - p[0], Z - p[1])
        J = np.c_[-(X - p[0]) / rho, -(Z - p[1]) / rho, -np.ones_like(X)]
        p = p - np.linalg.lstsq(J, rho - p[2], rcond=None)[0]
    res = np.hypot(X - p[0], Z - p[1]) - p[2]
    return p, np.linalg.pinv(J.T @ J) * np.sum(res ** 2) / max(len(X) - 3, 1)

def robust(x):
    return np.median(x), 1.4826 * np.median(np.abs(x - np.median(x)))

def rms(x):
    return np.sqrt(np.mean(x ** 2))

def project(fd):
    disp, P = np.load(fd / 'disp.npy'), np.load(fd / 'pts_cam.npy')
    P = P[P[:, 2] > 1]  # (N, 3) rectified left frame
    u = np.round(f_x * P[:, 0] / P[:, 2] + c_x).astype(int)
    v = np.round(f_x * P[:, 1] / P[:, 2] + c_y).astype(int)
    on = (u >= 0) & (u < disp.shape[1]) & (v >= 0) & (v < disp.shape[0])
    P, u, v = P[on], u[on], v[on]
    # (M, 3) columns: u - c_x, LiDAR disparity fB / Z, stereo minus LiDAR disparity; Z < 20 m
    e = np.column_stack([u - c_x, fB / P[:, 2], disp[v, u] - fB / P[:, 2]])
    return disp, P, u, v, e[np.isfinite(e[:, 2]) & (e[:, 1] > fB / 20)]

gt, e_d = {}, []
for fd in sorted(FR.glob('[0-9]*')):
    disp, P, u, v, e = project(fd)
    e_d.append(e)
    m_o = cv2.dilate(cv2.imread(str(fd / 'mask_oracle.png'), cv2.IMREAD_UNCHANGED), np.ones((1, 9), np.uint8))
    for k in range(1, m_o.max() + 1):
        s = P[m_o[v, u] == k]
        s = s[np.argsort(s[:, 2])]
        # depth groups split at gaps of 0.3 m, keep the nearest one with 8 points or more
        groups = np.split(s, np.flatnonzero(np.diff(s[:, 2]) > 0.3) + 1)
        groups = [g for g in groups if len(g) >= 8]
        if groups:
            g = groups[0]
            p, cov = circle_fit(g[:, 0], g[:, 2])
            gt[(int(fd.name), k)] = (p, cov, g)
e_d = np.concatenate(e_d)

lines = ['calibration check, every Livox point against SGBM, d_stereo - fB/Z_lidar [px], median and robust std']
for lo in range(80, 880, 160):
    s = (e_d[:, 1] < fB / 5) & (e_d[:, 0] + c_x >= lo) & (e_d[:, 0] + c_x < lo + 160)
    lines.append('  Z > 5 m, columns %3d-%3d  %+.3f  %.3f' % ((lo, lo + 160) + robust(e_d[s, 2])))

ok_gt = {}
for key, (p, cov, g) in gt.items():
    if 0.03 < p[2] < 0.6 and np.sqrt(cov[1, 1]) < 0.1:
        ok_gt[key] = (p, cov, g)
keys = [(int(f), int(k)) for f, k in zip(T['frame'], T['oracle_k'])]
X_L = np.full(len(keys), np.nan)
Z_L = np.full(len(keys), np.nan)
r_L = np.full(len(keys), np.nan)
s_GT = np.full(len(keys), np.nan)
for i, key in enumerate(keys):
    if key in ok_gt:
        p, cov, _ = ok_gt[key]
        X_L[i], Z_L[i], r_L[i] = p
        s_GT[i] = np.sqrt(cov[1, 1])
est = ['A', 'B', 'C']
dZ = {e: T['Z_' + e] - Z_L for e in est}
dX = {e: T['X_' + e] - X_L for e in est}
s_pred = np.sqrt(T['var_Z'] * T['N'] / T['N_eff'] + (T['Z_s'] / (2 * f_x)) ** 2)
ora = np.isfinite(Z_L) & (T['src'] == 0)
prd = np.isfinite(Z_L) & (T['src'] == 1)
pull = (dZ['C'] / np.hypot(s_pred, s_GT))[ora]

lines.append('LiDAR truth: %d oracle trunks with >= 8 points, %d pass radius and fit checks' % (len(gt), len(ok_gt)))
lines.append('YOLO covers %d of %d oracle trunks with truth' % (len({keys[i] for i in np.nonzero(prd)[0]}), ora.sum()))
for i in np.nonzero(ora)[0]:
    lines.append('%4d-%d  Z_L %5.2f  r_L %.3f  dZ A %+6.2f  B %+6.2f  C %+6.2f  dX C %+5.2f' % (
        T['frame'][i], T['k'][i], Z_L[i], r_L[i], dZ['A'][i], dZ['B'][i], dZ['C'][i], dX['C'][i]))
lines.append('C oracle: median dZ %+.2f m, pull mean %+.1f std %.1f n %d' % (
    np.median(dZ['C'][ora]), pull.mean(), pull.std(), len(pull)))
print('\n'.join(lines))
