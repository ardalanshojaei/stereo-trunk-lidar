import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path('results')
FR = Path('frames')
VIEW = 48  # frame for the bird's eye view
OCCLUDED = (168, 3)  # frame, trunk: the LiDAR picks up an object in front of it
bins = [0, 5, 10, 15, 30]

with open(OUT / 'trunks.csv') as f:
    names = f.readline().lstrip('# ').strip().split(',')
T = dict(zip(names, np.loadtxt(OUT / 'trunks.csv', delimiter=',').T))
cal = np.load(FR / f'{VIEW:04d}' / 'calib.npz')
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
    # smooth surface: stereo disparity spans less than 1 px over the 9x9 window around the pixel
    d_hi = cv2.dilate(np.nan_to_num(disp, nan=-100), np.ones((9, 9)))
    d_lo = -cv2.dilate(-np.nan_to_num(disp, nan=-100), np.ones((9, 9)))
    # (M, 4) columns: u - c_x, LiDAR disparity fB / Z, stereo minus LiDAR disparity, smooth; Z < 20 m
    e = np.column_stack([u - c_x, fB / P[:, 2], disp[v, u] - fB / P[:, 2], (d_hi - d_lo)[v, u] < 1])
    return disp, P, u, v, e[np.isfinite(e[:, 2]) & (e[:, 1] > fB / 20)]

gt, e_d = {}, []
for fd in sorted(FR.glob('[0-9]*')):
    disp, P, u, v, e = project(fd)
    e_d.append(e)
    # Livox is 0.15 m right of the camera, background past a trunk's right edge lands in the mask
    m_o = cv2.erode(cv2.imread(str(fd / 'mask_oracle.png'), cv2.IMREAD_UNCHANGED), np.ones((7, 7), np.uint8))
    for k in range(1, m_o.max() + 1):
        s = P[m_o[v, u] == k]
        s = s[np.argsort(s[:, 2])]
        # depth groups split at gaps of 0.3 m, keep the largest
        groups = np.split(s, np.flatnonzero(np.diff(s[:, 2]) > 0.3) + 1)
        g = max(groups, key=len)
        if len(g) >= 8:
            p, cov = circle_fit(g[:, 0], g[:, 2])
            gt[(int(fd.name), k)] = (p, cov, g)
e_d = np.concatenate(e_d)

held = sorted((FR / 'heldout').glob('*'))
H = []
for fd in held:
    _, _, _, _, e = project(fd)
    H.append(e)
H = np.concatenate(H)
H = H[(H[:, 1] < fB / 3) & (np.abs(H[:, 2]) < 3)]
A_h = np.column_stack([np.ones(len(H)), H[:, 0], H[:, 1]])
w = np.ones(len(H))
for _ in range(10):
    k_d = np.linalg.lstsq(A_h * w[:, None], H[:, 2] * w, rcond=None)[0]
    w = 1 / np.maximum(np.abs(H[:, 2] - A_h @ k_d), 0.3)

lines = ['calibration check, every Livox point against SGBM, d_stereo - fB/Z_lidar [px], median and robust std']
for lo in range(80, 880, 160):
    s = (e_d[:, 1] < fB / 5) & (e_d[:, 0] + c_x >= lo) & (e_d[:, 0] + c_x < lo + 160)
    f = s & (e_d[:, 3] > 0)
    lines.append('  Z > 5 m, columns %3d-%3d  %+.3f  %.3f   smooth surfaces only %+.3f  %.3f' % (
        (lo, lo + 160) + robust(e_d[s, 2]) + robust(e_d[f, 2])))
lines.append('held-out fit, %d frames, %d points: d_stereo - d_lidar = %+.3f %+.5f (u - c_x) %+.4f d_lidar'
             % (len(held), len(H), *k_d))

# corrected disparity (d - a - b (u - c_x)) / (1 + c), applied to the median disparity of each trunk
a, b, c = k_d
u_c = f_x * T['X_A'] / T['Z_A'] + c_x
Z_s = fB * (1 + c) / (fB / T['Z_s'] - a - b * (u_c - c_x))
shift = Z_s - T['Z_s']
for e in ['A', 'B', 'C']:
    Z_new = T['Z_' + e] + shift
    T['X_' + e + '*'] = T['X_' + e] * Z_new / T['Z_' + e]
    T['Z_' + e + '*'] = Z_new

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
est = ['A', 'B', 'C', 'A*', 'B*', 'C*']
dZ = {e: T['Z_' + e] - Z_L for e in est}
dX = {e: T['X_' + e] - X_L for e in est}
s_pred = np.sqrt(T['var_Z'] * T['N'] / T['N_eff'] + (T['Z_s'] / (2 * f_x)) ** 2)
ora = np.isfinite(Z_L) & (T['src'] == 0)
prd = np.isfinite(Z_L) & (T['src'] == 1)
pull = {e: (dZ[e] / np.hypot(s_pred, s_GT))[ora] for e in ('C', 'C*')}
clean = np.array([key != OCCLUDED for key in keys])

lines.append('LiDAR truth: %d oracle trunks with >= 8 points, %d pass radius and fit checks' % (len(gt), len(ok_gt)))
lines.append('LiDAR center sigma Z median %.3f m, p90 %.3f m; median LiDAR radius per Z bin %s m' % (
    np.median(s_GT[ora]), np.percentile(s_GT[ora], 90),
    ' '.join('%.3f' % np.median(r_L[ora & (Z_L >= lo) & (Z_L < hi)]) for lo, hi in zip(bins, bins[1:]))))
lines.append('C oracle: predicted sigma Z median %.3f m, of which the bound F^-1 alone %.4f m' % (
    np.median(s_pred[ora]), np.median(np.sqrt(T['var_Z'][ora]))))
lines.append('YOLO covers %d of %d oracle trunks with truth' % (len({keys[i] for i in np.nonzero(prd)[0]}), ora.sum()))
lines.append('* = held-out disparity correction; RMSE* without frame %d trunk %d (object in front)' % OCCLUDED)
lines.append('mask   est  Z bin     n  med dZ  RMSE dZ  RMSE*  robust  med dX  RMSE dX  B radius off by > r/2')
for src, name in ((ora, 'oracle'), (prd, 'yolo')):
    for e in est:
        for lo, hi in zip(bins, bins[1:]):
            s = src & (Z_L >= lo) & (Z_L < hi)
            if not s.any():
                continue
            b_off = ''
            if e == 'B':
                b_off = '%d of %d' % (np.sum(np.abs(T['r_B'][s] - r_L[s]) > r_L[s] / 2), s.sum())
            lines.append('%-6s %-2s  %2d-%2d  %4d  %+6.2f  %6.2f  %6.2f  %6.2f  %+6.2f  %6.2f   %s' % (
                name, e, lo, hi, s.sum(), np.median(dZ[e][s]), rms(dZ[e][s]), rms(dZ[e][s & clean]),
                robust(dZ[e][s])[1], np.median(dX[e][s]), rms(dX[e][s]), b_off))
s_d_eff = {}
for e in ['C', 'C*']:
    s_d_eff[e] = robust(dZ[e][ora] * fB / Z_L[ora] ** 2)[1]
    p_e = pull[e]
    lines.append(f'{e:<3} oracle: robust spread of dZ fB/Z^2 {s_d_eff[e]:.2f} px, pull mean {p_e.mean():+.1f} '
                 f'std {p_e.std():.1f} median {np.median(p_e):+.1f} n {len(p_e)}')
print('\n'.join(lines))
open(OUT / 'table.txt', 'w').write('\n'.join(lines) + '\n')

z = np.linspace(2, 17, 100)
band = z ** 2 * s_d_eff['C*'] / fB
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), layout='tight')
for a, err, label in ((ax[0], dZ, 'dZ = stereo - LiDAR [m]'), (ax[1], dX, 'dX = stereo - LiDAR [m]')):
    a.fill_between(z, -2 * band, 2 * band, color='0.9', lw=0)
    a.fill_between(z, -band, band, color='0.78', lw=0)
    a.plot(Z_L[ora], err['C'][ora], '+', label='C oracle')
    a.plot(Z_L[ora], err['C*'][ora], 'o', mfc='none', label='C* oracle')
    a.plot(Z_L[prd], err['C'][prd], 'x', label='C YOLO')
    a.set(xlabel='LiDAR Z [m]', ylabel=label, ylim=(-2.5, 1))
ax[0].legend(loc='lower left')
ax[0].set_title('bands: 1 and 2 x %.2f px (corrected spread), scaled by Z^2/fB' % s_d_eff['C*'], fontsize=9)
for e, label in (('C', 'published'), ('C*', 'corrected')):
    ax[2].hist(np.clip(pull[e], -10, 10), bins=np.arange(-10, 10.1, 0.5), density=True, alpha=0.5,
               label='%s: mean %+.1f std %.1f' % (label, pull[e].mean(), pull[e].std()))
x = np.linspace(-4, 4, 200)
ax[2].plot(x, np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi), 'k')
ax[2].legend(fontsize=8, title='pull dZ / sigma, C oracle, clipped at 10', title_fontsize=8)
fig.savefig(OUT / 'error_vs_range.png', dpi=110)

disp, P, _, _, _ = project(FR / f'{VIEW:04d}')
mask = cv2.imread(str(FR / f'{VIEW:04d}' / 'clean_oracle.png'), cv2.IMREAD_UNCHANGED)
s_d = np.median(T['s_d'][ora])
t = np.linspace(0, 2 * np.pi, 50)
fig, a = plt.subplots(figsize=(8, 8))
near = np.abs(P[:, 1]) < 1.5
a.plot(P[near, 0], P[near, 2], '.', ms=1, color='0.7')
u = np.nonzero(mask == 1)[1]
d = disp[mask == 1]
a.plot((u - c_x) * B / d, fB / d, '.', ms=1)
circle = np.vstack([np.cos(t), np.sin(t)])
for i in np.linspace(0, len(d) - 1, 6).astype(int):
    X_i = (u[i] - c_x) * B / d[i]
    Z_i = fB / d[i]
    J = np.array([[B / d[i], -(u[i] - c_x) * B / d[i] ** 2], [0, -fB / d[i] ** 2]])
    w, V = np.linalg.eigh(J @ np.diag([0.5 ** 2, s_d ** 2]) @ J.T)
    ell = V @ (2 * np.sqrt(w)[:, None] * circle)  # (2, 50) 2 sigma ellipse
    a.plot(X_i + ell[0], Z_i + ell[1], lw=0.6)
p_g, _, g = ok_gt[(VIEW, 1)]
X_g, Z_g, r_g = p_g
a.add_patch(plt.Circle((X_g, Z_g), r_g, fill=False, color='k'))
a.plot(g[:, 0], g[:, 2], 'k.', ms=3)
i = (T['frame'] == VIEW) & (T['src'] == 0) & (T['k'] == 1)
for e, mk in (('A', 'o'), ('B', 'x'), ('C', '+')):
    a.plot(T['X_' + e][i], T['Z_' + e][i], mk, color='r', mfc='none', ms=8)
a.set(xlabel='X [m]', ylabel='Z [m]', aspect='equal', xlim=(X_g - 0.6, X_g + 0.6), ylim=(Z_g - 0.8, Z_g + 0.4))
a.set_title('frame %d trunk 1: Livox (black: circle fit), stereo, 2 sigma ellipses, A o B x C +' % VIEW, fontsize=9)
fig.savefig(OUT / f'birds_eye_{VIEW:04d}.png', dpi=110)
