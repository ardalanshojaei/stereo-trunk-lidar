import numpy as np
import cv2
from pathlib import Path

FRAMES = sorted(Path('frames').glob('[0-9]*'))
s_u = 0.5
n_mc = 200
rng = np.random.default_rng(0)
np.seterr(invalid='ignore')

def ray_hit(u, X_c, Z_c, r, f_x, c_x):
    # rectified left frame; a ray that misses the circle gets its closest approach
    x_n = (u - c_x) / f_x
    # ray X = x_n Z against the circle: a Z^2 - 2 b Z + c = 0
    a = x_n ** 2 + 1
    b = x_n * X_c + Z_c
    c = X_c ** 2 + Z_c ** 2 - r ** 2
    disc = b ** 2 - a * c
    Z = (b - np.sqrt(np.maximum(disc, 0))) / a
    return Z, disc >= 0

def column(runs, key):
    return np.array([run[key] for run in runs])

def estimate(disp, m0, f_x, c_x, B, e):
    fB = f_x * B
    m_e = cv2.erode(np.uint8(m0), np.ones((2 * e + 1, 2 * e + 1), np.uint8)) > 0
    if np.isfinite(disp[m_e]).sum() < 30:
        return None
    d_med = np.nanmedian(disp[m_e])
    s_d = 1.4826 * np.nanmedian(np.abs(disp[m_e] - d_med))
    m = m_e & (np.abs(disp - d_med) <= 3 * max(s_d, 0.25))
    v, u = np.nonzero(m)
    d = disp[v, u]

    vv, uu = np.nonzero(m_e)
    ext = np.array([(uu[vv == k].min(), uu[vv == k].max()) for k in np.unique(vv)])
    w_e = np.median(ext[:, 1] - ext[:, 0] + 1)
    u_bar = np.median(ext.mean(1))
    cos_t = f_x / np.hypot(f_x, u_bar - c_x)
    Z_s = fB / np.median(d)
    w = w_e + 2 * e
    r = w * Z_s * cos_t / (2 * f_x)
    a = w_e / w
    Z_A = Z_s + r * np.sqrt(1 - a ** 2 / 4) * cos_t
    X_A = (u_bar - c_x) * Z_A / f_x

    X, Z = (u - c_x) * B / d, fB / d  # (N,) rectified left frame, horizontal plane
    A3 = np.column_stack([X, Z, np.ones_like(X)])
    b3 = -(X ** 2 + Z ** 2)
    tol = 2 * s_d * Z_s ** 2 / fB
    best, keep = 0, np.ones(len(X), bool)
    for _ in range(200):
        j = rng.choice(len(X), 3, replace=False)
        D, E, F = np.linalg.lstsq(A3[j], b3[j], rcond=None)[0]
        inl = np.abs(np.hypot(X + D / 2, Z + E / 2) - np.sqrt(D ** 2 / 4 + E ** 2 / 4 - F)) < tol
        if inl.sum() > best and -E / 2 > Z_s:
            best, keep = inl.sum(), inl
    D, E, F = np.linalg.lstsq(A3[keep], b3[keep], rcond=None)[0]
    X_B, Z_B, r_B = -D / 2, -E / 2, np.sqrt(D ** 2 / 4 + E ** 2 / 4 - F)

    # per-pixel covariance J diag(s_u^2, s_d^2) J^T of (X, Z)
    s_dp = max(s_d, 0.25)
    J11 = B / d
    J12 = -(u - c_x) * B / d ** 2
    J22 = -fB / d ** 2
    S_xx = J11 ** 2 * s_u ** 2 + J12 ** 2 * s_dp ** 2
    S_xz = J12 * J22 * s_dp ** 2
    S_zz = J22 ** 2 * s_dp ** 2
    c = np.array([X_A, Z_A])
    for _ in range(5):
        rho = np.hypot(X - c[0], Z - c[1])
        n_x, n_z = (X - c[0]) / rho, (Z - c[1]) / rho
        e_r = rho - r
        w_r = n_x ** 2 * S_xx + 2 * n_x * n_z * S_xz + n_z ** 2 * S_zz
        F_info = np.array([[np.sum(n_x * n_x / w_r), np.sum(n_x * n_z / w_r)],
                           [np.sum(n_x * n_z / w_r), np.sum(n_z * n_z / w_r)]])
        step = np.linalg.solve(F_info, [np.sum(n_x * e_r / w_r), np.sum(n_z * e_r / w_r)])
        c = c + step
    C = np.linalg.inv(F_info)  # (2, 2) CRLB of (X_c, Z_c) given r
    res = dict(N=len(d), s_d=s_d, Z_s=Z_s, X_A=X_A, Z_A=Z_A, r_A=r, X_B=X_B, Z_B=Z_B, r_B=r_B,
               X_C=c[0], Z_C=c[1], var_X=C[0, 0], var_Z=C[1, 1])
    return m, res

f_x, c_x, B, e = 655.3, 462.9, 0.2537, 3
print('synthetic cylinder, r = 0.15 m, sigma_d = 0.5 px, %d runs per row' % n_mc)
print(' Z_c  noise   bias A  bias C   std C   CRLB  std/CRLB  B: |r_B-r|>r/2  median r_B')
for Z_c in (3, 5, 8, 10, 15):
    for corr in (0, 1):
        Z_t, hit = ray_hit(np.arange(960), 0.2 * Z_c, Z_c, 0.15, f_x, c_x)
        disp0 = np.full((600, 960), np.nan)
        disp0[200:200 + int(f_x * 0.8 / Z_c)] = np.where(hit, f_x * B / Z_t, np.nan)
        runs = []
        for _ in range(n_mc):
            noise = rng.normal(0, 0.5, disp0.shape)
            if corr:
                noise = cv2.blur(noise, (5, 5)) * 5
            runs.append(estimate(disp0 + noise, np.isfinite(disp0), f_x, c_x, B, e)[1])
        Z_A, Z_C = column(runs, 'Z_A'), column(runs, 'Z_C')
        r_B, var_Z = column(runs, 'r_B'), column(runs, 'var_Z')
        print('%4.0f  %-6s %+7.3f %+7.3f %7.3f %6.3f %9.2f %10.2f %12.3f' % (
            Z_c, ('white', 'box5')[corr], np.mean(Z_A - Z_c), np.mean(Z_C - Z_c), np.std(Z_C),
            np.mean(np.sqrt(var_Z)), np.std(Z_C) / np.mean(np.sqrt(var_Z)),
            np.mean(np.abs(r_B - 0.15) > 0.075), np.median(r_B)))

for fd in FRAMES:
    cal = np.load(fd / 'calib.npz')
    disp = np.load(fd / 'disp.npy')
    m_o = cv2.imread(str(fd / 'mask_oracle.png'), cv2.IMREAD_UNCHANGED)
    f_x, c_x, B = float(cal['f_x']), float(cal['c_x']), float(cal['B'])
    e = (int(cal['w_b']) + 1) // 2
    for src, name in enumerate(('oracle', 'pred')):
        M = cv2.imread(str(fd / f'mask_{name}.png'), cv2.IMREAD_UNCHANGED)
        for k in range(1, M.max() + 1):
            # share of each oracle trunk covered by this mask
            hits = np.bincount(m_o[M == k], minlength=m_o.max() + 1)
            sizes = np.maximum(np.bincount(m_o.ravel()), 1)
            cover = (hits / sizes)[1:]
            if src == 0:
                j = k
            elif cover.max() >= 0.5:
                j = 1 + int(np.argmax(cover))
            else:
                j = 0
            out = estimate(disp, M == k, f_x, c_x, B, e)
            if out is not None:
                q = out[1]
                print(f"{fd.name} {name:<6} {k} -> {j}  N {q['N']:5d}  s_d {q['s_d']:.2f}  "
                      f"A ({q['X_A']:.2f}, {q['Z_A']:.2f}) r {q['r_A']:.3f}  "
                      f"B ({q['X_B']:.2f}, {q['Z_B']:.2f}) r {q['r_B']:.3f}  C ({q['X_C']:.2f}, {q['Z_C']:.2f})")
