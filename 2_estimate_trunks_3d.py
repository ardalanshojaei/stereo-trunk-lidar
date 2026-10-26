import numpy as np
import cv2
from pathlib import Path

FRAMES = sorted(Path('frames').glob('[0-9]*'))
s_u = 0.5
rng = np.random.default_rng(0)

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
    res = dict(N=len(d), s_d=s_d, Z_s=Z_s, X_A=X_A, Z_A=Z_A, r_A=r, X_B=X_B, Z_B=Z_B, r_B=r_B,
               s_Z=np.median(np.sqrt(S_zz)))
    return m, res

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
                print(f"{fd.name} {name:<6} {k} -> {j}  N {q['N']:5d}  s_d {q['s_d']:.2f}  sigma_Z/px {q['s_Z']:.3f}  "
                      f"A ({q['X_A']:.2f}, {q['Z_A']:.2f}) r {q['r_A']:.3f}  "
                      f"B ({q['X_B']:.2f}, {q['Z_B']:.2f}) r {q['r_B']:.3f}")
