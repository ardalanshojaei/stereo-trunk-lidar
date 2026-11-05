import numpy as np
import cv2
from pathlib import Path

FRAMES = sorted(Path('frames').glob('[0-9]*'))
OUT = Path('results')
s_u = 0.5
n_mc = 200
h_rho = 10
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

def autocorr(x, shape):
    f = np.fft.rfft2(x, shape)
    return np.fft.irfft2(f * f.conj(), shape)

def n_eff(eps, m):
    v, u = np.nonzero(m)
    v0, v1, u0, u1 = v.min(), v.max() + 1, u.min(), u.max() + 1
    E = np.where(m, eps, 0)[v0:v1, u0:u1]
    M = m[v0:v1, u0:u1].astype(float)
    # zero padding so that lags up to h_rho do not wrap around
    shape = (E.shape[0] + h_rho + 1, E.shape[1] + h_rho + 1)
    lag = list(range(h_rho + 1)) + list(range(-h_rho, 0))
    ac = autocorr(E, shape)[lag][:, lag]
    pairs = autocorr(M, shape)[lag][:, lag]
    rho = (ac / np.maximum(pairs, 1)) / (ac[0, 0] / pairs[0, 0])
    area = rho[pairs > 0.3 * pairs[0, 0]].sum()
    return len(v) / max(area, 1)

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

    def d_mod(c, du=0):
        return fB / ray_hit(u + du, c[0], c[1], r, f_x, c_x)[0]
    c = np.array([X_A, Z_A])
    s_dc = max(s_d, 0.25)
    for _ in range(2):
        lam = 1e-3
        for _ in range(100):
            d0 = d_mod(c)
            dd_dX = (d_mod(c + [1e-4, 0]) - d0) / 1e-4
            dd_dZ = (d_mod(c + [0, 1e-4]) - d0) / 1e-4
            G = np.column_stack([dd_dX, dd_dZ])  # (N, 2)
            w_d = s_dc ** 2 + ((d_mod(c, 0.5) - d_mod(c, -0.5)) * s_u) ** 2
            ok = np.isfinite(d0) & np.isfinite(G).all(1) & np.isfinite(w_d)
            Gw = G[ok] / w_d[ok, None]
            F_info = Gw.T @ G[ok]
            g = Gw.T @ (d - d0)[ok]
            chi2 = np.sum((d - d0)[ok] ** 2 / w_d[ok])
            n_hit = ray_hit(u, c[0], c[1], r, f_x, c_x)[1].sum()
            # LM, a step must lower the cost and not lose rays off the circle
            step = np.zeros(2)
            while lam < 1e10:
                trial = np.linalg.solve(F_info + lam * np.diag(np.diag(F_info)), g)
                hit_t = ray_hit(u, *(c + trial), r, f_x, c_x)[1].sum()
                if np.nansum((d - d_mod(c + trial))[ok] ** 2 / w_d[ok]) < chi2 and hit_t >= n_hit:
                    step = trial
                    break
                lam *= 10
            c = c + step
            lam = max(lam / 10, 1e-6)
            if np.hypot(*step) < 1e-4:
                break
        eps = np.full(disp.shape, np.nan)
        eps[v, u] = d - d_mod(c)
        s_dc = 1.4826 * np.nanmedian(np.abs(eps[v, u] - np.nanmedian(eps[v, u])))
    C = np.linalg.pinv(F_info)  # (2, 2) CRLB of (X_c, Z_c) given r
    for k in np.unique(u):
        eps[v[u == k], k] -= np.nanmedian(eps[v[u == k], k])
    N_eff = n_eff(np.nan_to_num(eps), m & np.isfinite(eps))
    res = dict(N=len(d), N_eff=N_eff, s_d=s_dc, Z_s=Z_s, X_A=X_A, Z_A=Z_A, r_A=r, X_B=X_B, Z_B=Z_B, r_B=r_B,
               X_C=c[0], Z_C=c[1], var_X=C[0, 0], var_Z=C[1, 1], cov_XZ=C[0, 1], last_step=np.hypot(*step),
               v_min=v.min(), v_max=v.max())
    return m, res

f_x, c_x, B, e = 655.3, 462.9, 0.2537, 3
print('synthetic cylinder, r = 0.15 m, sigma_d = 0.5 px, sigma_u = %.1f px, %d runs per row' % (s_u, n_mc))
print('std, CRLB and pred in mm; pred = CRLB sqrt(N/N_eff)')
print(' Z_c  noise   bias A  bias C   std C    CRLB    pred  std/CRLB  std/pred  N/Neff  conv  '
      'B: |r_B-r|>r/2  median r_B')
for Z_c in (3, 5, 8, 10, 15):
    for corr in (0, 1):
        rows_t = slice(200, 200 + int(f_x * 0.8 / Z_c))
        Z_t, hit = ray_hit(np.arange(960), 0.2 * Z_c, Z_c, 0.15, f_x, c_x)
        disp0 = np.full((600, 960), np.nan)
        disp0[rows_t] = np.where(hit, f_x * B / Z_t, np.nan)
        runs = []
        for _ in range(n_mc):
            # column jitter s_u, (rows, 960)
            u_j = np.arange(960) + rng.normal(0, s_u, disp0[rows_t].shape)
            disp_j = disp0.copy()
            disp_j[rows_t] = np.where(np.isfinite(disp0[rows_t]), f_x * B / ray_hit(u_j, 0.2 * Z_c, Z_c, 0.15, f_x,
                                                                                      c_x)[0], np.nan)
            noise = rng.normal(0, 0.5, disp0.shape)
            if corr:
                noise = cv2.blur(noise, (5, 5)) * 5
            runs.append(estimate(disp_j + noise, np.isfinite(disp0), f_x, c_x, B, e)[1])
        N, N_eff = column(runs, 'N'), column(runs, 'N_eff')
        Z_A, Z_C = column(runs, 'Z_A'), column(runs, 'Z_C')
        r_B, var_Z = column(runs, 'r_B'), column(runs, 'var_Z')
        s_crlb = np.mean(np.sqrt(var_Z))
        s_pred = np.mean(np.sqrt(var_Z * N / N_eff))
        print('%4.0f  %-6s %+7.3f %+7.3f %7.1f %7.1f %7.1f %9.2f %9.2f %7.1f %5.0f%% %10.2f %14.3f' % (
            Z_c, ('white', 'box5')[corr], np.mean(Z_A - Z_c), np.mean(Z_C - Z_c), 1e3 * np.std(Z_C),
            1e3 * s_crlb, 1e3 * s_pred, np.std(Z_C) / s_crlb, np.std(Z_C) / s_pred, np.mean(N / N_eff),
            100 * np.mean(column(runs, 'last_step') < 1e-4), np.mean(np.abs(r_B - 0.15) > 0.075), np.median(r_B)))

rows = []
for fd in FRAMES:
    cal = np.load(fd / 'calib.npz')
    disp = np.load(fd / 'disp.npy')
    m_o = cv2.imread(str(fd / 'mask_oracle.png'), cv2.IMREAD_UNCHANGED)
    f_x, c_x, B = float(cal['f_x']), float(cal['c_x']), float(cal['B'])
    e = (int(cal['w_b']) + 1) // 2
    for src, name in enumerate(('oracle', 'pred')):
        M = cv2.imread(str(fd / ('mask_%s.png' % name)), cv2.IMREAD_UNCHANGED)
        kept = np.zeros_like(M)
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
                kept[out[0]] = k
                rows.append([int(fd.name), src, k, j, cover.max()] + list(out[1].values()))
        cv2.imwrite(str(fd / f'clean_{name}.png'), kept)

OUT.mkdir(exist_ok=True)
header = ('frame,src,k,oracle_k,cover,N,N_eff,s_d,Z_s,X_A,Z_A,r_A,X_B,Z_B,r_B,'
          'X_C,Z_C,var_X,var_Z,cov_XZ,last_step,v_min,v_max')
np.savetxt(OUT / 'trunks.csv', np.array(rows), delimiter=',', fmt='%.6g', header=header)
print('%d trunk estimates written to %s' % (len(rows), OUT / 'trunks.csv'))
steps = np.array([row[header.split(',').index('last_step')] for row in rows])
print('C converged (last step < 0.1 mm) on %d of %d' % ((steps < 1e-4).sum(), len(rows)))
