import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from rosbags.highlevel import AnyReader

BAG = Path('data/example_filtered.bag')
FRAMES = [108, 162]
OUT = Path('out/1')
n_d, w_b, tol_lr = 128, 5, 1.0  # SGBM disparities, block size, left-right tolerance [px]
wh = (800, 592)

def quat_R(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

def draw_polygons(img):
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.title('click a trunk outline, Enter closes it; Enter with no clicks finishes')
    mask = np.zeros(img.shape[:2], np.uint8)
    uv = np.array(plt.ginput(n=-1, timeout=0))
    while len(uv) >= 3:
        cv2.fillPoly(mask, [np.round(uv).astype(np.int32)], int(mask.max()) + 1)
        plt.fill(uv[:, 0], uv[:, 1], alpha=0.3)
        uv = np.array(plt.ginput(n=-1, timeout=0))
    plt.close()
    return mask

def stamp(m):
    return m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec

t_img, t_cloud, cloud, cam, tf, img = [], [], [], {}, [], {}
with AnyReader([BAG]) as rd:
    con = {c.topic: c for c in rd.connections}
    topics = ['/nerian/left/image_raw', '/nerian/left/camera_info', '/nerian/right/camera_info',
              '/os1_cloud_node/points', '/tf_static']
    for c, t, raw in rd.messages(connections=[con[name] for name in topics]):
        m = rd.deserialize(raw, c.msgtype)
        if c.topic == '/tf_static':
            for x in m.transforms:
                p, q = x.transform.translation, x.transform.rotation
                tf.append((x.header.frame_id, x.child_frame_id, quat_R(q.w, q.x, q.y, q.z), np.array([p.x, p.y, p.z])))
        elif 'camera_info' in c.topic:
            cam.setdefault(c.topic.split('/')[2], (np.array(m.K).reshape(3, 3), np.array(m.D).reshape(1, -1)))
        elif c.topic == '/os1_cloud_node/points':
            t_cloud.append(stamp(m))
            cloud.append(bytes(m.data))
        else:
            t_img.append(stamp(m))
    want = [t_img[i] for i in FRAMES]
    for c, t, raw in rd.messages(connections=[con['/nerian/left/image_raw'], con['/nerian/right/image_raw']]):
        m = rd.deserialize(raw, c.msgtype)
        if stamp(m) in want:
            img[(c.topic.split('/')[2], stamp(m))] = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)

# pose of the left / right optical frame (X right, Y down, Z forward) in base_link
pose = {child: (R, t) for parent, child, R, t in tf if parent == 'base_link'}
R_bl, t_bl = pose['nerian_left']
R_br, t_br = pose['nerian_right']
R_lr = R_bl.T @ R_br
t_lr = R_bl.T @ (t_br - t_bl)
K_l, D_l = cam['left']
K_r, D_r = cam['right']
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K_l, D_l, K_r, D_r, wh, R_lr.T, (-R_lr.T @ t_lr).reshape(3, 1), alpha=0)
map_l = cv2.initUndistortRectifyMap(K_l, D_l, R1, P1, wh, cv2.CV_32FC1)
map_r = cv2.initUndistortRectifyMap(K_r, D_r, R2, P2, wh, cv2.CV_32FC1)
f_x, c_x, c_y = P1[0, 0], P1[0, 2], P1[1, 2]
B = -P2[0, 3] / P2[0, 0]
print('rectified f_x %.2f  c_x %.2f  c_y %.2f  B %.4f m  fB %.1f px m' % (f_x, c_x, c_y, B, f_x * B))

sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=n_d, blockSize=w_b, P1=24 * w_b ** 2, P2=96 * w_b ** 2,
                             uniquenessRatio=10, speckleWindowSize=100, speckleRange=2, disp12MaxDiff=-1,
                             mode=cv2.STEREO_SGBM_MODE_HH)
v, u = np.mgrid[0:wh[1], 0:wh[0]]
for i, st in zip(FRAMES, want):
    d = OUT / f'{i:04d}'
    d.mkdir(parents=True, exist_ok=True)
    L = cv2.remap(img[('left', st)], *map_l, cv2.INTER_LINEAR)
    R = cv2.remap(img[('right', st)], *map_r, cv2.INTER_LINEAR)
    g_l = cv2.cvtColor(L, cv2.COLOR_RGB2GRAY)
    g_r = cv2.cvtColor(R, cv2.COLOR_RGB2GRAY)
    disp = sgbm.compute(g_l, g_r).astype(np.float32) / 16
    disp_r = sgbm.compute(g_r[:, ::-1].copy(), g_l[:, ::-1].copy())[:, ::-1].astype(np.float32) / 16
    # rectified left frame: (u, v) in left matches (u - disp, v) in right
    d_back = disp_r[v, np.clip(np.round(u - disp).astype(int), 0, wh[0] - 1)]
    ok = (disp > 0) & (d_back > 0) & (np.abs(disp - d_back) <= tol_lr)
    disp[~ok] = np.nan

    if not (d / 'mask.png').exists():
        cv2.imwrite(str(d / 'mask.png'), draw_polygons(L))
    mask = cv2.imread(str(d / 'mask.png'), cv2.IMREAD_UNCHANGED)
    for k in range(1, mask.max() + 1):
        m_k = mask == k
        d_med = np.nanmedian(disp[m_k])
        print('frame %d  object %d  pixels %d  disp %.2f px  Z %.2f m' % (i, k, m_k.sum(), d_med, f_x * B / d_med))

    j = np.argmin(np.abs(np.array(t_cloud) - st))
    pts_lidar = np.frombuffer(cloud[j], np.float32).reshape(-1, 12)[:, :3]  # 48 bytes per point, x y z first
    pts_lidar = pts_lidar[np.linalg.norm(pts_lidar, axis=1) > 0.5]  # (N, 3) in os1_lidar
    print('frame %d  LR-valid %.1f %%  cloud dt %+.1f ms' % (i, 100 * ok.mean(), 1e3 * (t_cloud[j] - st)))
    cv2.imwrite(str(d / 'left_rect.png'), cv2.cvtColor(L, cv2.COLOR_RGB2BGR))
    np.save(d / 'disp.npy', disp)
    np.save(d / 'pts_lidar.npy', pts_lidar)
    # R1 rotates the raw left optical frame into the rectified left frame
    np.savez(d / 'calib.npz', f_x=f_x, c_x=c_x, c_y=c_y, B=B, R1=R1, stamp=st, dt_cloud=t_cloud[j] - st)
    np.savetxt(d / 'tf_static.txt', np.array([(p, c, *R.ravel(), *t) for p, c, R, t in tf], dtype=object), fmt='%s')
