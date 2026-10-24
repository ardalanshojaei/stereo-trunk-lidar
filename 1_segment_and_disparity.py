import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.rosbag1 import reader
from ultralytics import YOLO

BAG = Path('data/bgarden/1018_00_VLIO.bag')
CAL = 'data/bgarden/calib/'
OUT = Path('frames')
FRAMES = [24, 48, 72, 120, 144, 168, 192, 208, 228, 248, 300, 324, 348, 372, 396, 420, 444, 912, 940]
n_d, w_b, tol_lr = 80, 5, 1.0  # SGBM disparities, block size, left-right tolerance [px]
wh = (960, 600)

# index headers change field order at byte 1,384,229,779 and rosbags 0.11 caches the old offsets
read_index = reader.Reader.read_index_data
def read_index_fresh(self, pos, indexes):
    self.index_data_header_offsets = None
    return read_index(self, pos, indexes)
reader.Reader.read_index_data = read_index_fresh

def intrinsics(name):
    fs = cv2.FileStorage(CAL + 'camera_intrinsics/' + name, cv2.FILE_STORAGE_READ)
    p, k = fs.getNode('projection_parameters'), fs.getNode('distortion_parameters')
    K = np.array([[p.getNode('fx').real(), 0, p.getNode('cx').real()],
                  [0, p.getNode('fy').real(), p.getNode('cy').real()],
                  [0, 0, 1]])
    D = np.array([[k.getNode(n).real() for n in ('k1', 'k2', 'p1', 'p2')]])
    # 1920x1200 calibration, halved for the 960x600 bag images
    return np.diag([0.5, 0.5, 1]) @ K, D

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

t_img, odo, img, scan = [], [], {}, {}
with AnyReader([BAG]) as rd:
    con = {c.topic: c for c in rd.connections}
    for c, t, raw in rd.messages(connections=[con['/dalsa_color/left/image_raw'], con['/odom']]):
        m = rd.deserialize(raw, c.msgtype)
        if c.topic == '/odom':
            odo.append((stamp(m), m.twist.twist.linear.x, m.twist.twist.angular.z))
        else:
            t_img.append(stamp(m))
    want = [t_img[i] for i in FRAMES]
    topics = ['/dalsa_color/left/image_raw', '/dalsa_color/right/image_raw', '/livox/lidar']
    for c, t, raw in rd.messages(connections=[con[name] for name in topics]):
        if c.topic == '/livox/lidar' and np.abs(np.array(want) - 1e-9 * t).min() > 1:
            continue
        m = rd.deserialize(raw, c.msgtype)
        if c.topic == '/livox/lidar':
            t_b = 1e-9 * m.timebase
            for s in want:
                if t_b <= s < t_b + 0.1:
                    scan[s] = np.array([(p.x, p.y, p.z, t_b + 1e-9 * p.offset_time) for p in m.points])
        elif stamp(m) in want:
            img[(c.topic.split('/')[2], stamp(m))] = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
odo = np.array(odo)

K_l, D_l = intrinsics('dalsa_rgb0.yaml')
K_r, D_r = intrinsics('dalsa_rgb1.yaml')
fs = cv2.FileStorage(CAL + 'extrinsics/calib_chain.yaml', cv2.FILE_STORAGE_READ)
# T_01 maps rgb1 (right) points into rgb0 (left); T_cl maps Livox points into rgb0
T_01 = fs.getNode('T_rgb0_rgb1').mat()
T_cl = fs.getNode('T_rgb0_livox').mat()
R_10 = T_01[:3, :3].T
t_10 = -R_10 @ T_01[:3, 3:]
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K_l, D_l, K_r, D_r, wh, R_10, t_10, alpha=0)
map_l = cv2.initUndistortRectifyMap(K_l, D_l, R1, P1, wh, cv2.CV_32FC1)
map_r = cv2.initUndistortRectifyMap(K_r, D_r, R2, P2, wh, cv2.CV_32FC1)
f_x, c_x, c_y = P1[0, 0], P1[0, 2], P1[1, 2]
B = -P2[0, 3] / P2[0, 0]
print('rectified f_x %.2f  c_x %.2f  c_y %.2f  B %.4f m  fB %.1f px m' % (f_x, c_x, c_y, B, f_x * B))

sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=n_d, blockSize=w_b, P1=24 * w_b ** 2, P2=96 * w_b ** 2,
                             uniquenessRatio=10, speckleWindowSize=100, speckleRange=2, disp12MaxDiff=-1,
                             mode=cv2.STEREO_SGBM_MODE_HH)
yolo = YOLO('data/yolo/trunk.pt')
v, u = np.mgrid[0:wh[1], 0:wh[0]]
for i, st in zip(FRAMES, want):
    d = OUT / f'{i:04d}'
    d.mkdir(parents=True, exist_ok=True)
    L = cv2.remap(img[('left', st)], *map_l, cv2.INTER_LINEAR)
    R = cv2.remap(img[('right', st)], *map_r, cv2.INTER_LINEAR)
    g_l = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY)
    g_r = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)
    disp = sgbm.compute(g_l, g_r).astype(np.float32) / 16
    disp_r = sgbm.compute(g_r[:, ::-1].copy(), g_l[:, ::-1].copy())[:, ::-1].astype(np.float32) / 16
    # rectified left frame: (u, v) in left matches (u - disp, v) in right
    d_back = disp_r[v, np.clip(np.round(u - disp).astype(int), 0, wh[0] - 1)]
    ok = (disp > 0) & (d_back > 0) & (np.abs(disp - d_back) <= tol_lr)
    disp[~ok] = np.nan

    p = scan[st][np.linalg.norm(scan[st][:, :3], axis=1) > 0.5]
    dt = p[:, 3] - st
    v_x = np.interp(st, odo[:, 0], odo[:, 1])
    w_z = np.interp(st, odo[:, 0], odo[:, 2])
    q = p[:, :3] @ T_cl[:3, :3].T + T_cl[:3, 3]  # (N, 3) raw rgb0 frame, each point at its own time
    # rgb0 frame: odometry speed v_x along Z, yaw rate w_z about -Y, points moved to the image instant
    X = q[:, 0] - w_z * dt * q[:, 2]
    Z = q[:, 2] + w_z * dt * q[:, 0] + v_x * dt
    pts_cam = np.c_[X, q[:, 1], Z] @ R1.T  # (N, 3) rectified left frame at the image stamp
    np.save(d / 'disp.npy', disp)
    np.save(d / 'pts_cam.npy', pts_cam.astype(np.float32))
    np.savez(d / 'calib.npz', f_x=f_x, c_x=c_x, c_y=c_y, B=B, w_b=w_b, R1=R1, T_cl=T_cl, stamp=st, v_x=v_x, w_z=w_z)

    if not (d / 'mask_oracle.png').exists():
        cv2.imwrite(str(d / 'mask_oracle.png'), draw_polygons(L[:, :, ::-1]))
    mask_oracle = cv2.imread(str(d / 'mask_oracle.png'), cv2.IMREAD_UNCHANGED)
    res = yolo.predict(L, imgsz=1024, conf=0.25, retina_masks=True, verbose=False)[0]
    mask_pred = np.zeros(mask_oracle.shape, np.uint8)
    if res.masks is not None:
        for k, m_k in enumerate(res.masks.data.cpu().numpy() > 0.5, 1):
            mask_pred[m_k & (mask_pred == 0)] = k
    print('%d  LR %.1f %%  Livox %d  oracle %d  pred %d'
          % (i, 100 * ok.mean(), len(p), mask_oracle.max(), mask_pred.max()))
    cv2.imwrite(str(d / 'left_rect.png'), L)
    cv2.imwrite(str(d / 'mask_pred.png'), mask_pred)

    chk = L.copy()
    chk[mask_pred > 0] = L[mask_pred > 0] // 2 + np.uint8([0, 110, 0])
    edge = cv2.findContours(np.uint8(mask_oracle > 0), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    cv2.drawContours(chk, edge, -1, (0, 0, 255), 1)
    disp_8 = np.clip(np.nan_to_num(disp) * 3, 0, 255).astype(np.uint8)
    disp_c = cv2.applyColorMap(disp_8, cv2.COLORMAP_TURBO)
    cv2.imwrite(str(d / 'check.png'), np.vstack([chk, disp_c]))
