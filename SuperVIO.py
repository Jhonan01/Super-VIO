import os
import cv2
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from models.matching import Matching
from models.utils import frame2tensor
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.ion()  # Modo interativo
fig, ax = plt.subplots()
ax.set_title("Trajectory estimated vs Ground Truth")
ax.set_xlabel("East (m)")
ax.set_ylabel("North (m)")
ax.grid(True)

info_text = ax.text(
    0.02, 0.83, "",  # abaixa um pouco para dar mais espaço vertical
    transform=ax.transAxes, 
    verticalalignment='top', 
    bbox=dict(facecolor='white', alpha=0.7),
    fontsize=9
)

# Listas para armazenar pontos para plot
est_x, est_y = [0], [0]
gt_x, gt_y = [0], [0]

# Inicializa os plots
est_plot, = ax.plot(est_x, est_y, 'g-', label='VIO Estimated')
gt_plot, = ax.plot(gt_x, gt_y, 'r-', label='Ground Truth')
ax.legend()

# Variáveis para acumular posição estimada incrementalmente
est_pos_x, est_pos_y = 0.0, 0.0  # east, north

# Configuração do SuperGlue + SuperPoint
config = {
    'superpoint': {'nms_radius': 4, 'keypoint_threshold': 0.005, 'max_keypoints': 1024},
    'superglue': {
        'weights': 'outdoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
    }
}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
matching = Matching(config).eval().to(device)

image_dir = Path("/media/kj/NOVO VOLUME/nadir_images_03")
image_paths = sorted([str(p) for p in image_dir.glob("*.png")], key=lambda x: int(Path(x).stem))

csv_path = "/media/kj/NOVO VOLUME/fusion_data_03.csv"
df = pd.read_csv(csv_path)

gt_positions_abs = list(zip(
    df['real_north_position_meters'].astype(float),
    df['real_east_position_meters'].astype(float)
))

if len(gt_positions_abs) < len(image_paths):
    raise ValueError("Número de posições no CSV é menor que o número de imagens.")

K = np.array([[1109, 0, 960],
              [0, 1109, 540],
              [0, 0, 1]], dtype=np.float32)

pose = np.eye(4)

def process_pair(img0, img1):
    image0 = cv2.imread(img0, cv2.IMREAD_GRAYSCALE)
    image1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
    if image0 is None or image1 is None:
        raise ValueError(f"Erro ao carregar imagens {img0}, {img1}")

    inp0 = frame2tensor(image0, device)
    inp1 = frame2tensor(image1, device)
    pred = matching({'image0': inp0, 'image1': inp1})

    kpts0 = pred['keypoints0'][0].cpu().numpy()
    kpts1 = pred['keypoints1'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    matches_idx = np.where(valid)[0]

    if len(mkpts0) < 8:
        raise ValueError("Poucos matches para estimar homografia")

    H, mask_h = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC)
    if mask_h is None or np.count_nonzero(mask_h) < 4:
        raise ValueError("Menos de 4 inliers na homografia")

    inliers = mask_h.ravel().astype(bool)
    if not np.any(inliers):
        raise ValueError("Nenhum inlier na homografia")

    # Só os matches inliers pra desenhar
    inlier_matches_idx = matches_idx[inliers]

    # Converte keypoints para cv2.KeyPoint (necessário para drawMatches)
    keypoints0_cv = [cv2.KeyPoint(x=float(x), y=float(y), size=1) for x, y in kpts0]
    keypoints1_cv = [cv2.KeyPoint(x=float(x), y=float(y), size=1) for x, y in kpts1]

    # Construir lista de DMatch para os inliers
    cv_matches = [cv2.DMatch(_queryIdx=idx, _trainIdx=matches[idx], _distance=0) for idx in inlier_matches_idx]

    # Desenha correspondências inliers
    matched_img = cv2.drawMatches(image0, keypoints0_cv, image1, keypoints1_cv, cv_matches, None,
                                  matchColor=(0,255,0), singlePointColor=(255,0,0), flags=2)

    # Restante do cálculo PnP
    obj_pts = cv2.undistortPoints(np.expand_dims(mkpts0[inliers], 1), K, None)
    obj_pts = np.squeeze(obj_pts)
    obj_pts = np.hstack([obj_pts, np.zeros((obj_pts.shape[0], 1))])

    retval, rvec, tvec, inliers_pnp = cv2.solvePnPRansac(
        objectPoints=obj_pts.astype(np.float32),
        imagePoints=mkpts1[inliers].astype(np.float32),
        cameraMatrix=K,
        distCoeffs=None,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not retval:
        raise ValueError("solvePnPRansac falhou")

    R, _ = cv2.Rodrigues(rvec)
    return R, tvec, matched_img

for i in range(len(image_paths) - 1):
    img0 = image_paths[i]
    img1 = image_paths[i + 1]

    try:
        R, t, matched_img = process_pair(img0, img1)

        imu_north_delta = df.loc[i + 1, 'sensor_north_position_delta_meters']
        imu_east_delta = df.loc[i + 1, 'sensor_east_position_delta_meters']
        imu_distance = np.sqrt(imu_north_delta**2 + imu_east_delta**2)

        estimated_distance = np.linalg.norm([t[0], t[1]])
        scale_factor = imu_distance / estimated_distance if estimated_distance > 0 else 1.0
        t_scaled = t * scale_factor

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t_scaled.squeeze()
        pose = pose @ np.linalg.inv(T)

        delta_north = t_scaled[1][0]
        delta_east = t_scaled[0][0]

        # Atualiza posição estimada incrementalmente
        est_pos_x += delta_east
        est_pos_y += delta_north

        est_x.append(est_pos_x*(-1))
        est_y.append(est_pos_y)

        # Ground Truth acumulado (posição absoluta)
        north1, east1 = gt_positions_abs[i]
        north2, east2 = gt_positions_abs[i + 1]
        gt_dx = east2 - east1
        gt_dy = north2 - north1
        gt_x.append(east2 - gt_positions_abs[0][1])
        gt_y.append(north2 - gt_positions_abs[0][0])

        est_plot.set_data(est_x, est_y)
        gt_plot.set_data(gt_x, gt_y)

        margin = 5
        min_x = min(est_x + gt_x) - margin
        max_x = max(est_x + gt_x) + margin
        min_y = min(est_y + gt_y) - margin
        max_y = max(est_y + gt_y) + margin
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

        info_text.set_text(
            f"Estimated ΔNorth: {delta_north:.2f} m\n"
            f"Estimated ΔEast: {(-1)*delta_east:.2f} m\n"
            f"Ground Truth ΔNorth: {gt_dy:.2f} m\n"
            f"Ground Truth ΔEast: {gt_dx:.2f} m"
        )

        plt.draw()
        plt.pause(0.001)

        cv2.imshow("matches", cv2.resize(matched_img, (1000, 480), interpolation=cv2.INTER_AREA))
        cv2.waitKey(1)

        print(f"frame {i} - {i+1}: "
              f"Estimated (scaled IMU) -> ΔNorth = {delta_north:.2f} m, ΔEast = {(-1)*delta_east:.2f} m | "
              f"Ground Truth -> ΔNorth = {gt_dy:.2f} m, ΔEast = {gt_dx:.2f} m")

    except Exception as e:
        print(f"Erro ao processar par {img0} - {img1}: {e}")

plt.ioff()
plt.show()