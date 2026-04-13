import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def build_final_trajectory(num_images):
    orb = cv2.ORB_create(nfeatures=2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    trajectory = [[0, 0]]
    pos = np.array([[0.0], [0.0]])
    total_rotation = np.eye(2)

    prev_kp, prev_des = None, None

    print("Начинаю финальную сборку траектории...")

    for i in range(1, num_images + 1):
        filename = f'image_{i}.jpg'
        if not os.path.exists(filename):
            print(f"Пропуск: {filename} не найден")
            continue

        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (0, 0), fx=0.7, fy=0.7)

        kp, des = orb.detectAndCompute(img, None)

        if prev_des is not None and des is not None:
            matches = bf.match(prev_des, des)
            if len(matches) > 15:
                src_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                matrix, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)

                if matrix is not None:
                    R = matrix[:, :2]
                    T = matrix[:, 2:]

                    total_rotation = total_rotation @ np.linalg.inv(R)

                    movement = total_rotation @ T
                    pos -= movement

                    trajectory.append([pos[0, 0], pos[1, 0]])
                    print(f"Кадр {i}: Маневр зафиксирован")

        prev_kp, prev_des = kp, des

    return np.array(trajectory)


num_files = 21
path = build_final_trajectory(num_files)

plt.figure(figsize=(10, 10))
plt.plot(path[:, 0], path[:, 1], color='blue', linewidth=2, marker='o', label='Путь БПЛА')

plt.scatter(path[0, 0], path[0, 1], color='green', s=150, label='СТАРТ (image_1)', zorder=5)
if len(path) >= 5:
    plt.scatter(path[4, 0], path[4, 1], color='orange', s=150, label='Разворот 180° (image_5)', zorder=5)
plt.scatter(path[-1, 0], path[-1, 1], color='red', s=150, label='ФИНИШ (image_10)', zorder=5)

for i, pt in enumerate(path):
    plt.annotate(f' {i + 1}', (pt[0], pt[1]), fontsize=10, weight='bold')

plt.title("Итоговая траектория полета")
plt.xlabel("Относительное смещение X")
plt.ylabel("Относительное смещение Y")
plt.axis('equal')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()
