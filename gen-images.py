import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PARAMÈTRES
# ============================================================

N_SAMPLES = 20000
RECTANGLE_PCT = 0.3

RANDOM_SIZE = True
RANDOM_POSITION = True
RANDOM_ROTATION = True
RANDOM_COLOR = True

ADD_RANDOM_LINES = True
ADD_PIXEL_NOISE = True
ADD_BACKGROUND_CIRCLES = True

IMAGE_SIZE = 256
OUTPUT_DIR = "triangle_images"

SEED = 42
np.random.seed(SEED)

# ============================================================
# DOSSIERS
# ============================================================

os.makedirs(f"{OUTPUT_DIR}/1", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/0", exist_ok=True)

# ============================================================
# FONCTIONS
# ============================================================

def generate_right_triangle():
    width = np.random.uniform(40, 140)
    height = np.random.uniform(40, 140)

    return np.array([
        [0, 0],
        [width, 0],
        [0, height]
    ], dtype=float)


def generate_non_right_triangle():
    while True:
        points = np.random.uniform(0, 140, size=(3, 2))

        a = np.linalg.norm(points[0] - points[1])
        b = np.linalg.norm(points[1] - points[2])
        c = np.linalg.norm(points[2] - points[0])

        sides = sorted([a, b, c])

        if sides[0] + sides[1] > sides[2]:
            is_right = np.isclose(
                sides[0]**2 + sides[1]**2,
                sides[2]**2,
                atol=1e-2
            )

            if not is_right:
                return points


def rotate_points(points, angle_deg):
    angle_rad = np.deg2rad(angle_deg)

    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])

    center = points.mean(axis=0)
    return (points - center) @ rotation_matrix.T + center


def normalize_triangle(points):
    points = points - points.min(axis=0)

    max_value = points.max()

    target_size = np.random.uniform(80, 200) if RANDOM_SIZE else 160

    points = points / max_value * target_size

    if RANDOM_ROTATION:
        angle = np.random.uniform(0, 360)
        points = rotate_points(points, angle)
        points = points - points.min(axis=0)

    if RANDOM_POSITION:
        max_x = IMAGE_SIZE - points[:, 0].max() - 10
        max_y = IMAGE_SIZE - points[:, 1].max() - 10

        offset_x = np.random.uniform(10, max_x)
        offset_y = np.random.uniform(10, max_y)
    else:
        offset_x = (IMAGE_SIZE - points[:, 0].max()) / 2
        offset_y = (IMAGE_SIZE - points[:, 1].max()) / 2

    points[:, 0] += offset_x
    points[:, 1] += offset_y

    return points


def generate_random_color():
    if RANDOM_COLOR:
        return np.random.rand(3)

    return (0, 0, 0)


def add_background_circles(ax):
    if not ADD_BACKGROUND_CIRCLES:
        return

    n_circles = np.random.randint(3, 12)

    for _ in range(n_circles):
        x = np.random.uniform(0, IMAGE_SIZE)
        y = np.random.uniform(0, IMAGE_SIZE)
        radius = np.random.uniform(5, 35)
        color = np.random.rand(3)
        alpha = np.random.uniform(0.15, 0.45)

        circle = plt.Circle(
            (x, y),
            radius,
            color=color,
            alpha=alpha,
            fill=True
        )

        ax.add_patch(circle)


def add_random_lines(ax):
    if not ADD_RANDOM_LINES:
        return

    n_lines = np.random.randint(2, 10)

    for _ in range(n_lines):
        x1, y1 = np.random.uniform(0, IMAGE_SIZE, size=2)
        x2, y2 = np.random.uniform(0, IMAGE_SIZE, size=2)

        color = np.random.rand(3)
        linewidth = np.random.uniform(0.5, 2.0)
        alpha = np.random.uniform(0.15, 0.5)

        ax.plot(
            [x1, x2],
            [y1, y2],
            color=color,
            linewidth=linewidth,
            alpha=alpha
        )


def save_triangle_image(points, path, color):
    fig, ax = plt.subplots(figsize=(3, 3), dpi=IMAGE_SIZE // 3)

    add_background_circles(ax)
    add_random_lines(ax)

    triangle = plt.Polygon(
        points,
        closed=True,
        fill=False,
        edgecolor=color,
        linewidth=3
    )

    ax.add_patch(triangle)

    ax.set_xlim(0, IMAGE_SIZE)
    ax.set_ylim(0, IMAGE_SIZE)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ============================================================
# GÉNÉRATION
# ============================================================

n_right = int(N_SAMPLES * RECTANGLE_PCT)
n_non_right = N_SAMPLES - n_right

labels = [1] * n_right + [0] * n_non_right
np.random.shuffle(labels)

for i, label in enumerate(labels):

    if label == 1:
        points = generate_right_triangle()
        folder = "1"
    else:
        points = generate_non_right_triangle()
        folder = "0"

    points = normalize_triangle(points)
    color = generate_random_color()

    output_path = f"{OUTPUT_DIR}/{folder}/triangle_{i:05d}.png"

    save_triangle_image(
        points=points,
        path=output_path,
        color=color
    )

print("Images générées.")
print(f"Total : {N_SAMPLES}")
print(f"Triangles rectangles : {n_right}")
print(f"Triangles non rectangles : {n_non_right}")
print(f"Dossier : {OUTPUT_DIR}")