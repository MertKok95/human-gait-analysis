import json
import math
import matplotlib.pyplot as plt

target_file = '../data/raw_data/pose/PA006/UGS/WoJ_2_AlphaPose.json'
print(f"File being processed: {target_file}")

try:
    with open(target_file, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: {target_file} file not found. Please make sure the file is in the same directory as the code.")
    exit()
# -------------------------

lengths = []
frames = []

for item in data:
    if item.get('joints'):
        joints = item['joints']
        # Check the coordinates of the right hip and right knee.
        if 'r_hip' in joints and 'r_knee' in joints:
            rx1, ry1 = joints['r_hip']['x'], joints['r_hip']['y']
            rx2, ry2 = joints['r_knee']['x'], joints['r_knee']['y']
            
            if rx1 != 0 and ry1 != 0 and rx2 != 0 and ry2 != 0:
                dist = math.hypot(rx1 - rx2, ry1 - ry2)
                lengths.append((item['frame'], dist, item['joints']))

# Sort by length
lengths.sort(key=lambda x: x[1])

# Select the shortest and longest 5 frames to avoid anomalies
shortest = lengths[5]
longest = lengths[-5]

print(f"Shortest (Shortened): Square {shortest[0]}, Length {shortest[1]:.1f}")
print(f"Longest (Normal): Square {longest[0]}, Length {longest[1]:.1f}")


# --- PLOT SKELETONS ---
BONES = [
    ("l_shoulder", "r_shoulder", "gray"), ("l_shoulder", "l_hip", "gray"),
    ("r_shoulder", "r_hip", "gray"), ("l_hip", "r_hip", "gray"),
    ("l_shoulder", "l_elbow", "gray"), ("l_elbow", "l_wrist", "gray"),
    ("r_shoulder", "r_elbow", "gray"), ("r_elbow", "r_wrist", "gray"),
    ("l_hip", "l_knee", "#005b96"), ("l_knee", "l_ankle", "#005b96"), # Sol bacak mavi
    ("r_hip", "r_knee", "#d64161"), ("r_knee", "r_ankle", "#d64161")  # Sağ bacak kırmızı
]

fig, axes = plt.subplots(1, 2, figsize=(10, 6))

targets = [
    (longest, "Normal (Parallel) Limb\nClose to True Anatomical Size"), 
    (shortest, "Shortened (Angled) Limb\nPerspective Foreshortening")
]

for ax, (frame_info, title) in zip(axes, targets):
    frame_idx, length, joints = frame_info
    for start, end, color in BONES:
        if start in joints and end in joints:
            x1, y1 = joints[start]['x'], joints[start]['y']
            x2, y2 = joints[end]['x'], joints[end]['y']
            if x1 != 0 and y1 != 0 and x2 != 0 and y2 != 0:
                lw = 5 if start == 'r_hip' and end == 'r_knee' else 2.5
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, solid_capstyle='round')
    
    for j_name, coords in joints.items():
        if coords['x'] != 0 and coords['y'] != 0:
            ax.scatter(coords['x'], coords['y'], color='black', s=20, zorder=5)
            
    ax.set_title(f"{title}\nFrame: {frame_idx} | Right Thigh: {length:.1f} pixels", fontweight='bold')
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.axis('off')

plt.suptitle("Thesis Figure: Evidence of Perspective Foreshortening in 2D Cameras", fontsize=14, fontweight='bold', y=0.95)
plt.tight_layout()
plt.savefig('../data/outputs/figures/skeleton/tez_iskelet_kiyaslamasi.png', dpi=200, bbox_inches='tight')
print("Graph '/data/outputs/figures/skeleton/tez_iskelet_kiyaslamasi.png' saved.")