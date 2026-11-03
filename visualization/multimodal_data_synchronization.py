import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

file_path = '../data/raw_data/pose/PA007/FGS/WoJ_1_AlphaPose.json'

# 1. Extracting Skeleton Data from JSON for Frame 50 (FGS - PA007) (for an attendee)
with open(file_path, 'r') as file:
    pose_data = json.load(file)


# We find the 50th frame (because the person wasn't in the frame in the first few frames).
frame_50 = next(item for item in pose_data if item["frame"] == 50)
joints = frame_50["joints"]

# --- Virtual Neck Calculation ---
# We take the average of the X and Y coordinates of the left and right shoulders.
if 'l_shoulder' in joints and 'r_shoulder' in joints:
    joints['neck'] = {
        'x': (joints['l_shoulder']['x'] + joints['r_shoulder']['x']) / 2,
        'y': (joints['l_shoulder']['y'] + joints['r_shoulder']['y']) / 2
    }

# Joint connections that conform to human anatomy (lines connecting the points)
skeleton_edges = [
    # Head
    ('nose', 'l_eye'), ('l_eye', 'l_ear'), ('nose', 'r_eye'), ('r_eye', 'r_ear'),
    
    # Neck and Shoulders (calculated)
    ('nose', 'neck'), ('neck', 'l_shoulder'), ('neck', 'r_shoulder'),
    
    # Arms
    ('l_shoulder', 'l_elbow'), ('l_elbow', 'l_wrist'),
    ('r_shoulder', 'r_elbow'), ('r_elbow', 'r_wrist'), 
    
    # Body
    ('l_shoulder', 'l_hip'), ('r_shoulder', 'r_hip'), ('l_hip', 'r_hip'), 
    
    # Legs
    ('l_hip', 'l_knee'), ('l_knee', 'l_ankle'), 
    ('r_hip', 'r_knee'), ('r_knee', 'r_ankle')
]


# 2. Creating a 2x2 Grid
fig, axs = plt.subplots(2, 2, figsize=(12, 12))
fig.suptitle('Multimodal Gait Data Synchronization - Frame 50 (FGS - PA007)', fontsize=16, fontweight='bold')

# --- 1. Quarter: AlphaPose Skeleton (read from JSON) ---

for start, end in skeleton_edges:
    if start in joints and end in joints:
        axs[0, 0].plot([joints[start]['x'], joints[end]['x']], 
                       [joints[start]['y'], joints[end]['y']], 'b-', linewidth=2)
        

# Draw the joints.
for joint_name, coords in joints.items():
    axs[0, 0].plot(coords['x'], coords['y'], 'ro', markersize=5)


axs[0, 0].set_title('1. Pose Estimation (AlphaPose)')
axs[0, 0].set_xlim(0, 960)   # Make the x-axis the same as in images 2 and 3.
axs[0, 0].set_ylim(540, 0)   # Make the Y-axis the same and reverse it (0 up)

# --- 2. Quarter: YOLOV8 Silhouette (Adjust the example image path according to its own folder) ---
img_silhouette = mpimg.imread('../data/raw_data/silhouette/PA007/FGS/WoJ_1_YOLOV8/050.jpg')
axs[0, 1].imshow(img_silhouette, cmap='gray')
axs[0, 1].set_title('2. Silhouette (YOLOV8)')

# --- 3rd Quarter: DensePose Semantic Segmentation ---
img_semantic = mpimg.imread('../data/raw_data/semantic_segmentation/PA007/FGS/WoJ_1_DensePose/050.png')
axs[1, 0].imshow(img_semantic)
axs[1, 0].set_title('3. Semantic Segmentation (DensePose)')

# --- 4th Quarter: TVL1/GMFlow Optical Flow ---
img_flow = mpimg.imread('../data/raw_data/optical_flow/PA007/FGS/GMFLOW/WoJ_1/050.png')
axs[1, 1].imshow(img_flow)
axs[1, 1].set_title('4. Optical Flow (TVL1 / GMFlow)')

# Save and display the image.
plt.tight_layout(pad=3.0)
plt.savefig('../data/outputs/figures/diagrams/multimodal_sync_frame50.png', dpi=300)
print("saved location: ../data/outputs/figures/diagrams/multimodal_sync_frame50.png")
# plt.show()