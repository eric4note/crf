# all parameters shared in this project
import os

objects = [
    "automobile",
    "backhoe",
    "bulldozer",
    "concrete_bucket",
    "concrete_finishing",
    "concrete_mixer",
    "concrete_placing",
    "concrete_pump",
    "crane",
    "formwork",
    "formwork_slab_beam",
    "formwork_staircase",
    "formwork_wall_column",
    "rebar",
    "rebar_slab_beam",
    "rebar_wall_column",
    "scaffolding",
    "scaffolding_formwork_slab",
    "truck_dump",
    "truck_lorry",
    "van",
    "worker"
]

# 12 actions
actions = [
    "a01_standing",
    "a02_bending",
    "a03_squating",
    "a04_transporting",
    "a05_taking",
    "a06_pushing",
    "a07_pulling",
    "a08_leveling",
    "b01_moving",
    "b02_watching",
    "c01_standing",
    "c02_sitting"
]

# 17 activities
activities = [
    "A-Checking",
    "A-Moving",
    "A-Preparing",
    "A-Sitting",
    "A-Standing",
    "A-Taking-Dropping",
    "A-Transporting",
    "C-Compacting",
    "C-Leveling",
    "C-Placing",
    "C-Transporting",
    "F-Machining",
    "F-Placing-Fixing",
    "F-Transporting",
    "R-Machining",
    "R-Placing-Fixing",
    "R-Transporting"
]

param_fdp_root = "/media/xc/Local/Paper08"
param_fdp_segments = os.path.join(param_fdp_root, "segments")
param_fdp_videos = os.path.join(param_fdp_root, "videos")
param_fdp_images = os.path.join(param_fdp_root, "images")
param_fdp_labels = os.path.join(param_fdp_root, "labels")
param_fdp_clips = os.path.join(param_fdp_root, "clips")
param_fdp_results = os.path.join(param_fdp_root, "results")
param_fdp_data = os.path.join(param_fdp_root, "data")

param_padding = 0
param_maximum_wh = 480 + 2 * param_padding
param_minimum_wh = 32 + 2 * param_padding

NA1 = len(actions)      # 12, the number of actions
NA2 = len(activities)   # 17, the number of activities
NA3 = 3                 # the number of group activities

thresh_d=3              # maximum connection numbers
thresh_r=0.25           # minimum spatial relevance for a connection