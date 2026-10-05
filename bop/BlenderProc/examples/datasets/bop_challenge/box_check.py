import blenderproc as bproc
import pyrender
from pyrender.platforms import egl
import argparse
import os
import numpy as np
import glob
import re
import bpy

os.environ["PYOPENGL_PLATFORM"] = "egl"
pyrender.egl = None

parser = argparse.ArgumentParser()
parser.add_argument("bop_parent_path")
parser.add_argument("dataset_name")
parser.add_argument("cc_textures_path")
parser.add_argument("output_dir")
parser.add_argument("--num_scenes", type=int, default=1)
parser.add_argument("--sample_size", type=int, default=100)
parser.add_argument("--max_samples", type=int, default=128)
args = parser.parse_args()

FIXED_TARGET_DATASET = "hb"

ENABLE_PHYSICS = True
DROP_HEIGHT = 0.3
TRUE_Z_OFFSET = 0.02
TRUE_CASE_PROBABILITY = 0.2

FALSE_ROT_X_MIN = -15.0
FALSE_ROT_X_MAX = 15.0
FALSE_ROT_Y_MIN = -15.0
FALSE_ROT_Y_MAX = 15.0
FALSE_ROT_Z_MIN = -20.0
FALSE_ROT_Z_MAX = 20.0

CAMERA_DISTANCE_MIN = 0.5
CAMERA_DISTANCE_MAX = 0.8
CAMERA_AZIMUTH_MIN = -180.0
CAMERA_AZIMUTH_MAX = 180.0
CAMERA_ELEVATION_MIN = 25.0
CAMERA_ELEVATION_MAX = 45.0
CAMERA_INPLANE_MIN = -5.0
CAMERA_INPLANE_MAX = 5.0
CAMERA_LOOK_X_MIN = -0.05
CAMERA_LOOK_X_MAX = 0.05
CAMERA_LOOK_Y_MIN = -0.04
CAMERA_LOOK_Y_MAX = 0.04
CAMERA_LOOK_Z_OFFSET = 0.05
CAMERA_AZIMUTH_JITTER = 1.5

TARGET_X_MIN = -0.12
TARGET_X_MAX = 0.12
TARGET_Y_MIN = -0.10
TARGET_Y_MAX = 0.10

LIGHT_X_MIN = -1.5
LIGHT_X_MAX = 1.5
LIGHT_Y_MIN = -1.5
LIGHT_Y_MAX = 1.5
LIGHT_Z_MIN = 2.0
LIGHT_Z_MAX = 4.0
LIGHT_ENERGY_MIN = 100.0
LIGHT_ENERGY_MAX = 500.0

PHYSICS_SUBSTEPS = int(os.environ.get("PHYSICS_SUBSTEPS", "20"))
PHYSICS_SOLVER_ITERS = int(os.environ.get("PHYSICS_SOLVER_ITERS", "25"))

CPU_THREADS = os.environ.get("BLENDER_THREADS")
if CPU_THREADS:
    try:
        CPU_THREADS = max(1, int(CPU_THREADS))
    except ValueError:
        CPU_THREADS = None

print(f"[CPU] threads={CPU_THREADS if CPU_THREADS else 'AUTO'}, physics={PHYSICS_SUBSTEPS}/{PHYSICS_SOLVER_ITERS}")

cc_textures = []
room_planes = []
light_point = None
box = None
box_center = None
box_top = None
box_bottom = None
box_z_min = None


def resolve_dataset_path(base_path, dataset_name):
    path = os.path.join(base_path, dataset_name)
    if os.path.isdir(path):
        return path
    raise FileNotFoundError(f"Dataset not found: {path}")


def sample_scene_rotation():
    return np.array([
        np.random.choice([0.0, np.pi]),
        0.0,
        np.random.choice([0.0, np.pi])
    ], dtype=np.float32)


def randomize_background():
    if not cc_textures:
        return
    material = np.random.choice(cc_textures)
    for plane in room_planes:
        plane.replace_materials(material)


def randomize_light():
    light_point.set_location([
        np.random.uniform(LIGHT_X_MIN, LIGHT_X_MAX),
        np.random.uniform(LIGHT_Y_MIN, LIGHT_Y_MAX),
        np.random.uniform(LIGHT_Z_MIN, LIGHT_Z_MAX)
    ])
    light_point.set_energy(np.random.uniform(LIGHT_ENERGY_MIN, LIGHT_ENERGY_MAX))


def prepare_target(obj, obj_id):
    obj.set_cp("category_id", obj_id)
    obj.set_cp("bop_dataset_name", FIXED_TARGET_DATASET)
    obj.set_cp("object_type", "target")
    obj.set_cp("original_bbox", np.array(obj.get_bound_box()).tolist())
    obj.set_shading_mode("auto")

    mats = obj.get_materials()
    if not mats:
        mat = bproc.material.create("auto_mat")
        obj.replace_materials(mat)
    else:
        mat = mats[0]

    mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
    mat.set_principled_shader_value("Specular IOR Level", np.random.uniform(0, 1.0))


def sample_false_pose(obj, scene_rotation):
    rotation = scene_rotation.copy()
    rotation += np.deg2rad([
        np.random.uniform(FALSE_ROT_X_MIN, FALSE_ROT_X_MAX),
        np.random.uniform(FALSE_ROT_Y_MIN, FALSE_ROT_Y_MAX),
        np.random.uniform(FALSE_ROT_Z_MIN, FALSE_ROT_Z_MAX)
    ])

    bbox = np.array(obj.get_bound_box())
    target_z_min = bbox[:, 2].min()

    obj.set_location([
        box_center[0] + np.random.uniform(TARGET_X_MIN, TARGET_X_MAX),
        box_center[1] + np.random.uniform(TARGET_Y_MIN, TARGET_Y_MAX),
        box_top + DROP_HEIGHT - target_z_min
    ])
    obj.set_rotation_euler(rotation)


def restore_initial_pose(obj, transform, scene_rotation):
    location = transform["location"].copy()
    location[2] += TRUE_Z_OFFSET
    obj.set_location(location)
    obj.set_rotation_euler(transform["rotation"] + scene_rotation)


def add_camera_poses(bvh):
    azimuths = np.linspace(CAMERA_AZIMUTH_MIN, CAMERA_AZIMUTH_MAX, args.sample_size, endpoint=False)
    np.random.shuffle(azimuths)
    cam_poses = 0

    for azimuth_deg in azimuths:
        azimuth_deg += np.random.uniform(-CAMERA_AZIMUTH_JITTER, CAMERA_AZIMUTH_JITTER)
        distance = np.random.uniform(CAMERA_DISTANCE_MIN, CAMERA_DISTANCE_MAX)
        elevation = np.deg2rad(np.random.uniform(CAMERA_ELEVATION_MIN, CAMERA_ELEVATION_MAX))
        azimuth = np.deg2rad(azimuth_deg)
        horizontal_distance = distance * np.cos(elevation)

        location = box_center + np.array([
            horizontal_distance * np.sin(azimuth),
            -horizontal_distance * np.cos(azimuth),
            distance * np.sin(elevation)
        ])

        poi = box_center.copy()
        poi += np.array([
            np.random.uniform(CAMERA_LOOK_X_MIN, CAMERA_LOOK_X_MAX),
            np.random.uniform(CAMERA_LOOK_Y_MIN, CAMERA_LOOK_Y_MAX),
            CAMERA_LOOK_Z_OFFSET
        ])

        rot = bproc.camera.rotation_from_forward_vec(
            poi - location,
            inplane_rot=np.deg2rad(np.random.uniform(CAMERA_INPLANE_MIN, CAMERA_INPLANE_MAX))
        )

        cam2world = bproc.math.build_transformation_mat(location, rot)

        if bproc.camera.perform_obstacle_in_view_check(cam2world, {"min": 0.3}, bvh):
            bproc.camera.add_camera_pose(cam2world)
            cam_poses += 1

    return cam_poses


bproc.init()
bproc.renderer.set_render_devices(use_only_cpu=False)

if CPU_THREADS is not None:
    scene = bpy.context.scene
    scene.render.threads_mode = "FIXED"
    scene.render.threads = CPU_THREADS
    print(f"[CPU] Blender render threads: {CPU_THREADS}")

target_dataset_path = resolve_dataset_path(args.bop_parent_path, FIXED_TARGET_DATASET)
models_dir = os.path.join(target_dataset_path, "models")

blend_files = glob.glob(os.path.join(models_dir, "obj_*.blend"))
blend_map = {}

for path in blend_files:
    match = re.search(r"obj_(\d+)\.blend", os.path.basename(path))
    if match:
        blend_map[int(match.group(1))] = path

print(f"[INFO] Found blend objects: {sorted(blend_map.keys())}")

if 2 not in blend_map:
    raise FileNotFoundError(f"obj_000002.blend not found in {models_dir}")

candidate_ids = sorted(obj_id for obj_id in blend_map if obj_id != 2)

if not candidate_ids:
    raise RuntimeError("No target objects found besides obj_000002.")

print(f"[INFO] Candidate target IDs: {candidate_ids}")

box_objs = bproc.loader.load_blend(blend_map[2], obj_types="mesh")

if not box_objs:
    raise RuntimeError(f"No mesh in {blend_map[2]}")

box = box_objs[0]
box.set_cp("category_id", 2)
box.set_cp("bop_dataset_name", FIXED_TARGET_DATASET)
box.set_cp("object_type", "box")
box.set_shading_mode("auto")

box_bbox = np.array(box.get_bound_box())
box_z_min = box_bbox[:, 2].min()

box.set_location([0, 0, -box_z_min])
box.set_rotation_euler([0, 0, 0])

box.enable_rigidbody(False, collision_shape="MESH", friction=100)

box_bbox = np.array(box.get_bound_box())
box_center = box_bbox.mean(axis=0)
box_top = box_bbox[:, 2].max()
box_bottom = box_bbox[:, 2].min()

print("Box center:", box_center)
print("Box top:", box_top)
print("Box bottom:", box_bottom)

width = 1440
height = 1080

K = np.array([
    [1070.7898, 0.0, 722.30939],
    [0.0, 1070.5270, 550.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

bproc.camera.set_intrinsics_from_K_matrix(K, width, height)

box.hide(False)

room_planes = [
    bproc.object.create_primitive("PLANE", scale=[2, 2, 1]),
    bproc.object.create_primitive("PLANE", scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.5708, 0, 0]),
    bproc.object.create_primitive("PLANE", scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.5708, 0, 0]),
    bproc.object.create_primitive("PLANE", scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.5708, 0]),
    bproc.object.create_primitive("PLANE", scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.5708, 0])
]

for plane in room_planes:
    plane.enable_rigidbody(
        False,
        collision_shape="BOX",
        mass=1.0,
        friction=100.0,
        linear_damping=0.99,
        angular_damping=0.99
    )

light_point = bproc.types.Light()
light_point.set_type("POINT")
light_point.set_energy(200)

cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_path)

print(f"[INFO] Loaded {len(cc_textures)} CC materials")

if not cc_textures:
    print("[WARNING] No CC materials found.")

bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(args.max_samples)

for i in range(args.num_scenes):
    scene = bpy.context.scene
    scene.frame_end = 0

    print(f"\n================ Scene {i + 1}/{args.num_scenes} ================\n")

    cam_ob = scene.camera

    if cam_ob is not None and cam_ob.animation_data is not None:
        cam_ob.animation_data_clear()

    scene_rotation = sample_scene_rotation()

    print(
        f"[SCENE {i}] Base rotation "
        f"X={np.rad2deg(scene_rotation[0]):.0f}°, "
        f"Z={np.rad2deg(scene_rotation[2]):.0f}°"
    )

    is_scene_b = np.random.random() < TRUE_CASE_PROBABILITY

    print(
        f"[SCENE {i}] "
        f"{'B - Initial position' if is_scene_b else 'A - Random placement'}"
    )

    randomize_background()
    randomize_light()

    selected_id = int(np.random.choice(candidate_ids))
    selected_blend = blend_map[selected_id]

    print(f"[TARGET] selected object: obj_{selected_id:06d}.blend")

    target_objs = bproc.loader.load_blend(selected_blend, obj_types="mesh")

    if not target_objs:
        raise RuntimeError(f"No mesh in {selected_blend}")

    target_bop_objs = []

    for obj in target_objs:
        prepare_target(obj, selected_id)
        obj.hide(False)
        target_bop_objs.append(obj)

    target_initial_transforms = [
        {
            "location": np.array(obj.get_location()).copy(),
            "rotation": np.array(obj.get_rotation_euler()).copy()
        }
        for obj in target_bop_objs
    ]

    box.hide(False)
    box.disable_rigidbody()
    box.enable_rigidbody(False, collision_shape="MESH", friction=100)

    box.set_location([0, 0, -box_z_min])
    box.set_rotation_euler([0, 0, 0])

    if is_scene_b:
        for obj, transform in zip(target_bop_objs, target_initial_transforms):
            obj.disable_rigidbody()
            restore_initial_pose(obj, transform, scene_rotation)

        print("[SCENE] Target collision: DISABLED")
    else:
        if ENABLE_PHYSICS:
            for obj in target_bop_objs:
                obj.enable_rigidbody(
                    True,
                    mass=1.0,
                    friction=100.0,
                    linear_damping=0.99,
                    angular_damping=0.99
                )

        bproc.object.sample_poses(
            objects_to_sample=target_bop_objs,
            sample_pose_func=lambda obj: sample_false_pose(obj, scene_rotation),
            max_tries=1000
        )

        print("[SCENE] Target collision: ENABLED")

    if ENABLE_PHYSICS and not is_scene_b:
        bproc.object.simulate_physics_and_fix_final_poses(
            min_simulation_time=3,
            max_simulation_time=10,
            check_object_interval=1,
            substeps_per_frame=PHYSICS_SUBSTEPS,
            solver_iters=PHYSICS_SOLVER_ITERS
        )

    all_objs = target_bop_objs + [box]

    bvh = bproc.object.create_bvh_tree_multi_objects(all_objs)
    cam_poses = add_camera_poses(bvh)

    print(
        f"[SCENE {i}] Rendering {cam_poses} camera poses "
        f"(requested {args.sample_size}, 360° horizontal coverage)"
    )

    if cam_poses == 0:
        raise RuntimeError(f"[SCENE {i}] No valid camera poses found.")

    data = bproc.renderer.render()

    print(f"[SCENE {i}] BOP objects:")

    for obj in all_objs:
        print(
            "    ",
            obj.get_name(),
            "category_id =",
            obj.get_cp("category_id"),
            "object_type =",
            obj.get_cp("object_type")
        )

    bproc.writer.write_bop(
        os.path.join(args.output_dir, "bop_data"),
        target_objects=all_objs,
        dataset=FIXED_TARGET_DATASET,
        depth_scale=0.1,
        depths=data["depth"],
        colors=data["colors"],
        color_file_format="JPEG",
        ignore_dist_thres=10
    )

    for obj in target_bop_objs:
        obj.hide(True)
        obj.disable_rigidbody()

    box.hide(True)
    box.disable_rigidbody()

    print(f"[SCENE {i}] Done.")

print("Done")