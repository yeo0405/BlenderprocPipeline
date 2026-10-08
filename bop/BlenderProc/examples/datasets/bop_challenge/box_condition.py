import blenderproc as bproc
import argparse
import glob
import os
import re

os.environ["PYOPENGL_PLATFORM"] = "egl"

import bpy
import numpy as np
import pyrender
from pyrender.platforms import egl

# ==============================================================================
# Adjustable Parameters & Global Configurations
# ==============================================================================

# Dataset & Command Line Defaults
FIXED_TARGET_DATASET = "hb"
DEFAULT_NUM_SCENES = 1
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_MAX_SAMPLES = 128

# Render Resolution & Intrinsics
IMAGE_WIDTH, IMAGE_HEIGHT = 1440, 1080
CAMERA_FX, CAMERA_FY = 1070.7898, 1070.5270
CAMERA_CX, CAMERA_CY = 722.30939, 550.0

# Depth / BOP Output
BOP_DEPTH_SCALE = 0.1
BOP_IGNORE_DIST_THRESHOLD = 10
BOP_COLOR_FORMAT = "JPEG"
BOP_NUM_WORKERS = 1

# Physics & Rigidbody
ENABLE_PHYSICS = True
DROP_HEIGHT = 0.3
TRUE_Z_OFFSET = 0.02
TRUE_CASE_PROBABILITY = 0.2
PHYSICS_MIN_SIMULATION_TIME = 3
PHYSICS_MAX_SIMULATION_TIME = 10
PHYSICS_CHECK_OBJECT_INTERVAL = 1
PHYSICS_SUBSTEPS = int(os.environ.get("PHYSICS_SUBSTEPS", "20"))
PHYSICS_SOLVER_ITERS = int(os.environ.get("PHYSICS_SOLVER_ITERS", "25"))

RIGIDBODY_MASS = 1.0
RIGIDBODY_FRICTION = 100.0
RIGIDBODY_LINEAR_DAMPING = 0.99
RIGIDBODY_ANGULAR_DAMPING = 0.99

# CPU Threads
CPU_THREADS = os.environ.get("BLENDER_THREADS")
if CPU_THREADS:
    try:
        CPU_THREADS = max(1, int(CPU_THREADS))
    except ValueError:
        CPU_THREADS = None

# Randomization Ranges
FALSE_ROT_X_MIN, FALSE_ROT_X_MAX = -15.0, 15.0
FALSE_ROT_Y_MIN, FALSE_ROT_Y_MAX = -15.0, 15.0
FALSE_ROT_Z_MIN, FALSE_ROT_Z_MAX = -20.0, 20.0

BOX_X_MIN, BOX_X_MAX = -0.5, 0.5
BOX_Y_MIN, BOX_Y_MAX = -0.5, 0.5
BOX_Z_ROT_MIN, BOX_Z_ROT_MAX = -180.0, 180.0

# obj_000003 relative to Box (meters / degrees)
OBJ3_RELATIVE_LOCATION = [0.00000000, 0.06704373, 0.12656543] #0.12656543
OBJ3_RELATIVE_ROTATION_DEG = [0.0, 0.0, 0.0]

CAMERA_DISTANCE_MIN, CAMERA_DISTANCE_MAX = 0.5, 0.8
CAMERA_AZIMUTH_MIN, CAMERA_AZIMUTH_MAX = -180.0, 180.0
CAMERA_AZIMUTH_JITTER = 1.5
CAMERA_ELEVATION_MIN, CAMERA_ELEVATION_MAX = 32.0, 50.0
CAMERA_INPLANE_MIN, CAMERA_INPLANE_MAX = -5.0, 5.0

CAMERA_LOOK_X_MIN, CAMERA_LOOK_X_MAX = -0.20, 0.20
CAMERA_LOOK_Y_MIN, CAMERA_LOOK_Y_MAX = -0.15, 0.15
CAMERA_LOOK_Z_OFFSET = 0.05
CAMERA_OBSTACLE_MIN_DISTANCE = 0.3
CAMERA_MAX_ATTEMPTS_PER_POSE = 100
CAMERA_LOOK_MARGIN = 0.05

TARGET_X_MIN, TARGET_X_MAX = -0.12, 0.12
TARGET_Y_MIN, TARGET_Y_MAX = -0.10, 0.10

DISTRACTOR_OBJECT_COUNT = 3
DISTRACTOR_MARGIN = 0.05
DISTRACTOR_MAX_TRIES = 100
DISTRACTOR_XY_MARGIN = 0.05
DISTRACTOR_Z_MIN, DISTRACTOR_Z_MAX = 0.03, 0.15

LIGHT_X_MIN, LIGHT_X_MAX = -1.5, 1.5
LIGHT_Y_MIN, LIGHT_Y_MAX = -1.5, 1.5
LIGHT_Z_MIN, LIGHT_Z_MAX = 2.0, 4.0
LIGHT_ENERGY_MIN, LIGHT_ENERGY_MAX = 100.0, 500.0
LIGHT_TYPE = "POINT"
LIGHT_DEFAULT_ENERGY = 200.0

MATERIAL_ROUGHNESS_MIN, MATERIAL_ROUGHNESS_MAX = 0.0, 1.0
MATERIAL_SPECULAR_MIN, MATERIAL_SPECULAR_MAX = 0.0, 1.0

# Geometry Defaults
TABLE_SCALE = [2, 2, 1]
ROOM_PLANE_SCALE = [2, 2, 1]
ROOM_BACK_LOCATION, ROOM_BACK_ROTATION = [0, -2, 2], [-1.5708, 0, 0]
ROOM_FRONT_LOCATION, ROOM_FRONT_ROTATION = [0, 2, 2], [1.5708, 0, 0]
ROOM_RIGHT_LOCATION, ROOM_RIGHT_ROTATION = [2, 0, 2], [0, -1.5708, 0]
ROOM_LEFT_LOCATION, ROOM_LEFT_ROTATION = [-2, 0, 2], [0, 1.5708, 0]

SCENE_ROTATION_X_CHOICES = [0.0, np.pi]
SCENE_ROTATION_Z_CHOICES = [0.0, np.pi]

DEPTH_ANTIALIASING = False
RENDER_USE_ONLY_CPU = False


# ==============================================================================
# Helper Functions & Object Setup
# ==============================================================================

def resolve_dataset_path(base_path, dataset_name):
    path = os.path.join(base_path, dataset_name)
    if os.path.isdir(path):
        return path
    raise FileNotFoundError(f"Dataset not found: {path}")


def clear_bop_properties(obj):
    blender_obj = obj.blender_obj
    for key in ("category_id", "bop_dataset_name", "object_type", "original_bbox"):
        if key in blender_obj:
            del blender_obj[key]


def set_bop_properties(obj, category_id, object_type):
    clear_bop_properties(obj)
    blender_obj = obj.blender_obj
    blender_obj["category_id"] = int(category_id)
    blender_obj["bop_dataset_name"] = str(FIXED_TARGET_DATASET)
    blender_obj["object_type"] = str(object_type)
    blender_obj["original_bbox"] = np.asarray(obj.get_bound_box(), dtype=float).tolist()


def set_object_type(obj, object_type):
    blender_obj = obj.blender_obj
    blender_obj.pop("object_type", None)
    blender_obj["object_type"] = str(object_type)


def setup_static_collision(obj, collision_shape="MESH"):
    obj.hide(False)
    obj.disable_rigidbody()
    obj.enable_rigidbody(
        False,
        collision_shape=collision_shape,
        friction=RIGIDBODY_FRICTION,
        linear_damping=RIGIDBODY_LINEAR_DAMPING,
        angular_damping=RIGIDBODY_ANGULAR_DAMPING,
    )


def setup_dynamic_collision(obj):
    obj.enable_rigidbody(
        True,
        mass=RIGIDBODY_MASS,
        friction=RIGIDBODY_FRICTION,
        linear_damping=RIGIDBODY_LINEAR_DAMPING,
        angular_damping=RIGIDBODY_ANGULAR_DAMPING,
    )


def apply_random_material_properties(obj, mat_name="auto_mat"):
    mats = obj.get_materials()
    mat = mats[0] if mats else bproc.material.create(mat_name)
    if not mats:
        obj.replace_materials(mat)

    mat.set_principled_shader_value("Roughness", np.random.uniform(MATERIAL_ROUGHNESS_MIN, MATERIAL_ROUGHNESS_MAX))
    mat.set_principled_shader_value("Specular IOR Level", np.random.uniform(MATERIAL_SPECULAR_MIN, MATERIAL_SPECULAR_MAX))


def prepare_target(obj, obj_id):
    set_bop_properties(obj, obj_id, "target")
    obj.set_shading_mode("auto")
    apply_random_material_properties(obj, "auto_mat")


def prepare_box(obj):
    set_bop_properties(obj, 2, "box")
    obj.set_shading_mode("auto")


def prepare_obj3(obj):
    set_object_type(obj, "reference")
    obj.set_shading_mode("auto")


def prepare_distractor(obj):
    set_object_type(obj, "distractor")
    obj.set_shading_mode("auto")
    apply_random_material_properties(obj, "distractor_mat")


def prepare_table(obj):
    set_object_type(obj, "table")
    obj.set_shading_mode("auto")


def rotate_xy(offset, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * offset[0] - s * offset[1], s * offset[0] + c * offset[1]])


def sample_scene_rotation():
    return np.array(
        [np.random.choice(SCENE_ROTATION_X_CHOICES), 0.0, np.random.choice(SCENE_ROTATION_Z_CHOICES)],
        dtype=np.float32,
    )


def randomize_background(cc_textures, room_planes):
    if cc_textures:
        material = np.random.choice(cc_textures)
        for plane in room_planes:
            plane.replace_materials(material)


def randomize_light(light_point):
    light_point.set_location([
        np.random.uniform(LIGHT_X_MIN, LIGHT_X_MAX),
        np.random.uniform(LIGHT_Y_MIN, LIGHT_Y_MAX),
        np.random.uniform(LIGHT_Z_MIN, LIGHT_Z_MAX),
    ])
    light_point.set_energy(np.random.uniform(LIGHT_ENERGY_MIN, LIGHT_ENERGY_MAX))


def randomize_box_pose(box, table_top):
    angle = np.deg2rad(np.random.uniform(BOX_Z_ROT_MIN, BOX_Z_ROT_MAX))
    xy = np.array([np.random.uniform(BOX_X_MIN, BOX_X_MAX), np.random.uniform(BOX_Y_MIN, BOX_Y_MAX)])

    box.set_rotation_euler([0.0, 0.0, angle])
    box.set_location([xy[0], xy[1], 0.0])

    current_z_min = np.asarray(box.get_bound_box(), dtype=float)[:, 2].min()
    box.set_location([xy[0], xy[1], table_top - current_z_min])

    bbox = np.asarray(box.get_bound_box(), dtype=float)
    center = bbox.mean(axis=0)
    print(f"[BOX] location=({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}), rotation_z={np.rad2deg(angle):.2f}°")
    return center, angle


def place_obj3_relative_to_box(obj3, box):
    rel_loc = np.asarray(OBJ3_RELATIVE_LOCATION, dtype=float)
    rel_rot = np.deg2rad(np.asarray(OBJ3_RELATIVE_ROTATION_DEG, dtype=float))

    box_loc = np.asarray(box.get_location(), dtype=float)
    box_rot = np.asarray(box.get_rotation_euler(), dtype=float)

    angle_z = box_rot[2]
    cos_z = np.cos(angle_z)
    sin_z = np.sin(angle_z)

    world_rel = np.array([
        cos_z * rel_loc[0] - sin_z * rel_loc[1],
        sin_z * rel_loc[0] + cos_z * rel_loc[1],
        rel_loc[2],
    ])

    obj3.set_location(box_loc + world_rel)
    obj3.set_rotation_euler(box_rot + rel_rot)


def sample_false_pose(obj, scene_rotation, box, box_center, box_top):
    rotation = scene_rotation.copy() + np.deg2rad([
        np.random.uniform(FALSE_ROT_X_MIN, FALSE_ROT_X_MAX),
        np.random.uniform(FALSE_ROT_Y_MIN, FALSE_ROT_Y_MAX),
        np.random.uniform(FALSE_ROT_Z_MIN, FALSE_ROT_Z_MAX),
    ])
    local_xy = np.array([
        np.random.uniform(TARGET_X_MIN, TARGET_X_MAX),
        np.random.uniform(TARGET_Y_MIN, TARGET_Y_MAX),
    ])

    box_angle = box.get_rotation_euler()[2]
    world_xy = rotate_xy(local_xy, box_angle) + box_center[:2]

    obj.set_rotation_euler(rotation)
    target_z_min = np.asarray(obj.get_bound_box(), dtype=float)[:, 2].min()
    obj.set_location([world_xy[0], world_xy[1], box_top + DROP_HEIGHT - target_z_min])


def restore_initial_pose(obj, transform, scene_rotation, box, box_center, ref_box_center):
    local_xy = transform["location"][:2] - ref_box_center[:2]
    box_angle = box.get_rotation_euler()[2]
    world_xy = rotate_xy(local_xy, box_angle) + box_center[:2]

    location = transform["location"].copy()
    location[:2] = world_xy
    location[2] += (box_center[2] - ref_box_center[2] + TRUE_Z_OFFSET)

    obj.set_location(location)
    obj.set_rotation_euler(transform["rotation"] + scene_rotation + np.array([0.0, 0.0, box_angle]))


def get_xy_aabb(obj):
    bbox = np.asarray(obj.get_bound_box(), dtype=float)
    return bbox[:, 0].min(), bbox[:, 0].max(), bbox[:, 1].min(), bbox[:, 1].max()


def aabb_overlap(a, b, margin=0.0):
    return not (a[1] + margin < b[0] or a[0] - margin > b[1] or a[3] + margin < b[2] or a[2] - margin > b[3])


def sample_distractor_pose(obj, box, table_bounds):
    xmin, xmax, ymin, ymax, table_top = table_bounds
    box_aabb = get_xy_aabb(box)

    for _ in range(DISTRACTOR_MAX_TRIES):
        obj.set_rotation_euler(bproc.sampler.uniformSO3())
        z_min = np.asarray(obj.get_bound_box(), dtype=float)[:, 2].min()

        x = np.random.uniform(xmin + DISTRACTOR_XY_MARGIN, xmax - DISTRACTOR_XY_MARGIN)
        y = np.random.uniform(ymin + DISTRACTOR_XY_MARGIN, ymax - DISTRACTOR_XY_MARGIN)
        z = table_top + np.random.uniform(DISTRACTOR_Z_MIN, DISTRACTOR_Z_MAX) - z_min

        obj.set_location([x, y, z])
        if not aabb_overlap(get_xy_aabb(obj), box_aabb, DISTRACTOR_MARGIN):
            return True
    return False


def add_camera_poses(bvh, box_center, target_count):
    if target_count <= 0:
        raise ValueError(f"sample_size must be > 0, got {target_count}")

    azimuths = np.linspace(CAMERA_AZIMUTH_MIN, CAMERA_AZIMUTH_MAX, target_count, endpoint=False)
    np.random.shuffle(azimuths)

    for pose_index, azimuth_base in enumerate(azimuths, start=1):
        accepted = False
        for attempt in range(1, CAMERA_MAX_ATTEMPTS_PER_POSE + 1):
            azimuth_deg = azimuth_base + np.random.uniform(-CAMERA_AZIMUTH_JITTER, CAMERA_AZIMUTH_JITTER)
            distance = np.random.uniform(CAMERA_DISTANCE_MIN, CAMERA_DISTANCE_MAX)
            elevation_deg = np.random.uniform(CAMERA_ELEVATION_MIN, CAMERA_ELEVATION_MAX)

            elevation, azimuth = np.deg2rad(elevation_deg), np.deg2rad(azimuth_deg)
            horizontal_distance = distance * np.cos(elevation)

            location = box_center + np.array([
                horizontal_distance * np.cos(azimuth),
                horizontal_distance * np.sin(azimuth),
                distance * np.sin(elevation),
            ])

            look_offset = np.array([
                np.random.uniform(CAMERA_LOOK_X_MIN, CAMERA_LOOK_X_MAX),
                np.random.uniform(CAMERA_LOOK_Y_MIN, CAMERA_LOOK_Y_MAX),
                CAMERA_LOOK_Z_OFFSET,
            ])
            poi = box_center + look_offset
            forward = poi - location

            if np.linalg.norm(forward) < 1e-6:
                continue

            inplane_rot = np.deg2rad(np.random.uniform(CAMERA_INPLANE_MIN, CAMERA_INPLANE_MAX))
            rot = bproc.camera.rotation_from_forward_vec(forward, inplane_rot=inplane_rot)
            cam2world = bproc.math.build_transformation_mat(location, rot)

            obstacle_ok = bproc.camera.perform_obstacle_in_view_check(
                cam2world, {"min": CAMERA_OBSTACLE_MIN_DISTANCE}, bvh
            )

            print(f"[CAMERA] pose={pose_index}/{target_count} attempt={attempt} azimuth={azimuth_deg:.2f}° "
                  f"elevation={elevation_deg:.2f}° distance={distance:.3f} obstacle_check={obstacle_ok}")

            if obstacle_ok:
                bproc.camera.add_camera_pose(cam2world)
                accepted = True
                print(f"[CAMERA] accepted {pose_index}/{target_count}")
                break

        if not accepted:
            raise RuntimeError(f"[CAMERA] Failed pose {pose_index}/{target_count} after {CAMERA_MAX_ATTEMPTS_PER_POSE} tries.")

    print(f"[CAMERA] Successfully generated exactly {target_count} camera poses.")
    return target_count


# ==============================================================================
# Main Execution Pipeline
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bop_parent_path")
    parser.add_argument("dataset_name")
    parser.add_argument("cc_textures_path")
    parser.add_argument("output_dir")
    parser.add_argument("--num_scenes", type=int, default=DEFAULT_NUM_SCENES)
    parser.add_argument("--sample_size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--max_samples", type=int, default=DEFAULT_MAX_SAMPLES)
    args = parser.parse_args()

    print(f"[CPU] threads={CPU_THREADS if CPU_THREADS else 'AUTO'}, physics={PHYSICS_SUBSTEPS}/{PHYSICS_SOLVER_ITERS}")

    bproc.init()
    bproc.renderer.set_render_devices(use_only_cpu=RENDER_USE_ONLY_CPU)

    if CPU_THREADS is not None:
        bpy.context.scene.render.threads_mode = "FIXED"
        bpy.context.scene.render.threads = CPU_THREADS
        print(f"[CPU] Blender render threads: {CPU_THREADS}")

    # Camera Intrinsics
    K = np.array([[CAMERA_FX, 0.0, CAMERA_CX], [0.0, CAMERA_FY, CAMERA_CY], [0.0, 0.0, 1.0]], dtype=np.float32)
    bproc.camera.set_intrinsics_from_K_matrix(K, IMAGE_WIDTH, IMAGE_HEIGHT)

    # Load Dataset Models
    target_dataset_path = resolve_dataset_path(args.bop_parent_path, FIXED_TARGET_DATASET)
    models_dir = os.path.join(target_dataset_path, "models")
    blend_files = glob.glob(os.path.join(models_dir, "obj_*.blend"))

    blend_map = {
        int(match.group(1)): path
        for path in blend_files
        if (match := re.search(r"obj_(\d+)\.blend", os.path.basename(path)))
    }
    print(f"[INFO] Found blend objects: {sorted(blend_map.keys())}")

    if 2 not in blend_map:
        raise FileNotFoundError(f"obj_000002.blend not found in {models_dir}")
    if 3 not in blend_map:
        raise FileNotFoundError(f"obj_000003.blend not found in {models_dir}")

    candidate_ids = sorted(obj_id for obj_id in blend_map if obj_id not in (2, 3))
    if not candidate_ids:
        raise RuntimeError("No target objects found besides obj_000002 and obj_000003.")
    print(f"[INFO] Candidate target IDs: {candidate_ids}")

    # Load Distractors
    distractor_dir = os.path.join(args.bop_parent_path, "distractor_objs")
    distractor_blend_files = glob.glob(os.path.join(distractor_dir, "*.blend"))
    distractor_library = []
    for path in distractor_blend_files:
        for obj in bproc.loader.load_blend(path, obj_types="mesh"):
            prepare_distractor(obj)
            obj.hide(True)
            obj.disable_rigidbody()
            distractor_library.append(obj)
    print(f"[INFO] Loaded distractor objects: {len(distractor_library)}")

    # Load Table Geometry
    table_path = os.path.join(args.bop_parent_path, "desk.blend")
    if not os.path.exists(table_path):
        raise FileNotFoundError(f"desk.blend not found: {table_path}")
    table_objs = bproc.loader.load_blend(table_path, obj_types="mesh")
    if not table_objs:
        raise RuntimeError(f"No mesh in {table_path}")

    table = table_objs[0]
    prepare_table(table)
    table_bbox = np.asarray(table.get_bound_box(), dtype=float)
    table.set_location([0.0, 0.0, -table_bbox[:, 2].min()])
    table.set_rotation_euler([0.0, 0.0, 0.0])
    setup_static_collision(table, "MESH")

    table_bbox = np.asarray(table.get_bound_box(), dtype=float)
    table_bounds = (
        table_bbox[:, 0].min(), table_bbox[:, 0].max(),
        table_bbox[:, 1].min(), table_bbox[:, 1].max(),
        table_bbox[:, 2].max(),
    )
    table_top = table_bounds[4]
    print(f"[TABLE] center={table_bbox.mean(axis=0)}, top={table_top:.4f}")

    # Load Box Geometry
    box_objs = bproc.loader.load_blend(blend_map[2], obj_types="mesh")
    if not box_objs:
        raise RuntimeError(f"No mesh in {blend_map[2]}")

    box = box_objs[0]
    prepare_box(box)
    box_bbox = np.asarray(box.get_bound_box(), dtype=float)
    box_ref_z_min, box_ref_center = box_bbox[:, 2].min(), box_bbox.mean(axis=0)

    box.set_location([0.0, 0.0, -box_ref_z_min])
    box.set_rotation_euler([0.0, 0.0, 0.0])
    setup_static_collision(box, "MESH")
    print(f"[BOX] reference center={box_ref_center}")

    # Load obj_000003
    obj3_objs = bproc.loader.load_blend(blend_map[3], obj_types="mesh")
    if not obj3_objs:
        raise RuntimeError(f"No mesh in {blend_map[3]}")

    obj3 = obj3_objs[0]
    prepare_obj3(obj3)
    obj3.hide(False)
    setup_static_collision(obj3, "MESH")
    print("[OBJ3] Loaded obj_000003.blend")

    # Background Planes & Lighting
    room_planes = [
        bproc.object.create_primitive("PLANE", scale=TABLE_SCALE),
        bproc.object.create_primitive("PLANE", scale=ROOM_PLANE_SCALE, location=ROOM_BACK_LOCATION, rotation=ROOM_BACK_ROTATION),
        bproc.object.create_primitive("PLANE", scale=ROOM_PLANE_SCALE, location=ROOM_FRONT_LOCATION, rotation=ROOM_FRONT_ROTATION),
        bproc.object.create_primitive("PLANE", scale=ROOM_PLANE_SCALE, location=ROOM_RIGHT_LOCATION, rotation=ROOM_RIGHT_ROTATION),
        bproc.object.create_primitive("PLANE", scale=ROOM_PLANE_SCALE, location=ROOM_LEFT_LOCATION, rotation=ROOM_LEFT_ROTATION),
    ]
    for plane in room_planes:
        setup_static_collision(plane, "BOX")

    light_point = bproc.types.Light()
    light_point.set_type(LIGHT_TYPE)
    light_point.set_energy(LIGHT_DEFAULT_ENERGY)

    cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_path)
    print(f"[INFO] Loaded {len(cc_textures)} CC materials")

    # Renderer Config
    bproc.renderer.enable_depth_output(activate_antialiasing=DEPTH_ANTIALIASING)
    bproc.renderer.set_max_amount_of_samples(args.max_samples)
    print("[RENDERER] Depth output: ENABLED")

    # Scene Generation Loop
    for i in range(args.num_scenes):
        scene = bpy.context.scene
        scene.frame_end = 0
        print(f"\n================ Scene {i + 1}/{args.num_scenes} ================\n")

        if scene.camera and scene.camera.animation_data:
            scene.camera.animation_data_clear()

        scene_rotation = sample_scene_rotation()
        print(f"[SCENE {i}] Base rotation X={np.rad2deg(scene_rotation[0]):.0f}°, Z={np.rad2deg(scene_rotation[2]):.0f}°")

        is_scene_b = np.random.random() < TRUE_CASE_PROBABILITY
        print(f"[SCENE {i}] {'B - Initial position' if is_scene_b else 'A - Random placement'}")

        # Reset Static Objects
        box.hide(False)
        table.hide(False)
        obj3.hide(False)
        setup_static_collision(box, "MESH")
        setup_static_collision(obj3, "MESH")
        setup_static_collision(table, "MESH")

        for obj in distractor_library:
            obj.hide(True)
            obj.disable_rigidbody()

        randomize_background(cc_textures, room_planes)
        randomize_light(light_point)

        # Randomize Box & Place obj_000003 Relative to Box
        box_center, box_angle = randomize_box_pose(box, table_top)
        box_bbox = np.asarray(box.get_bound_box(), dtype=float)
        box_center, box_top, box_bottom = box_bbox.mean(axis=0), box_bbox[:, 2].max(), box_bbox[:, 2].min()

        place_obj3_relative_to_box(obj3, box)

        # Target Setup
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
            {"location": np.asarray(obj.get_location()).copy(), "rotation": np.asarray(obj.get_rotation_euler()).copy()}
            for obj in target_bop_objs
        ]

        # Distractor Sampling
        count = min(DISTRACTOR_OBJECT_COUNT, len(distractor_library)) if distractor_library else 0
        selected_distractors = list(np.random.choice(distractor_library, size=count, replace=False)) if count else []
        print("[DISTRACTOR] selected: " + (", ".join(o.get_name() for o in selected_distractors) if selected_distractors else "NONE"))

        for obj in selected_distractors:
            obj.hide(False)
            obj.disable_rigidbody()

        # Target Placement
        if is_scene_b:
            for obj, transform in zip(target_bop_objs, target_initial_transforms):
                obj.disable_rigidbody()
                restore_initial_pose(obj, transform, scene_rotation, box, box_center, box_ref_center)
            print("[SCENE] Target collision: DISABLED")
        else:
            if ENABLE_PHYSICS:
                for obj in target_bop_objs:
                    setup_dynamic_collision(obj)
            bproc.object.sample_poses(
                objects_to_sample=target_bop_objs,
                sample_pose_func=lambda obj: sample_false_pose(obj, scene_rotation, box, box_center, box_top),
                max_tries=1000,
            )
            print("[SCENE] Target collision: ENABLED")

        # Distractor Placement
        valid_distractors = []
        for obj in selected_distractors:
            if sample_distractor_pose(obj, box, table_bounds):
                valid_distractors.append(obj)
            else:
                print(f"[DISTRACTOR] Could not place {obj.get_name()} outside Box; hiding.")
                obj.hide(True)

        selected_distractors = valid_distractors

        if ENABLE_PHYSICS:
            for obj in selected_distractors:
                setup_dynamic_collision(obj)

        # Physics Simulation
        if ENABLE_PHYSICS and not is_scene_b:
            bproc.object.simulate_physics_and_fix_final_poses(
                min_simulation_time=PHYSICS_MIN_SIMULATION_TIME,
                max_simulation_time=PHYSICS_MAX_SIMULATION_TIME,
                check_object_interval=PHYSICS_CHECK_OBJECT_INTERVAL,
                substeps_per_frame=PHYSICS_SUBSTEPS,
                solver_iters=PHYSICS_SOLVER_ITERS,
            )

        # BVH & Camera Poses
        scene_objects = target_bop_objs + selected_distractors + [box, obj3, table]
        bvh = bproc.object.create_bvh_tree_multi_objects(scene_objects)
        cam_poses = add_camera_poses(bvh, box_center, args.sample_size)

        print(f"[SCENE {i}] Rendering {cam_poses} camera poses (requested {args.sample_size}, 360° horizontal coverage)")
        if cam_poses == 0:
            raise RuntimeError(f"[SCENE {i}] No valid camera poses found.")

        # Render & Check BOP Objects
        data = bproc.renderer.render()
        print(f"[SCENE {i}] Render keys: {list(data.keys())}")

        if "colors" not in data or "depth" not in data:
            raise RuntimeError("[RENDER] RGB or Depth colors were not generated.")

        print(f"[RENDER] RGB frames: {len(data['colors'])}, Depth frames: {len(data['depth'])}")

        for obj in target_bop_objs + [box]:
            category_id = obj.get_cp("category_id")
            print(f"    {obj.get_name()} category_id = {category_id} type = {type(category_id)} object_type = {obj.get_cp('object_type')}")
            if not isinstance(category_id, (int, np.integer)):
                raise TypeError(f"category_id of {obj.get_name()} is not an integer: {category_id!r}")

        # BOP Output Writing
        bop_output_dir = os.path.join(args.output_dir, "bop_data")
        print(f"[BOP] Writing to: {bop_output_dir}")

        bproc.writer.write_bop(
            bop_output_dir,
            target_objects=(target_bop_objs + [box]),
            dataset=FIXED_TARGET_DATASET,
            depth_scale=BOP_DEPTH_SCALE,
            depths=data["depth"],
            colors=data["colors"],
            color_file_format=BOP_COLOR_FORMAT,
            ignore_dist_thres=BOP_IGNORE_DIST_THRESHOLD,
            calc_mask_info_coco=True,
            num_worker=BOP_NUM_WORKERS,
        )

        print(f"[SCENE {i}] BOP writer finished.")

        # Cleanup Frame
        for obj in (target_bop_objs + selected_distractors + [obj3, box, table]):
            obj.hide(True)
            obj.disable_rigidbody()

        print(f"[SCENE {i}] Done.")

    print("Done")


if __name__ == "__main__":
    main()