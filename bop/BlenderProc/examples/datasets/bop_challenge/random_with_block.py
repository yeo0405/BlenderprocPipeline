import blenderproc as bproc
import pyrender
from pyrender.platforms import egl
import argparse
import os
import numpy as np
import glob
import re
import bpy

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
TARGET_OBJECT_COUNT = 3

POSE_MODE_NORMAL_PHYSICS = 0
POSE_MODE_FRONT_UP_PHYSICS = 1
POSE_MODE_FLOAT = 2

LIGHT_INTENSITY_SCALE = 1.0
LIGHT_COLOR_TEMPERATURE_ENABLED = True
LIGHT_COLOR_TEMPERATURE_MIN = 2800
LIGHT_COLOR_TEMPERATURE_MAX = 7500
LIGHT_COLOR_TEMPERATURE_PROBABILITY = 0.85
LIGHT_WARM_COOL_STRENGTH = 0.65

LIGHT_MAIN_ENERGY_MIN = 1.0
LIGHT_MAIN_ENERGY_MAX = 15.0
LIGHT_POINT_ENERGY_MIN = 30.0
LIGHT_POINT_ENERGY_MAX = 900.0
LIGHT_SECONDARY_ENERGY_MIN = 0.0
LIGHT_SECONDARY_ENERGY_MAX = 400.0
LIGHT_FILL_ENERGY_MIN = 0.0
LIGHT_FILL_ENERGY_MAX = 250.0
LIGHT_DIRECTION_RANDOMNESS = 1.0
LIGHT_POINT_COUNT = 3

LIGHT_MODE_WEIGHTS = {
    "normal": 35,
    "bright": 10,
    "dark": 10,
    "warm": 10,
    "cool": 10,
    "strong_side": 10,
    "backlight": 10,
    "top": 5
}

cpu_threads = os.environ.get("BLENDER_THREADS")
if cpu_threads:
    try:
        cpu_threads = max(1, int(cpu_threads))
    except ValueError:
        cpu_threads = None

physics_substeps = int(
    os.environ.get("PHYSICS_SUBSTEPS", "20")
)
physics_solver_iters = int(
    os.environ.get("PHYSICS_SOLVER_ITERS", "25")
)

print(
    f"[CPU] threads="
    f"{cpu_threads if cpu_threads else 'AUTO'}, "
    f"physics={physics_substeps}/"
    f"{physics_solver_iters}"
)


def resolve_dataset_path(base_path: str, dataset_name: str) -> str:
    path = os.path.join(base_path, dataset_name)
    if os.path.isdir(path):
        return path
    raise FileNotFoundError(f"Dataset not found: {path}")


bproc.init()

if cpu_threads is not None:
    scene = bpy.context.scene
    scene.render.threads_mode = "FIXED"
    scene.render.threads = cpu_threads
    print(
        f"[CPU] Blender render threads: {cpu_threads}"
    )

bproc.renderer.set_render_devices(use_only_cpu=False)

target_dataset_path = resolve_dataset_path(
    args.bop_parent_path,
    FIXED_TARGET_DATASET
)

distractor_dir = os.path.join(
    args.bop_parent_path,
    "distractor_objs"
)

blend_files = glob.glob(
    os.path.join(distractor_dir, "*.blend")
)
distractor_objs = []

for blend in blend_files:
    distractor_objs.extend(
        bproc.loader.load_blend(
            blend,
            obj_types="mesh"
        )
    )

models_dir = os.path.join(
    target_dataset_path,
    "models"
)

blend_files = glob.glob(
    os.path.join(models_dir, "obj_*.blend")
)

obj_ids = []
blend_map = {}

for f in blend_files:
    m = re.search(
        r"obj_(\d+)\.blend",
        os.path.basename(f)
    )
    if m:
        obj_id = int(m.group(1))
        obj_ids.append(obj_id)
        blend_map[obj_id] = f

obj_ids.sort()
print(f"[INFO] Found blend objects: {obj_ids}")

target_bop_objs = []

for obj_id in obj_ids:
    objs = bproc.loader.load_blend(
        blend_map[obj_id],
        obj_types="mesh"
    )

    if not objs:
        print(
            f"[WARNING] No mesh in "
            f"{blend_map[obj_id]}"
        )
        continue

    for obj in objs:
        obj.set_cp("category_id", obj_id)
        obj.set_cp(
            "bop_dataset_name",
            FIXED_TARGET_DATASET
        )
        target_bop_objs.append(obj)

print(
    f"[INFO] Loaded "
    f"{len(target_bop_objs)} mesh objects"
)

if not target_bop_objs:
    raise RuntimeError(
        "No target objects were loaded."
    )

print(
    f"[INFO] Target objects per scene: "
    f"{min(TARGET_OBJECT_COUNT, len(target_bop_objs))}"
)

width = 1440
height = 1080

K = np.array([
    [1070.7898, 0.0, 722.30939],
    [0.0, 1070.5270, 550.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

bproc.camera.set_intrinsics_from_K_matrix(
    K,
    width,
    height
)

for obj in target_bop_objs:
    obj.set_shading_mode("auto")
    obj.hide(True)

for obj in distractor_objs:
    obj.set_shading_mode("auto")
    obj.hide(True)

room_planes = [
    bproc.object.create_primitive(
        "PLANE",
        scale=[2, 2, 1]
    ),
    bproc.object.create_primitive(
        "PLANE",
        scale=[2, 2, 1],
        location=[0, -2, 2],
        rotation=[-1.5708, 0, 0]
    ),
    bproc.object.create_primitive(
        "PLANE",
        scale=[2, 2, 1],
        location=[0, 2, 2],
        rotation=[1.5708, 0, 0]
    ),
    bproc.object.create_primitive(
        "PLANE",
        scale=[2, 2, 1],
        location=[2, 0, 2],
        rotation=[0, -1.5708, 0]
    ),
    bproc.object.create_primitive(
        "PLANE",
        scale=[2, 2, 1],
        location=[-2, 0, 2],
        rotation=[0, 1.5708, 0]
    )
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

table_path = os.path.join(
    args.bop_parent_path,
    "desk.blend"
)

if not os.path.exists(table_path):
    table_path = (
        "/home/yeo/Downloads/Data_Generation/"
        "bop/BlenderProc/desk.blend"
    )

table = bproc.loader.load_blend(
    table_path,
    obj_types="mesh"
)[0]

bbox = np.array(table.get_bound_box())
z_min = bbox[:, 2].min()

table.set_location([0, 0, -z_min])
table.enable_rigidbody(
    False,
    collision_shape="BOX",
    friction=100
)

xmin = bbox[:, 0].min()
xmax = bbox[:, 0].max()
ymin = bbox[:, 1].min()
ymax = bbox[:, 1].max()
table_top = bbox[:, 2].max()

occlusion_cube = bproc.object.create_primitive(
    "CUBE",
    scale=[1, 1, 1],
    location=[0, 0, -10]
)

occlusion_cube.set_name("occlusion_cube")
occlusion_cube.hide(True)
occlusion_cube.disable_rigidbody()
occlusion_cube.set_shading_mode("auto")

occlusion_material = bproc.material.create(
    "occlusion_material"
)
occlusion_material.set_principled_shader_value(
    "Base Color",
    [0.5, 0.5, 0.5, 1]
)
occlusion_material.set_principled_shader_value(
    "Roughness",
    0.8
)
occlusion_cube.replace_materials(
    occlusion_material
)


def place_random_occlusion_cube(target_obj, cube):
    bbox = np.array(target_obj.get_bound_box())

    xmin = bbox[:, 0].min()
    xmax = bbox[:, 0].max()
    ymin = bbox[:, 1].min()
    ymax = bbox[:, 1].max()
    zmin = bbox[:, 2].min()
    zmax = bbox[:, 2].max()

    target_size = np.array([
        xmax - xmin,
        ymax - ymin,
        zmax - zmin
    ])

    target_center = np.array([
        (xmin + xmax) * 0.5,
        (ymin + ymax) * 0.5,
        (zmin + zmax) * 0.5
    ])

    scale_xy = 0.60
    cube_size_x = target_size[0] * scale_xy
    cube_size_y = target_size[1] * scale_xy
    cube_size_z = max(
        target_size[2] * np.random.uniform(0.3, 0.8),
        0.01
    )

    cube.set_scale([
        cube_size_x / 2,
        cube_size_y / 2,
        cube_size_z / 2
    ])

    max_offset_x = target_size[0] * 0.35
    max_offset_y = target_size[1] * 0.35

    offset_x = np.random.uniform(
        -max_offset_x,
        max_offset_x
    )
    offset_y = np.random.uniform(
        -max_offset_y,
        max_offset_y
    )

    cube_z = zmax + cube_size_z * 0.25

    cube.set_location([
        target_center[0] + offset_x,
        target_center[1] + offset_y,
        cube_z
    ])

    cube.set_rotation_euler([
        0,
        0,
        np.random.uniform(-np.pi, np.pi)
    ])

    cube.hide(False)
    cube.disable_rigidbody()

    print(
        "[OCCLUSION] "
        f"target_size={target_size} "
        f"cube_size={cube_size_x:.4f}, "
        f"{cube_size_y:.4f}, "
        f"{cube_size_z:.4f} "
        f"offset={offset_x:.4f}, "
        f"{offset_y:.4f}"
    )


obj_bbox = [
    np.array(obj.get_bound_box())
    for obj in target_bop_objs
]

obj_bbox = np.concatenate(obj_bbox, axis=0)

obj_size = np.max(
    obj_bbox.max(axis=0) -
    obj_bbox.min(axis=0)
)

print("Object size:", obj_size)

light_plane = bproc.object.create_primitive(
    "PLANE",
    scale=[3, 3, 1],
    location=[0, 0, 10]
)

light_plane.set_name("light_plane")

light_plane_material = bproc.material.create(
    "light_material"
)

light_point = bproc.types.Light()
light_point_2 = bproc.types.Light()
light_point_3 = bproc.types.Light()

cc_textures = bproc.loader.load_ccmaterials(
    args.cc_textures_path
)

margin = 0.05


def sample_pose_func(obj, mode):
    obj.set_location(
        np.random.uniform(
            [
                xmin + margin,
                ymin + margin,
                table_top + 0.05
            ],
            [
                xmax - margin,
                ymax - margin,
                table_top + 0.30
            ]
        )
    )

    if mode == POSE_MODE_FRONT_UP_PHYSICS:
        obj.set_rotation_euler([
            0,
            0,
            np.random.uniform(-np.pi, np.pi)
        ])
    else:
        obj.set_rotation_euler(
            bproc.sampler.uniformSO3()
        )


def kelvin_to_rgb(kelvin):
    k = np.clip(kelvin, 1000, 40000) / 100.0

    if k <= 66:
        r = 255
        g = (
            99.4708025861 * np.log(k)
            - 161.1195681661
        )
        b = 0 if k <= 19 else (
            138.5177312231 * np.log(k - 10)
            - 305.0447927307
        )
    else:
        r = (
            329.698727446 *
            ((k - 60) ** -0.1332047592)
        )
        g = (
            288.1221695283 *
            ((k - 60) ** -0.0755148492)
        )
        b = 255

    rgb = np.array([
        np.clip(r, 0, 255),
        np.clip(g, 0, 255),
        np.clip(b, 0, 255)
    ]) / 255.0

    return np.clip(rgb, 0, 1)


def random_rgb(min_value=0.7, max_value=1.0):
    return np.random.uniform(
        min_value,
        max_value,
        3
    )


def apply_color_temperature(
    rgb,
    kelvin,
    strength
):
    temperature_rgb = kelvin_to_rgb(kelvin)

    return np.clip(
        rgb * (1.0 - strength)
        + temperature_rgb * strength,
        0,
        1
    )


def random_light_position(mode):
    if mode == "strong_side":
        side = np.random.choice([-1, 1])
        return [
            side * np.random.uniform(2.0, 4.0),
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(1.0, 4.0)
        ]

    if mode == "backlight":
        return [
            np.random.uniform(-1.0, 1.0),
            np.random.choice([-1, 1])
            * np.random.uniform(2.0, 4.0),
            np.random.uniform(2.0, 5.0)
        ]

    if mode == "top":
        return [
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(4.0, 6.0)
        ]

    return bproc.sampler.shell(
        center=[0, 0, 1],
        radius_min=1.0,
        radius_max=4.0,
        elevation_min=20,
        elevation_max=85
    )


def setup_random_lighting():
    modes = list(LIGHT_MODE_WEIGHTS.keys())
    weights = np.array(
        list(LIGHT_MODE_WEIGHTS.values()),
        dtype=float
    )
    weights /= weights.sum()

    lighting_mode = np.random.choice(
        modes,
        p=weights
    )

    print(f"[LIGHT] mode={lighting_mode}")

    main_energy = np.random.uniform(
        LIGHT_MAIN_ENERGY_MIN,
        LIGHT_MAIN_ENERGY_MAX
    )
    point_energy = np.random.uniform(
        LIGHT_POINT_ENERGY_MIN,
        LIGHT_POINT_ENERGY_MAX
    )
    secondary_energy = np.random.uniform(
        LIGHT_SECONDARY_ENERGY_MIN,
        LIGHT_SECONDARY_ENERGY_MAX
    )
    fill_energy = np.random.uniform(
        LIGHT_FILL_ENERGY_MIN,
        LIGHT_FILL_ENERGY_MAX
    )

    main_color = random_rgb(0.75, 1.0)
    point_color = random_rgb(0.70, 1.0)
    secondary_color = random_rgb(0.60, 1.0)
    fill_color = random_rgb(0.60, 1.0)

    if lighting_mode == "dark":
        main_energy *= np.random.uniform(0.25, 0.55)
        point_energy *= np.random.uniform(0.20, 0.45)
        secondary_energy *= np.random.uniform(0.0, 0.3)
        fill_energy *= np.random.uniform(0.0, 0.25)

    elif lighting_mode == "bright":
        main_energy *= np.random.uniform(1.3, 2.0)
        point_energy *= np.random.uniform(1.2, 1.8)
        secondary_energy *= np.random.uniform(1.2, 1.8)
        fill_energy *= np.random.uniform(1.2, 1.8)

    elif lighting_mode == "strong_side":
        main_energy *= np.random.uniform(1.0, 1.5)
        point_energy *= np.random.uniform(1.2, 1.8)
        secondary_energy *= np.random.uniform(0.0, 0.4)
        fill_energy *= np.random.uniform(0.0, 0.3)

    elif lighting_mode == "backlight":
        main_energy *= np.random.uniform(0.7, 1.2)
        point_energy *= np.random.uniform(1.2, 1.8)
        secondary_energy *= np.random.uniform(0.5, 1.0)
        fill_energy *= np.random.uniform(0.0, 0.4)

    elif lighting_mode == "top":
        main_energy *= np.random.uniform(1.0, 1.5)
        point_energy *= np.random.uniform(1.0, 1.5)
        secondary_energy *= np.random.uniform(0.0, 0.5)
        fill_energy *= np.random.uniform(0.0, 0.4)

    temperature = None

    if (
        LIGHT_COLOR_TEMPERATURE_ENABLED
        and np.random.rand()
        < LIGHT_COLOR_TEMPERATURE_PROBABILITY
    ):
        temperature = np.random.uniform(
            LIGHT_COLOR_TEMPERATURE_MIN,
            LIGHT_COLOR_TEMPERATURE_MAX
        )

        main_color = apply_color_temperature(
            main_color,
            temperature,
            LIGHT_WARM_COOL_STRENGTH
        )
        point_color = apply_color_temperature(
            point_color,
            temperature,
            LIGHT_WARM_COOL_STRENGTH
        )
        secondary_color = apply_color_temperature(
            secondary_color,
            temperature,
            LIGHT_WARM_COOL_STRENGTH
        )
        fill_color = apply_color_temperature(
            fill_color,
            temperature,
            LIGHT_WARM_COOL_STRENGTH
        )

    if lighting_mode == "warm":
        warm_temperature = np.random.uniform(
            LIGHT_COLOR_TEMPERATURE_MIN,
            min(4000, LIGHT_COLOR_TEMPERATURE_MAX)
        )
        main_color = apply_color_temperature(
            main_color,
            warm_temperature,
            0.85
        )
        point_color = apply_color_temperature(
            point_color,
            warm_temperature,
            0.85
        )

    elif lighting_mode == "cool":
        cool_temperature = np.random.uniform(
            max(5000, LIGHT_COLOR_TEMPERATURE_MIN),
            LIGHT_COLOR_TEMPERATURE_MAX
        )
        main_color = apply_color_temperature(
            main_color,
            cool_temperature,
            0.85
        )
        point_color = apply_color_temperature(
            point_color,
            cool_temperature,
            0.85
        )

    scale = LIGHT_INTENSITY_SCALE

    main_energy *= scale
    point_energy *= scale
    secondary_energy *= scale
    fill_energy *= scale

    light_plane_material.make_emissive(
        emission_strength=main_energy,
        emission_color=np.append(
            main_color,
            1.0
        )
    )

    light_plane.replace_materials(
        light_plane_material
    )

    light_point.set_energy(point_energy)
    light_point.set_color(point_color)
    light_point.set_location(
        random_light_position(lighting_mode)
    )

    light_point_2.set_energy(secondary_energy)
    light_point_2.set_color(secondary_color)
    light_point_2.set_location(
        random_light_position("normal")
    )

    light_point_3.set_energy(fill_energy)
    light_point_3.set_color(fill_color)
    light_point_3.set_location(
        random_light_position("normal")
    )

    if temperature is not None:
        print(
            f"[LIGHT] temperature="
            f"{temperature:.0f}K"
        )

    print(
        "[LIGHT] "
        f"main={main_energy:.2f}, "
        f"point={point_energy:.2f}, "
        f"secondary={secondary_energy:.2f}, "
        f"fill={fill_energy:.2f}"
    )


bproc.renderer.enable_depth_output(
    activate_antialiasing=False
)

bproc.renderer.set_max_amount_of_samples(
    args.max_samples
)

for i in range(args.num_scenes):
    print(
        f"\n================ Scene "
        f"{i + 1}/{args.num_scenes} ================\n"
    )

    scene = bpy.context.scene
    scene.frame_end = 0

    occlusion_cube.hide(True)
    occlusion_cube.disable_rigidbody()

    cam_ob = scene.camera

    if (
        cam_ob is not None
        and cam_ob.animation_data is not None
    ):
        cam_ob.animation_data_clear()

    target_count = min(
        TARGET_OBJECT_COUNT,
        len(target_bop_objs)
    )

    sampled_target_bop_objs = list(
        np.random.choice(
            target_bop_objs,
            size=target_count,
            replace=False
        )
    )

    print(
        "[TARGET] selected: "
        + ", ".join(
            str(obj.get_cp("category_id"))
            for obj in sampled_target_bop_objs
        )
    )

    num_dist = min(3, len(distractor_objs))

    if num_dist > 0:
        sampled_distractors = list(
            np.random.choice(
                distractor_objs,
                size=num_dist,
                replace=False
            )
        )
    else:
        sampled_distractors = []

    for obj in (
        sampled_target_bop_objs
        + sampled_distractors
    ):
        mats = obj.get_materials()

        if not mats:
            mat = bproc.material.create("auto_mat")
            obj.replace_materials(mat)
        else:
            mat = mats[0]

        mat.set_principled_shader_value(
            "Roughness",
            np.random.uniform(0, 1.0)
        )
        mat.set_principled_shader_value(
            "Specular IOR Level",
            np.random.uniform(0, 1.0)
        )

        obj.enable_rigidbody(
            True,
            mass=1.0,
            friction=100.0,
            linear_damping=0.99,
            angular_damping=0.99
        )
        obj.hide(False)

    setup_random_lighting()

    tex = np.random.choice(cc_textures)

    for plane in room_planes:
        plane.replace_materials(tex)

    rand = np.random.rand()

    if rand < 1 / 3:
        pose_mode = POSE_MODE_NORMAL_PHYSICS
    elif rand < 2 / 3:
        pose_mode = POSE_MODE_FRONT_UP_PHYSICS
    else:
        pose_mode = POSE_MODE_FLOAT

    print(f"[POSE] mode={pose_mode}")

    all_objs = (
        sampled_target_bop_objs
        + sampled_distractors
    )

    bproc.object.sample_poses(
        objects_to_sample=all_objs,
        sample_pose_func=lambda obj:
            sample_pose_func(obj, pose_mode),
        max_tries=1000
    )

    if pose_mode != POSE_MODE_FLOAT:
        bproc.object.simulate_physics_and_fix_final_poses(
            min_simulation_time=3,
            max_simulation_time=10,
            check_object_interval=1,
            substeps_per_frame=physics_substeps,
            solver_iters=physics_solver_iters
        )

    if pose_mode == POSE_MODE_FRONT_UP_PHYSICS:
        target_obj = np.random.choice(
            sampled_target_bop_objs
        )
        place_random_occlusion_cube(
            target_obj,
            occlusion_cube
        )

    bvh = bproc.object.create_bvh_tree_multi_objects(
        all_objs + [table]
    )

    cam_poses = 0

    while cam_poses < args.sample_size:
        focus_obj = np.random.choice(
            sampled_target_bop_objs
        )

        object_center = bproc.object.compute_poi(
            [focus_obj]
        )

        location = bproc.sampler.shell(
            center=object_center,
            radius_min=obj_size * 2.5,
            radius_max=obj_size * 4,
            elevation_min=25,
            elevation_max=70
        )

        poi_offset = np.random.uniform(
            [-1.5, -1.5, -0.8],
            [1.5, 1.5, 0.8]
        ) * obj_size

        poi = object_center + poi_offset

        rot = bproc.camera.rotation_from_forward_vec(
            poi - location,
            inplane_rot=np.random.uniform(
                -np.pi / 8,
                np.pi / 8
            )
        )

        cam2world = bproc.math.build_transformation_mat(
            location,
            rot
        )

        if bproc.camera.perform_obstacle_in_view_check(
            cam2world,
            {"min": 0.3},
            bvh
        ):
            bproc.camera.add_camera_pose(cam2world)
            cam_poses += 1

    data = bproc.renderer.render()

    bproc.writer.write_bop(
        os.path.join(
            args.output_dir,
            "bop_data"
        ),
        target_objects=sampled_target_bop_objs,
        dataset="hb",
        depth_scale=0.1,
        depths=data["depth"],
        colors=data["colors"],
        color_file_format="JPEG",
        ignore_dist_thres=10
    )

    for obj in (
        sampled_target_bop_objs
        + sampled_distractors
    ):
        obj.hide(True)
        obj.disable_rigidbody()

    occlusion_cube.hide(True)
    occlusion_cube.disable_rigidbody()

print("Done")