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
parser.add_argument('bop_parent_path')
parser.add_argument('dataset_name')
parser.add_argument('cc_textures_path')
parser.add_argument('output_dir')
parser.add_argument('--num_scenes', type=int, default=1)
parser.add_argument('--sample_size', type=int, default=100)
parser.add_argument('--max_samples', type=int, default=128)
args = parser.parse_args()

FIXED_TARGET_DATASET = 'hb'

def resolve_dataset_path(base_path: str, dataset_name: str) -> str:
    path = os.path.join(base_path, dataset_name)
    if os.path.isdir(path):
        return path
    raise FileNotFoundError(f"Dataset not found: {path}")

# ================= initial =================
bproc.init()
bproc.renderer.set_render_devices(use_only_cpu=False)

target_dataset_path = resolve_dataset_path(args.bop_parent_path, FIXED_TARGET_DATASET)

distractor_dir = os.path.join(args.bop_parent_path,"distractor_objs_fix")
blend_files = glob.glob(os.path.join(distractor_dir, "*.blend"))
distractor_objs = []
for blend in blend_files:
    objs = bproc.loader.load_blend(blend,obj_types="mesh")
    distractor_objs.extend(objs)


# ================= load .blend =================
models_dir = os.path.join(target_dataset_path, 'models')
blend_files = glob.glob(os.path.join(models_dir, 'obj_*.blend'))

obj_ids = []
blend_map = {}

for f in blend_files:
    m = re.search(r'obj_(\d+)\.blend', os.path.basename(f))
    if m:
        obj_id = int(m.group(1))
        obj_ids.append(obj_id)
        blend_map[obj_id] = f

obj_ids.sort()
print(f"[INFO] Found blend objects: {obj_ids}")

target_bop_objs = []

for obj_id in obj_ids:
    blend_path = blend_map[obj_id]

    objs = bproc.loader.load_blend(
        blend_path,
        obj_types="mesh"
    )

    if len(objs) == 0:
        print(f"[WARNING] No mesh in {blend_path}")
        continue

    for obj in objs:
        obj.set_cp("category_id", obj_id)
        obj.set_cp("bop_dataset_name", FIXED_TARGET_DATASET)

        # if model is mm in unit，cancel conment
        # obj.set_scale([0.001, 0.001, 0.001])

        target_bop_objs.append(obj)

print(f"[INFO] Loaded {len(target_bop_objs)} mesh objects")
ripcord_root = bproc.object.create_empty("ripcord_root")
for obj in target_bop_objs:
    obj.set_parent(ripcord_root)

# ================= camera  =================
width = 1920
height = 1080

K = np.array([
    [1077.4795,    0.0,      968.95593],
    [0.0,       1077.4795,   563.32214],
    [0.0,          0.0,        1.0]
], dtype=np.float32)

bproc.camera.set_intrinsics_from_K_matrix(K,width,height)

# ================= set shading =================
for obj in target_bop_objs:
    obj.set_shading_mode('auto')
    obj.hide(True)

for obj in distractor_objs:
    obj.set_shading_mode("auto")
    obj.hide(True)

# ================= build room =================
room_planes = [
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.5708, 0, 0]),
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.5708, 0, 0]),
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.5708, 0]),
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.5708, 0])
]
for plane in room_planes:
    plane.enable_rigidbody(False, collision_shape='BOX', mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)


# ================= table =================
table_path = '/home/yeo/Downloads/Data_Generation/bop/BlenderProc/desk.blend'
table = bproc.loader.load_blend(table_path, obj_types="mesh")[0]
bbox = np.array(table.get_bound_box())
z_min = bbox[:,2].min()
table.set_location([0,0,-z_min])
table.enable_rigidbody(False,collision_shape="BOX",friction=100)

xmin = bbox[:,0].min()
xmax = bbox[:,0].max()
ymin = bbox[:,1].min()
ymax = bbox[:,1].max()
table_top = bbox[:,2].max()

# ================= object size =================

obj_bbox = []
for obj in target_bop_objs:
    bbox = np.array(obj.get_bound_box())
    obj_bbox.append(bbox)

obj_bbox = np.concatenate(obj_bbox, axis=0)
obj_size = np.max(obj_bbox.max(axis=0) -obj_bbox.min(axis=0))

print("Object size:", obj_size)

# ================= light =================
light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
light_plane.set_name('light_plane')
light_plane_material = bproc.material.create('light_material')

light_point = bproc.types.Light()
light_point.set_energy(200)

# ================= texture =================
cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_path)


# ================= mesh visibility =================

def sample_ripcord_visibility(objects):
    visible_objects = []

    for obj in objects:
        name = obj.get_name()
        if name == "barcode":
            obj.hide(False)
            visible_objects.append(obj)
            print(f"[SHOW] {name}")
        elif np.random.rand() < 0.5:
            obj.hide(True)
            print(f"[HIDE] {name}")
        else:
            obj.hide(False)
            visible_objects.append(obj)
            print(f"[SHOW] {name}")

    if len(visible_objects) == 0:
        obj = np.random.choice(objects)
        obj.hide(False)
        visible_objects.append(obj)
        print(f"[FORCE SHOW] {obj.get_name()}")

    return visible_objects

RIPCORD_ORDER = [
    "ripcord_4",
    "ripcord_3",
    "ripcord_2",
    "ripcord"
]

def get_obj_id(visible_objects):
    visible_names = {obj.get_name() for obj in visible_objects}
    for index, name in enumerate(RIPCORD_ORDER, start=1):
        if name in visible_names:
            return index
    return len(RIPCORD_ORDER)

# ================= pose =================

margin = 0.05

center = np.array([
    (xmin+xmax)/2,
    (ymin+ymax)/2,
    table_top+0.05
])

def sample_ripcord_pose():
    ripcord_root.set_location(center)
    ripcord_root.set_rotation_euler([
        0,
        0,
        0
    ])

# ================= render settings =================
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(args.max_samples)

# ================= main loop =================
for i in range(args.num_scenes):
    scene = bpy.context.scene
    scene.frame_end = 0
    cam_ob = scene.camera
    if cam_ob is not None and cam_ob.animation_data is not None:
        cam_ob.animation_data_clear()

    sampled_target_bop_objs = sample_ripcord_visibility(target_bop_objs)
    num_dist = min(3, len(distractor_objs))
    
    sampled_distractors = list(np.random.choice(distractor_objs, size=num_dist, replace=False))


    for obj in sampled_distractors:

        obj.hide(False)

        obj.enable_rigidbody(
            True,
            mass=1.0,
            friction=100,
            linear_damping=0.99,
            angular_damping=0.99
        )

    # light
    light_plane_material.make_emissive(
        emission_strength=np.random.uniform(3,6),
        emission_color=np.random.uniform([0.5,0.5,0.5,1],[1,1,1,1])
    )
    light_plane.replace_materials(light_plane_material)

    light_point.set_color(np.random.uniform([0.5,0.5,0.5],[1,1,1]))
    light_point.set_location(
        bproc.sampler.shell([0,0,0], 1, 1.5, 5, 89)
    )

    # ground texture
    tex = np.random.choice(cc_textures)
    for plane in room_planes:
        plane.replace_materials(tex)

    # object pose

    all_objs = sampled_target_bop_objs + sampled_distractors
    sample_ripcord_pose()

    # Physics Positioning
    USE_PHYSICS = np.random.rand() < 1.0
    if USE_PHYSICS:
        bproc.object.simulate_physics_and_fix_final_poses(
            min_simulation_time=3,
            max_simulation_time=5,
            check_object_interval=1,
            substeps_per_frame=20,
            solver_iters=25
        )
        pass

    bvh = bproc.object.create_bvh_tree_multi_objects(all_objs + [table])

    cam_poses = 0

    while cam_poses < args.sample_size:
        focus_obj = np.random.choice(sampled_target_bop_objs)
        object_center = bproc.object.compute_poi([focus_obj])
        # camera distance
        location = bproc.sampler.shell(
            center=object_center,
            radius_min=obj_size*2.5,
            radius_max=obj_size*4,
            elevation_min=25,
            elevation_max=70
        )
        # shift looking point
        poi_offset = np.random.uniform(
            [-1.5,-1.5,-0.8],
            [1.5,1.5,0.8]
        ) * obj_size
        poi = object_center
        rot = bproc.camera.rotation_from_forward_vec(
            poi-location,
            inplane_rot=np.random.uniform(-np.pi/8,np.pi/8)
        )
        cam2world = bproc.math.build_transformation_mat(location,rot)

        if bproc.camera.perform_obstacle_in_view_check(cam2world,{"min":0.3},bvh):
            bproc.camera.add_camera_pose(cam2world)
            cam_poses += 1

    data = bproc.renderer.render()


    # duplicate visible objects
    copies = []
    for obj in sampled_target_bop_objs:
        bpy_obj = obj.blender_obj
        dup = bpy_obj.copy()
        dup.data = bpy_obj.data.copy()
        bpy.context.collection.objects.link(dup)
        copies.append(dup)

    bpy.ops.object.select_all(action='DESELECT')
    for obj in copies:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = copies[0]
    bpy.ops.object.join()

    joined_blender_obj = bpy.context.view_layer.objects.active
    joined = bproc.types.MeshObject(joined_blender_obj)

    obj_id = get_obj_id(sampled_target_bop_objs)
    joined.set_cp("category_id", obj_id)
    joined.set_cp("bop_dataset_name",FIXED_TARGET_DATASET)

    data = bproc.renderer.render()

    bproc.writer.write_bop(os.path.join(args.output_dir, 'bop_data'),
                           target_objects = [joined],
                           dataset = 'hb',
                           depth_scale = 0.1,
                           depths = data["depth"],
                           colors = data["colors"], 
                           color_file_format = "JPEG",
                           ignore_dist_thres = 10)
    bpy.data.objects.remove(joined_blender_obj,do_unlink=True)

    for obj in sampled_target_bop_objs + sampled_distractors:
        obj.hide(True)
        obj.disable_rigidbody()

print("✅ Done")