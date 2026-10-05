"""
Parameterized BlenderProc script for generating training data with annotated models.
This script wraps the original main_hb.py to make it more flexible and reusable.
"""

import blenderproc as bproc
import argparse
import os
import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_required_files(dataset_path: str, dataset_name: str) -> bool:
    """Validate that required files exist for the dataset."""
    required_files = {
        'camera.json': os.path.join(dataset_path, dataset_name, 'camera.json'),
        'test_targets_bop19.json': os.path.join(dataset_path, dataset_name, 'test_targets_bop19.json'),
        'models_info.json': os.path.join(dataset_path, dataset_name, 'models', 'models_info.json'),
    }
    
    missing_files = []
    for file_type, file_path in required_files.items():
        if not os.path.exists(file_path):
            missing_files.append(f"{file_type}: {file_path}")
    
    if missing_files:
        logger.error("Missing required files:")
        for missing in missing_files:
            logger.error(f"  - {missing}")
        return False
    
    logger.info(f"All required files found for dataset: {dataset_name}")
    return True


def validate_models(dataset_path: str, dataset_name: str) -> List[str]:
    """Validate and list available models in the dataset."""
    models_dir = os.path.join(dataset_path, dataset_name, 'models')
    models = []
    
    if not os.path.exists(models_dir):
        logger.error(f"Models directory not found: {models_dir}")
        return models
    
    # List all .ply files in the models directory
    for file in os.listdir(models_dir):
        if file.endswith('.ply'):
            models.append(file)
    
    if models:
        logger.info(f"Found {len(models)} model(s): {models}")
    else:
        logger.warning(f"No .ply model files found in {models_dir}")
    
    return models


def generate_training_data(
    bop_parent_path: str,
    dataset_name: str,
    cc_textures_path: str,
    output_dir: str,
    num_scenes: int = 2000,
    num_images_per_scene: int = 25,
    seed: Optional[int] = None
) -> bool:
    """
    Generate training data using BlenderProc.
    
    Args:
        bop_parent_path: Path to the BOP datasets parent directory
        dataset_name: Name of the specific dataset to use (e.g., 'hb', 'tless', 'ycbv')
        cc_textures_path: Path to downloaded CC textures
        output_dir: Path where final files will be saved
        num_scenes: Number of scenes to generate
        num_images_per_scene: Number of images per scene (not used in this script but for reference)
        seed: Random seed for reproducibility
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            logger.info(f"Set random seed to {seed}")
        
        # Validate required files
        dataset_path = bop_parent_path
        if not validate_required_files(dataset_path, dataset_name):
            return False
        
        # Validate models
        models = validate_models(dataset_path, dataset_name)
        if not models:
            logger.warning("No models found, but continuing anyway...")
        
        logger.info(f"Initializing BlenderProc...")
        bproc.init()
        
        # Load BOP objects for target dataset
        logger.info(f"Loading target BOP objects from {dataset_name}...")
        target_bop_objs = bproc.loader.load_bop_objs(
            bop_dataset_path=os.path.join(dataset_path, dataset_name),
            mm2m=True
        )
        logger.info(f"Loaded {len(target_bop_objs)} target objects")
        
        # Load distractor BOP objects (optional - only if they exist)
        distractor_datasets = ['tless', 'ycbv', 'tyol']
        all_distractor_objs = []
        
        for dist_dataset in distractor_datasets:
            dist_path = os.path.join(dataset_path, dist_dataset)
            if os.path.exists(dist_path):
                try:
                    if dist_dataset == 'tless':
                        dist_objs = bproc.loader.load_bop_objs(
                            bop_dataset_path=dist_path,
                            model_type='cad',
                            mm2m=True
                        )
                    else:
                        dist_objs = bproc.loader.load_bop_objs(
                            bop_dataset_path=dist_path,
                            mm2m=True
                        )
                    all_distractor_objs.extend(dist_objs)
                    logger.info(f"Loaded {len(dist_objs)} distractor objects from {dist_dataset}")
                except Exception as e:
                    logger.warning(f"Could not load {dist_dataset}: {str(e)}")
            else:
                logger.warning(f"Distractor dataset not found: {dist_path}")
        
        # Load BOP dataset intrinsics
        logger.info(f"Loading BOP intrinsics from {dataset_name}...")
        bproc.loader.load_bop_intrinsics(
            bop_dataset_path=os.path.join(dataset_path, dataset_name)
        )
        
        # Set shading and hide objects
        for obj in (target_bop_objs + all_distractor_objs):
            obj.set_shading_mode('auto')
            obj.hide(True)
        
        logger.info("Creating room environment...")
        # Create room
        room_planes = [
            bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
            bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]),
            bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]),
            bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]),
            bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0])
        ]
        
        # Create light plane
        light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
        light_plane.set_name('light_plane')
        light_plane_material = bproc.material.create('light_material')
        
        # Sample point light
        light_point = bproc.types.Light()
        light_point.set_energy(200)
        
        # Load CC textures
        logger.info(f"Loading CC textures from {cc_textures_path}...")
        if not os.path.exists(cc_textures_path):
            logger.warning(f"CC textures path not found: {cc_textures_path}")
            cc_textures = []
        else:
            cc_textures = bproc.loader.load_ccmaterials(cc_textures_path)
            logger.info(f"Loaded {len(cc_textures)} CC textures")
        
        # Define pose sampling function
        def sample_pose_func(obj: bproc.types.MeshObject):
            min_pos = np.random.uniform([-0.3, -0.3, 0.0], [-0.2, -0.2, 0.0])
            max_pos = np.random.uniform([0.2, 0.2, 0.4], [0.3, 0.3, 0.6])
            obj.set_location(np.random.uniform(min_pos, max_pos))
            obj.set_rotation_euler(bproc.sampler.uniformSO3())
        
        # Enable rendering options
        logger.info("Configuring renderer...")
        bproc.renderer.enable_depth_output(activate_antialiasing=False)
        bproc.renderer.set_max_amount_of_samples(50)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate scenes
        logger.info(f"Starting generation of {num_scenes} scenes...")
        for i in range(num_scenes):
            if i % 100 == 0:
                logger.info(f"Generating scene {i + 1}/{num_scenes}")
            
            # Sample target object
            sampled_target_objs = list(np.random.choice(target_bop_objs, size=1, replace=False))
            
            # Sample distractor objects
            sampled_distractor_objs = []
            if len(all_distractor_objs) > 0:
                num_distractors = min(3, len(all_distractor_objs))
                sampled_distractor_objs = list(
                    np.random.choice(all_distractor_objs, size=num_distractors, replace=False)
                )
            
            # Randomize materials
            for obj in (sampled_target_objs + sampled_distractor_objs):
                mat = obj.get_materials()[0] if obj.get_materials() else None
                if mat:
                    if obj.get_cp("bop_dataset_name") in ['itodd', 'tless']:
                        grey_col = np.random.uniform(0.1, 0.9)
                        mat.set_principled_shader_value("Base Color", [grey_col, grey_col, grey_col, 1])
                    mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
                    mat.set_principled_shader_value("Specular IOR Level", np.random.uniform(0, 1.0))
                obj.hide(False)
            
            # Sample light sources
            light_plane_material.make_emissive(
                emission_strength=np.random.uniform(3, 6),
                emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0])
            )
            light_plane.replace_materials(light_plane_material)
            light_point.set_color(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))
            location = bproc.sampler.shell(
                center=[0, 0, 0],
                radius_min=1,
                radius_max=1.5,
                elevation_min=5,
                elevation_max=89
            )
            light_point.set_location(location)
            
            # Apply textures to room planes
            if cc_textures:
                random_cc_texture = np.random.choice(cc_textures)
                for plane in room_planes:
                    plane.replace_materials(random_cc_texture)
            
            # Sample object poses
            bproc.object.sample_poses(
                objects_to_sample=sampled_target_objs + sampled_distractor_objs,
                sample_pose_func=sample_pose_func,
                max_tries=1000
            )
            
            # Define initial pose sampling
            def sample_initial_pose(obj: bproc.types.MeshObject):
                obj.set_location(bproc.sampler.upper_region(
                    objects_to_sample_on=room_planes[0:1],
                    min_height=1,
                    max_height=4,
                    face_sample_range=[0.4, 0.6]
                ))
                obj.set_rotation_euler(np.random.uniform([0, 0, 0], [0, 0, np.pi * 2]))
            
            # Sample poses on surface
            bproc.object.sample_poses_on_surface(
                objects_to_sample=sampled_target_objs + sampled_distractor_objs,
                surface=room_planes[0],
                sample_pose_func=sample_initial_pose,
                min_distance=0.01,
                max_distance=0.2
            )
            
            # Create BVH tree for camera checks
            bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(
                sampled_target_objs + sampled_distractor_objs
            )
            
            # Sample camera poses
            cam_poses = 0
            max_camera_samples = 100
            
            while cam_poses < max_camera_samples:
                location = bproc.sampler.shell(
                    center=[0, 0, 0],
                    radius_min=0.35,
                    radius_max=1.5,
                    elevation_min=5,
                    elevation_max=89
                )
                
                poi = bproc.object.compute_poi(
                    np.random.choice(sampled_target_objs, size=1, replace=False)
                )
                
                rotation_matrix = bproc.camera.rotation_from_forward_vec(
                    poi - location,
                    inplane_rot=np.random.uniform(-0.7854, 0.7854)
                )
                
                cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
                
                if bproc.camera.perform_obstacle_in_view_check(
                    cam2world_matrix,
                    {"min": 0.3},
                    bop_bvh_tree
                ):
                    bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)
                    cam_poses += 1
            
            # Render
            data = bproc.renderer.render()
            
            # Write BOP format output
            bproc.writer.write_bop(
                os.path.join(output_dir, 'bop_data'),
                target_objects=sampled_target_objs,
                dataset=dataset_name,
                depth_scale=0.1,
                depths=data["depth"],
                colors=data["colors"],
                color_file_format="JPEG",
                ignore_dist_thres=10
            )
            
            # Hide objects for next iteration
            for obj in (sampled_target_objs + sampled_distractor_objs):
                obj.hide(True)
        
        logger.info(f"Successfully generated {num_scenes} scenes!")
        return True
        
    except Exception as e:
        logger.error(f"Error during data generation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Generate annotated training data using BlenderProc'
    )
    parser.add_argument('bop_parent_path', help="Path to the BOP datasets parent directory")
    parser.add_argument('dataset_name', help="Name of the dataset (e.g., 'hb', 'tless', 'ycbv')")
    parser.add_argument('cc_textures_path', nargs='?', default="resources/cctextures",
                       help="Path to downloaded CC textures (default: resources/cctextures)")
    parser.add_argument('output_dir', help="Path where the final files will be saved")
    parser.add_argument('--num_scenes', type=int, default=2000,
                       help="Number of scenes to generate (default: 2000)")
    parser.add_argument('--num_images_per_scene', type=int, default=25,
                       help="Number of images per scene (default: 25)")
    parser.add_argument('--seed', type=int, default=None,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    success = generate_training_data(
        bop_parent_path=args.bop_parent_path,
        dataset_name=args.dataset_name,
        cc_textures_path=args.cc_textures_path,
        output_dir=args.output_dir,
        num_scenes=args.num_scenes,
        num_images_per_scene=args.num_images_per_scene,
        seed=args.seed
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
