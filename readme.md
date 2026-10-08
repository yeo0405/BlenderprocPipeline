# 6 DOF Training Pipeline

## Start Genrated

```bash
cd WebUI
python run.py
```

### Step 1: Create Dataset & Upload Models

**.blend Model Files**

- must be name as obj_000001, obj_000002 ...

### Step 2: Genenrate Training Data

- Number of Scenes
- Camera Poses Per Scene 

if number of scenes and camera poses per scene both value 10, the generated data will be 100 (10 * 10)

### bash
```
box=bpy.context.scene.objects["Box 1_v1"]; obj3=bpy.context.scene.objects["IDG_PART_MESH_Box 1_v1.002"]; r=box.matrix_world.inverted()@obj3.matrix_world; rr=r.to_euler("XYZ"); import math; print("BOX =",box.name); print("OBJ3 =",obj3.name); print(f"OBJ3_RELATIVE_LOCATION = [{r.translation.x:.8f}, {r.translation.y:.8f}, {r.translation.z:.8f}]"); print(f"OBJ3_RELATIVE_ROTATION_DEG = [{math.degrees(rr.x):.8f}, {math.degrees(rr.y):.8f}, {math.degrees(rr.z):.8f}]")
```