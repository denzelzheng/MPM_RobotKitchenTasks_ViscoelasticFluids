import re
import yourdfpy
import json
import numpy as np
import trimesh
from icecream import ic
from yourdfpy import URDF
from config import shadow_urdf_path, points_data_path
from vedo import show

shadow_hand: URDF = URDF.load(shadow_urdf_path)
shadow_joint_names = shadow_hand.actuated_joint_names
lower_config = {k: getattr(v.limit, "lower", 0) for k, v in shadow_hand.joint_map.items()}
upper_config = {k: getattr(v.limit, "upper", 0) for k, v in shadow_hand.joint_map.items()}
shadow_hand.update_cfg(lower_config)
shadow_hand.show()
shadow_hand.update_cfg(upper_config)
shadow_hand.show()

point_index_reg = re.compile(r'\((\d+),\s(\d+)\)')
with open(points_data_path, 'r') as f:
    points_info = json.load(f)

link_points_dict = {k: [] for k in shadow_hand.link_map.keys()}
for k, v in points_info.items():
    if k.startswith('$'):
        continue
    pind = point_index_reg.match(k).groups()
    pind = (int(pind[0]), int(pind[1]))
    link_name = v['Item2']
    pos = v['Item3']
    pos = np.asarray([pos['z'], -pos['x'], pos['y']])
    link_points_dict[link_name].append(pos)

scene: trimesh.Scene = shadow_hand._scene.copy()

for k, v in link_points_dict.items():
    if not len(v):
        continue
    k += '.stl'
    pcd = trimesh.PointCloud(np.stack(v, axis=0))
    scene.add_geometry(pcd, node_name=k + '_pcd', parent_node_name=k)

scene.show()
meshes = scene.dump(False)
show(meshes)
for geom in meshes:
    ic(geom.vertices.max(axis=0), geom.vertices.min(axis=0))
for _, link in shadow_hand.link_map.items():
    ic(_, link.visuals[0].geometry.mesh.filename, scene.graph.get(_)[0])
