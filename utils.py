import taichi as ti
import trimesh
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as R
from icecream import ic
import igl

T = ti.f32
vec3 = ti.types.vector(3, T)
mat3 = ti.types.matrix(3, 3, dtype=T)
mat4 = ti.types.matrix(4, 4, T)

vecs = ti.Vector.field
mats = ti.Matrix.field
scalars = ti.field


def trimesh_show(geoms):
    scene = trimesh.Scene()
    for g in geoms:
        scene.add_geometry(g)
    scene.show()


def transform_pos(pos: np.ndarray, transform: np.ndarray):
    assert (len(pos.shape) <= 2)
    assert (pos.shape[-1] == 3)
    assert (transform.shape == (4, 4))
    return pos @ transform[:3, :3].T + transform[:3, 3]


@ti.func
def inv_quat_ti(quat):
    return ti.Vector([quat[0], -quat[1], -quat[2], -quat[3]]).normalized()

@ti.func
def transform_pos_ti(pos: vec3, transform: mat4):
    _pos = ti.Vector([pos[0], pos[1], pos[2], 1], T)
    return (transform @ _pos)[:3]


@ti.func
def rotate_pos_ti(pos, quat):
    qvec = ti.Vector([quat[1], quat[2], quat[3]])
    uv = qvec.cross(pos)
    uuv = qvec.cross(uv)
    return pos + 2 * (quat[0] * uv + uuv)


@ti.func 
def transform3_pos_ti(pos, trans, quat):
    return rotate_pos_ti(pos, quat) + trans


@ti.func
def inv_transform3_pos_ti(pos, trans, quat):
    return rotate_pos_ti(pos - trans, inv_quat_ti(quat))


left_tf = np.empty((4, 4), float)
right_tf = np.empty((4, 4), float)
left_tf[:3, :3] = R.from_euler('z', -np.pi/2).as_matrix()
right_tf[:3, :3] = R.from_euler('z', np.pi/2).as_matrix()
left_tf[:3, 3] = right_tf[:3, 3] = left_tf[3, :3] = right_tf[3, :3] = 0
left_tf[3, 3] = right_tf[3, 3] = 1.


def fix_unity_urdf_tf(tf):
    return left_tf @ tf @ right_tf


class TetMesh:
    def __init__(self, verts: np.ndarray,
                 surf_tris: np.ndarray,
                 surf_vert_inds: np.ndarray, 
                 tets: np.ndarray) -> None:
        self.verts = verts
        self.surf_tris = surf_tris
        self.surf_vert_inds = surf_vert_inds
        self.tets = tets
        self.surf_mesh = trimesh.Trimesh(self.verts, self.surf_tris)

    @property
    def n_verts(self):
        return self.verts.shape[0]

    @property
    def n_tets(self):
        return self.tets.shape[0]

    @property
    def show(self):
        trimesh_show([self.surf_mesh])


def read_tet(path):
    tet_mesh = pv.read(path)
    points = np.asarray(tet_mesh.points)
    tets = np.asarray(tet_mesh.cells_dict[10])

    n_points = tet_mesh.points.shape[0]
    tet_mesh['orig_inds'] = np.arange(n_points, dtype=np.int32)
    surf_mesh = tet_mesh.extract_surface()
    surf_tris = np.asarray(surf_mesh.faces).reshape(-1, 4)[:, 1:]
    surf_tris = np.asarray(surf_mesh['orig_inds'][surf_tris])
    surf_vert_inds = np.asarray(surf_mesh['orig_inds'])

    return TetMesh(points, surf_tris, surf_vert_inds, tets)


def interpolate_from_mesh(query_points: np.ndarray,
                          mesh: trimesh.Trimesh,
                          mesh_value: np.ndarray,
                          dist_thresh=0.005,
                          default_value=0):
    assert (len(query_points.shape) == 2)
    assert (query_points.shape[1] == 3)
    if len(mesh_value.shape) == 1:
        mesh_value = mesh_value[:, None]
    closest, distance, tri_inds = trimesh.proximity.closest_point(
        mesh, query_points)
    tri_vert_inds = mesh.faces[tri_inds]
    tri_verts = mesh.vertices[tri_vert_inds]
    bc_weights = trimesh.triangles.points_to_barycentric(
        tri_verts, closest)
    query_value = (mesh_value[tri_vert_inds] *
                   bc_weights[:, :, None]).sum(axis=1)
    query_value[distance > dist_thresh] = default_value
    return query_value


@ti.func 
def normalize_ti(v, eps=1e-14):
    return v / (v.norm() + eps)


@ti.data_oriented
class SDF:
    def __init__(self, voxels: np.ndarray,
                 T_mesh2vox: np.ndarray):
        self.np_voxels = voxels
        self.np_T_mesh2vox = T_mesh2vox
        self.res = voxels.shape[0]
        self.voxels = scalars(T, shape=self.np_voxels.shape)
        self.T_mesh2vox = mats(4, 4, T, shape=())

        self.voxels.from_numpy(self.np_voxels)
        self.T_mesh2vox.from_numpy(self.np_T_mesh2vox)


    @ti.func
    def transform_pos_mesh_to_voxels(self, pos_mesh):
        return transform_pos_ti(pos_mesh, self.T_mesh2vox[None])


    @ti.func 
    def value(self, pos_in, meshSpace=True):
        pos_voxels = self.transform_pos_mesh_to_voxels(pos_in) if meshSpace else pos_in
        # sdf value from voxels coordinate
        base = ti.floor(pos_voxels, ti.i32)
        signed_dist = ti.cast(0.0, T)
        if (base >= self.res - 1).any() or (base < 0).any():
            signed_dist = 1.0
        else:
            signed_dist = 0.0
            for offset in ti.static(ti.grouped(ti.ndrange(2, 2, 2))):
                voxel_pos = base + offset
                w_xyz = 1 - ti.abs(pos_voxels - voxel_pos)
                w = w_xyz[0] * w_xyz[1] * w_xyz[2]
                signed_dist += w * self.voxels[voxel_pos]

        return signed_dist
    
    @ti.func
    def normal(self, pos_in, meshSpace=True):
        pos_voxels = self.transform_pos_mesh_to_voxels(pos_in) if meshSpace else pos_in
        # since we are in voxels frame, delta can be a relatively big value
        delta = ti.cast(1e-2, T)
        normal_vec = ti.Vector([0, 0, 0], dt=T)

        for i in ti.static(range(3)):
            inc = pos_voxels
            dec = pos_voxels
            inc[i] += delta
            dec[i] -= delta
            normal_vec[i] = (self.value(inc, False) - self.value(dec, False)) / (2 * delta)

        if meshSpace:
            R_voxels_to_mesh = self.T_mesh2vox[None][:3, :3].inverse()
            normal_vec = R_voxels_to_mesh @ normal_vec

        normal_vec = normalize_ti(normal_vec)
        return normal_vec


def sdf_from_mesh(mesh: trimesh.Trimesh,
                  res=128):
    # if is list of Trimesh
    if isinstance(mesh, trimesh.Scene):
        meshes = mesh.dump()
        mesh = meshes.sum()
    else:
        meshes = [mesh]
    aabb = np.stack([mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)])
    voxels_center = aabb.mean(axis=0)
    voxels_radius = (aabb[1] - aabb[0]).max() / 2 * 1.2 # pick longest axis
    x = np.linspace(-voxels_radius, voxels_radius, res)
    y = np.linspace(-voxels_radius, voxels_radius, res)
    z = np.linspace(-voxels_radius, voxels_radius, res)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    query_points = np.stack([X, Y, Z], axis=-1).reshape((-1, 3)) + voxels_center

    voxels = np.stack([igl.signed_distance(query_points, m.vertices, m.faces)[0] for m in meshes])
    voxels = voxels.min(axis=0)

    voxels = voxels.reshape([res, res, res])

    T_mesh2vox = np.eye(4)
    scale = (res - 1) / (voxels_radius * 2)
    T_mesh2vox[:3, :3] *= scale
    T_mesh2vox[:3, 3] = (res - 1) / 2 - voxels_center * scale

    return SDF(voxels, T_mesh2vox)
