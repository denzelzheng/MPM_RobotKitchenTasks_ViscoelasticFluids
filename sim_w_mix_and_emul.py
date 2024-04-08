from os.path import join as pjoin
import time
import trimesh
import taichi as ti
import numpy as np
from utils import mat3, scalars, vecs, mats, T, TetMesh, sdf_from_mesh, normalize_ti,\
                  transform3_pos_ti, inv_transform3_pos_ti, rotate_pos_ti
from typing import Optional, List
from vedo import show
from icecream import ic
import igl
# current: only neo-hookean + rigid body


@ti.func
def compute_P_hat(sig, mu, lam):
    epsilon = ti.Vector([ti.log(sig[0, 0]), ti.log(sig[1, 1]), ti.log(sig[2, 2])])
    sum_log = epsilon[0] + epsilon[1] + epsilon[2]
    psi_0 = (2 * mu * epsilon[0] + lam * sum_log) / sig[0, 0]
    psi_1 = (2 * mu * epsilon[1] + lam * sum_log) / sig[1, 1]
    psi_2 = (2 * mu * epsilon[2] + lam * sum_log) / sig[2, 2]
    P_hat =  ti.Vector([psi_0, psi_1, psi_2])
    return P_hat



@ti.func
def compute_emulsion_viscosity(viscosity, emul, phi_w=0.5, phi_m=0.64, viscosity_intrinsic=2.5, lambda_decay=2):

    # Original fluid phase viscosity
    viscosity_o = (1 - phi_w * emul) * viscosity * ti.exp(-lambda_decay * emul)
    
    # Suspension viscosity
    viscosity_s = viscosity * (1 - (phi_w * emul) / phi_m)**(-viscosity_intrinsic * phi_m)
    
    # Emulsion viscosity
    viscosity_e = (1 - phi_w * emul) * viscosity_o + (phi_w * emul) * viscosity_s
    
    # if(emul == 1):
    #     print(viscosity_e)
    return viscosity_e



@ti.data_oriented
class Material:
    pass

    @ti.func
    def compute_kirchhoff_stress(self, F, dt, C, emul):
        raise NotImplementedError




# NOTE: for rigid bodies, now only support kinematic movements


class NeoHookean(Material):
    def __init__(self, E: float = 5e3, nu: float = 0.2, fluid: bool = False) -> None:
        super().__init__()
        self.mu, self.lam = E / (2 * (1 + nu)), E * \
            nu / ((1+nu) * (1 - 2 * nu))
        self.fluid = fluid
        
    @ti.func
    def compute_kirchhoff_stress(self, F, dt, C, emul):
        U, sig, V = ti.svd(F)
        J = F.determinant()
        if self.fluid:
            F = ti.Matrix.identity(float, 3) * ti.pow(J, 1/3)
        kirchhoff_stress = 2 * self.mu * (F - U @ V.transpose()) @ \
                    F.transpose() + ti.Matrix.identity(float, 3) * \
                    self.lam * J * (J - 1)
        return kirchhoff_stress, F



class StVK_with_Hecky_strain(Material):
    def __init__(self, E: float = 5e3, nu: float = 0.2, fluid: bool = False) -> None:
        super().__init__()
        self.mu, self.lam = E / (2 * (1 + nu)), E * \
            nu / ((1+nu) * (1 - 2 * nu))
        self.fluid = fluid
    
    @ti.func
    def compute_kirchhoff_stress(self, F, dt, C, emul):
        
        J = F.determinant()
        if self.fluid:
            F = ti.Matrix.identity(float, 3) * ti.pow(J, 1/3)
        U, sig, V = ti.svd(F)
        P_hat = compute_P_hat(sig, self.mu, self.lam)
        P = U @ ti.Matrix([[P_hat[0], 0.0, 0.0], [0.0, P_hat[1], 0.0], [0.0, 0.0, P_hat[2]]]) @ V.transpose()
        kirchhoff_stress = P @ F.transpose()
        return kirchhoff_stress, F




class visco_StVK_with_Hecky_strain(Material):
    def __init__(self, E: float = 5e3, nu: float = 0.2, viscosity_v: float = 1, fluid: bool = False) -> None:
        super().__init__()
        self.mu, self.lam = E / (2 * (1 + nu)), E * \
            nu / ((1+nu) * (1 - 2 * nu))
        self.viscosity_v = viscosity_v
        self.viscosity_d = viscosity_v  # 假设体积粘度与偏差粘度相同
        self.fluid = fluid
    
    @ti.func
    def compute_kirchhoff_stress(self, F, dt, C, emul):

        new_viscosity_v = self.viscosity_v
        new_viscosity_d = self.viscosity_d

        J = F.determinant()
        
        U, sig, V = ti.svd(F)
        epsilon = ti.Vector([ti.log(sig[0, 0]), ti.log(sig[1, 1]), ti.log(sig[2, 2])])
        alpha = 2.0 * self.mu / new_viscosity_d
        beta = 2.0 * (2.0 * self.mu + self.lam * 3) / (9.0 * new_viscosity_v) - 2.0 * self.mu / (new_viscosity_d * 3)
        A = 1 / (1 + dt * alpha)
        B = dt * beta / (1 + dt * (alpha + 3 * beta))
        epsilon_trace = ti.log(sig[0, 0]) + ti.log(sig[1, 1]) + ti.log(sig[2, 2])
        temp_epsilon = A * (epsilon - ti.Vector([B * epsilon_trace, B * epsilon_trace, B * epsilon_trace]) )  
        d = ti.exp(temp_epsilon)
        new_sig = ti.Matrix([[d[0], 0.0, 0.0], [0.0, d[1], 0.0], [0.0, 0.0, d[2]]])
        new_F = U @ new_sig @ V.transpose()
        P_hat = compute_P_hat(new_sig, self.mu, self.lam)
        P = U @ ti.Matrix([[P_hat[0], 0.0, 0.0], [0.0, P_hat[1], 0.0], [0.0, 0.0, P_hat[2]]]) @ V.transpose()
        kirchhoff_stress_visco = P @ new_F.transpose()

        if self.fluid:
            F = ti.Matrix.identity(float, 3) * ti.pow(J, 1/3)
        
        return kirchhoff_stress_visco, new_F



class cross_visco_StVK_with_Hecky_strain(Material):
    def __init__(self, E: float = 5e3, nu: float = 0.2, viscosity_v: float = 1, 
                 viscosity_inf: float = 0, K: float = 0.1, m: float = 1, fluid: bool = False) -> None:
        super().__init__()
        self.mu, self.lam = E / (2 * (1 + nu)), E * \
            nu / ((1+nu) * (1 - 2 * nu))
        self.viscosity_v = viscosity_v
        self.viscosity_d = viscosity_v # 假设体积粘度与偏差粘度相同
        self.fluid = fluid
        self.K = K # 时间常数的倒数
        self.m = m # 控制剪切变稀强度的参数
        self.viscosity_v_inf = viscosity_inf # 无限剪切速率下的粘度
        self.viscosity_d_inf = viscosity_inf # 假设体积粘度与偏差粘度相同
        self.viscosity_inf = viscosity_inf 

    @ti.func
    def compute_kirchhoff_stress(self, F, dt, C, emul):
        
        J = F.determinant()
        if self.fluid:
            F = ti.Matrix.identity(float, 3) * ti.pow(J, 1/3)

        U, sig, V = ti.svd(F)
        D = (C + C.transpose()) / 2.0
        shear_rate = ti.sqrt(2.0 * (D[0, 1] ** 2 + D[0, 2] ** 2 + D[1, 2] ** 2))



        viscosity_d_0 = compute_emulsion_viscosity(self.viscosity_d, emul)
        viscosity_v_0 = self.viscosity_v
        new_viscosity_d = self.viscosity_d_inf + (viscosity_d_0 - self.viscosity_d_inf) / (1.0 + ti.pow(self.K * shear_rate, self.m))
        new_viscosity_v = self.viscosity_d_inf + (viscosity_v_0 - self.viscosity_d_inf) / (1.0 + ti.pow(self.K * shear_rate, self.m))





        epsilon = ti.Vector([ti.log(sig[0, 0]), ti.log(sig[1, 1]), ti.log(sig[2, 2])])
        alpha = 2.0 * self.mu / new_viscosity_d
        beta = 2.0 * (2.0 * self.mu + self.lam * 3) / (9.0 * new_viscosity_v) - 2.0 * self.mu / (new_viscosity_d * 3)
        A = 1 / (1 + dt * alpha)
        B = dt * beta / (1 + dt * (alpha + 3 * beta))
        epsilon_trace = ti.log(sig[0, 0]) + ti.log(sig[1, 1]) + ti.log(sig[2, 2])
        temp_epsilon = A * (epsilon - ti.Vector([B * epsilon_trace, B * epsilon_trace, B * epsilon_trace]) )  
        d = ti.exp(temp_epsilon)
        new_sig = ti.Matrix([[d[0], 0.0, 0.0], [0.0, d[1], 0.0], [0.0, 0.0, d[2]]])
        new_F = U @ new_sig @ V.transpose()
        P_hat = compute_P_hat(new_sig, self.mu, self.lam)
        P = U @ ti.Matrix([[P_hat[0], 0.0, 0.0], [0.0, P_hat[1], 0.0], [0.0, 0.0, P_hat[2]]]) @ V.transpose()
        kirchhoff_stress_visco = P @ new_F.transpose()
        
        return kirchhoff_stress_visco, new_F


class Boundary:
    def __init__(self, 
                 mesh: trimesh.Trimesh, 
                 friction: float = 0.5,
                 sdf_res: int = 128) -> None:
        self.sdf = sdf_from_mesh(mesh, sdf_res)

        if isinstance(mesh, trimesh.Scene):
            meshes = mesh.dump()
            mesh = trimesh.util.concatenate(meshes)

        self.n_vertices = mesh.vertices.shape[0]
        self.n_faces = mesh.faces.shape[0]
        self.init_vertices = vecs(3, T, shape=(self.n_vertices))
        self.vertices = vecs(3, T, shape=(self.n_vertices))
        self.faces = scalars(ti.i32, shape=(self.n_faces * 3))
        self.init_vertices.from_numpy(mesh.vertices)
        self.vertices.copy_from(self.init_vertices)
        self.faces.from_numpy(mesh.faces.flatten())

        self.friction: float = friction


@ti.data_oriented
class StaticBoundary(Boundary):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @ti.func 
    def signed_dist(self, pos_world): # pos_mesh = pos_world
        return self.sdf.value(pos_world)
    
    @ti.func
    def normal(self, pos_world): # pos_mesh = pos_world
        return self.sdf.normal(pos_world)
    
    @ti.func
    def collide(self, 
                pos_world, 
                vel):
        signed_dist = self.signed_dist(pos_world)
        if signed_dist <= 0:
            normal_vec = self.normal(pos_world)

            # v w.r.t collider
            rel_v = vel
            normal_component = rel_v.dot(normal_vec)

            # remove inward velocity, if any
            rel_v_t = rel_v - ti.min(normal_component, 0) * normal_vec
            rel_v_t_norm = rel_v_t.norm()

            # tangential component after friction (if friction exists)
            rel_v_t_friction = rel_v_t / rel_v_t_norm * ti.max(0, rel_v_t_norm + normal_component * self.friction)

            # tangential component after friction
            flag = ti.cast(normal_component < 0 and rel_v_t_norm > 1e-30, T)
            rel_v_t = rel_v_t_friction * flag + rel_v_t * (1 - flag)
            vel = rel_v_t

        return vel



@ti.data_oriented
class DynamicBoundary(Boundary):
    def __init__(self, 
                 collide_type: str = 'particle',
                 **kwargs) -> None:
            super().__init__(**kwargs)
            self.collide_type = collide_type

            self.pos = vecs(3, T, shape=())
            self.quat = vecs(4, T, shape=())
            self.target_pos = vecs(3, T, shape=())
            self.target_quat = vecs(4, T, shape=())
            self.force = vecs(3, T, shape=())
            self.work = scalars(T, shape=())

            self.pos.from_numpy(np.array([0, 0, 0]))
            self.quat.from_numpy(np.array([1, 0, 0, 0]))
            self.target_pos.copy_from(self.pos)
            self.target_quat.copy_from(self.quat)

    def toward_target(self):
        self.pos.copy_from(self.target_pos)
        self.quat.copy_from(self.target_quat)
        self.update_vertices()

    @ti.kernel
    def update_vertices(self):
        # print("updating vertices...") 
        # print("current pos: ", self.pos[None][0], self.pos[None][1], self.pos[None][2])
        # print("current quat: ", self.quat[None][0], self.quat[None][1], self.quat[None][2], self.quat[None][3])
        for i in self.vertices:
            self.vertices[i] = ti.cast(
                transform3_pos_ti(self.init_vertices[i], self.pos[None], self.quat[None]), 
                self.vertices.dtype)

    @ti.func
    def collider_v(self, pos_world, dt):
        pos_mesh = inv_transform3_pos_ti(pos_world, self.pos[None], self.quat[None])
        pos_world_new = transform3_pos_ti(pos_mesh, self.target_pos[None], self.target_quat[None])
        collider_v = (pos_world_new - pos_world) / dt
        return collider_v
    
    @ti.func
    def transform_pos_from_world_to_mesh(self, pos_world):
        return inv_transform3_pos_ti(pos_world, self.pos[None], self.quat[None])
    
    @ti.func
    def transform_pos_from_mesh_to_world(self, pos_mesh):
        return transform3_pos_ti(pos_mesh, self.pos[None], self.quat[None])
    
    @ti.func
    def signed_dist(self, pos_world):
        pos_mesh = self.transform_pos_from_world_to_mesh(pos_world)
        return self.sdf.value(pos_mesh)
    
    @ti.func
    def normal(self, pos_world):
        pos_mesh = self.transform_pos_from_world_to_mesh(pos_world)
        normal_mesh = self.sdf.normal(pos_mesh)
        return rotate_pos_ti(normal_mesh, self.quat[None])

    @ti.func
    def collide(self, pos_world, vel, dt, mass):
        # if ti.static(self.has_dynamics):
        if ti.static(True):
            signed_dist = self.signed_dist(pos_world)
            if signed_dist <= 0:
                vel_in = vel
                collider_v = self.collider_v(pos_world, dt)

                if ti.static(self.friction > 10.0):
                    vel = collider_v
                else:
                    # v w.r.t collider
                    rel_v = vel - collider_v
                    normal_vec = self.normal(pos_world)
                    normal_component = rel_v.dot(normal_vec)

                    # remove inward velocity, if any
                    rel_v_t = rel_v - ti.min(normal_component, 0) * normal_vec
                    rel_v_t_norm = rel_v_t.norm()

                    # tangential component after friction (if friction exists)
                    rel_v_t_friction = rel_v_t / rel_v_t_norm * ti.max(0, rel_v_t_norm + normal_component * self.friction)

                    # tangential component after friction
                    flag = ti.cast(normal_component < 0 and rel_v_t_norm > 1e-30, T)
                    rel_v_t = rel_v_t_friction * flag + rel_v_t * (1 - flag)
                    vel = collider_v + rel_v_t
                    
                # compute force (impulse)
                force = -(vel - vel_in) * mass
                self.force[None] += force.norm(1e-14)
                self.work[None] += ti.math.dot(-force, collider_v)

        return vel
    
    def set_target(self, target_pos: np.ndarray, target_quat: np.ndarray):
        self.target_pos.from_numpy(target_pos)
        self.target_quat.from_numpy(target_quat)


class Body:
    def __init__(self) -> None:
        pass

    @property
    def n_pars(self):
        pass


class SoftBody(Body):
    def __init__(self, rest_pars_pos: np.ndarray, material: Material, color: np.ndarray, 
                 emulsification_efficiency: float, emulsifier_capacity: float, density: float) -> None:
        self.rest_pos: np.ndarray = rest_pars_pos
        self.material: Material = material
        self.color: np.ndarray = color
        self.emulsification_efficiency: float = emulsification_efficiency
        self.emulsifier_capacity: float = emulsifier_capacity
        self.rho: float = density


    @property
    def n_pars(self):
        return self.rest_pos.shape[0]
    

class RigidBody(Body):
    def __init__(self, mesh: trimesh.Trimesh, dx=1/128) -> None:
        self.mesh: trimesh.Trimesh = mesh.copy()
        n_points = int(3 * mesh.area / dx ** 2)
        points, face_inds = trimesh.sample.sample_surface_even(
            self.mesh, n_points)
        self.sample_faces_verts = self.mesh.vertices[self.mesh.faces[face_inds]]
        self.sample_bc_weights = trimesh.triangles.points_to_barycentric(
            self.sample_faces_verts, points)
        self.rest_pos: np.ndarray = points
        self.tri_inds: np.ndarray = face_inds
        self.target_pos = self.rest_pos

    @property
    def n_pars(self):
        return self.rest_pos.shape[0]

    @property
    def n_tris(self):
        return self.mesh.faces.shape[0]

    def set_target(self, target_vert_pos: np.ndarray):
        target_faces_verts = target_vert_pos[self.mesh.faces[self.tri_inds]]
        self.target_pos = (self.sample_bc_weights[:, :, None] *
                           target_faces_verts).sum(axis=1)


# TODO: CPIC
# NOTE: ignore gravity for now
@ti.data_oriented
class MpmSim:
    def __init__(self, dt: float = 1e-4,
                 origin: np.ndarray = np.zeros((3,), float),
                 gravity: np.ndarray = np.array([0, -9.8, 0], float),
                 box_bound_rel: float = 0.1,
                 ground_friction: float = 0.1,
                 n_grids: int = 128,
                 scale: float = 1.0) -> None:
        self.dt = dt
        self.n_grids = n_grids
        self.box_bound = max(int(n_grids * box_bound_rel), 0)
        self.scale = scale
        self.dx = 1. / self.n_grids
        self.inv_dx = self.n_grids
        self.origin = origin
        self.gravity = ti.Vector(gravity)
        self.ground_friction = ground_friction
        self.reset()

        # TODO: align these parameters in the future
        self.default_p_vol = (self.dx * 0.5)**2
        self.rp_rho = 1e2
        self.rp_vol = self.default_p_vol
        self.rp_mass = self.rp_vol * self.rp_rho

        self.coloring_mixing_alpha = 0.6
        self.p_c_L1_distance_criterion = 0.2  # assess phases mixing uniformity
        self.uniform_coloring_mixing_alpha = 3.0
        self.emulsified_droplets_vol_ratio  = 1

        # max_alpha = 3e-4
        self.alpha_for_emul_rate = 5e-3
        self.critical_concentration = 0.03
        # self.emul_rate_constant0 = 1e6
        # self.emul_rate_constant2 = 0.001
        # self.emul_rate_constant1 = (self.emul_rate_constant0 - (1 / self.alpha_for_emul_rate)) / \
        #     ti.log(self.critical_concentration * self.emul_rate_constant2 + 1.0)

    @property
    def n_static_bounds(self):
        return len(self.static_bounds)
    
    @property
    def n_dynamic_bounds(self):
        return len(self.dynamic_bounds)
    
    @property
    def n_rigid(self):
        return len(self.rigid_bodies)

    def reset(self):
        self.window = ti.ui.Window("CPIC-Scene", (768, 768))
        self.canvas = self.window.get_canvas()
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.Camera()
        self.camera.position(2, 4, 0)
        self.camera.lookat(0.5, 0.5, 0.5)

        self.static_bounds = []
        self.dynamic_bounds = []
        self.rigid_bodies = []
        self.rigid_tris_inds = []
        self.rigid_tris_vinds = []
        self.rigid_pars_offsets = []
        self.rigid_sizes = []
        self.deformable_bodies = []
        self.materials = []
        self.body_pars = []
        self.body_colors = []
        self.body_emulsifier_capacities = []
        self.body_emulsification_efficiencies = []
        self.body_rhos = []

        # fileds
        self.x: Optional[vecs] = None
        self.emul: Optional[scalars] = None   # degree of emulsification
        self.x_body_id: Optional[scalars] = None
        self.x_color: Optional[vecs] = None
        self.v: Optional[vecs] = None
        self.p_rho: Optional[scalars] = None
        self.p_vol: Optional[scalars] = None
        self.C: Optional[mats] = None
        self.F: Optional[mats] = None
        self.p_c: Optional[vecs] = None     # phase_counts for phase_concentrations
        self.p_c_global: Optional[scalars] = None
        self.e_c: Optional[scalars] = None   # emulsifier_capacity
        self.e_e: Optional[scalars] = None   # emulsification_efficiency
        self.Jp: Optional[scalars] = None
        self.grid_v: Optional[vecs] = None
        self.grid_m: Optional[scalars] = None
        self.grid_p_c: Optional[scalars] = None
        self.grid_c: Optional[vecs] = None    # grid_color



        self.n_soft_pars: int = 0
        self.n_rigid_tris: int = 0
        self.n_rigid_pars: int = 0

        self.n_lag_verts: int = 0
        self.n_lag_tris: int = 0

        self.x_rp: Optional[vecs] = None
        self.v_rp: Optional[vecs] = None  # for the current naive coupling
        # not using the true-rigid-coupling for now
        # self.x_rt: Optional[vecs] = None
        # self.x_rp2t: Optional[vecs] = None

        self.x_lag: Optional[vecs] = None
        self.v_lag: Optional[vecs] = None
        self.C_lag: Optional[mats] = None
        self.restInvT_lag: Optional[mats] = None
        self.energy_lag: Optional[scalars] = None
        self.tris_lag: Optional[scalars] = None
        self.nrm_lag: Optional[vecs] = None
        self.tris_area_lag: Optional[scalars] = None
        self.tris_lag_expanded: Optional[scalars] = None

    def init_system(self):

        if self.n_lag_verts:
            self.x_lag = vecs(3, T, self.n_lag_verts, needs_grad=True)
            self.v_lag = vecs(3, T, self.n_lag_verts)
            self.C_lag = mats(3, 3, T, self.n_lag_verts)
            self.restInvT_lag = mats(2, 2, T, self.n_lag_tris)
            self.energy_lag = scalars(T, shape=(), needs_grad=True)
            self.tris_lag = scalars(int, (self.n_lag_tris, 3))
            self.nrm_lag = vecs(3, T, self.n_lag_tris)
            self.tris_area_lag = scalars(float, (self.n_lag_tris,))
            self.tris_lag_expanded = scalars(int, self.n_lag_tris * 3)

            self.x_lag.from_numpy(np.asarray(
                self.lag_mesh.vertices) - self.origin)
            self.tris_lag.from_numpy(np.asarray(self.lag_mesh.faces))





        if self.n_soft_pars:   
            self.n_phases = len(self.deformable_bodies)
            self.x = vecs(3, T, self.n_soft_pars)
            self.emul = scalars(T, self.n_soft_pars)
            self.x_color = vecs(3, T, self.n_soft_pars)
            self.p_c = vecs(self.n_phases, T, self.n_soft_pars)
            self.p_c_global = vecs(self.n_phases, T, ())
            self.v = vecs(3, T, self.n_soft_pars)
            self.C = mats(3, 3, T, self.n_soft_pars)
            self.F = mats(3, 3, T, self.n_soft_pars)



            self.body_x = {
                i: ti.Vector.field(3, dtype=T, shape=self.body_pars[i])
                for i in range(self.n_phases)
            }

            self.body_x_color = {
                i: ti.Vector.field(3, dtype=T, shape=self.body_pars[i])
                for i in range(self.n_phases)
            }

            self.x_body_id = scalars(ti.i32, self.n_soft_pars)
            self.Jp = scalars(T, self.n_soft_pars)
            self.p_vol = scalars(T, self.n_soft_pars)
            self.p_rho = scalars(T, self.n_soft_pars)
            self.e_c = scalars(T, self.n_phases)
            self.e_e = scalars(T, self.n_phases)
            np_x = np.concatenate(
                [b.rest_pos for b in self.deformable_bodies], axis=0) 
            self.x.from_numpy(np_x - self.origin)

            np_body_id = np.concatenate([np.full(pars, i) for i, pars in enumerate(self.body_pars)])
            self.x_body_id.from_numpy(np_body_id)
            
            self.e_e.from_numpy(np.array(self.body_emulsification_efficiencies))
            self.e_c.from_numpy(np.array(self.body_emulsifier_capacities))


            np_colors = np.concatenate([np.tile(color, (pars, 1)) for color, pars in zip(self.body_colors, self.body_pars)])
            self.x_color.from_numpy(np.array(np_colors))

        self.grid_v = vecs(3, T, (self.n_grids, self.n_grids, self.n_grids))
        self.grid_c = vecs(3, T, (self.n_grids, self.n_grids, self.n_grids))
        self.grid_m = scalars(T, (self.n_grids, self.n_grids, self.n_grids))
        self.grid_p_c = vecs(n=self.n_phases, dtype=T, shape=(self.n_grids, self.n_grids, self.n_grids))
  
        if self.n_rigid_pars:      
            self.x_rp = vecs(3, T, self.n_rigid_pars)
            self.v_rp = vecs(3, T, self.n_rigid_pars)
            self.r_sizes = scalars(int, self.n_rigid)
            self.rp_offsets = scalars(int, self.n_rigid)
            # self.x_rt = vecs(3, T, self.n_rigid_tris)
            # self.x_rp2t = scalars(int, self.n_rigid_pars)
            np_x_rp = np.concatenate(
                [b.rest_pos for b in self.rigid_bodies], axis=0) 
            self.x_rp.from_numpy(np_x_rp - self.origin)
            np_r_sizes = np.asarray(self.rigid_sizes)
            self.r_sizes.from_numpy(np_r_sizes)
            np_rp_offsets = np.asarray(self.rigid_pars_offsets)
            self.rp_offsets.from_numpy(np_rp_offsets)

        # for bi in range(self.n_soft_bends):
        #     face_inds = self.bending_faces[bi]
        #     n0 = self.compute_normal_lag(face_inds[0])
        #     n1 = self.compute_normal_lag(face_inds[1])
        #     theta = ti.acos(n0.dot(n1))
        #     theta = ti.max(theta, ti.abs(self.eps))
        #     edge_inds = self.bending_edges[bi]
        #     edge = (self.x_lag[edge_inds[1]] -
        #             self.x_lag[edge_inds[0]]).normalized()
        #     sin_theta = n0.cross(n1).dot(edge)
        #     if sin_theta < 0:
        #         theta = - theta
        #     self.rest_bending_lag[bi] = theta

        self.clear_fields()

    @ti.func
    def compute_T_lag(self, i):
        a, b, c = self.tris_lag[i,
            0], self.tris_lag[i, 1], self.tris_lag[i, 2]
        xab = self.x_lag[b] - self.x_lag[a]
        xac = self.x_lag[c] - self.x_lag[a]
        return ti.Matrix([
            [xab[0], xac[0]],
            [xab[1], xac[1]],
            [xab[2], xac[2]]
        ])
    

    @ti.func
    def compute_emulsion_stress(self, V0, F, Emul, Emul_Eff):
        # epsilon = 1e-3
        # sigma = 1e-2
        # d0 = 1e-3
        alpha = 5e-5
        cauchy_green = F.transpose() @ F  
        i_cauchy = cauchy_green.trace()  
        # phi1 = 4 * Emul * epsilon / V0 * 12 * (sigma / d0) ** 12 * i_cauchy **(-7) * d0
        # phi2 = 4 * Emul * epsilon / V0 * 6 * (sigma / d0) ** 6 * i_cauchy **(-4) * d0
        # stress = (phi1 - phi2) * F @ cauchy_green
        stress = alpha * Emul_Eff * (1 - Emul) * ti.pow(i_cauchy, -3) * F @ cauchy_green
        return stress
    
    @ti.func
    def compute_area_lag(self, i):
        a, b, c = self.tris_lag[i,
            0], self.tris_lag[i, 1], self.tris_lag[i, 2]
        xab = self.x_lag[b] - self.x_lag[a]
        xac = self.x_lag[c] - self.x_lag[a]
        return 0.5 * xab.cross(xac).norm()
    
    @ti.func
    def compute_normal_lag(self, i):
        a, b, c = self.tris_lag[i,
            0], self.tris_lag[i, 1], self.tris_lag[i, 2]
        xab = self.x_lag[b] - self.x_lag[a]
        xac = self.x_lag[c] - self.x_lag[a]
        return xab.cross(xac).normalized()


    @ti.kernel
    def clear_fields(self):
        if ti.static(self.n_soft_pars):
            for i in ti.ndrange(self.n_soft_pars):
                self.v[i] = ti.Vector.zero(T, 3)
                self.F[i] = ti.Vector.identity(T, 3)
                self.C[i] = ti.Matrix.zero(float, 3, 3)
                self.p_rho[i] = 1.0
                self.p_vol[i] = self.default_p_vol
        if ti.static(self.n_lag_verts):
            for i in ti.ndrange(self.n_lag_verts):
                self.v_lag[i] = ti.Vector([0, 0, 0], T)
                self.C_lag[i] = ti.Matrix.zero(T, 3, 3)
            for i in range(self.n_lag_tris):
                ds = self.compute_T_lag(i)
                ds0 = ti.Vector([ds[0, 0], ds[1, 0], ds[2, 0]])
                ds1 = ti.Vector([ds[0, 1], ds[1, 1], ds[2, 1]])
                ds0_norm = ds0.norm()
                IB = ti.Matrix([
                    [ds0_norm, ds0.dot(ds1) / ds0_norm],
                    [0, ds0.cross(ds1).norm() / ds0_norm]
                ]).inverse()
                if ti.math.isnan(IB).sum():
                    print('[nan detected during IB computation]')
                    IB = ti.Matrix.zero(T, 2, 2)
                self.restInvT_lag[i] = IB
                self.tris_area_lag[i] = self.compute_area_lag(i)
                self.nrm_lag[i] = self.compute_normal_lag(i)
            for i, j in ti.ndrange(self.n_lag_tris, 3):
                self.tris_lag_expanded[i * 3 + j] = self.tris_lag[i, j]



    def set_camera_pos(self, x, y, z):
        self.camera.position(x, y, z)

    def camera_lookat(self, x, y, z):
        self.camera.lookat(x, y, z)

    def substep(self):
        # TODO
        # reference: taichi mpm128 https://github.com/taichi-dev/taichi/tree/master/python/taichi/examples/mpm128.py
        self.init_step()
        if self.n_lag_verts:
            self.energy_lag[None] = 0
            with ti.ad.Tape(self.energy_lag):
                self.compute_energy_lag()
        self.P2G()
        self.grid_op()
        self.G2P()
        self.emulsification_update()
        self.compute_global_phase_concentration()

    @ti.kernel
    def init_step(self):
        for i, j, k in self.grid_m:
            self.grid_v[i, j, k] = [0, 0, 0]
            self.grid_c[i, j, k] = [0, 0, 0]
            self.grid_m[i, j, k] = 0
            for l in ti.static(range(self.n_phases)):
                self.grid_p_c[i, j, k][l] = 0

    def update_scene(self):
        self.scene.set_camera(self.camera)
        self.scene.ambient_light((0.8, 0.8, 0.8))
        self.scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
        if self.x:
            self.scene.particles(self.x, color=(
                0.68, 0.26, 0.19), radius=0.002)

        if self.x:

            default_radius=0.0035
            np_x_body_id = self.x_body_id.to_numpy()
            np_x = self.x.to_numpy()
            np_x_color = self.x_color.to_numpy()
            np_p_vol = self.p_vol.to_numpy()
            unique_body_ids = np.unique(np_x_body_id)
            x_by_body_id = {}
            x_color_by_body_id = {}
            p_vol_by_body_id = {}
            for body_id in unique_body_ids:
                mask = np_x_body_id == body_id
                x_by_body_id[body_id] = np_x[mask]
                x_color_by_body_id[body_id] = np_x_color[mask]
                p_vol_by_body_id[body_id] = np_p_vol[mask]

                tmp_radius = (p_vol_by_body_id[body_id][0] / self.default_p_vol) ** (1/3) * default_radius
                tmp_radius = float(round(tmp_radius, 3))
                if tmp_radius < 0.002: # bug of taichi ti.ui.Scene().particles()
                    tmp_radius = 0.002
                self.body_x[body_id].from_numpy(np_x[mask])
                self.body_x_color[body_id].from_numpy(np_x_color[mask])
                self.scene.particles(self.body_x[body_id], per_vertex_color=self.body_x_color[body_id], radius=tmp_radius)


            # print(self.p_c.to_numpy())
            # print(self.p_c_global.to_numpy())
            # print(self.emul.to_numpy(), np.sum(self.emul.to_numpy(), axis=0))
            # print(self.e_c.to_numpy())

        if self.x_rp:
            self.scene.particles(self.x_rp, color=(
                0.19, 0.26, 0.68), radius=0.002)
        if self.n_static_bounds:
            for b in self.static_bounds:
                self.scene.mesh(b.vertices, b.faces, color=(0.5, 0.5, 0.5))
        if self.n_dynamic_bounds:
            for b in self.dynamic_bounds:
                self.scene.mesh(b.vertices, b.faces, color=(0.25, 0.25, 0.25))
        if self.n_lag_verts:
            self.scene.mesh(self.x_lag, self.tris_lag_expanded, color=(0.3, 0.5, 0.3))


    def show(self):
        self.canvas.scene(self.scene)
        self.window.show()

    @ti.kernel
    def P2G(self):
        if ti.static(self.n_soft_pars):
            for p in self.x:  # Particle state update and scatter to grid (P2G)
                base = (self.x[p] * self.inv_dx - 0.5).cast(int)
                fx = self.x[p] * self.inv_dx - base.cast(float)
                # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
                w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1)
                    ** 2, 0.5 * (fx - 0.5) ** 2]
                # deformation gradient update
                self.F[p] = (ti.Matrix.identity(float, 3) +
                            self.dt * self.C[p]) @ self.F[p]

                
                particle_phase = ti.Vector.zero(T, self.n_phases)
                new_F = ti.Matrix.identity(T, 3)
                stress = ti.Matrix.identity(T, 3)
                for i in ti.static(range(len(self.materials))):
                    if i == self.x_body_id[p]:
                    
                        particle_phase[i] = 1
                        self.p_rho[p] = self.body_rhos[i]
                        stress, new_F = self.materials[i].compute_kirchhoff_stress(self.F[p], self.dt, self.C[p], self.emul[p])


                        # todo: better design for emulsion p_vol
                        self.p_vol[p] = self.default_p_vol * (1 - (1 - self.emulsified_droplets_vol_ratio) * self.body_emulsification_efficiencies[i])

                        # stress += self.compute_emulsion_stress(self.body_pars[i] * self.p_vol[p], 
                        #                 new_F, self.emul[p], self.body_emulsification_efficiencies[i])


                p_mass = self.p_rho[p] * self.p_vol[p]
                self.F[p] = new_F
                stress = (-self.dt * self.p_vol[p] * 4 * self.inv_dx ** 2) * stress
                affine = stress + p_mass * self.C[p]
                for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                    # Loop over 3x3 grid node neighborhood
                    offset = ti.Vector([i, j, k])
                    position = base + offset
                    dpos = (offset.cast(float) - fx) * self.dx
                    weight = w[i][0] * w[j][1] * w[k][2]
                    self.grid_v[position] += weight * \
                        (p_mass * self.v[p] + affine @ dpos)
                    self.grid_m[position] += weight * p_mass
                    self.grid_p_c[position] += particle_phase
                    self.grid_c[position] += self.x_color[p]


        if ti.static(self.n_lag_verts):
            for p in self.x_lag:
                base = ti.cast(self.x_lag[p] * self.inv_dx - 0.5, ti.i32)
                fx = self.x_lag[p] * self.inv_dx - ti.cast(base, float)
                w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1)
                    ** 2, 0.5 * (fx - 0.5) ** 2]
                affine = self.rp_mass * self.C_lag[p]
                for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                    offset = ti.Vector([i, j, k])
                    dpos = (offset.cast(float) - fx) * \
                        self.dx
                    position = base + offset
                    weight = w[i][0] * w[j][1] * w[k][2]
                    if not ti.math.isnan(self.x_lag.grad[p]).sum():
                        self.grid_v[position] += weight * (
                            self.rp_mass * self.v_lag[p] - 
                            self.dt * self.x_lag.grad[p] + affine @ dpos)
                        self.grid_m[position] += weight * self.rp_mass

    @ti.kernel
    def grid_op(self):
        for I in ti.grouped(self.grid_m):
            if self.grid_m[I] > 0:  # No need for epsilon here
                # Momentum to velocity
                v = (1 / self.grid_m[I]) * self.grid_v[I]
                v += self.gravity * self.dt  # gravity
                # v *= 0.999  # damping

                # collide with statics
                if ti.static(self.n_static_bounds > 0):
                    for i in ti.static(range(self.n_static_bounds)):
                        v = self.static_bounds[i].collide(I * self.dx, v)
                        # print("{}-th sdf value for grid [{}, {}, {}] is {}".format(i, I[0], I[1], I[2], self.static_bounds[i].signed_dist(I * self.dx)))

                # collide with dynamics
                if ti.static(self.n_dynamic_bounds > 0):
                    for i in ti.static(range(self.n_dynamic_bounds)):
                        if ti.static(self.dynamic_bounds[i].collide_type in ["grid", "both"]):
                            # v = self.dynamic_bounds[i].collide(I * self.dx, v, self.dt, p_mass)
                            v = self.dynamic_bounds[i].collide(I * self.dx, v, self.dt, self.grid_m[I])

                if ti.static(self.box_bound > 0):
                    eps = ti.cast(1e-30, T)
                    for d in ti.static(range(3)):
                        if I[d] < self.box_bound and v[d] < 0:
                            if ti.static(d != 1 or self.ground_friction == 0): # non-ground or no friction
                                # print("set velocity to zero!")
                                v[d] = 0  # Boundary conditions
                                if ti.static(d == 1):
                                    v = ti.Vector.zero(T, 3)
                            else:
                                if ti.static(self.ground_friction < 10):
                                    # TODO: 1e-30 problems ...
                                    normal = ti.Vector.zero(T, 3)
                                    normal[d] = 1.
                                    lin = v.dot(normal) + eps # ||v_n||
                                    vit = v - lin * normal - normal * eps # v_t
                                    lit = normalize_ti(vit) # ||v_t||
                                    v = ti.max(1. + ti.static(self.ground_friction) * lin / lit, 0.) * (vit + normal * eps)
                                else:
                                    v = ti.Vector.zero(T, 3)
                        if I[d] > self.n_grids - self.box_bound and v[d] > 0: 
                            v[d] = 0

                self.grid_v[I] = v

    @ti.kernel
    def G2P(self):
        if ti.static(self.n_soft_pars):
            for p in self.x:  # grid to particle (G2P)
                base = (self.x[p] * self.inv_dx - 0.5).cast(int)
                fx = self.x[p] * self.inv_dx - base.cast(float)
                w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0)
                    ** 2, 0.5 * (fx - 0.5) ** 2]
                new_v = ti.Vector.zero(float, 3)
                new_C = ti.Matrix.zero(float, 3, 3)
                new_p_c = ti.Vector.zero(float, self.n_phases)
                new_c = ti.Vector.zero(float, 3)
                for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                    # loop over 3x3 grid node neighborhood
                    dpos = ti.Vector([i, j, k]).cast(float) - fx
                    position = base + ti.Vector([i, j, k])
                    g_v = self.grid_v[position]
                    g_p_c = self.grid_p_c[position]
                    g_c = self.grid_c[position]
                    # todo: collect local_c_p
                    weight = w[i][0] * w[j][1] * w[k][2]
                    new_v += weight * g_v
                    new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)
                    new_p_c += g_p_c
                    new_c += g_c

                # collide with dynamics

                p_mass = self.p_vol[p] * self.p_rho[p]
                if ti.static(self.n_dynamic_bounds > 0):
                    for i in ti.static(range(self.n_dynamic_bounds)):
                        if ti.static(self.dynamic_bounds[i].collide_type in ["particle", "both"]):
                            new_x_tmp = self.x[p] + self.dt * new_v
                            new_v = self.dynamic_bounds[i].collide(new_x_tmp, new_v, self.dt, p_mass)

                new_p_c_sum = 0
                for q in ti.static(range(self.n_phases)):
                    new_p_c_sum += new_p_c[q]
                new_c = new_c / new_p_c_sum   
                self.p_c[p] = new_p_c / new_p_c_sum 

                alpha = self.coloring_mixing_alpha
                delta_p_c = self.p_c[p] - self.p_c_global[None]
                p_c_L1_distance = 0.0
                for q in ti.static(range(self.n_phases)):
                    p_c_L1_distance += ti.abs(delta_p_c[q])
                if p_c_L1_distance < self.p_c_L1_distance_criterion:
                    alpha = self.uniform_coloring_mixing_alpha
                # if self.emul[p] == 1:
                #     alpha = 1e3
                self.x_color[p] = self.x_color[p] + alpha * self.dt * (new_c - self.x_color[p])
                self.v[p], self.C[p] = new_v, new_C
                self.x[p] += self.dt * self.v[p]  # advection


        if ti.static(self.n_rigid_pars):
            for p in self.x_rp:
                self.x_rp[p] += self.dt * self.v_rp[p]

        if ti.static(self.n_lag_verts):
            for p in self.x_lag:
                base = ti.cast(self.x_lag[p] * self.inv_dx - 0.5, ti.i32)
                fx = self.x_lag[p] * self.inv_dx - float(base)
                w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0)
                    ** 2, 0.5 * (fx - 0.5) ** 2]
                new_v = ti.Vector.zero(T, 3)
                new_C = ti.Matrix.zero(T, 3, 3)
                for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                    dpos = ti.Vector([i, j, k]).cast(float) - fx
                    g_v = self.grid_v[base + ti.Vector([i, j, k])]
                    weight = w[i][0] * w[j][1] * w[k][2]
                    new_v += weight * g_v
                    new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)
                self.v_lag[p], self.C_lag[p] = new_v, new_C
                self.x_lag[p] += self.v_lag[p] * self.dt



# todo: refine the model
    @ti.kernel
    def emulsification_update(self):
        if ti.static(self.n_soft_pars):
            for p in self.x:
                emulsifier_concentration = 0.0  # warning: must be 0.0 instead of 0
                for q in ti.static(range(self.n_phases)): 
                    emulsifier_concentration += self.p_c[p][q] * self.e_c[q]
                # emulsifier_concentration = emulsifier_concentration

                emulsion_color = ti.Vector([1.0, 1.0, 1.0])
                emul_eff = 0.0 
                for q in ti.static(range(len(self.materials))):
                    if q == self.x_body_id[p]:
                        emul_eff = self.e_e[q]    


                D = (self.C[p] + self.C[p].transpose()) / 2.0
                shear_rate = ti.sqrt(2.0 * (D[0, 1] ** 2 + D[0, 2] ** 2 + D[1, 2] ** 2))  

                prev_emul = self.emul[p]

                if emulsifier_concentration >= self.critical_concentration:
                    self.emul[p] += self.dt * self.alpha_for_emul_rate * shear_rate ** 2 * emul_eff

                if self.emul[p] >= 1.0:
                    self.emul[p] = 1.0
                
                self.x_color[p] += (self.emul[p] - prev_emul) * (emulsion_color - self.x_color[p])

    @ti.kernel
    def compute_global_phase_concentration(self):
        if ti.static(self.n_soft_pars):
            new_p_c_global = ti.Vector.zero(T, self.n_phases)
            for p in self.x:
                new_p_c_global += self.p_c[p] / self.n_soft_pars
            self.p_c_global[None] = new_p_c_global
            # np_pc = self.p_c.to_numpy()
            # np_pc_global = np.sum(np_pc, axis=0) / self.n_soft_pars
            # self.p_c_global.from_numpy(np_pc_global)

                
                






    @ti.kernel
    def compute_energy_lag(self):
        for i in range(self.n_lag_tris):
            Ds = self.compute_T_lag(i)
            F = Ds @ self.restInvT_lag[i]
            f0 = ti.Vector([F[0, 0], F[1, 0], F[2, 0]])
            f1 = ti.Vector([F[0, 1], F[1, 1], F[2, 1]])
            Estretch = self.lag_mu * self.tris_area_lag[i] * \
                ((f0.norm() - 1) ** 2 + (f1.norm() - 1) ** 2)
            Eshear = self.lag_mu * 0.3 * self.tris_area_lag[i] * f0.dot(f1) ** 2
            self.energy_lag[None] += Eshear + Estretch


    def add_body(self, body: Body):
        if isinstance(body, RigidBody):
            self.rigid_bodies.append(body)
            self.rigid_pars_offsets.append(self.n_rigid_pars)
            self.rigid_tris_inds.append(
                self.n_rigid_tris + np.asarray(body.tri_inds))
            self.rigid_tris_vinds.append(
                self.n_rigid_pars + np.asarray(body.mesh.faces))
            self.n_rigid_pars += body.n_pars
            self.n_rigid_tris += body.n_tris
            self.rigid_sizes.append(body.n_pars)
        elif isinstance(body, SoftBody):
            self.deformable_bodies.append(body)
            self.n_soft_pars += body.n_pars
            self.body_pars.append(body.n_pars)
            self.materials.append(body.material)
            self.body_colors.append(body.color)
            self.body_emulsification_efficiencies.append(body.emulsification_efficiency)
            self.body_emulsifier_capacities.append(body.emulsifier_capacity)
            self.body_rhos.append(body.rho)
        else:
            raise NotImplementedError()
        
    def add_lag_body(self, lag_mesh: trimesh.Trimesh, lag_E, lag_nu):
        self.lag_mesh = lag_mesh
        self.n_lag_verts = lag_mesh.vertices.shape[0]
        self.n_lag_tris = lag_mesh.faces.shape[0]
        self.lag_mu, self.lag_lam = lag_E / (2 * (1 + lag_nu)), lag_E * \
            lag_nu / ((1 + lag_nu) * (1 - 2 * lag_nu))
        self.eps = 1e-6
        pos_mask = (lag_mesh.vertices - self.origin) < 0
        pos_mask *= (lag_mesh.vertices - self.origin) > 1
        if pos_mask.sum() > 0:
            print(
                'MpmSim: lag mesh body trying to be added is out of the bounding box!')


    
    def add_boundary(self, bound: Boundary):
        if isinstance(bound, StaticBoundary):
            self.static_bounds.append(bound)
        elif isinstance(bound, DynamicBoundary):
            self.dynamic_bounds.append(bound)
        else:
            raise NotImplementedError()

    def toward_target(self, substeps=1000):
        # TODO: compute v_rp
        if self.n_rigid_pars > 0:
            np_v_rp = np.concatenate(
                [b.target_pos for b in self.rigid_bodies], axis=0) - self.origin
            np_v_rp = (np_v_rp - self.x_rp.to_numpy()) / (self.dt * substeps)
            self.v_rp.from_numpy(np_v_rp)

        # dynamic boundaries
        for b in self.dynamic_bounds:
            b.toward_target()



def test_mpm():
    ti.init(arch=ti.cuda)
    dt = 1e-4
    substeps = 10
    sim = MpmSim(origin=np.asarray([-0.5,] * 3), dt=dt)
    nhk = NeoHookean()
    cube: trimesh.Trimesh = trimesh.creation.box((0.1,) * 3)
    cube_points = cube.sample(8192)
    # cube_pcd: trimesh.PointCloud = trimesh.PointCloud(cube.sample(8192))
    sponge_box = SoftBody(cube_points, nhk)
    wrist_mesh = trimesh.load_mesh('./data/Mano_URDF/meshes/m_avg_R_Wrist.stl')
    # pos = np.asarray([pos['z'], -pos['x'], pos['y']])
    wrist_verts = np.asarray(wrist_mesh.vertices)
    wrist_mesh.vertices = np.concatenate(
        [wrist_verts[:, [2]], -wrist_verts[:, [0]], wrist_verts[:, [1]]], axis=1)
    wrist_mesh.apply_translation(np.asarray([0., 0.2, 0.15]))
    rigid_wrist = RigidBody(wrist_mesh)

    sim.add_body(sponge_box)
    sim.add_body(rigid_wrist)
    sim.init_system()

    steps = 0
    while not sim.window.is_pressed(ti.GUI.ESCAPE):
        if (steps // 100) % 2:
            wrist_mesh.apply_translation(np.asarray([0., 0.001, 0.]))
        else:
            wrist_mesh.apply_translation(np.asarray([0., -0.001, 0.]))
        rigid_wrist.set_target(wrist_mesh.vertices)
        for s in range(substeps):
            sim.substep()
        sim.update_scene()
        sim.show()
        sim.toward_target(substeps)
        steps += 1



if __name__ == '__main__':
    test_mpm()
