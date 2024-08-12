import taichi as ti



ti.init(arch=ti.cuda, device_memory_GB=8)

total_mass = 1
tool_total_mass = 0.5
density = 500

visco_lower_bound = 0.01
visco_upper_bound = 10

gravity = 9.81
dim = 3
bound = 3

@ti.data_oriented
class ParticleSystem:
    def __init__(self, n_particles, n_tool_particles, max_steps, container_length, container_width, container_height):
        self.n_particles = n_particles
        self.n_tool_particles = n_tool_particles
        self.n_grid = 64
        self.dt = 1e-4
        self.dx, self.inv_dx = 1 / self.n_grid, float(self.n_grid)
        self.p_mass = total_mass / n_particles
        self.p_vol = (self.dx * 0.5)**2
        self.p_vol = self.p_mass / density
        self.tool_p_mass = tool_total_mass / n_tool_particles
        self.max_steps = max_steps


        self.container_width = container_width
        self.container_length = container_length
        self.container_height = container_height


        # Main particles
        self.x = ti.Vector.field(dim, dtype=ti.f32, shape=(max_steps, n_particles))
        self.v = ti.Vector.field(dim, dtype=ti.f32, shape=(max_steps, n_particles))
        self.C = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=(max_steps, n_particles))
        self.F = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=(max_steps, n_particles))

        # Tool particles
        self.tool_x = ti.Vector.field(dim, dtype=ti.f32, shape=(max_steps, n_tool_particles))
        self.tool_v = ti.Vector.field(dim, dtype=ti.f32, shape=(max_steps, n_tool_particles))
        self.tool_C = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=(max_steps, n_tool_particles))
        self.tool_F = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=(max_steps, n_tool_particles))
        
        # For SVD
        self.F_tmp = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
        self.U = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
        self.V = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
        self.sig = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)

        # Non-equilibrated part
        self.non_equilibrated_x = ti.Vector.field(dim, dtype=ti.f32, shape=(max_steps, n_particles))
        self.non_equilibrated_v = ti.Vector.field(dim, dtype=ti.f32, shape=(max_steps, n_particles))
        self.non_equilibrated_C = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=(max_steps, n_particles))
        self.non_equilibrated_F = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=(max_steps, n_particles))
        
        self.non_equilibrated_F_tmp = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
        self.non_equilibrated_U = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
        self.non_equilibrated_V = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
        self.non_equilibrated_sig = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)

        # Grid
        self.grid_v = ti.Vector.field(3, dtype=ti.f32, shape=(self.n_grid, self.n_grid, self.n_grid))
        self.grid_m = ti.field(dtype=ti.f32, shape=(self.n_grid, self.n_grid, self.n_grid))
        self.grid_v_out = ti.Vector.field(3, dtype=ti.f32, shape=(self.n_grid, self.n_grid, self.n_grid))

        # Initial position fields
        self.initial_x = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)
        self.final_x = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)
        self.initial_tool_x = ti.Vector.field(dim, dtype=ti.f32, shape=n_tool_particles)
        
        # Other fields
        self.tool_traj = ti.Vector.field(dim, dtype=ti.f32, shape=max_steps)
        self.tool_force = ti.field(dtype=ti.f32, shape=max_steps)

        # Constitutive parameters
        self.E = ti.field(dtype=ti.f32, shape=())
        self.nu = ti.field(dtype=ti.f32, shape=())
        self.yield_stress = ti.field(dtype=ti.f32, shape=())
        self.viscosity = ti.field(dtype=ti.f32, shape=())
        self.loss = ti.field(dtype=ti.f32, shape=())

        ti.root.lazy_grad()

    def initialize_objects(self, initial_positions, final_positions, initial_tool_positions):
        self.initial_x.from_numpy(initial_positions)
        self.initial_tool_x.from_numpy(initial_tool_positions)
        self.final_x.from_numpy(final_positions)

    def set_mechanical_motion(self, tool_traj_sequence, tool_force_sequence):
        self.tool_traj.from_numpy(tool_traj_sequence)
        self.tool_force.from_numpy(tool_force_sequence)

    def set_constitutive_parameters(self, E, nu, yield_stress, viscosity):
        self.E[None] = E
        self.nu[None] = nu
        self.yield_stress[None] = yield_stress
        self.viscosity[None] = viscosity

    def export_deformation(self):
        self.particles_np = self.x.to_numpy()
        self.tool_particles_np = self.tool_x.to_numpy()
        return self.particles_np, self.tool_particles_np

    def zero_vec(self):
        return [0.0, 0.0, 0.0]


    def zero_matrix(self):
        return [self.zero_vec(), self.zero_vec(), self.zero_vec()]


    @ti.func
    def make_matrix_from_diag(self, d):
        return ti.Matrix([[d[0], 0.0, 0.0], [0.0, d[1], 0.0], [0.0, 0.0, d[2]]], dt=ti.f32)



    @ti.kernel
    def initialize_ti_field(self):
        for i in range(self.n_particles):
            self.x[0, i] = self.initial_x[i]
            self.v[0, i] = ti.Vector([0.0, 0.0, 0.0])
            self.F[0, i] = ti.Matrix.identity(float, dim)
            self.non_equilibrated_x[0, i] = self.initial_x[i]
            self.non_equilibrated_v[0, i] = ti.Vector([0.0, 0.0, 0.0])
            self.non_equilibrated_F[0, i] = ti.Matrix.identity(float, dim)

        for i in range(self.n_tool_particles):
            self.tool_x[0, i] = self.initial_tool_x[i]
            self.tool_v[0, i] = ti.Vector([0.0, 0.0, 0.0])
            self.tool_F[0, i] = ti.Matrix.identity(float, dim)


    @ti.kernel
    def clear_grid(self):
        for i, j, k in self.grid_m:
            self.grid_v[i, j, k] = [0, 0, 0]
            self.grid_m[i, j, k] = 0
            self.grid_v.grad[i, j, k] = [0, 0, 0]
            self.grid_m.grad[i, j, k] = 0

    @ti.kernel
    def clear_particles(self):
        for i in range(self.n_particles):
            for j in range(self.max_steps):
                self.x[j, i] = [0, 0, 0]
                self.non_equilibrated_x[j, i] = [0, 0, 0]

    @ti.kernel
    def compute_F_tmp(self, f: ti.i32):
        for p in range(0, self.n_particles):  # Particle state update and scatter to grid (P2G)
            self.F_tmp[p] = (ti.Matrix.identity(ti.f32, dim) + self.dt * self.C[f, p]) @ self.F[f, p]
            self.non_equilibrated_F_tmp[p] = (ti.Matrix.identity(ti.f32, dim) +
                                        self.dt * self.non_equilibrated_C[f, p]) @ self.non_equilibrated_F[f, p]


    @ti.kernel
    def svd(self):
        for p in range(0, self.n_particles):
            self.U[p], self.sig[p], self.V[p] = ti.svd(self.F_tmp[p])
            self.non_equilibrated_U[p], self.non_equilibrated_sig[p], \
                self.non_equilibrated_V[p] = ti.svd(self.non_equilibrated_F_tmp[p])


    @ti.kernel
    def clear_SVD_grad(self):
        zero = ti.Matrix.zero(ti.f32, dim, dim)
        for i in range(0, self.n_particles):
            self.U.grad[i] = zero
            self.sig.grad[i] = zero
            self.V.grad[i] = zero
            self.F_tmp.grad[i] = zero
            self.non_equilibrated_U.grad[i] = zero
            self.non_equilibrated_sig.grad[i] = zero
            self.non_equilibrated_V.grad[i] = zero
            self.non_equilibrated_F_tmp.grad[i] = zero

    @ti.func
    def clamp(self, a):
        # remember that we don't support if-return in taichi
        # stop the gradient ...
        if a >= 0:
            a = max(a, 1e-6)
        else:
            a = min(a, -1e-6)
        return a
    
    @ti.func
    def backward_svd(self, gu, gsigma, gv, u_matrix, sigma, v_matrix):
        # https://github.com/pytorch/pytorch/blob/ab0a04dc9c8b84d4a03412f1c21a6c4a2cefd36c/tools/autograd/templates/Functions.cpp
        vt = v_matrix.transpose()
        ut = u_matrix.transpose()
        sigma_term = u_matrix @ gsigma @ vt

        s = ti.Vector.zero(ti.f32, dim)
        s = ti.Vector([sigma[0, 0], sigma[1, 1], sigma[2, 2]]) ** 2
        f = ti.Matrix.zero(ti.f32, dim, dim)
        for i, j in ti.static(ti.ndrange(dim, dim)):
            if i == j:
                f[i, j] = 0
            else:
                f[i, j] = 1. / self.clamp(s[j] - s[i])
        u_term = u_matrix @ ((f * (ut @ gu - gu.transpose() @ u_matrix)) @ sigma) @ vt
        v_term = u_matrix @ (sigma @ ((f * (vt @ gv - gv.transpose() @ v_matrix)) @ vt))
        return u_term + v_term + sigma_term


    @ti.kernel
    def svd_grad(self):
        for p in range(0, self.n_particles):
            self.F_tmp.grad[p] += self.backward_svd(self.U.grad[p], self.sig.grad[p], self.V.grad[p], self.U[p], self.sig[p], self.V[p])
            self.non_equilibrated_F_tmp.grad[p] += \
                self.backward_svd(self.non_equilibrated_U.grad[p], self.non_equilibrated_sig.grad[p],
                            self.non_equilibrated_V.grad[p], self.non_equilibrated_U[p],
                            self.non_equilibrated_sig[p], self.non_equilibrated_V[p])


    @ti.kernel
    def compute_target_loss(self, f: ti.i32):
        for target in range(self.n_particles):
            dist = self.final_x[target] - self.x[f, target]
            dist = dist ** 2 / self.n_particles
            if abs((dist(0) + dist(1) + dist(2)) / 3) > 0:
                self.loss[None] += (dist(0) + dist(1) + dist(2)) / (3)


    @ti.func
    def compute_P_hat(self, sig, mu, la):
        epsilon = ti.Vector([ti.log(sig[0, 0]), ti.log(sig[1, 1]), ti.log(sig[2, 2])])
        sum_log = epsilon[0] + epsilon[1] + epsilon[2]
        psi_0 = (2 * mu * epsilon[0] + la * sum_log) / sig[0, 0]
        psi_1 = (2 * mu * epsilon[1] + la * sum_log) / sig[1, 1]
        psi_2 = (2 * mu * epsilon[2] + la * sum_log) / sig[2, 2]
        P_hat = ti.Vector([psi_0, psi_1, psi_2])
        return P_hat
    

    @ti.func
    def norm(self, x, eps=1e-8):
        return ti.sqrt(x.dot(x) + eps)


    @ti.ad.grad_replaced
    def forward(self, s):
        self.clear_grid()
        self.compute_F_tmp(s)
        self.svd()
        self.p2g(s)
        self.grid_op(s)
        self.g2p(s)

    @ti.ad.grad_for(forward)
    def backward(self, s):
        self.clear_grid()
        self.clear_SVD_grad()
        self.compute_F_tmp(s)
        self.svd()
        self.p2g(s)
        self.grid_op(s)

        self.g2p.grad(s)
        self.grid_op.grad(s)
        self.p2g.grad(s)
        self.svd_grad()
        self.compute_F_tmp.grad(s)

    def run_simulation(self, end_step):
        self.clear_grid()
        self.clear_particles()
        self.initialize_ti_field()
        with ti.ad.Tape(loss=self.loss):
            self.loss[None] = 0
            for f in range(end_step):
                self.forward(f)
            self.compute_target_loss(end_step)
        return self.loss[None]

    
    @ti.kernel
    def p2g(self, f: ti.i32):
        for p in range(0, self.n_particles):

            base = ti.cast(self.non_equilibrated_x[f, p] * self.inv_dx - 0.5, ti.i32)
            fx = self.non_equilibrated_x[f, p] * self.inv_dx - ti.cast(base, ti.i32)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

            mu = self.E[None] / (2 * (1 + self.nu[None]))
            la = self.E[None] * self.nu[None] / (
                    (1 + self.nu[None]) * (1 - 2 * self.nu[None]))

            temp_sig = self.non_equilibrated_sig[p]
            epsilon = ti.Vector([ti.log(temp_sig[0, 0]), ti.log(temp_sig[1, 1]), ti.log(temp_sig[2, 2])])
            alpha = 2.0 * mu / self.viscosity[None]
            beta = 2.0 * (2.0 * mu + la * dim) / (9.0 * self.viscosity[None]) - \
                2.0 * mu / (self.viscosity[None] * dim)

            A = 1 / (1 + self.dt * alpha)
            B = self.dt * beta / (1 + self.dt * (alpha + dim * beta))
            epsilon_trace = ti.log(temp_sig[0, 0]) + ti.log(temp_sig[1, 1]) + ti.log(temp_sig[2, 2])
            temp_epsilon = A * (epsilon - ti.Vector([B * epsilon_trace, B * epsilon_trace, B * epsilon_trace]))
            new_sig = self.make_matrix_from_diag(ti.exp(temp_epsilon))
            new_non_equilibrated_F = self.non_equilibrated_U[p] @ new_sig @ self.non_equilibrated_V[p].transpose()
            non_equilibrated_P_hat = self.compute_P_hat(self.non_equilibrated_sig[p], mu, la)
            non_equilibrated_P = self.non_equilibrated_U[p] @ ti.Matrix([[non_equilibrated_P_hat[0], 0.0, 0.0],
                [0.0, non_equilibrated_P_hat[1], 0.0], [0.0, 0.0, non_equilibrated_P_hat[2]]]) @ self.non_equilibrated_V[p].transpose()
            non_equilibrated_kirchhoff_stress = non_equilibrated_P @ new_non_equilibrated_F.transpose()
            stress = -(self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx) * non_equilibrated_kirchhoff_stress
            affine = stress + self.p_mass * self.non_equilibrated_C[f, p]
            temp_v = self.non_equilibrated_v[f, p] + [0, self.dt * (-gravity), 0]
            self.non_equilibrated_F[f + 1, p] = new_non_equilibrated_F

            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)):
                        offset = ti.Vector([i, j, k])
                        dpos = (ti.cast(ti.Vector([i, j, k]), ti.f32) - fx) * self.dx
                        weight = w[i](0) * w[j](1) * w[k](2)
                        ti.atomic_add(self.grid_v[base + offset],
                                    weight * (self.p_mass * temp_v + affine @ dpos))
                        ti.atomic_add(self.grid_m[base + offset], weight * self.p_mass)




        for p in range(0, self.n_particles):

            base = ti.cast(self.x[f, p] * self.inv_dx - 0.5, ti.i32)
            fx = self.x[f, p] * self.inv_dx - ti.cast(base, ti.i32)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

            mu = self.E[None] / (2 * (1 + self.nu[None]))
            la = self.E[None] * self.nu[None] / (
                    (1 + self.nu[None]) * (1 - 2 * self.nu[None]))
            new_F = self.F_tmp[p]

            # von-mises
            temp_sig = ti.max(self.sig[p], 0.005)
            epsilon = ti.Vector([ti.log(temp_sig[0, 0]), ti.log(temp_sig[1, 1]), ti.log(temp_sig[2, 2])])
            epsilon_hat = epsilon - (epsilon.sum() / dim)
            epsilon_hat_norm = self.norm(epsilon_hat)
            delta_gamma = epsilon_hat_norm - self.yield_stress[None] / (2 * mu)
            if delta_gamma > 0:  # Yields
                epsilon -= (delta_gamma / epsilon_hat_norm) * epsilon_hat
                temp_sig = self.make_matrix_from_diag(ti.exp(epsilon))
                new_F = self.U[p] @ temp_sig @ self.V[p].transpose()

            self.F[f + 1, p] = new_F
            J = new_F.determinant()
            ident = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            kirchhoff = mu * (new_F @ new_F.transpose()) + ti.Matrix(ident) * (
                    la * ti.log(J) - mu)

            stress = -(self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx) * kirchhoff
            affine = stress + self.p_mass * self.C[f, p]
            temp_v = self.v[f, p] + [0, self.dt * (-gravity), 0]

            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)):
                        offset = ti.Vector([i, j, k])
                        dpos = (ti.cast(ti.Vector([i, j, k]), ti.f32) - fx) * self.dx
                        weight = w[i](0) * w[j](1) * w[k](2)
                        ti.atomic_add(self.grid_v[base + offset],
                                    weight * (self.p_mass * temp_v + affine @ dpos))
                        ti.atomic_add(self.grid_m[base + offset], weight * self.p_mass)


        for p in range(0, self.n_tool_particles):

            base = ti.cast(self.tool_x[f, p] * self.inv_dx - 0.5, ti.i32)
            fx = self.tool_x[f, p] * self.inv_dx - ti.cast(base, ti.i32)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            direction = self.tool_v[f, p] / self.norm(self.tool_v[f, p])
            impulse = direction * self.tool_force[f] / self.n_tool_particles
            momentum = self.tool_p_mass * self.tool_v[f, p] + impulse
            affine = self.tool_p_mass * self.tool_C[f, p]
            
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)):
                        offset = ti.Vector([i, j, k])
                        dpos = (ti.cast(ti.Vector([i, j, k]), ti.f32) - fx) * self.dx
                        weight = w[i](0) * w[j](1) * w[k](2)
                        ti.atomic_add(self.grid_v[base + offset],
                                    weight * (momentum + affine @ dpos))
                        ti.atomic_add(self.grid_m[base + offset], weight * self.tool_p_mass)




    
    @ti.kernel
    def grid_op(self, f: ti.i32):
        for i, j, k in self.grid_m:
            inv_m = 1 / (self.grid_m[i, j, k] + 1e-10)
            v_out = inv_m * self.grid_v[i, j, k]

            x = i * self.dx
            y = j * self.dx
            z = k * self.dx

            # Container (hole) boundaries
            x_min = 0.5 - self.container_width / 2
            x_max = 0.5 + self.container_width / 2
            z_min = 0.5 - self.container_length / 2
            z_max = 0.5 + self.container_length / 2

            # Check if the point is inside the container (hole)
            in_container = (x_min <= x <= x_max and
                            z_min <= z <= z_max and
                            y <= self.container_height + bound * self.dx)

            if in_container:
                # Bottom of the container
                if y < bound * self.dx:
                    # v_out[0] = 0
                    v_out[1] = 0
                    # v_out[2] = 0

                # x direction (width)
                if x < x_min and v_out[0] < 0:
                    v_out[0] = 0
                if x > x_max and v_out[0] > 0:
                    v_out[0] = 0

                # z direction (depth)
                if z < z_min and v_out[2] < 0:
                    v_out[2] = 0
                if z > z_max and v_out[2] > 0:
                    v_out[2] = 0

            else:
                if i < bound and v_out[0] < 0:
                    v_out[0] = 0
                    # v_out[1] = 0
                    # v_out[2] = 0
                if i > self.n_grid - bound and v_out[0] > 0:
                    v_out[0] = 0
                    # v_out[1] = 0
                    # v_out[2] = 0
                if k < bound and v_out[2] < 0:
                    # v_out[0] = 0
                    # v_out[1] = 0
                    v_out[2] = 0
                if k > self.n_grid - bound and v_out[2] > 0:
                    # v_out[0] = 0
                    # v_out[1] = 0
                    v_out[2] = 0
                if j < bound and v_out[1] < 0:
                    # v_out[0] = 0
                    v_out[1] = 0
                    # v_out[2] = 0
                if j > self.n_grid - bound and v_out[1] > 0:
                    # v_out[0] = 0
                    v_out[1] = 0
                    # v_out[2] = 0

            self.grid_v_out[i, j, k] = v_out



    @ti.kernel
    def g2p(self, f: ti.i32):

        for p in range(self.n_particles):
            base = ti.cast(self.x[f, p] * self.inv_dx - 0.5, ti.i32)
            fx = self.x[f, p] * self.inv_dx - ti.cast(base, ti.f32)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.Vector([0.0, 0.0, 0.0])
            new_C = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)):
                        dpos = ti.cast(ti.Vector([i, j, k]), ti.f32) - fx
                        g_v = self.grid_v_out[base(0) + i, base(1) + j, base(2) + k]
                        weight = w[i](0) * w[j](1) * w[k](2)
                        new_v += weight * g_v
                        new_C += 4 * weight * g_v.outer_product(dpos) * self.inv_dx

            self.v[f + 1, p] = new_v
            self.x[f + 1, p] = self.x[f, p] + self.dt * self.v[f + 1, p]
            self.C[f + 1, p] = new_C

        for p in range(self.n_particles):
                base = ti.cast(self.non_equilibrated_x[f, p] * self.inv_dx - 0.5, ti.i32)
                fx = self.non_equilibrated_x[f, p] * self.inv_dx - ti.cast(base, ti.f32)
                w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
                new_v = ti.Vector([0.0, 0.0, 0.0])
                new_C = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

                for i in ti.static(range(3)):
                    for j in ti.static(range(3)):
                        for k in ti.static(range(3)):
                            dpos = ti.cast(ti.Vector([i, j, k]), ti.f32) - fx
                            g_v = self.grid_v_out[base(0) + i, base(1) + j, base(2) + k]
                            weight = w[i](0) * w[j](1) * w[k](2)
                            new_v += weight * g_v
                            new_C += 4 * weight * g_v.outer_product(dpos) * self.inv_dx

                self.non_equilibrated_v[f + 1, p] = new_v
                # self.non_equilibrated_x[f + 1, p] = self.x[f, p] + dt * self.v[f + 1, p]
                self.non_equilibrated_x[f + 1, p] = self.non_equilibrated_x[f, p] + self.dt * self.non_equilibrated_v[f + 1, p]
                self.non_equilibrated_C[f + 1, p] = new_C

        for p in range(self.n_tool_particles):
            base = ti.cast(self.tool_x[f, p] * self.inv_dx - 0.5, ti.i32)
            fx = self.tool_x[f, p] * self.inv_dx - ti.cast(base, ti.f32)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_tool_v = ti.Vector([0.0, 0.0, 0.0])
            new_tool_C = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)):
                        dpos = ti.cast(ti.Vector([i, j, k]), ti.f32) - fx
                        g_v = self.grid_v_out[base(0) + i, base(1) + j, base(2) + k]
                        weight = w[i](0) * w[j](1) * w[k](2)
                        new_tool_v += weight * g_v
                        new_tool_C += 4 * weight * g_v.outer_product(dpos) * self.inv_dx

            self.tool_v[f + 1, p] = (self.tool_traj[f + 1] - self.tool_traj[f]) / self.dt
            self.tool_x[f + 1, p] = self.tool_x[0, p] + self.tool_traj[f + 1] - self.tool_traj[0]
            self.tool_C[f + 1, p] = new_tool_C



    def initialize_optimizer(self, optimizer_type):
        if optimizer_type == 'adam':
            self.adam_m = ti.field(ti.f32, shape=())
            self.adam_v = ti.field(ti.f32, shape=())
            self.adam_m[None] = 0
            self.adam_v[None] = 0
        elif optimizer_type == 'sgd':
            self.sgd_momentum = ti.field(ti.f32, shape=())
            self.sgd_momentum[None] = 0
        self.visco_lower_bound = visco_lower_bound
        self.visco_upper_bound = visco_upper_bound

    @ti.func
    def adam_update(self, t, g, m, v, alpha, beta1, beta2, eps):
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        m_hat = m / (1 - beta1 ** (t + 1))
        v_hat = v / (1 - beta2 ** (t + 1))
        update = alpha * m_hat / (ti.sqrt(v_hat) + eps)
        return update, m, v

    @ti.kernel
    def adam_step(self, t: ti.i32, alpha: ti.f32, beta1: ti.f32, beta2: ti.f32, eps: ti.f32):
        g = self.viscosity.grad[None]
        update, self.adam_m[None], self.adam_v[None] = self.adam_update(t, g, self.adam_m[None], self.adam_v[None], alpha, beta1, beta2, eps)
        self.viscosity[None] -= update
        self.viscosity[None] = ti.max(self.visco_lower_bound, ti.min(self.viscosity[None], self.visco_upper_bound))

    @ti.kernel
    def sgd_step(self, learning_rate: ti.f32, momentum: ti.f32):
        self.sgd_momentum[None] = momentum * self.sgd_momentum[None] + learning_rate * self.viscosity.grad[None]
        self.viscosity[None] -= self.sgd_momentum[None]
        self.viscosity[None] = ti.max(self.visco_lower_bound, ti.min(self.viscosity[None], self.visco_upper_bound))
    
    
    def optimize_viscosity(self, num_iterations, end_step, optimizer_type='adam', **kwargs):
        self.initialize_optimizer(optimizer_type)

        if optimizer_type == 'adam':
            alpha = kwargs.get('alpha', 0.001)
            beta1 = kwargs.get('beta1', 0.9)
            beta2 = kwargs.get('beta2', 0.999)
            eps = kwargs.get('eps', 1e-8)
        elif optimizer_type == 'sgd':
            learning_rate = kwargs.get('learning_rate', 0.01)
            momentum = kwargs.get('momentum', 0.9)

        best_loss = float('inf')
        best_viscosity = self.viscosity[None]

        for iteration in range(num_iterations):
            loss = self.run_simulation(end_step)

            print(f"Iteration {iteration + 1}/{num_iterations}")
            print(f"Loss: {loss}")
            print(f"Current viscosity: {self.viscosity[None]}")

            # Update best loss and viscosity if current loss is lower
            if loss < best_loss:
                best_loss = loss
                best_viscosity = self.viscosity[None]

            
            if optimizer_type == 'adam':
                self.adam_step(iteration, alpha, beta1, beta2, eps)
            elif optimizer_type == 'sgd':
                self.sgd_step(learning_rate, momentum)
        print(f"Optimization completed.")

        # Set the viscosity to the best found value
        self.viscosity[None] = best_viscosity

        # Run the simulation again with the best viscosity
        final_loss = self.run_simulation(end_step)

        print(f"Final simulation run with best viscosity:")
        print(f"Final loss: {final_loss}")
        print(f"Final viscosity: {self.viscosity[None]}")

        return best_viscosity, final_loss

