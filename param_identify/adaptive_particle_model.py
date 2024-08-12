import numpy as np
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull

class AdaptiveParticleModel:
    def __init__(self, center, n_particles=5000, length=1.0, width=0.8, height=0.6):
        self.n_particles = n_particles
        self.center = center
        self.length = length
        self.width = width
        self.height = height
        self.initial_particles = self._initialize_cuboid_particles()
        self.particles = self.initial_particles.copy()

    def _initialize_cuboid_particles(self):
        """初始化长方体粒子"""
        particles = np.random.rand(self.n_particles, 3)
        particles = particles - np.mean(particles) + self.center
        particles[:, 0] *= self.length
        particles[:, 1] *= self.width
        particles[:, 2] *= self.height
        return particles

    def update_model(self, surface_points, rebuild_from_cuboid=False):
        """更新粒子模型以适应新的表面点云"""
        if rebuild_from_cuboid:
            self.particles = self.initial_particles.copy()

        # 第一阶段：初步形状构建
        self.particles, _ = self._initial_shape_formation(surface_points)
        
        # 第二阶段：优化粒子分布
        self.particles, _ = self._optimize_particle_distribution(surface_points)

        return self.particles

    def _initial_shape_formation(self, surface_points):
        """初步形状构建"""
        surface_tree = cKDTree(surface_points)
        particle_tree = cKDTree(self.particles)

        new_positions = []
        used_particles = set()
        history = [self.particles.copy()]

        for point in surface_points:
            _, indices = particle_tree.query(point, k=len(self.particles))
            for idx in indices:
                if idx not in used_particles:
                    new_positions.append(point)
                    used_particles.add(idx)
                    break

        new_positions = np.array(new_positions)
        remaining_particles = self.particles[list(set(range(len(self.particles))) - used_particles)]

        new_particles = np.vstack((new_positions, remaining_particles))
        history.append(new_particles)

        return new_particles, history

    def _optimize_particle_distribution(self, surface_points, iterations=5):
        """优化粒子分布，考虑不规则表面"""
        history = [self.particles.copy()]
        original_count = len(self.particles)

        for _ in range(iterations):
            self.particles = self._fill_gaps_and_connect(surface_points)
            self.particles = self._resample_particles(original_count)
            history.append(self.particles.copy())

        return self.particles, history

    def _fill_gaps_and_connect(self, surface_points):
        """在空隙处填充粒子并创建到不规则表面的连接"""
        particle_tree = cKDTree(self.particles)
        surface_tree = cKDTree(surface_points)
        new_particles = []

        # 填充表面附近的空隙
        for point in surface_points:
            neighbors = particle_tree.query_ball_point(point, r=0.1)
            if len(neighbors) < 5:
                for _ in range(5 - len(neighbors)):
                    new_particle = point + np.random.normal(0, 0.05, 3)
                    if surface_tree.query(new_particle)[0] < 0.2:
                        new_particles.append(new_particle)

        # 创建连接到不规则表面的路径
        for particle in self.particles:
            distance, nearest_surface_index = surface_tree.query(particle)
            if distance > 0.2:  # 如果粒子离表面较远
                nearest_surface = surface_points[nearest_surface_index]

                # 创建连接路径
                path = self._create_adaptive_path(particle, nearest_surface, surface_points)
                for point in path[1:-1]:  # 排除起点和终点
                    if particle_tree.query(point)[0] > 0.05:  # 如果路径上的点周围没有其他粒子
                        new_particles.append(point)

        return np.vstack((self.particles, new_particles)) if new_particles else self.particles

    def _create_adaptive_path(self, start, end, surface_points):
        """创建一条适应不规则表面的路径"""
        direct_path = np.linspace(start, end, num=10)
        adaptive_path = [start]

        surface_tree = cKDTree(surface_points)

        for point in direct_path[1:-1]:
            _, nearest_surface_index = surface_tree.query(point)
            nearest_surface = surface_points[nearest_surface_index]

            # 将路径点稍微拉向最近的表面点
            adapted_point = point + 0.3 * (nearest_surface - point)
            adaptive_path.append(adapted_point)

        adaptive_path.append(end)
        return np.array(adaptive_path)

    def _resample_particles(self, target_count):
        """在 bbox 空间上均匀重新采样粒子以达到目标数量"""
        if len(self.particles) <= target_count:
            return self.particles

        # 计算当前粒子的 bounding box
        min_coords = np.min(self.particles, axis=0)
        max_coords = np.max(self.particles, axis=0)
        
        # 计算每个维度的分割数
        dims = len(min_coords)
        particles_per_dim = int(np.ceil(target_count ** (1/dims)))
        
        # 创建网格
        grids = [np.linspace(min_coords[i], max_coords[i], num=particles_per_dim) for i in range(dims)]
        mesh = np.array(np.meshgrid(*grids))
        
        # 重塑网格点为一个列表
        grid_points = mesh.reshape(dims, -1).T
        
        # 如果网格点数量超过目标数量，随机选择目标数量的点
        if len(grid_points) > target_count:
            selected_indices = np.random.choice(len(grid_points), target_count, replace=False)
            grid_points = grid_points[selected_indices]
        
        # 对于每个网格点，找到最近的原始粒子
        tree = cKDTree(self.particles)
        _, indices = tree.query(grid_points)
        
        # 选择最接近网格点的原始粒子
        self.particles = self.particles[indices]
        
        return self.particles
    

import numpy as np
from scipy.spatial import ConvexHull

class SurfaceToParticleModel:
    def __init__(self, bottom_thickness=0.1, particle_count=1000):
        self.bottom_thickness = bottom_thickness
        self.particle_count = particle_count

    def convert(self, surface_points):
        # 计算边界框（bbox）
        bbox_min = np.min(surface_points, axis=0)
        bbox_max = np.max(surface_points, axis=0)
        
        # 计算点云密度
        hull = ConvexHull(surface_points[:, [0, 2]])  # 使用x和z坐标计算凸包
        surface_area = hull.area
        point_density = len(surface_points) / surface_area

        print(f"Surface area: {surface_area}, Point density: {point_density}")

        # 找到最底部的点并创建底面
        bottom_y = bbox_min[1]
        bottom_plane_y = bottom_y - self.bottom_thickness

        # 为每个点创建随机插值粒子
        interpolated_particles = []
        for point in surface_points:
            distance_to_bottom = point[1] - bottom_plane_y
            volume = distance_to_bottom * (1 / point_density)  # 假设每个点代表的体积
            num_particles = max(1, int(volume * point_density * 1.5))  # 确保至少生成1个粒子
            
            for _ in range(num_particles):
                random_y = np.random.uniform(bottom_plane_y, point[1])
                random_x = np.random.normal(point[0], self.bottom_thickness / 2)
                random_z = np.random.normal(point[2], self.bottom_thickness / 2)
                interpolated_particle = [random_x, random_y, random_z]
                interpolated_particles.append(interpolated_particle)

        print(f"Number of interpolated particles: {len(interpolated_particles)}")

        # 合并原始表面点和插值粒子
        if len(interpolated_particles) > 0:
            all_particles = np.vstack([surface_points, np.array(interpolated_particles)])
        else:
            all_particles = surface_points

        # 随机采样 particle_count 个粒子
        if len(all_particles) > self.particle_count:
            sampled_indices = np.random.choice(len(all_particles), self.particle_count, replace=False)
            particle_model = all_particles[sampled_indices]
        else:
            particle_model = all_particles

        print(f"Final particle model shape: {particle_model.shape}")

        return particle_model

    def update_parameters(self, surface_points, bottom_thickness=None, particle_count=None):
        if bottom_thickness is not None:
            self.bottom_thickness = bottom_thickness
        if particle_count is not None:
            self.particle_count = particle_count
        
        return self.convert(surface_points)