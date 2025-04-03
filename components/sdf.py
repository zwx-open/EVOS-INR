import torch
from torch.utils.data import Dataset
import mcubes
import trimesh
import math
import numpy as np
import os

class MeshSDF(Dataset):
    ''' convert point cloud to SDF '''

    def __init__(self, dataset_configs, input_output_configs):
        super().__init__()
        self.config = dataset_configs
        self.coord_mode = input_output_configs.coord_mode
        self.data_range = input_output_configs.data_range

        self.num_samples = self.config.num_samples
        self.pointcloud_path = self.config.xyz_file
        self.coarse_scale = self.config.coarse_scale
        self.fine_scale = self.config.fine_scale
        self.normalize = True
        self.dim_in = 3
        self.dim_out = 1
        self.out_range = None

        # load gt point cloud with normals
        self.load_mesh(self.config.xyz_file)
        
        # precompute sdf and occupancy grid
        self.render_resolution = self.config.render_resolution
        self.load_precomputed_occu_grid(self.config.xyz_file, self.render_resolution)

    def load_precomputed_occu_grid(self, xyz_file, render_resolution):
        # load from files if exists
        sdf_file = xyz_file.replace('.xyz', f'_{render_resolution}_sdf.npy')
        if os.path.exists(sdf_file):
            self.sdf = np.load(sdf_file)
        else:
            self.sdf = self.build_sdf(render_resolution)
            np.save(sdf_file, self.sdf)

        occu_grid = (self.sdf <= 0)
        self.occu_grid = occu_grid

    def build_sdf(self, render_resolution):
        N = render_resolution
        # build grid
        x = torch.arange(-N//2, N//2) / N
        x = x.float()
        x, y, z = torch.meshgrid(x, x, x)
        render_coords = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=-1).cuda()
        vox_centers = render_coords.cpu().numpy()

        # use KDTree to get nearest neighbours and estimate the normal
        _, idx = self.kd_tree.query(vox_centers, k=3)
        avg_normal = np.mean(self.n[idx], axis=1)
        sdf = np.sum((vox_centers - self.v[idx][:, 0]) * avg_normal, axis=-1)
        sdf = sdf.reshape(N, N, N)
        return sdf
    
    def build_grid_coords(self, render_resolution):
        N = render_resolution
        x = torch.arange(-N//2, N//2) / N
        x = x.float()
        x, y, z = torch.meshgrid(x, x, x)
        coords = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=-1).cuda()
        return coords.cpu().numpy()

    def load_mesh(self, pointcloud_path):
        from pykdtree.kdtree import KDTree
        npy_file = pointcloud_path.replace('.xyz', '.npy')
        if os.path.exists(npy_file):
            pointcloud = np.load(npy_file)
        else:
            pointcloud = np.genfromtxt(pointcloud_path)
            np.save(pointcloud_path.replace('.xyz', '.npy'), pointcloud)
        self.pointcloud = pointcloud
        print("No. of points: ", pointcloud.shape[0])
        
        # cache to speed up loading
        self.v = pointcloud[:, :3]
        self.n = pointcloud[:, 3:]

        n_norm = (np.linalg.norm(self.n, axis=-1)[:, None])
        n_norm[n_norm == 0] = 1.
        self.n = self.n / n_norm
        self.v = self.normalize_coords(self.v)
        self.kd_tree = KDTree(self.v)
        print('finish loading pc')

    def normalize_coords(self, coords):
        coords -= np.mean(coords, axis=0, keepdims=True)
        coord_max = np.amax(coords)
        coord_min = np.amin(coords)
        coords = (coords - coord_min) / (coord_max - coord_min) * 0.9
        coords -= 0.45
        return coords

    def sample_surface(self, use_all=False):
        if use_all:
            points =  np.copy(self.v) # clone
            index = None
        else:
            index = np.random.randint(0, self.v.shape[0], self.num_samples)
            points = self.v[index]
        
        points[::2] += np.random.laplace(scale=self.coarse_scale, size=(points.shape[0] - points.shape[0]//2, points.shape[-1]))
        points[1::2] += np.random.laplace(scale=self.fine_scale, size=(points.shape[0]//2, points.shape[-1]))

        # wrap around any points that are sampled out of bounds
        points[points > 0.5] -= 1
        points[points < -0.5] += 1

        # use KDTree to get distance to surface and estimate the normal
        sdf, idx = self.kd_tree.query(points, k=3)
        avg_normal = np.mean(self.n[idx], axis=1)
        sdf = np.sum((points - self.v[idx][:, 0]) * avg_normal, axis=-1)
        sdf = sdf[..., None]

        return points, sdf, index

    def __getitem__(self, idx):
        batch_size = 262144
        start_idx = idx * batch_size
        coords, sdf = self.get_data(start_idx, batch_size)
        return coords, sdf
    
    def __len__(self):
        return 1
    
    def get_data(self):
        coords, sdf, idx = self.sample_surface()
        return torch.from_numpy(coords).float(), torch.from_numpy(sdf).float(), torch.from_numpy(idx.astype(np.int32))
    
    def get_all_data(self):
        coords, sdf, _ = self.sample_surface(use_all=True)
        return torch.from_numpy(coords).float(), torch.from_numpy(sdf).float()


def generate_mesh(model, N=512, return_sdf=False, device='cuda'):
    num_outputs = 1     # hard code this because the current models generate only one output
    # write output
    x = torch.linspace(-0.5, 0.5, N)
    if return_sdf:
        x = torch.arange(-N//2, N//2) / N
        x = x.float()
    x, y, z = torch.meshgrid(x, x, x)
    render_coords = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=-1).to(device)
    
    sdf_values = [np.zeros((N**3, 1)) for i in range(num_outputs)]

    # render in mini batch to save memory
    bsize = int(400**2)
    model.eval()
    for i in range(math.ceil(N**3 / bsize)):
        coords = render_coords[i*bsize:(i+1)*bsize, :]
        with torch.no_grad():
            out = model(coords)

        if not isinstance(out, list):
            out = [out, ]

        for idx, sdf in enumerate(out):
            sdf_values[idx][i*bsize:(i+1)*bsize] = sdf.detach().cpu().numpy()

    for idx, sdf in enumerate(sdf_values):
        sdf = sdf.reshape(N, N, N)
        vertices, triangles = mcubes.marching_cubes(-sdf, 0)
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        mesh.vertices = (mesh.vertices / N - 0.5) + 0.5/N
    model.train()

    if return_sdf:
        return mesh, sdf
    else:
        return mesh
    

