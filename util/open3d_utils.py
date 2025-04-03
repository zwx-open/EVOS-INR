import open3d as o3d
import numpy as np
from tqdm import tqdm
import torch

def create_o3d_pointcloud(coords, norms=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    if norms is not None:
        pcd.normals = o3d.utility.Vector3dVector(norms)
    return pcd

def trimesh_to_o3d_mesh(mesh):
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    return o3d_mesh

def mesh_to_voxelGrid(mesh, voxel_size=1/512):
    voxel = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=voxel_size)
    return voxel

def pointcloud_to_voxelGrid(pcd, voxel_size=1/512):
    voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    return voxel

def save_pointcloud(pcd, path):
    o3d.io.write_point_cloud(path, pcd)

def save_mesh(mesh, path):
    o3d.io.write_triangle_mesh(path, mesh)

def save_voxelGrid(voxel, path):
    o3d.io.write_voxel_grid(path, voxel)

def load_pointcloud(path):
    pcd = o3d.io.read_point_cloud(path)
    return pcd

def load_mesh(path):
    mesh = o3d.io.read_triangle_mesh(path)
    return mesh

def load_voxelGrid(path):
    voxel = o3d.io.read_voxel_grid(path)
    return voxel

def visualize(obj):
    o3d.visualization.draw_geometries([obj])

def visualize_voxelgrid(voxel):
    # Create a coordinate frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    voxel.origin = np.array([0,0,0])
    o3d.visualization.draw_geometries([voxel, frame])

def compute_voxel_grid_iou(vg1, vg2, N=512):
    # compute the iou between two voxel grids
    # convert to numpy array
    intersection, union = 0, 0
    x = torch.linspace(-0.5, 0.5, N)
    x, y, z = torch.meshgrid(x, x, x)
    coords = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=-1)


    occupancy1 = np.stack(list([vx.grid_index for vx in vg1.get_voxels()]))
    occupancy2 = np.stack(list([vx.grid_index for vx in vg2.get_voxels()]))

    # process in batch
    batch_size = 128**2
    for i in tqdm(range(0, coords.size(0), batch_size)):
        c = coords[i:i+batch_size, :]
        gt_hit = [vg2.check_if_included(x.numpy()) for x in c]
        pred_hit = [vg1.check_if_included(x.numpy()) for x in c]
        batch_intersection = torch.sum(torch.stack([torch.tensor(gt_hit), torch.tensor(pred_hit)], dim=0).all(dim=0))
        batch_union = torch.sum(torch.stack([torch.tensor(gt_hit), torch.tensor(pred_hit)], dim=0).any(dim=0))
        intersection += batch_intersection.item()
        union += batch_union.item()
        

    unique_oc1 = set(tuple(x) for x in occupancy1)
    unique_oc2 = set(tuple(x) for x in occupancy2)
    intersection = len(unique_oc1.intersection(unique_oc2))
    union = len(occupancy1) + len(occupancy2) - intersection
    iou = intersection / union
    return iou