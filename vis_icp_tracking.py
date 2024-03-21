import os
import copy

import open3d as o3d
import numpy as np
import torch
from tqdm import tqdm
from matplotlib import cm
import pickle
from pytorch3d.ops.points_alignment import iterative_closest_point, _apply_similarity_transform

from fusion import Fusion
from utils.draw_utils import aggr_point_cloud_from_data
from utils.track_vis import TrackVis

fusion = Fusion(num_cam=1)

num_cam = 4

x_upper = 1.5
x_lower = 0.0
y_upper = 0.5
y_lower = -0.5
z_upper = 0.5
z_lower = 0.2

boundaries = {'x_lower': x_lower,
              'x_upper': x_upper,
              'y_lower': y_lower,
              'y_upper': y_upper,
              'z_lower': z_lower,
              'z_upper': z_upper,}

kypts_boundaries = {'x_lower': x_lower,
                    'x_upper': x_upper,
                    'y_lower': y_lower,
                    'y_upper': y_upper,
                    'z_lower': z_lower,
                    'z_upper': z_upper,}

vis_o3d = True

def gen_dense_kypts(data_path, src_feat_info):
    # colors = np.stack([cv2.imread(os.path.join(data_path, f'camera_{i}', 'color', f'0.png')) for i in range(num_cam)], axis=0)# [N, H, W, C]
    # depths = np.stack([cv2.imread(os.path.join(data_path, f'camera_{i}', 'depth', f'0.png'), cv2.IMREAD_ANYDEPTH) for i in range(num_cam)], axis=0) / 1000. # [N, H, W]

    # extrinsics = np.stack([np.load(os.path.join(data_path, f'camera_{i}', 'camera_extrinsics.npy')) for i in range(num_cam)])
    # cam_param = np.stack([np.load(os.path.join(data_path, f'camera_{i}', 'camera_params.npy')) for i in range(num_cam)])

    # intrinsics = np.zeros((num_cam, 3, 3))
    # intrinsics[:, 0, 0] = cam_param[:, 0]
    # intrinsics[:, 1, 1] = cam_param[:, 1]
    # intrinsics[:, 0, 2] = cam_param[:, 2]
    # intrinsics[:, 1, 2] = cam_param[:, 3]
    # intrinsics[:, 2, 2] = 1

    colors_all = np.load('/home/ywang/Downloads/data/color', allow_pickle = True)[:, None].astype(np.uint8) # (T, N, H, W, 3)
    depths_all = np.load('/home/ywang/Downloads/data/depth', allow_pickle = True)[:, None] / 1000. # (T, N, H, W)
    T = colors_all.shape[0]
    extrinsics = np.linalg.inv(np.array([[ 9.99928880e-01, -1.05292680e-03, 1.18796708e-02, 7.40902314e-01],
                                         [-1.28010915e-03, -9.99816142e-01, 1.91322552e-02, -2.35826574e-03],
                                         [ 1.18573418e-02, -1.91461019e-02, -9.99746383e-01, 1.76778660e+00],
                                         [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]))
    extrinsics = extrinsics[None] # (N, 4, 4)

    intrinsics = np.array([[391.05383301,   0.        , 322.77679443],
                           [  0.        , 391.05383301, 242.89453125],
                           [  0.        ,   0.        ,   1.        ]])
    intrinsics = intrinsics[None] # (N, 3, 3)


    # multi-category tracking
    query_texts = list(src_feat_info.keys())
    query_thresholds = [src_feat_info[k]['params']['sam_threshold'] for k in query_texts]

    # create output dir
    full_pts_path = os.path.join(data_path, 'obj_kypts') # list of (ptcl_num, 3) for each push, indexed by push_num
    os.system(f'mkdir -p {full_pts_path}')
    
    track_vis = TrackVis(poses=extrinsics, Ks=intrinsics, output_dir=full_pts_path, vis_o3d=vis_o3d)
    
    time_skip = 1
    times = list(range(T))
    
    for t in tqdm(times[::time_skip]):
        colors = colors_all[t]
        depths = depths_all[t]

        if vis_o3d:
            pcd = aggr_point_cloud_from_data(colors[..., ::-1], depths, intrinsics, extrinsics, downsample=True) # , boundaries=boundaries)
            # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            # o3d.visualization.draw_geometries([pcd, origin])
        else:
            pcd = None
        
        obs = {
            'color': colors,
            'depth': depths,
            'pose': extrinsics[:, :3], # (N, 3, 4)
            'K': intrinsics,
        }
        
        fusion.update(obs)
        fusion.text_queries_for_inst_mask(query_texts, query_thresholds, boundaries=boundaries)
        
        # initialize for tracking
        obj_pcd = fusion.extract_masked_pcd(list(range(1, fusion.get_inst_num())), boundaries=boundaries)
        if t == 0:
            rand_ptcl_num = 100
            src_feats_list, src_pts_list, color_list = fusion.select_features_from_pcd(obj_pcd, rand_ptcl_num, per_instance=True)
            
            # save src_feats_list and src_pts_list
            src_feats_np_list = [src_feats.detach().cpu().numpy() for src_feats in src_feats_list]
            src_pts_np_list = [src_pts for src_pts in src_pts_list]
            pickle.dump(src_feats_np_list, open(os.path.join(full_pts_path, f'src_feats_list.pkl'), 'wb'))
            pickle.dump(src_pts_np_list, open(os.path.join(full_pts_path, f'src_pts_list.pkl'), 'wb'))
            
            # save make label
            pickle.dump(fusion.curr_obs_torch['mask_label'][0], open(os.path.join(full_pts_path, f'mask_label.pkl'), 'wb'))
            
            last_k = ""
            rep_idx = 0
            for k_i, k in enumerate(fusion.curr_obs_torch['mask_label'][0][1:]):
                if k == last_k:
                    rep_idx += 1
                    src_feat_info[k+f'_{rep_idx}'] = copy.copy(src_feat_info[k])
                    src_feat_info[k+f'_{rep_idx}']['src_feats'] = src_feats_list[k_i]
                    # src_feat_info[k+f'_{rep_idx}']['src_color'] = color_list[k_i]
                    src_feat_info[k+f'_{rep_idx}']['src_pts'] = src_pts_list[k_i]
                    src_feat_loc_norm = (src_feat_info[k]['src_pts'][:, 0] - src_feat_info[k]['src_pts'][:, 0].min()) / \
                        (src_feat_info[k]['src_pts'][:, 0].max() - src_feat_info[k]['src_pts'][:, 0].min())
                    cmap = cm.get_cmap('viridis')
                    colors = (cmap(src_feat_loc_norm)[:, :3] * 255).astype(np.uint8)[:, ::-1]
                    src_feat_info[k+f'_{rep_idx}']['src_pts_color'] = colors
                else:
                    rep_idx = 0
                    src_feat_info[k]['src_feats'] = src_feats_list[k_i]
                    # src_feat_info[k]['src_color'] = color_list[k_i]
                    src_feat_info[k]['src_pts'] = src_pts_list[k_i]
                    src_feat_loc_norm = (src_feat_info[k]['src_pts'][:, 0] - src_feat_info[k]['src_pts'][:, 0].min()) / \
                        (src_feat_info[k]['src_pts'][:, 0].max() - src_feat_info[k]['src_pts'][:, 0].min())
                    cmap = cm.get_cmap('viridis')
                    colors = (cmap(src_feat_loc_norm)[:, :3] * 255).astype(np.uint8)[:, ::-1]
                    src_feat_info[k]['src_pts_color'] = colors
                last_k = k
            match_pts_list = src_pts_list.copy()
        
        match_pts_tensor = torch.from_numpy(match_pts_list[0]).to(device=fusion.device, dtype=torch.float32)[None]
        obj_pcd_tensor = torch.from_numpy(obj_pcd).to(device=fusion.device, dtype=torch.float32)[None]
        ICPSol = iterative_closest_point(match_pts_tensor, obj_pcd_tensor)
        
        match_pts_tensor_tf = _apply_similarity_transform(match_pts_tensor, ICPSol.RTs.R, ICPSol.RTs.T, ICPSol.RTs.s)
        match_pts_list[0] = match_pts_tensor_tf[0].detach().cpu().numpy()

        track_vis.visualize_match_pts(match_pts_list, pcd, obs['color'][..., ::-1], src_feat_info)
        
        pickle.dump(match_pts_list, open(os.path.join(full_pts_path, f'{t:06d}.pkl'), 'wb'))

if __name__ == '__main__':
    src_feat_info = {
        'circle':
            {'params': {'sam_threshold': 0.3},
            'src_feats_path': None},
    }
    gen_dense_kypts('/home/ywang/Downloads', src_feat_info)
