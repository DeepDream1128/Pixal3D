from typing import *
import json
from abc import abstractmethod
import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class StandardDatasetBase(Dataset):
    """
    Base class for standard datasets.

    Args:
        roots (str): paths to the dataset
        skip_list (str, optional): path to a file containing sha256 hashes to skip (one per line)
                                   Format: "dataset/sha256" (e.g., "ABO/6a79dbb5...")
        skip_aesthetic_score_datasets (list, optional): list of dataset names to skip aesthetic score check
                                                        (e.g., ["texverse"] for datasets without aesthetic_score)
    """

    def __init__(self,
        roots: str,
        skip_list: Optional[str] = None,
        skip_aesthetic_score_datasets: Optional[List[str]] = None,
    ):
        super().__init__()
        
        # Datasets to skip aesthetic score check
        self.skip_aesthetic_score_datasets = set(skip_aesthetic_score_datasets or [])
        
        # Load skip list if provided
        self.skip_set = set()
        if skip_list is not None and os.path.exists(skip_list):
            with open(skip_list, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        self.skip_set.add(line)
            print(f'Loaded {len(self.skip_set)} items from skip_list: {skip_list}')
        
        try:
            self.roots = json.loads(roots)
            root_type = 'obj'
        except:
            self.roots = roots.split(',')
            root_type = 'list'
        self.instances = []
        self.metadata = pd.DataFrame()
        
        self._stats = {}
        if root_type == 'obj':
            for key, root in self.roots.items():
                self._stats[key] = {}
                metadata = pd.DataFrame(columns=['sha256']).set_index('sha256')
                
                # Only merge key fields from ss_latent and render_cond
                # Exclude base, because cond_rendered=False in base/metadata.csv would incorrectly overwrite real values
                for sub_key, r in root.items():
                    if sub_key == 'base':
                        continue  # Skip base directory
                    metadata_file = os.path.join(r, 'metadata.csv')
                    if os.path.exists(metadata_file):
                        metadata = metadata.combine_first(pd.read_csv(metadata_file).set_index('sha256'))
                
                # Read aesthetic_score separately from base (avoid reading other potentially conflicting columns)
                if 'base' in root:
                    base_metadata_file = os.path.join(root['base'], 'metadata.csv')
                    if os.path.exists(base_metadata_file):
                        base_df = pd.read_csv(base_metadata_file).set_index('sha256')
                        if 'aesthetic_score' in base_df.columns and 'aesthetic_score' not in metadata.columns:
                            metadata['aesthetic_score'] = base_df['aesthetic_score']
                
                self._stats[key]['Total'] = len(metadata)
                metadata, stats = self.filter_metadata(metadata, dataset_name=key)
                self._stats[key].update(stats)
                
                # Filter out items in skip_list
                skipped_count = 0
                for sha256 in metadata.index.values:
                    skip_key = f'{key}/{sha256}'
                    if skip_key in self.skip_set:
                        skipped_count += 1
                    else:
                        self.instances.append((root, sha256, key))
                if skipped_count > 0:
                    self._stats[key]['Skipped (skip_list)'] = skipped_count
                    self._stats[key]['After skip_list'] = len(metadata) - skipped_count
                
                self.metadata = pd.concat([self.metadata, metadata])
        else:
            for root in self.roots:
                key = os.path.basename(root)
                self._stats[key] = {}
                metadata = pd.read_csv(os.path.join(root, 'metadata.csv'))
                self._stats[key]['Total'] = len(metadata)
                metadata, stats = self.filter_metadata(metadata, dataset_name=key)
                self._stats[key].update(stats)
                
                # Filter out items in skip_list
                skipped_count = 0
                for sha256 in metadata['sha256'].values:
                    skip_key = f'{key}/{sha256}'
                    if skip_key in self.skip_set:
                        skipped_count += 1
                    else:
                        self.instances.append((root, sha256, key))
                if skipped_count > 0:
                    self._stats[key]['Skipped (skip_list)'] = skipped_count
                    self._stats[key]['After skip_list'] = len(metadata) - skipped_count
                metadata.set_index('sha256', inplace=True)
                self.metadata = pd.concat([self.metadata, metadata])
            
    @abstractmethod
    def filter_metadata(self, metadata: pd.DataFrame, dataset_name: str = None) -> Tuple[pd.DataFrame, Dict[str, int]]:
        pass
    
    @abstractmethod
    def get_instance(self, root, instance: str) -> Dict[str, Any]:
        pass
        
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index) -> Dict[str, Any]:
        try:
            root, instance, dataset_name = self.instances[index]
            pack = self.get_instance(root, instance)
            pack['_dataset_name'] = dataset_name
            pack['_sha256'] = instance
            return pack
        except Exception as e:
            print(f'Error loading {self.instances[index][1]}: {e}')
            return self.__getitem__(np.random.randint(0, len(self)))
        
    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f'  - Total instances: {len(self)}')
        lines.append(f'  - Sources:')
        for key, stats in self._stats.items():
            lines.append(f'    - {key}:')
            for k, v in stats.items():
                lines.append(f'      - {k}: {v}')
        return '\n'.join(lines)


class ImageConditionedMixin:
    def __init__(self, roots, *, image_size=518, **kwargs):
        self.image_size = image_size
        super().__init__(roots, **kwargs)
    
    def filter_metadata(self, metadata, dataset_name=None):
        metadata, stats = super().filter_metadata(metadata, dataset_name=dataset_name)
        metadata = metadata[metadata['cond_rendered'].notna()]
        stats['Cond rendered'] = len(metadata)
        return metadata, stats
    
    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)
       
        image_root = os.path.join(root['render_cond'], instance)
        with open(os.path.join(image_root, 'transforms.json')) as f:
            metadata = json.load(f)
        n_views = len(metadata['frames'])
        view = np.random.randint(n_views)
        metadata = metadata['frames'][view]

        image_path = os.path.join(image_root, metadata['file_path'])
        image = Image.open(image_path)

        alpha = np.array(image.getchannel(3))
        bbox = np.array(alpha).nonzero()
        bbox = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
        aug_hsize = hsize
        aug_center_offset = [0, 0]
        aug_center = [center[0] + aug_center_offset[0], center[1] + aug_center_offset[1]]
        aug_bbox = [int(aug_center[0] - aug_hsize), int(aug_center[1] - aug_hsize), int(aug_center[0] + aug_hsize), int(aug_center[1] + aug_hsize)]
        image = image.crop(aug_bbox)

        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha = image.getchannel(3)
        image = image.convert('RGB')
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        image = image * alpha.unsqueeze(0)
        pack['cond'] = image
       
        return pack


class ViewImageConditionedMixin:
    """
    Mixin for view-based image-conditioned datasets.
    
    This mixin is designed for datasets where ss_latent is stored per-view (view{XX}.npz),
    and needs to load the corresponding view image and scale from view{XX}_scale.json.
    
    Args:
        image_size: Target image size
        load_camera_info: Whether to load camera information for view-aligned conditioning
    """
    def __init__(self, roots, *, image_size=518, load_camera_info=False, **kwargs):
        self.image_size = image_size
        # self.load_camera_info = load_camera_info
        super().__init__(roots, **kwargs)
    
    def filter_metadata(self, metadata, dataset_name=None):
        metadata, stats = super().filter_metadata(metadata, dataset_name=dataset_name)
        metadata = metadata[metadata['cond_rendered'].notna()]
        stats['Cond rendered'] = len(metadata)
        return metadata, stats
    
    def get_instance(self, root, instance):
        """
        Get instance with view-aligned image and camera info.
        
        Expects parent class to set:
            - pack['x_0']: the latent tensor
            - self._current_view_idx: the selected view index
            - self._current_latent_dir: the latent directory path
        """
        pack = super().get_instance(root, instance)
        
        # Get view_idx from parent class (set by SparseStructureLatentView)
        if not hasattr(self, '_current_view_idx'):
            raise RuntimeError("Parent class must set '_current_view_idx' before calling ViewImageConditionedMixin.get_instance")
        if not hasattr(self, '_current_latent_dir'):
            raise RuntimeError("Parent class must set '_current_latent_dir' before calling ViewImageConditionedMixin.get_instance")
        view_idx = self._current_view_idx
        latent_dir = self._current_latent_dir
        
        # Load image metadata
        image_root = os.path.join(root['render_cond'], instance)
        with open(os.path.join(image_root, 'transforms.json')) as f:
            metadata = json.load(f)
        
        # Load corresponding image for this view
        frame_metadata = metadata['frames'][view_idx]
        image_path = os.path.join(image_root, frame_metadata['file_path'])
        image = Image.open(image_path)

        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha = image.getchannel(3)
        image = image.convert('RGB')
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        image = image * alpha.unsqueeze(0)
        pack['cond'] = image
        
        # Load camera info if requested
   
        # camera_angle_x: check frame first, then root metadata
        if 'camera_angle_x' in frame_metadata:
            camera_angle_x = float(frame_metadata['camera_angle_x'])
        elif 'camera_angle_x' in metadata:
            camera_angle_x = float(metadata['camera_angle_x'])
        else:
            raise KeyError(f"'camera_angle_x' not found in transforms.json for {instance}")
        pack['camera_angle_x'] = torch.tensor(camera_angle_x, dtype=torch.float32)
        
        # transform_matrix
        if 'transform_matrix' not in frame_metadata:
            raise KeyError(f"'transform_matrix' not found in frame {view_idx} for {instance}")
        transform_matrix = torch.tensor(frame_metadata['transform_matrix'], dtype=torch.float32)
        distance = torch.norm(transform_matrix[:3, 3]).item()
            
        pack['camera_distance'] = torch.tensor(distance, dtype=torch.float32)
        # NOTE: Do NOT pass transform_matrix to ProjGrid.
        # shape_latent space objects are already rotated to front-view by transform_mesh,
        # so ProjGrid should use the default front_view_transform_matrix + distance.
        # pack['transform_matrix'] = transform_matrix
        
        # Load mesh_scale from ss_latent directory's view{XX}_scale.json
        scale_json_path = os.path.join(latent_dir, f'view{view_idx:02d}_scale.json')
        if not os.path.exists(scale_json_path):
            raise FileNotFoundError(f"Scale file not found: {scale_json_path}")
        with open(scale_json_path) as f:
            scale_data = json.load(f)
        if 'total_scale' not in scale_data:
            raise KeyError(f"'total_scale' not found in {scale_json_path}")
        pack['mesh_scale'] = torch.tensor(float(scale_data['total_scale']), dtype=torch.float32)
       
        return pack


class MultiViewProjImageConditionedMixin:
    """
    Multi-view image-conditioned mixin for proj-mode (Pixal3D § 3.2.3 / § 3.4).

    For each sample, randomly samples ``V`` views in ``[min_views, max_views]``
    (default [2, 6], following the paper). View 0 is *always* the **primary**
    view that owns the latent (i.e. this mixin must be combined with a
    parent dataset whose ``get_instance`` already loaded a view-aligned latent
    via ``self._current_view_idx`` and ``self._current_latent_dir`` — exactly
    the same hooks used by ``ViewImageConditionedMixin``).

    Output shape (always padded to ``max_views``):

      pack['cond']             : [V_max, 3, H, W]   stacked images
      pack['camera_angle_x']   : [V_max]            FOV per view
      pack['camera_distance']  : [V_max]            translation norm in *world* frame
      pack['mesh_scale']       : [V_max]            primary's total_scale broadcast
      pack['transform_matrix'] : [V_max, 4, 4]      aux: T in *primary canonical frame*
                                                    primary slot is identity-like (will be
                                                    treated as ``None`` downstream)
      pack['view_valid']       : [V_max]            1 = real view, 0 = padding
      pack['num_views']        : int                actual V for this sample

    Args:
        image_size:    DINOv3 input resolution (default 518)
        min_views:     Minimum sampled views (default 2)
        max_views:     Maximum sampled views (default 6) — also the padding length
    """
    # Sentinel for "primary view, use front_view default"
    _PRIMARY_TM_SENTINEL = 'primary'

    def __init__(self, roots, *, image_size=518, min_views: int = 2, max_views: int = 6, **kwargs):
        self.image_size = image_size
        self.mv_min_views = int(min_views)
        self.mv_max_views = int(max_views)
        assert 1 <= self.mv_min_views <= self.mv_max_views, \
            f'invalid (min_views, max_views) = ({min_views}, {max_views})'
        super().__init__(roots, **kwargs)

    def filter_metadata(self, metadata, dataset_name=None):
        metadata, stats = super().filter_metadata(metadata, dataset_name=dataset_name)
        metadata = metadata[metadata['cond_rendered'].notna()]
        stats['Cond rendered'] = len(metadata)
        return metadata, stats

    @staticmethod
    def _front_view_transform_matrix(distance: float) -> torch.Tensor:
        """Match ProjGrid.front_view_transform_matrix for the given distance."""
        T = torch.tensor([
            [1.0, 0.0,  0.0,         0.0],
            [0.0, 0.0, -1.0, -float(distance)],
            [0.0, 1.0,  0.0,         0.0],
            [0.0, 0.0,  0.0,         1.0],
        ], dtype=torch.float32)
        return T

    def _load_view_image(self, image_root: str, frame_metadata: dict) -> torch.Tensor:
        """Load and (alpha-)resize one view to [3, image_size, image_size]."""
        image_path = os.path.join(image_root, frame_metadata['file_path'])
        image = Image.open(image_path)
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha = image.getchannel(3)
        image = image.convert('RGB')
        image_t = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha_t = torch.tensor(np.array(alpha)).float() / 255.0
        return image_t * alpha_t.unsqueeze(0)

    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)
        if not hasattr(self, '_current_view_idx'):
            raise RuntimeError(
                "Parent class must set '_current_view_idx' before "
                "calling MultiViewProjImageConditionedMixin.get_instance"
            )
        if not hasattr(self, '_current_latent_dir'):
            raise RuntimeError(
                "Parent class must set '_current_latent_dir' before "
                "calling MultiViewProjImageConditionedMixin.get_instance"
            )
        primary_idx = int(self._current_view_idx)
        latent_dir = self._current_latent_dir

        image_root = os.path.join(root['render_cond'], instance)
        with open(os.path.join(image_root, 'transforms.json')) as f:
            tmeta = json.load(f)
        all_frames = tmeta['frames']
        n_views = len(all_frames)
        if n_views < 1 or primary_idx >= n_views:
            raise RuntimeError(
                f'Bad view setup for {instance}: primary={primary_idx}, n_views={n_views}'
            )

        # Sample V (number of *real* views, including primary)
        v_max_real = min(self.mv_max_views, n_views)
        v_min_real = min(self.mv_min_views, v_max_real)
        V = int(np.random.randint(v_min_real, v_max_real + 1))

        # Pick V-1 aux view indices (≠ primary), no replacement
        candidate_aux = [i for i in range(n_views) if i != primary_idx]
        if V - 1 > len(candidate_aux):
            V = len(candidate_aux) + 1
        aux_indices = list(np.random.choice(candidate_aux, size=V - 1, replace=False)) if V > 1 else []
        view_indices = [primary_idx] + [int(a) for a in aux_indices]

        # Primary canonical reference (matches ProjGrid's default semantics)
        primary_frame = all_frames[primary_idx]
        primary_T_world = torch.tensor(primary_frame['transform_matrix'], dtype=torch.float32)
        primary_distance = float(torch.linalg.norm(primary_T_world[:3, 3]).item())
        T_canon_primary = self._front_view_transform_matrix(primary_distance)

        # mesh_scale from primary's view_idx_scale.json (consistent with single-view path)
        scale_path = os.path.join(latent_dir, f'view{primary_idx:02d}_scale.json')
        if not os.path.exists(scale_path):
            raise FileNotFoundError(f'Scale file not found: {scale_path}')
        with open(scale_path) as f:
            scale_data = json.load(f)
        if 'total_scale' not in scale_data:
            raise KeyError(f"'total_scale' not found in {scale_path}")
        primary_mesh_scale = float(scale_data['total_scale'])

        # ---- Build per-view tensors ------------------------------------
        Vmax = self.mv_max_views
        H = W = self.image_size
        images = torch.zeros(Vmax, 3, H, W, dtype=torch.float32)
        cam_angle = torch.zeros(Vmax, dtype=torch.float32)
        cam_dist = torch.zeros(Vmax, dtype=torch.float32)
        mesh_scl = torch.full((Vmax,), primary_mesh_scale, dtype=torch.float32)
        # transform_matrix[v=0] is identity (sentinel; downstream will pass None for primary)
        # transform_matrix[v>=1] is aux's c2w in primary's canonical frame
        T_mv = torch.zeros(Vmax, 4, 4, dtype=torch.float32)
        T_mv[:] = torch.eye(4)
        T_mv[0] = T_canon_primary
        view_valid = torch.zeros(Vmax, dtype=torch.bool)

        for slot, vid in enumerate(view_indices):
            frame = all_frames[vid]
            images[slot] = self._load_view_image(image_root, frame)
            # FOV: per-frame > root-level
            if 'camera_angle_x' in frame:
                cam_angle[slot] = float(frame['camera_angle_x'])
            elif 'camera_angle_x' in tmeta:
                cam_angle[slot] = float(tmeta['camera_angle_x'])
            else:
                raise KeyError(f"camera_angle_x missing for {instance} view {vid}")
            T_w = torch.tensor(frame['transform_matrix'], dtype=torch.float32)
            cam_dist[slot] = float(torch.linalg.norm(T_w[:3, 3]).item())
            if slot == 0:
                # primary slot: T_canon_p (treated as None downstream)
                T_mv[slot] = T_canon_primary
            else:
                # aux: T_canon_p @ inv(T_world_p) @ T_world_aux
                T_mv[slot] = T_canon_primary @ torch.linalg.inv(primary_T_world) @ T_w
            view_valid[slot] = True

        # Padded slots: copy primary image / camera (DINOv3 will still run on it,
        # but valid mask zeros it out during aggregation).
        for slot in range(V, Vmax):
            images[slot] = images[0]
            cam_angle[slot] = cam_angle[0]
            cam_dist[slot] = cam_dist[0]
            T_mv[slot] = T_mv[0]
            # view_valid[slot] stays False

        pack['cond'] = images
        pack['camera_angle_x'] = cam_angle
        pack['camera_distance'] = cam_dist
        pack['mesh_scale'] = mesh_scl
        pack['transform_matrix'] = T_mv
        pack['view_valid'] = view_valid
        pack['num_views'] = V
        return pack


class MultiImageConditionedMixin:
    def __init__(self, roots, *, image_size=518, max_image_cond_view = 4, **kwargs):
        self.image_size = image_size
        self.max_image_cond_view = max_image_cond_view
        super().__init__(roots, **kwargs)

    def filter_metadata(self, metadata, dataset_name=None):
        metadata, stats = super().filter_metadata(metadata, dataset_name=dataset_name)
        metadata = metadata[metadata['cond_rendered'].notna()]
        stats['Cond rendered'] = len(metadata)
        return metadata, stats
    
    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)
       
        image_root = os.path.join(root['render_cond'], instance)
        with open(os.path.join(image_root, 'transforms.json')) as f:
            metadata = json.load(f)

        n_views = len(metadata['frames'])
        n_sample_views = np.random.randint(1, self.max_image_cond_view+1)

        assert n_views >= n_sample_views, f'Not enough views to sample {n_sample_views} unique images.'

        sampled_views = np.random.choice(n_views, size=n_sample_views, replace=False)

        cond_images = []
        for v in sampled_views:
            frame_info = metadata['frames'][v]
            image_path = os.path.join(image_root, frame_info['file_path'])
            image = Image.open(image_path)

            alpha = np.array(image.getchannel(3))
            bbox = np.array(alpha).nonzero()
            bbox = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]
            center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
            aug_hsize = hsize
            aug_center = center
            aug_bbox = [
                int(aug_center[0] - aug_hsize),
                int(aug_center[1] - aug_hsize),
                int(aug_center[0] + aug_hsize),
                int(aug_center[1] + aug_hsize),
            ]

            img = image.crop(aug_bbox)
            img = img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            alpha = img.getchannel(3)
            img = img.convert('RGB')
            img = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
            alpha = torch.tensor(np.array(alpha)).float() / 255.0
            img = img * alpha.unsqueeze(0)

            cond_images.append(img)

        pack['cond'] = [torch.stack(cond_images, dim=0)]  # (V,3,H,W)
        return pack
