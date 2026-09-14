from numpy import ndarray
from typing import Optional, Tuple, Dict

import numpy as np

def assert_ndarray(arr, name: str="arr", shape: Optional[Tuple[int, ...]]=None, dtype=None):
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"{name} must be a numpy.ndarray or None, got {type(arr)}")
    if shape is not None:
        # shape may contain None as wildcard
        if len(shape) != arr.ndim:
            raise ValueError(f"{name}: expected shape length {len(shape)} but array ndim is {arr.ndim}")
        for i, (exp, actual) in enumerate(zip(shape, arr.shape)):
            if exp > 0 and exp != actual:
                raise ValueError(f"{name} shape mismatch at axis {i}: expected {exp}, got {actual}")
    if dtype is not None:
        if not np.issubdtype(arr.dtype, dtype):
            raise ValueError(f"{name} dtype must be {dtype}, got {arr.dtype}")

# 三角面片随机采样
def sample_surface(
    num_samples: int,
    vertices: ndarray,
    faces: ndarray,
    mask: Optional[ndarray]=None,
    face_index: Optional[ndarray]=None,
    random_lengths: Optional[ndarray]=None,
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    '''
    Randomly pick samples proportional to face area.
    
    See sample_surface: https://github.com/mikedh/trimesh/blob/main/trimesh/sample.py
    
    Args:
        mask: (num_faces,), only sample points on the faces where value is True.
    Return:
        vertex_samples: sampled vertices
        
        original_face_index: on which face is sampled
        
        face_index: sampled faces
        
        random_lengths: sampled vectors on face
    '''
    original_face_indices = np.arange(len(faces))
    # sample according to mask
    if mask is not None:
        original_face_indices = original_face_indices[mask]
        faces = faces[mask]
    if face_index is None:
        # get face area
        offset_0 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
        offset_1 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
        face_weight = np.linalg.norm(np.cross(offset_0, offset_1, axis=-1), axis=-1)
        
        weight_cum = np.cumsum(face_weight, axis=0)
        face_pick = np.random.rand(num_samples) * weight_cum[-1]
        _face_index = np.searchsorted(weight_cum, face_pick)
    else:
        _face_index = face_index
    # map face_index back to original indices
    original_face_index = original_face_indices[_face_index]
    
    # pull triangles into the form of an origin + 2 vectors
    tri_origins = vertices[faces[:, 0]]
    tri_vectors = vertices[faces[:, 1:]]
    tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))

    # pull the vectors for the faces we are going to sample from
    tri_origins = tri_origins[_face_index]
    tri_vectors = tri_vectors[_face_index]
    
    if random_lengths is None:
        # randomly generate two 0-1 scalar components to multiply edge vectors b
        random_lengths = np.random.rand(len(tri_vectors), 2, 1)
    
    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)
    
    sample_vector = (tri_vectors * random_lengths).sum(axis=1)
    vertex_samples = sample_vector + tri_origins
    return vertex_samples, original_face_index, _face_index, random_lengths

def sample_barycentric(
    vertex_group: ndarray,
    faces: ndarray,
    face_index: ndarray,
    random_lengths: ndarray,
) -> ndarray:
    v_origins = vertex_group[faces[face_index, 0]]
    v_vectors = vertex_group[faces[face_index, 1:]]
    v_vectors -= v_origins[:, np.newaxis, :]
    
    sample_vector = (v_vectors * random_lengths).sum(axis=1)
    v_samples = sample_vector + v_origins
    return v_samples

def sample_vertex_groups(
    vertices: ndarray,
    faces: ndarray,
    num_samples: int,
    num_vertex_samples: Optional[int]=None,
    vertex_normals: Optional[ndarray]=None,
    face_normals: Optional[ndarray]=None,
    vertex_groups: Optional[ndarray]=None,
    face_mask: Optional[ndarray]=None,
    deterministic_params: Optional[Dict[str, ndarray]]=None,
) -> Tuple[ndarray, Optional[ndarray], Optional[ndarray], Dict[str, ndarray]]:
    """
    Choose num_samples samples on the mesh and get their positions and normals.
    If vertex_group is provided, get its weights using barycentric sampling.
    
    Return:
        sampled_vertices, sampled_normals, sampled_vertex_groups, deterministic_params
    
    Args:
        vertices: (N, 3)
        
        faces: (F, 3)
        
        num_samples: how many samples
        
        num_vertex_samples:
            At most num_vertex_samples unique vertices to be included,
            these points will be concatenated in the last (if shuffle is False).
        
        vertex_normals: (N, 3), sampled_normals will be None if not provided
        
        face_normals: (N, 3), sampled_normals will be None if not provided
        
        vertex_groups: (N, m), sampled_vertex_groups will be None if not provided
        
        face_mask:
            (F,) or (F, m), if shape is (F,), use the same mask across all
            vertex groups. Only sample on faces where value is True.
        
        deterministic_params:
            A dict of parameters to be used directly instead of random sampling.
    """
    
    if num_vertex_samples is None:
        num_vertex_samples = 0
    if num_vertex_samples > num_samples:
        raise ValueError(f"num_vertex_samples cannot be larger than num_samples, found: {num_vertex_samples} > {num_samples}")
    
    # 原始顶点采样
    def get_mask_perm(mask: Optional[ndarray]):
        if mask is None:
            vertex_mask = np.arange(vertices.shape[0])
        else:
            vertex_mask = np.unique(mask)
        perm = np.random.permutation(vertex_mask.shape[0])
        return vertex_mask[perm[:num_vertex_samples]]
    
    if vertex_groups is not None:
        if vertex_groups.ndim == 1:
            assert_ndarray(arr=vertex_groups, name="vertex_groups", shape=(vertices.shape[0],))
            vertex_groups = vertex_groups[:, None]
        else:
            assert_ndarray(arr=vertex_groups, name="vertex_groups", shape=(vertices.shape[0], -1))
            vertex_groups = vertex_groups
    
    if vertex_groups is not None:
        if face_mask is not None:
            assert_ndarray(arr=face_mask, name="mask", shape=(faces.shape[0],))
        perm = None
        _mask = None
        if deterministic_params is not None:
            perm = deterministic_params['perm']
            origin_face_index = deterministic_params['original_face_index']
            face_index = deterministic_params['face_index']
            random_lengths = deterministic_params['random_lengths']
            _num_samples = num_samples - len(perm)
            face_vertices, origin_face_index, face_index, random_lengths = sample_surface(
                num_samples=_num_samples,
                vertices=vertices,
                faces=faces,
                mask=_mask,
                face_index=face_index,
                random_lengths=random_lengths,
            )
        else:
            if face_mask is not None:
                assert face_mask.ndim == 1
                perm = get_mask_perm(faces[face_mask])
                _mask = face_mask
            else:
                perm = get_mask_perm(None)
                _mask = None
            _num_samples = num_samples - len(perm)
            
            face_vertices, origin_face_index, face_index, random_lengths = sample_surface(
                num_samples=_num_samples,
                vertices=vertices,
                faces=faces,
                mask=_mask,
            )
        
        sampled_vertices = np.concatenate([vertices[perm], face_vertices], axis=0)
        if vertex_normals is not None and face_normals is not None:
            sampled_normals = np.concatenate([vertex_normals[perm], face_normals[origin_face_index]], axis=0)
        else:
            sampled_normals = None
        
        g = sample_barycentric(
            vertex_group=vertex_groups,
            faces=faces,
            face_index=face_index,
            random_lengths=random_lengths,
        )
        sampled_vertex_groups = np.concatenate([vertex_groups[perm], g], axis=0)
        
    else: # otherwise only sample vertices and normals
        if deterministic_params is not None:
            perm = deterministic_params['perm']
            face_index = deterministic_params['face_index']
            origin_face_index = deterministic_params['original_face_index']
            random_lengths = deterministic_params['random_lengths']
            num_samples -= len(perm)
            face_vertices, origin_face_index, face_index, random_lengths = sample_surface(
                num_samples=num_samples,
                vertices=vertices,
                faces=faces,
                mask=face_mask,
                face_index=face_index,
                random_lengths=random_lengths,
            )
        else:
            if face_mask is not None:
                assert_ndarray(arr=face_mask, name="mask", shape=(faces.shape[0],))
                perm = get_mask_perm(faces[face_mask])
            else:
                perm = get_mask_perm(None)
            num_samples -= len(perm)
            face_vertices, origin_face_index, face_index, random_lengths = sample_surface(
                num_samples=num_samples,
                vertices=vertices,
                faces=faces,
                mask=face_mask,
            )
        n_vertex = vertices[perm]
        sampled_vertices = np.concatenate([n_vertex, face_vertices], axis=0)
        if vertex_normals is not None and face_normals is not None:
            sampled_normals = np.concatenate([vertex_normals[perm], face_normals[origin_face_index]], axis=0)
        else:
            sampled_normals = None
        sampled_vertex_groups = None
    d = {
        "perm": perm,
        "original_face_index": origin_face_index,
        "face_index": face_index,
        "random_lengths": random_lengths,
    }
    return sampled_vertices, sampled_normals, sampled_vertex_groups, d

def random_euler_rotation(
    batch_size: int=1,
    x_range=(0, 0), # degree
    y_range=(0, 0), # degree
    z_range=(0, 0), # degree
    degrees=True,
    return_4x4=True,
):
    from scipy.spatial.transform import Rotation
    """
    Generate random rotation matrices with independent angle control per axis.
    x_range/y_range/z_range: tuple or list (min_deg, max_deg)
    return_4x4: if True, returns (B, 4, 4) homogeneous matrices
    """
    # random degrees for each axis
    x_deg = np.random.uniform(*x_range, size=batch_size)
    y_deg = np.random.uniform(*y_range, size=batch_size)
    z_deg = np.random.uniform(*z_range, size=batch_size)
    
    # build rotations: Z * Y * X
    rot = Rotation.from_euler("ZYX", np.vstack([z_deg, y_deg, x_deg]).T, degrees=degrees)
    mats = rot.as_matrix().astype(np.float32)  # (B,3,3)

    if not return_4x4:
        return mats
    
    # convert to homogeneous matrices
    mats4 = np.zeros((batch_size, 4, 4), dtype=np.float32)
    mats4[:, :3, :3] = mats
    mats4[:, 3, 3] = 1.0
    return mats4
