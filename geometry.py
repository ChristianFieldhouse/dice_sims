import numpy as np
import json
import os

def fix_winding_order(vertices, faces):
    """
    Ensures all faces of a convex polyhedron have vertices ordered counter-clockwise
    so their normals point strictly outward.
    """
    center = np.mean(vertices, axis=0)
    fixed_faces = []
    for f in faces:
        v0 = np.array(vertices[f[0]])
        v1 = np.array(vertices[f[1]])
        v2 = np.array(vertices[f[2]])
        n = np.cross(v1 - v0, v2 - v0)
        
        # Check dot product with outward vector
        outward = v0 - center
        if np.dot(n, outward) < 0:
            # Reverse winding order
            if len(f) == 3:
                fixed_faces.append([f[0], f[2], f[1]])
            elif len(f) == 4:
                fixed_faces.append([f[0], f[3], f[2], f[1]])
            else:
                fixed_faces.append(list(reversed(f)))
        else:
            fixed_faces.append(f)
    return fixed_faces

def to_obj(vertices, faces, face_uvs=None):
    lines = []
    for v in vertices:
        lines.append(f"v {v[0]} {v[1]} {v[2]}")
    
    if face_uvs:
        for uvs in face_uvs:
            for u, v_coord in uvs:
                lines.append(f"vt {u} {v_coord}")
        
        vt_idx = 1
        for f, uvs in zip(faces, face_uvs):
            f_parts = []
            for v_idx in f:
                f_parts.append(f"{v_idx+1}/{vt_idx}")
                vt_idx += 1
            lines.append(f"f {' '.join(f_parts)}")
    else:
        for f in faces:
            if isinstance(f, (list, np.ndarray)):
                lines.append(f"f {' '.join(str(i+1) for i in f)}")
    return "\n".join(lines)

def generate_uvs(vertices, faces, dice_type=None):
    """
    Generates high-quality, aspect-ratio-preserving UV coordinates for every face.
    Projects each face onto its own 2D plane and fits it into a texture cell.
    """
    num_faces = len(faces)
    cols = int(np.ceil(np.sqrt(num_faces)))
    rows = int(np.ceil(num_faces / cols))
    
    face_uvs = []
    for i in range(num_faces):
        # 1. Get face vertices
        f = faces[i]
        fv = np.array([vertices[idx] for idx in f])
        
        # 2. Define local coordinate system for the face
        v0 = fv[0]
        v1 = fv[1]
        v2 = fv[2]
        normal = np.cross(v1 - v0, v2 - v0)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-9:
            # Degenerate face fallback
            face_uvs.append([[0, 0]] * len(f))
            continue
        normal /= norm_len
        
        # e1 and e2 axes for the 2D plane
        e1 = v1 - v0
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(normal, e1)
        e2 /= np.linalg.norm(e2)
        
        # 3. Project to 2D
        pts2d = []
        for v in fv:
            vec = v - v0
            pts2d.append([np.dot(vec, e1), np.dot(vec, e2)])
        pts2d = np.array(pts2d)
        
        # 4. Find bounding box in 2D
        p_min = np.min(pts2d, axis=0)
        p_max = np.max(pts2d, axis=0)
        p_rng = p_max - p_min
        p_rng[p_rng < 1e-9] = 1e-9 # avoid div zero
        
        # 5. Determine aspect ratio and scaling to fit in [0, 1] square
        # We want to fit the face into a cell while keeping it proportional.
        # We'll use a larger margin (0.95) to give the face more territory
        # which makes the centered text look smaller/better-fitted.
        margin = 0.95
        aspect = p_rng[0] / p_rng[1]
        
        if aspect > 1.0:
            scale = margin / p_rng[0]
            off_x = (1.0 - margin) / 2
            off_y = (1.0 - p_rng[1] * scale) / 2
        else:
            scale = margin / p_rng[1]
            off_y = (1.0 - margin) / 2
            off_x = (1.0 - p_rng[0] * scale) / 2
            
        # 6. Map to cell [u_min..u_max, v_min..v_max]
        r = i // cols
        c = i % cols
        u_min, u_max = c / cols, (c + 1) / cols
        v_min, v_max = 1.0 - (r + 1) / rows, 1.0 - r / rows
        
        uvs = []
        for p in pts2d:
            # Local cell coords [0..1]
            local_u = off_x + (p[0] - p_min[0]) * scale
            local_v = off_y + (p[1] - p_min[1]) * scale
            # Global texture coords
            global_u = u_min + local_u * (1.0 / cols)
            global_v = v_min + local_v * (1.0 / rows)
            uvs.append([global_u, global_v])
        face_uvs.append(uvs)
        
    return face_uvs, cols, rows

def get_d4():
    v = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
    f = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]
    return v, f

def get_d6():
    v = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
    ], dtype=float)
    f = [
        [0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
        [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]
    ]
    return v, f

def get_d8():
    v = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
    ], dtype=float)
    f = [
        [4, 0, 2], [4, 2, 1], [4, 1, 3], [4, 3, 0],
        [5, 2, 0], [5, 1, 2], [5, 3, 1], [5, 0, 3]
    ]
    return v, f

def get_d10():
    # To make the kite faces perfectly planar, the ratio of H to Z must be
    # (1 + cos(36)) / (1 - cos(36)) approx 9.472
    h = 1.4
    z = h / 9.4721359
    r = 1.0
    v = [[0, 0, h], [0, 0, -h]]
    for i in range(5):
        a = 2 * np.pi * i / 5
        v.append([r * np.cos(a), r * np.sin(a), z])
    for i in range(5):
        a = 2 * np.pi * (i + 0.5) / 5
        v.append([r * np.cos(a), r * np.sin(a), -z])
    v = np.array(v)
    f = []
    for i in range(5):
        # Upper faces (connected to vertex 0) - kite quad
        u_idx, u_next = i+2, ((i+1)%5)+2
        l_idx = i+7
        f.append([0, u_idx, l_idx, u_next])
        
        # Lower faces (connected to vertex 1) - kite quad
        l_idx, l_prev = i+7, ((i-1)%5)+7
        u_curr = i+2
        f.append([1, l_idx, u_curr, l_prev])
    return v, f

def get_d12():
    phi = (1 + np.sqrt(5)) / 2
    inv_phi = 1 / phi
    
    # 20 vertices
    v = [
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
        [0, -inv_phi, -phi], [0, -inv_phi, phi], [0, inv_phi, -phi], [0, inv_phi, phi],
        [-inv_phi, -phi, 0], [-inv_phi, phi, 0], [inv_phi, -phi, 0], [inv_phi, phi, 0],
        [-phi, 0, -inv_phi], [phi, 0, -inv_phi], [-phi, 0, inv_phi], [phi, 0, inv_phi]
    ]
    v = np.array(v)
    
    # Faces (12 pentagons)
    f = [
        [8, 10, 2, 16, 0], [8, 4, 14, 12, 0], [8, 10, 6, 17, 4],
        [9, 11, 3, 18, 1], [9, 5, 14, 12, 1], [9, 11, 7, 19, 5],
        [10, 2, 13, 15, 6], [11, 3, 13, 15, 7], [12, 0, 16, 18, 1],
        [13, 2, 16, 18, 3], [14, 4, 17, 19, 5], [15, 6, 17, 19, 7]
    ]
    return v, f

def get_d20():
    phi = (1 + np.sqrt(5)) / 2
    v = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    f = [
        [0, 8, 4], [0, 4, 6], [0, 6, 10], [0, 10, 2], [0, 2, 8],
        [8, 9, 4], [4, 1, 6], [6, 11, 10], [10, 7, 2], [2, 5, 8],
        [3, 9, 1], [3, 1, 11], [3, 11, 7], [3, 7, 5], [3, 5, 9],
        [4, 9, 1], [6, 1, 11], [10, 11, 7], [2, 7, 5], [8, 5, 9]
    ]
    return v, f

def get_cuboctahedron():
    v = np.array([
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1]
    ], dtype=float)
    f = [
        # 8 octants (triangles)
        [0, 4, 8], [1, 4, 10], [2, 6, 8], [3, 6, 10],
        [0, 5, 9], [1, 5, 11], [2, 7, 9], [3, 7, 11],
        # 6 planes (squares)
        [0, 4, 1, 5], [2, 6, 3, 7],     # x planes
        [0, 8, 2, 9], [1, 10, 3, 11],   # y planes
        [4, 8, 6, 10], [5, 9, 7, 11]    # z planes
    ]
    return v, f

def compute_convex_hull(vertices):
    """
    Computes faces of a convex hull from a point cloud.
    Returns a list of faces, where each face is a list of vertex indices.
    """
    n = len(vertices)
    if n < 4: return []
    
    triangles = []
    center = np.mean(vertices, axis=0)
    
    # 1. Find all triplets that form a hull face
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v0, v1, v2 = vertices[i], vertices[j], vertices[k]
                norm = np.cross(v1 - v0, v2 - v0)
                l = np.linalg.norm(norm)
                if l < 1e-9: continue
                norm /= l
                
                # Ensure normal points away from center
                if np.dot(norm, v0 - center) < 0:
                    norm = -norm
                    
                # Check if all other points are "behind" the plane
                dots = np.dot(vertices - v0, norm)
                if np.all(dots < 1e-5):
                    # Potential face
                    triangles.append({'indices': [i, j, k], 'normal': norm})
    
    # 2. Merge coplanar triangles into larger polygons
    # For Archimedean solids, neighbors are coplanar if they share an edge and have same normal
    faces = []
    used_triangles = [False] * len(triangles)
    
    for i in range(len(triangles)):
        if used_triangles[i]: continue
        
        current_face_indices = set(triangles[i]['indices'])
        current_normal = triangles[i]['normal']
        used_triangles[i] = True
        
        # Grow the face by adding coplanar triangles that share at least 2 vertices
        changed = True
        while changed:
            changed = False
            for j in range(len(triangles)):
                if used_triangles[j]: continue
                if np.dot(current_normal, triangles[j]['normal']) > 0.9999:
                    shared = current_face_indices.intersection(triangles[j]['indices'])
                    if len(shared) >= 2:
                        current_face_indices.update(triangles[j]['indices'])
                        used_triangles[j] = True
                        changed = True
        
        # 3. Order vertices of the merged polygon
        # This is a bit tricky. We have a set of vertices on a plane.
        # We'll project to 2D and sort by angle from centroid.
        face_list = list(current_face_indices)
        if len(face_list) < 3: continue
        
        face_v = vertices[face_list]
        face_center = np.mean(face_v, axis=0)
        
        # Local 2D space
        v0 = vertices[face_list[0]]
        e1 = vertices[face_list[1]] - v0
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(current_normal, e1)
        
        angles = []
        for idx in face_list:
            vec = vertices[idx] - face_center
            angles.append(np.arctan2(np.dot(vec, e2), np.dot(vec, e1)))
            
        sorted_indices = [face_list[idx] for idx in np.argsort(angles)]
        faces.append(sorted_indices)
        
    return faces

_archimedean_data = None
_hull_cache = {}

def load_archimedean():
    global _archimedean_data
    if _archimedean_data is not None: return _archimedean_data
    
    path = os.path.join(os.path.dirname(__file__), "archimedean.json")
    if not os.path.exists(path):
        _archimedean_data = {}
        return {}
        
    with open(path, "r") as f:
        _archimedean_data = json.load(f)
    return _archimedean_data

def get_dice_geometry(name):
    lower_name = name.lower()
    if lower_name == 'd4': v, f = get_d4()
    elif lower_name == 'd6': v, f = get_d6()
    elif lower_name == 'd8': v, f = get_d8()
    elif lower_name == 'd10': v, f = get_d10()
    elif lower_name == 'd12': v, f = get_d12()
    elif lower_name == 'd20': v, f = get_d20()
    elif lower_name == 'cuboctahedron': v, f = get_cuboctahedron()
    else:
        # Check hull cache first
        if lower_name in _hull_cache:
            return _hull_cache[lower_name]
            
        data = load_archimedean()
        # Search by name (case-insensitive)
        match = None
        for key in data:
            if key.lower() == lower_name:
                match = data[key]
                break
        
        if match:
            v = np.array(match['vertices'])
            f = compute_convex_hull(v)
            _hull_cache[lower_name] = (v, f)
        else:
            raise ValueError(f"Unknown dice: {name}")
    
    if f is not None:
        f = fix_winding_order(v, f)
    return v, f

def get_dice_obj_string(name):
    v, f = get_dice_geometry(name)
    if f is None:
        return to_obj(v, []), (1, 1), 0
    uvs, cols, rows = generate_uvs(v, f, name)
    return to_obj(v, f, uvs), (cols, rows), len(f)
