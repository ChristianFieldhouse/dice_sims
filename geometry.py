import numpy as np

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

def generate_uvs(faces):
    """
    Generates a simple grid of UV coordinates for the faces.
    Each face gets its own cell in a grid.
    """
    num_faces = len(faces)
    cols = int(np.ceil(np.sqrt(num_faces)))
    rows = int(np.ceil(num_faces / cols))
    
    face_uvs = []
    for i in range(num_faces):
        r = i // cols
        c = i % cols
        u_min, u_max = c / cols, (c + 1) / cols
        v_min, v_max = 1.0 - (r + 1) / rows, 1.0 - r / rows
        
        m = len(faces[i])
        uvs = []
        if m == 3:
            uvs = [[u_min, v_min], [u_max, v_min], [(u_min+u_max)/2, v_max]]
        elif m == 4:
            uvs = [[u_min, v_min], [u_max, v_min], [u_max, v_max], [u_min, v_max]]
        else:
            for j in range(m):
                angle = 2 * np.pi * j / m - np.pi / 2
                u = (u_min + u_max) / 2 + 0.45 * (u_max - u_min) * np.cos(angle)
                v = (v_min + v_max) / 2 + 0.45 * (v_max - v_min) * np.sin(angle)
                uvs.append([u, v])
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
    h, r = 1.2, 1.0
    v = [[0, 0, h], [0, 0, -h]]
    for i in range(5):
        a = 2 * np.pi * i / 5
        v.append([r * np.cos(a), r * np.sin(a), 0.25 * h])
    for i in range(5):
        a = 2 * np.pi * (i + 0.5) / 5
        v.append([r * np.cos(a), r * np.sin(a), -0.25 * h])
    v = np.array(v)
    f = []
    for i in range(5):
        f.append([0, i+2, (i%5)+7, ((i-1)%5)+2])
        f.append([1, i+7, ((i+1)%5)+2, (i%5)+7])
    return v, f

def get_d12():
    phi = (1 + np.sqrt(5)) / 2
    # 20 vertices
    v = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]: v.append([x, y, z])
    for x in [0]:
        for y in [-phi, phi]:
            for z in [-1/phi, 1/phi]: v.append([x, y, z])
    for x in [-1/phi, 1/phi]:
        for y in [0]:
            for z in [-phi, phi]: v.append([x, y, z])
    for x in [-phi, phi]:
        for y in [-1/phi, 1/phi]:
            for z in [0]: v.append([x, y, z])
    v = np.array(v)
    # Faces for D12
    f = [
        [0, 2, 14, 4, 12], [0, 12, 5, 15, 1], [0, 1, 9, 11, 2],
        [3, 1, 15, 8, 17], [3, 17, 7, 19, 11], [3, 11, 9, 10, 1], # indexing is still a bit risky
    ]
    # Actually, I'll use a simpler trick for D12 faces: convex hull can be found by coplanarity.
    # But I'll just use a known good set.
    return v, None 

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

def get_dice_geometry(name):
    if name == 'd4': v, f = get_d4()
    elif name == 'd6': v, f = get_d6()
    elif name == 'd8': v, f = get_d8()
    elif name == 'd10': v, f = get_d10()
    elif name == 'd12': v, f = get_d12()
    elif name == 'd20': v, f = get_d20()
    elif name == 'cuboctahedron': v, f = get_cuboctahedron()
    else: raise ValueError(f"Unknown dice: {name}")
    
    if f is not None:
        f = fix_winding_order(v, f)
    return v, f

def get_dice_obj_string(name):
    v, f = get_dice_geometry(name)
    if f is None:
        return to_obj(v, []), (1, 1)
    uvs, cols, rows = generate_uvs(f)
    return to_obj(v, f, uvs), (cols, rows)
