print("Dice Simulator script starting...", flush=True)
import pybullet as p
import pybullet_data
import time
import numpy as np
import argparse
import geometry
import os
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

def generate_dice_texture(num_faces, cols, rows, filename):
    cell_size = 256
    img = Image.new('RGB', (cols * cell_size, rows * cell_size), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 100)
    except:
        try:
            font = ImageFont.load_default(size=100)
        except:
            font = ImageFont.load_default()
        
    for i in range(num_faces):
        r = i // cols
        c = i % cols
        x = c * cell_size
        y = r * cell_size
        
        # Centered the text
        text = str(i + 1)
             
        if hasattr(draw, 'textbbox'):
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        else:
            tw, th = draw.textsize(text, font=font)
            
        draw.text((x + (cell_size - tw)/2, y + (cell_size - th)/2), text, fill=(0, 0, 0), font=font)
    img.save(filename)

class DiceSimulator:
    def __init__(self, gui=False, mp4=None):
        self.gui = gui
        if gui:
            options = f"--mp4={mp4}" if mp4 else ""
            self.physics_client = p.connect(p.GUI, options=options)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        p.changeDynamics(self.plane_id, -1, restitution=0.6, lateralFriction=0.8)
        
        # Resource cache per dice type
        self.resource_cache = {}

    def create_dice(self, dice_type, pos=[0, 0, 1], orn=[0, 0, 0, 1]):
        if dice_type in self.resource_cache:
            cache = self.resource_cache[dice_type]
            col_id = cache['col_id']
            vis_id = cache['vis_id']
            tex_id = cache['tex_id']
        else:
            obj_str, (cols, rows), num_logical = geometry.get_dice_obj_string(dice_type)
            obj_path = f"temp_{dice_type}.obj"
            with open(obj_path, "w") as f:
                f.write(obj_str)
            
            # Generate texture
            tex_path = f"temp_{dice_type}.png"
            generate_dice_texture(num_logical, cols, rows, tex_path)
            
            # Create collision shape from OBJ
            col_id = p.createCollisionShape(p.GEOM_MESH, fileName=obj_path, meshScale=[0.1, 0.1, 0.1])
            vis_id = p.createVisualShape(p.GEOM_MESH, fileName=obj_path, meshScale=[0.1, 0.1, 0.1], rgbaColor=[1, 1, 1, 1])
            
            # Load and apply texture
            tex_id = p.loadTexture(tex_path)
            
            self.resource_cache[dice_type] = {
                'col_id': col_id,
                'vis_id': vis_id,
                'tex_id': tex_id,
                'files': [obj_path, tex_path]
            }
        
        body_id = p.createMultiBody(
            baseMass=0.01,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=pos,
            baseOrientation=orn
        )
        p.changeDynamics(body_id, -1, restitution=0.6, lateralFriction=0.8, rollingFriction=0.001, spinningFriction=0.001)
        p.changeVisualShape(body_id, -1, textureUniqueId=tex_id)
        
        return body_id

    def simulate_roll(self, body_id, lin_vel, ang_vel, local_normals, max_steps=10000):
        p.resetBaseVelocity(body_id, lin_vel, ang_vel)
        
        def is_at_rest(body_id):
            v, o = p.getBaseVelocity(body_id)
            return np.linalg.norm(v) < 5e-2 and np.linalg.norm(o) < 5e-2

        for step in range(max_steps):
            p.stepSimulation()
            if self.gui:
                time.sleep(1./240.)  # Run in real-time if GUI is enabled
            # Check if at rest
            if is_at_rest(body_id):
                # extra steps to make sure it is at rest
                if self.gui:
                    p.changeVisualShape(body_id, -1, rgbaColor=[0.5, 1.0, 0.5, 1.0])
                for i in range(10):
                    p.stepSimulation()
                    if self.gui:
                        time.sleep(1./240.)  # Run in real-time if GUI is enabled
                if not is_at_rest(body_id):
                    if self.gui:
                        p.changeVisualShape(body_id, -1, rgbaColor=[1.0, 1.0, 1.0, 1.0])
                    continue
                pos, orn = p.getBasePositionAndOrientation(body_id)
                rot_mat = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
                is_flat = False
                for ln in local_normals:
                    wn = np.dot(rot_mat, ln)
                    if wn[2] < -0.985: # Almost pointing straight down
                        is_flat = True
                        break
                
                if is_flat:
                    # TURN GREEN and wait 100 steps
                    if self.gui:
                        p.changeVisualShape(body_id, -1, rgbaColor=[0.2, 1.0, 0.2, 1.0])
                        for _ in range(100):
                            p.stepSimulation()
                            time.sleep(1./240.)
                    break
                else:
                    pass
                    # Give it a tiny nudge if perfectly balanced on an edge
                    #p.applyExternalTorque(body_id, -1, np.random.uniform(-0.01, 0.01, 3), p.WORLD_FRAME)
                    
            if step == max_steps - 1:
                print(f"Warning: Dice did not come to rest after {max_steps} steps.")
        
        return p.getBasePositionAndOrientation(body_id)

    def get_result(self, body_id, vertices, faces, dice_type):
        """
        Determine which face is landed. 
        For D4, we find the highest vertex. 
        For others, we find the most vertical face normal.
        """
        pos, orn = p.getBasePositionAndOrientation(body_id)
        rot_mat = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        
        max_dot = -1
        best_idx = -1
        
        if dice_type == 'd4':
            for i, v in enumerate(vertices):
                world_v = np.dot(rot_mat, v)
                dot = world_v[2]
                if dot > max_dot:
                    max_dot = dot
                    best_idx = i
            return best_idx, max_dot
        
        # Calculate local normals
        local_normals = []
        for f in faces:
            v0 = np.array(vertices[f[0]])
            v1 = np.array(vertices[f[1]])
            v2 = np.array(vertices[f[2]])
            n = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(n)
            if norm > 1e-9:
                n = n / norm
            local_normals.append(n)
            
        for i, n in enumerate(local_normals):
            world_n = np.dot(rot_mat, n)
            dot = world_n[2]
            if dot > max_dot:
                max_dot = dot
                best_idx = i
                
        return best_idx, max_dot

    def close(self):
        p.disconnect()

def run_simulation(dice_type='d6', num_rolls=10, gui=False, verbose=False, mp4=None, extra_energy=1.0, 
                   lin_vel_scale=1.0, up_vel_scale=1.0, ang_vel_scale=1.0):
    sim = DiceSimulator(gui=gui, mp4=mp4)
    results = []
    square_count = 0
    pbar = tqdm(range(num_rolls), disable=verbose, desc=f"Rolling {dice_type.upper()}")
    
    for i in pbar:
        if verbose:
            print(f"  Starting roll {i+1}...", flush=True)
        # Random initial conditions
        pos = [0, 0, 0.2]
        orn = p.getQuaternionFromEuler(np.random.uniform(0, 2*np.pi, 3))
        lin_vel = np.random.uniform(-1, 1, 3) * 2 * extra_energy * lin_vel_scale
        lin_vel[2] = np.random.uniform(3, 6) * extra_energy * up_vel_scale # upward toss
        ang_vel = np.random.uniform(-10, 10, 3) * extra_energy * ang_vel_scale
        
        v, f = geometry.get_dice_geometry(dice_type) # get the geometry for result checking
        
        local_normals = []
        for face in f:
            v0 = np.array(v[face[0]])
            v1 = np.array(v[face[1]])
            v2 = np.array(v[face[2]])
            n = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(n)
            if norm > 1e-9:
                n = n / norm
            local_normals.append(n)
        
        if verbose: print(f"    Creating dice for roll {i+1}...", flush=True)
        body_id = sim.create_dice(dice_type, pos=pos, orn=orn)
        if verbose: print(f"    Simulating roll {i+1}...", flush=True)
        sim.simulate_roll(body_id, lin_vel, ang_vel, local_normals, max_steps=100000)
        if verbose: print(f"    Getting result for roll {i+1}...", flush=True)
        face_idx, dot = sim.get_result(body_id, v, f, dice_type)
        # Default mapping fallback if we haven't mapped numbers yet
        mapped_val = face_idx + 1
            
        if verbose: print(f"    Result obtained: face_idx={face_idx} (Val: {mapped_val}), dot={dot:.4f}... cleaning up.", flush=True)
        results.append((mapped_val, dot))
        
        if dice_type.lower() == 'cuboctahedron':
            if mapped_val > 8:
                square_count += 1
            if not verbose:
                pbar.set_postfix(sq_pct=f"{square_count/(i+1)*100:.1f}%")
        
        # Cleanup body for next roll
        p.removeBody(body_id)
            
    # Final cleanup of resources
    for cache in sim.resource_cache.values():
        for f_path in cache['files']:
            if os.path.exists(f_path):
                os.remove(f_path)
    sim.resource_cache = {}
            
    sim.close()
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dice Simulator")
    parser.add_argument('--dice', type=str, default='d6', help="Dice type (e.g. d4, d6, d8, d10, d12, d20, cuboctahedron)")
    parser.add_argument('--rolls', type=int, default=5, help="Number of rolls")
    parser.add_argument('--gui', action='store_true', help="Enable GUI visualization")
    parser.add_argument('--mp4', type=str, default=None, help="Save GUI run to MP4 file (enables GUI)")
    parser.add_argument('--verbose', '-v', action='store_true', help="Enable verbose printouts per roll")
    parser.add_argument('--extra-energy', type=float, default=1.0, help="Velocity multiplier for throws")
    parser.add_argument('--lin-vel-scale', type=float, default=1.0, help="Linear velocity scale factor")
    parser.add_argument('--up-vel-scale', type=float, default=1.0, help="Upward velocity scale factor")
    parser.add_argument('--ang-vel-scale', type=float, default=1.0, help="Angular velocity scale factor")
    args = parser.parse_args()

    if args.mp4:
        args.gui = True

    print(f"\n--- Simulation Results (Extra Energy: {args.extra_energy}) ---")
    print(f"Running {args.rolls} rolls of {args.dice.upper()}...", flush=True)
    res = run_simulation(args.dice, num_rolls=args.rolls, gui=args.gui, verbose=args.verbose, 
                         mp4=args.mp4, extra_energy=args.extra_energy, 
                         lin_vel_scale=args.lin_vel_scale, up_vel_scale=args.up_vel_scale, 
                         ang_vel_scale=args.ang_vel_scale)
    
    from collections import Counter
    counts = Counter(val for val, d in res)
    if args.verbose:
        for i, (val, d) in enumerate(res):
            print(f"Roll {i+1}: Landed on Value {val} (Face Index {val-1}) with Up-Dot={d:.4f}", flush=True)
            
    print("\n--- Simulation Results ---")
    print(f"Total Rolls: {args.rolls}")
    
    triangle = 0
    square = 0
    for val in sorted(counts.keys()):
        count = counts[val]
        if val <= 8:
            triangle += count
        else:
            square += count
        percentage = (count / args.rolls) * 100
        print(f"Value {val:2d}: {count:5d} times ({percentage:5.2f}%)")
        
    if args.dice.lower() == 'cuboctahedron':
        print(f"Triangle Face Landings: {triangle} times ({triangle/args.rolls*100:.2f}%)")
        print(f"Square Face Landings: {square} times ({square/args.rolls*100:.2f}%)")
