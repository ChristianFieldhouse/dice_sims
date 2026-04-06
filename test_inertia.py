import pybullet as p, geometry
p.connect(p.DIRECT)
v, f = geometry.get_dice_geometry('d6')
obj = geometry.get_dice_obj_string('d6')
with open('temp_test.obj', 'w') as fh: fh.write(obj)
col = p.createCollisionShape(p.GEOM_MESH, fileName='temp_test.obj', meshScale=[0.1,0.1,0.1])
body = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=col)
dyn = p.getDynamicsInfo(body, -1)
print(f"Mass: {dyn[0]}, Inertia: {dyn[2]}")
