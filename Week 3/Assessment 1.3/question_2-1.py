import numpy as np
import spatialmath as sm
import spatialmath.base as smb
import roboticstoolbox as rtb

# Replicate what the autograder sample test almost certainly does
wTa = sm.SE3()
wTb = sm.SE3.Tx(2.0)
n = 5

# Build world traj using mtraj on twist
Ta = wTa.A
aTb = wTa.inv().A @ wTb.A
twist_ab = smb.trlog(aTb, twist=True)
traj_data = rtb.mtraj(rtb.quintic, np.zeros(6), twist_ab, n)
world_traj = sm.SE3.Empty()
for q in traj_data.q:
    world_traj.append(sm.SE3(Ta @ smb.trexp(q), check=False))

# Build relative traj using SE3_traj_relative
rel_traj = sm.SE3.Empty()
for i in range(1, len(world_traj)):
    rel = world_traj[i-1].inv() * world_traj[i]
    rel_traj.append(rel)

# Now simulate rocket_diagnostics looking for world_traj[2]
target = world_traj[2].A
current = np.eye(4)

print(f"Target pose x={target[0,3]:.6f}")
print(f"Identity matches target: {np.allclose(current, target, atol=1e-4)}")

for i, pose in enumerate(rel_traj):
    current = current @ pose.A
    match = np.allclose(current, target, atol=1e-4)
    print(f"After step i={i}: x={current[0,3]:.6f}, match={match}, time=(i+1)*0.5={(i+1)*0.5}")
    if match:
        print(f"  --> Returns {(i+1)*0.5}")
        break