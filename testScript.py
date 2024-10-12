# file สำหรับตรวจคำตอบ
# ในกรณีที่มีการสร้าง function อื่น ๆ ให้ระบุว่า input-output คืออะไรด้วย

import FRA333_HW3_6534_6535 as HW3
import numpy as np
import math
import HW3_utils

import roboticstoolbox as rtb
from spatialmath import SE3

'''
ชื่อ_รหัส(ex: ธนวัฒน์_6461)
1. นันท์นภัส_6534
2. บัซลาอ์_6535
'''
# Define Robot
d_1 = 0.0892
a_2 = -0.425
a_3 = -0.39243
d_4 = 0.109
d_5 = 0.093
d_6 = 0.082

# Define a 3-DOF revolute robot using MDH parameters
T3_e = SE3(a_3-d_6,-d_5,d_4) * SE3.RPY(0,-np.pi/2,0)

robot = rtb.DHRobot([
    rtb.RevoluteMDH(d=d_1, offset=np.pi),  # Link 1
    rtb.RevoluteMDH(alpha=np.pi/2),          # Link 2
    rtb.RevoluteMDH(a=a_2)           # Link 3
]
, tool = T3_e, name="3R robot")

print(robot)

# #===========================================<ตรวจคำตอบข้อ 1>====================================================#
#code here

print("------Prove endEffectorJacobianHW3 function-------")

# Define the joint angles (q) for prove
q = [np.pi/4, np.pi/6, np.pi/3]

# Calculate the Jacobian at base using endEffectorJacobianHW3 function
J_sol = HW3.endEffectorJacobianHW3(q)
print("Jacobian calculate from endEffectorJacobianHW3:")
print(J_sol)
print()

# Calculate the Jacobian at base using roboticstoolbox
jacobian_toolbox = robot.jacob0(q)
print("Jacobian calculate from roboticstoolbox:")
print(jacobian_toolbox)
print()

# Element-wise comparison with a tolerance
if np.allclose(J_sol, jacobian_toolbox, atol=1e-6):
    print("The two Jacobians are very close.\n")
else:
    print("The two Jacobians are different.\n")

#==============================================================================================================#
#===========================================<ตรวจคำตอบข้อ 2>====================================================#
#code here

print("------Prove checkSingularityHW3 function-------\n")

# check case of singularity
print("------check case of singularity-------")
q = [0,-np.pi/2 - 0.1,0] # Define q that make robot be singularity

# check singularity using checkSingularityHW3 function
checkSingularity = HW3.checkSingularityHW3(q)
print("checkSingularity function:", checkSingularity)

# check singularity using roboticstoolbox
print("check singularity using roboticstoolbox:")
# Calculate the Jacobian at base using roboticstoolbox
jacobian_toolbox = robot.jacob0(q)
velocity_jacobian = jacobian_toolbox[:3, :] # reduce jacobian
determinant = np.linalg.det(velocity_jacobian) # calculate det(J)
print(f"Determinant of the velocity Jacobian (3x3): {determinant}")

if np.isclose(determinant, 0, atol=0.001):
    print("singularity\n")
else:
    print("not singularity\n")

# check case of not singularity
print("------check case of not singularity-------")
q = [np.pi/4, np.pi/6, np.pi/3] # Define q that make robot not singularity

# check singularity using checkSingularityHW3 function
checkSingularity = HW3.checkSingularityHW3(q)
print("checkSingularity function:", checkSingularity)

# check singularity using roboticstoolbox
print("check singularity using roboticstoolbox:")
# Calculate the Jacobian at base using roboticstoolbox
jacobian_toolbox = robot.jacob0(q)
velocity_jacobian = jacobian_toolbox[:3, :] # reduce jacobian
determinant = np.linalg.det(velocity_jacobian) # calculate det(J)
print(f"Determinant of the velocity Jacobian (3x3): {determinant}")

if np.isclose(determinant, 0, atol=0.001):
    print("singularity\n")
else:
    print("not singularity\n")

#==============================================================================================================#
#===========================================<ตรวจคำตอบข้อ 3>====================================================#
#code here

print("------Prove computeEffortHW3 function-------\n")

q = [0,math.pi/2,0]
w = [5,10,0,0,1,0] # [Fx, Fy, Fz, Nx, Ny, Nz] (reference from end-effector)

# Calculate torques using computeEffortHW3 function
torque = HW3.computeEffortHW3(q, w)
print("Torque From computeEffortHW3 function:")
print(torque) # แรงปฏิกิริยาที่ข้อต่อออก เพื่อต้านแรงภายนอกให้เกิดสมดุล
print()

# Compute Jacobian reference form end-effector
j = robot.jacobe(q) 

# Calculate torques using Robotics Toolbox
tau_rtb = robot.pay(w, J=j, frame=1) # torque ภายนอกที่มากระทำกับแต่ละข้อต่อ
print("Torque From Robotics Toolboxt: ")
print(-tau_rtb) # เติมลบ เพื่อให้ได้แรงปฏิกิริยาที่ข้อต่อต้านแรงภายนอกให้เกิดสมดุล

#==============================================================================================================#