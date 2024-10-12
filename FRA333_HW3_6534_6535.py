# file สำหรับเขียนคำตอบ
# ในกรณีที่มีการสร้าง function อื่น ๆ ให้ระบุว่า input-output คืออะไรด้วย
'''
ชื่อ_รหัส
1. นันท์นภัส_6534
2. บัซลาอ์_6535
'''

import HW3_utils
import math
import numpy as np

#=============================================<คำตอบข้อ 1>======================================================#
#code here

# Define cross vertor function 
def cos_vector(a, b):
    ans = [0,0,0]
    ans[0] = a[1]*b[2] - b[1]*a[2]
    ans[1] = b[0]*a[2] - a[0]*b[2]
    ans[2] = a[0]*b[1] - b[0]*a[1]
    return ans

def endEffectorJacobianHW3(q:list[float])->list[float]:

    R,P,R_e,p_e = HW3_utils.FKHW3(q)

    [q1,q2,q3] = q

    # Calculate Jacobian relative to the base frame (0)
    Pe_P1 = [p_e[0]-P[0][0], p_e[1]-P[1][0], p_e[2]-P[2][0]]
    Pe_P2 = [p_e[0]-P[0][1], p_e[1]-P[1][1], p_e[2]-P[2][1]]
    Pe_P3 = [p_e[0]-P[0][2], p_e[1]-P[1][2], p_e[2]-P[2][2]]

    # Define z1 z2 z3 reference from frame 0
    z1 = [0, 0, 1]
    z2 = [-math.sin(q1), math.cos(q1), 0]
    z3 = [-math.sin(q1), math.cos(q1), 0]

    z1_cross_PeP1 = cos_vector(z1, Pe_P1)
    z2_cross_PeP2 = cos_vector(z2, Pe_P2)
    z3_cross_PeP3 = cos_vector(z3, Pe_P3)

    # Fist column of Jacobian
    v_xe_q1 = z1_cross_PeP1[0]
    v_ye_q1 = z1_cross_PeP1[1]
    v_ze_q1 = z1_cross_PeP1[2]
    w_xe_q1 = z1[0]
    w_ye_q1 = z1[1]
    w_ze_q1 = z1[2]

    # Second column of Jacobian
    v_xe_q2 = z2_cross_PeP2[0]
    v_ye_q2 = z2_cross_PeP2[1]
    v_ze_q2 = z2_cross_PeP2[2]
    w_xe_q2 = z2[0]
    w_ye_q2 = z2[1]
    w_ze_q2 = z2[2]

    # Third column of Jacobian
    v_xe_q3 = z3_cross_PeP3[0]
    v_ye_q3 = z3_cross_PeP3[1]
    v_ze_q3 = z3_cross_PeP3[2]
    w_xe_q3 = z3[0]
    w_ye_q3 = z3[1]
    w_ze_q3 = z3[2]

    J = np.array([v_xe_q1, v_xe_q2, v_xe_q3,  
                  v_ye_q1, v_ye_q2, v_ye_q3,
                  v_ze_q1, v_ze_q2, v_ze_q3,
                  w_xe_q1, w_xe_q2, w_xe_q3,
                  w_ye_q1, w_ye_q2, w_ye_q3,
                  w_ze_q1, w_ze_q2, w_ze_q3]).reshape(6,3)
    
    return J

#==============================================================================================================#
#=============================================<คำตอบข้อ 2>======================================================#
#code here
def checkSingularityHW3(q:list[float])->bool:

    J = endEffectorJacobianHW3(q)

    # reduce jacobian
    J_3x3 = J[:3, :]
    
    # Calculate the determinant of J
    det_J = np.linalg.det(J_3x3)
    
    # Set a threshold for singularity
    e = 0.001
    
    # Check if the norm of the determinant is below the threshold
    if abs(det_J) < e:
        return True
    else:
        return False

#==============================================================================================================#
#=============================================<คำตอบข้อ 3>======================================================#
#code here
def computeEffortHW3(q:list[float], w:list[float])->list[float]:
    J = endEffectorJacobianHW3(q)
    J_transposed = np.transpose(J)

    R,P,R_e,p_e = HW3_utils.FKHW3(q)

    # Transform w from the base frame (frame 0) to the end-effector frame.
    f_e = w[:3]
    n_e = w[3:]

    f_0 = R_e @ f_e 
    n_0 = R_e @ n_e

    w_0 = np.hstack((f_0, n_0)).reshape(6, 1)

    # calculate torque
    tau = J_transposed @ w_0

    return tau

#==============================================================================================================#

# #=================<ทดสอบฟังก์ชันข้อ 1>=========================
# J_sol = endEffectorJacobianHW3([0.0,-math.pi/2,-0.2])
# print("---------------J--------------")
# print(J_sol)

# #=================<ทดสอบฟังก์ชันข้อ 2>=========================
# checkSingularity = checkSingularityHW3([0,0,0])
# print("--------checkSingularity--------")
# print(checkSingularity)

# #=================<ทดสอบฟังก์ชันข้อ 3>=========================
# torque = computeEffortHW3([0,math.pi/2,0], [5,10,0,0,1,0])
# print("--------------Torque-----------")
# print(torque)