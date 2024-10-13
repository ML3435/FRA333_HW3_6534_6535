จากโจทย์ในไฟล์ Instruction Kinematics HW 3.html
# แนวคิดในการแก้โจทย์

<img width="806" alt="Screenshot 2567-10-12 at 23 23 14" src="https://github.com/user-attachments/assets/ab610409-e9f6-4ba5-a3c2-584c56234b21">

## คำถามข้อที่ 1: จงเขียนฟังก์ชันในการ Jacobian ของหุ่นยนต์ตัวนี้ให้อยู่ในฟังก์ชั่นต่อไปนี้

J_e = endEffectorJacobianHW3(q) //การหา Jacobian ของ end-effector ที่ reference frame 0

<img width="891" alt="Screenshot 2567-10-13 at 12 30 53" src="https://github.com/user-attachments/assets/e74c3edc-0409-4168-8a55-7e704f0dbfdf">

```python
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
```
## Prove คำตอบของคำถามข้อที่ 1 โดยเปรียบเทียบกับ Jacobian ที่ได้จาก Roboticstoolbox

เริ่มจากการสร้างหุ่นยนต์ด้วย Roboticstoolbox ดังนี้

![image](https://github.com/user-attachments/assets/d859a8f9-7653-43cb-8db8-6b8c6d73a63c)

จากนั้นเขียนโปรแกรมเพื่อเปรียบเทียบค่า Jacobian ที่ได้จากฟังก์ชัน robot.jacob0(q) กับ Jacobian ที่ได้จากฟังก์ชัน endEffectorJacobianHW3(q)

```python
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
```
จะได้
![image](https://github.com/user-attachments/assets/f55ebf5f-dfa1-4c8e-b180-6ba81de25273)

ซึ่งเมื่อนำคำตอบมาเปรียบเทียบกันแล้วพบว่าเหมือนกัน จึงสามารถ Proove ได้ว่า Jacobian ของ end-effector ที่ reference frame 0 ที่ได้จากฟังก์ชัน endEffectorJacobianHW3(q) ถูกต้อง
----------------------------------------------------------------------------------------------------------------------------------------
## คำถามข้อที่ 2: จงเขียนฟังก์ชันในการหาสภาวะ Singularity ด้วยสมการ ∣∣det(J*(q))∣∣<ε โดยที่ค่า ε มีค่า 0.001 และ J*(⋅) คือเมตริกซ์จาโคเบียนที่ถูกลดรูปแล้ว
flag = checkSingularityHW3(q)
โดยที่  flag ∈ 0,1 เป็น scalar ที่มีค่าเท่ากับ 1 ก็ต่อเมื่ออยู่ตำแหน่งใกล้สภาวะ Singularity หรือมีค่าเท่ากับ 0 เมื่อแขนกลอยู่ในสภาวะปกติ

```python
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
```
## Prove คำตอบของคำถามข้อที่ 2 โดยเปรียบเทียบกับการใช้ Jacobian ที่ได้จาก Roboticstoolbox จะได้ผลลัพธ์คือ

![image](https://github.com/user-attachments/assets/fd0946b3-39d7-4962-9458-b0efb3faf589)

จะเห็นได้ว่ามีผลลัพธ์ที่เหมือนกัน จึงสามารถสรุปได้ว่าฟังก์ชัน checkSingularityHW3(q) สามารถหาสภาวะ Singularity ของหุ่นยนต์ได้
----------------------------------------------------------------------------------------------------------------------------------------
## คำถามข้อที่ 3: เขียนฟังก์ชันในการหา effort ของแต่ละข้อต่อเมื่อมี wrench มากระทำกับจุดกึ่งกลางของเฟรมพิกัด Fe

tau = computeEffortHW3(q,w)

จากสมการ

  τ = computeEffort(q,w)
  
โดยที่
  tau เป็นเวกเตอร์หลักของ double ที่มีขนาดเท่ากับ 3 ที่แสดงถึงค่า Effort ของแต่ละข้อต่อ
  
  q เป็นเวกเตอร์หลักของ double ที่มีขนาดเท่ากับ 3 ที่แสดงถึง configuration ของหุ่นยนต์ (Joint Configuration)
  
  w เป็นเวกเตอร์หลักของ double ที่มีขนาดเท่ากับ 6 ที่แสดงโมเมนท์และแรงที่อ้างอิงกับเฟรมพิกัด Fe
  
ซึ่งการจะหา torque ได้นั้น Jacobian และ w ต้องมี reference frame เดียวกัน ซึ่งเราจะยึดตาม frame ของ Jacobian จึงต้องคูณ Rotation matrix ที่ frame e โดยอ้างอิง frame 0 จะได่้

![image](https://github.com/user-attachments/assets/ccd1d687-48e9-47d3-84af-f03bf3fe3e7f)

```python
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
```
## Prove คำตอบของคำถามข้อที่ 3 โดยเปรียบเทียบกับ Jacobian ที่ได้จาก Roboticstoolbox

เริ่มจากการกำหนดค่าให้กับ q และ w

q = [0,math.pi/2,0]

w = [5,10,0,0,1,0]
  
จากนั้นเขียนโปรแกรมเพื่อเปรียบเทียบค่า torque ที่ได้จากฟังก์ชัน jacobe(q) กับ torque ที่ได้จากฟังก์ชัน computeEffortHW3(q, w)

```python
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
```
จะได้ผลลัพธ์คือ

![image](https://github.com/user-attachments/assets/e1504a81-7476-4f55-9c81-03d942bb628f)

จะเห็นได้ว่าเมื่อนำคำตอบมาเปรียบเทียบกันแล้วพบว่าค่าที่ได้เหมือนกัน จึงสามารถ Proove ได้ว่า torque ที่ได้จากฟังก์ชัน computeEffortHW3(q, w) ถูกต้อง
