#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 17:34:05 2022

@author: luyin
"""
import taichi as ti

@ti.func
def compute_stress(F:ti.template(), F_plastic:ti.template(), mu, lam):
    Q, R = QR3(F)
    Q_T = Q.transpose()
    R_T = R.transpose()
    Q_T_inv = Q_T.inverse()
    R_T_inv = R_T.inverse()
    r11 = R[0,0]
    r12 = R[0,1]
    r13 = R[0,2]
    r22 = R[1,1] 
    r23 = R[1,2]
    r33 = R[2,2]
    
    # fixed corotated potential
    F_2d = ti.Matrix([[r11, r12], [0.0, r22]])
    R_2d, S_2d = ti.polar_decompose(F_2d) # R rotation, S strech
    J_2d = F_2d.determinant()
    #F_2d_T = F_2d.transpose()
    #F_2d_inv_T = F_2d_T.inverse()
    #P1 = 2 * mu * (F_2d - R_2d) + lam * (J_2d - 1) * J_2d * F_2d_inv_T
    d_F_2d_d_F = ti.Matrix([[r22, 0.0], [0.0, r11]])
    P1 = 2 * mu * (F_2d - R_2d) + lam * (J_2d - 1) * d_F_2d_d_F
    
    # shearing of normal to surface
    gamma = 0
    P2 = [gamma * r13, gamma * r23]
    
    # compression
    k = 1e4
    P3 = 0.0

    if r33 <= 1 :
        P3 = - k * (1 - r33) ** 2
    
    dphi_dR = ti.Matrix([
        [P1[0,0], P1[0,1], P2[0]],
        [P1[1,0], P1[1,1], P2[1]],
        [0.0, 0.0, P3]])
    
    #print('dphi_dR: ', dphi_dR)
    # print('F: ', F)
    # print('Q: ', Q)
    # print('R:', R)
    upper = dphi_dR @ R_T
    
    QT_P_RT = ti.Matrix([
        [upper[0,0], upper[0,1], upper[0,2]],
        [upper[0,1], upper[1,1], upper[1,2]],
        [upper[0,2], upper[1,2], upper[2,2]]
        ])
    
    P = (Q_T_inv @ (QT_P_RT @ R_T_inv))
    

    

    
    if r33 > 1:
        r13 = R[0,2] = 0.0
        r23 = R[1,2] = 0.0
        r33 = R[2,2] = 1.0
    
    if r33 < 1:
        f_n = k / (r11 * r22) * (r33 - 1) ** 2
        f_f = gamma / (r11 * r12) * (r13 ** 2 + r23 ** 2) ** 0.5
        cf = 0.2
        if f_f > cf * f_n:
            r13 = R[0,2] = cf * f_n / f_f * r13
            r23 = R[1,2] = cf * f_n / f_f * r23
            
    
    F_elastic = Q @ R
    F_plastic = F_elastic.inverse() @ F
        
    return P, F_elastic, F_plastic
    #return P, F, F_plastic


    

@ti.func
 #3x3 mat, Gram–Schmidt Orthogonalization
def QR3(Mat:ti.template()) :
    a1 = ti.Vector([Mat[0,0],Mat[1,0],Mat[2,0]])
    a2 = ti.Vector([Mat[0,1],Mat[1,1],Mat[2,1]])
    a3 = ti.Vector([Mat[0,2],Mat[1,2],Mat[2,2]])
    u1 = a1
    e1 = u1/u1.norm(1e-6)
    u2 = a2 - projection(u1,a2)
    e2 = u2/u2.norm(1e-6)
    u3 = a3 - projection(u1,a3)-projection(u2,a3)
    e3 = u3/u3.norm(1e-6)
    r11, r12, r13, r22, r23, r33 = e1.transpose()@a1, e1.transpose()@a2, e1.transpose()@a3, e2.transpose()@a2, e2.transpose()@a3, e3.transpose()@a3
    R = ti.Matrix([[r11[0],r12[0],r13[0]],[0.0,r22[0],r23[0]],[0.0,0.0,r33[0]]])
    Q = ti.Matrix.cols([e1,e2,e3])
    return Q,R

@ti.func
def projection(u,a):
    proj = ti.Vector([0.0,0.0,0.0])
    upper = u.transpose()@a
    lower = u.transpose()@u
    compute = upper[0]/lower[0]*u
    proj[0],proj[1],proj[2] = compute[0],compute[1],compute[2]
    return proj