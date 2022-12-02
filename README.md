# 편미분방정식의 수치해석
### Index
1. lecture 01 : Computer Architecture
2. lecture 02 : Interpolation
3. lecture 03 : Differentiation and Integration
4. lecture 04 : Methods of least squares and Non-linear equations
5. lecture 05 : Linear systems and Conjugate Gradient method(addition)
6. lecture 06 : Ordinary differential equation with initial value problem
7. lecture 07 : Boundary Value Problem with linear shooting and nonlinear shooting
8. lecture 08 : Partial Differential Equation
9. lecture 09 : Finite Difference Method
10. lecture 10 : Finite Element Method 1
11. lecture 11 : Lattice Boltzmann Method
12. lecture 12 : Finite Element Method 2 

### Homework
1. HW01
- part a : Take 4 (x,y) points (y = sin(πx), 0<x<2, choose x values arbitrarily) and find the polynomial that interpolates these points. Add one more point (obeying the above condition) and find the polynomial that interpolates your 5 points.(Choose Handwriting or Word-processing - First part of your report)
- part b : Make your code (any algorithm is okay) to compute integration of y = sin(πx) from 0 to 2 and integration of your interpolation polynomials. Compare your results while increasing the node number for the integration of y = sin(πx). (Second part of your report)

2. HW02
- part a : Consider a PDE u,tt = u,xx - (1+x2)u (0≤x≤1) with u(0,x) = 0, u(t,x) = u(t+2π,x), u(t,0) = sin(t), ∂xu(t,1) = 0. 
First, apply separation of variables. Next, describe how to solve the space-part equation, numerically.

- part b : Make a FDM code for this PDE. u,t + m·x·u,x = 0 (m = 1 or -1) 
    Initial condition: u(0,x) = exp[-(x-x0)2]
    Domain: -L≤x≤L (You can choose L, but L >> |x0|)
    Boundary conditions: ∂xu(0,-L) = ∂xu(0,L) = 0
Draw or plot the state after some time. Discuss dependence on m and x0.


### Reference
- Lecture Notes on 편미분방정식의 수치해석 in Seoul National University
- An introduction to numerical method, Wen Shen