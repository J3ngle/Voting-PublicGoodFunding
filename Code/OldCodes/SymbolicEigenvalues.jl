
using Symbolics, LinearAlgebra, BenchmarkTools
# Define symbolic variables
@variables c g v v_c v_g v_z z
# Population values
c = 1
g = 0
z = 0
v = 1

# Fitness functions
f_c = 4 * (v - 0.5)^2
f_g = 1 - 4 * (v - 0.5)^2
f_z_one = v^2
f_z_two = (1 - v)^2

# Changes in behavior
v_c = (v^2) / (2*v^2 - 2*v + 1)
v_g = (1 - v)^2 / (2*v^2 - 2*v + 1)
v_z = 1

# Jacobian Elements
a = g*f_c + 8*c*g*(v - 0.5)*v_c - (g*f_g - 8*c*g*(v - 0.5)*v_c) + z*f_c + 8*c*z*(v - 0.5)*v_c - z*f_z_one + c*z*(2*z)*v_c + (1 - 2*c - g - z)*f_c + 8*(v - 0.5)*v_c*(1 - c - g - z)*c + 2*(1 - v)*v_c*c*(1 - c - g - z) - (1 - 2*c)*f_z_two
b = c*f_c + c*g*8*(v - 0.5)*v_g - c*f_g + c*g*(-8)*(v - 0.5)*v_g + c*z*8*(v - 0.5)*v_g - c*z*2*v*v_g - c*f_c + c*(1 - c - g - z)*8*(v - 0.5)*v_g - (-c*f_z_two - 2*(1 - v)*v_g*c*(1 - c - g - z))
d = c*g*8*(v - 0.5)*v_z + c*z*8*(v - 0.5)*v_z + c*f_c - (c*f_z_one + c*z*(-2)*(1 - v)*v_z) + c*(1 - c - g - z)*8*(v - 0.5)*v_z - c*f_c - (c*(1 - c - g - z)*(-2)*(1 - v)*v_z - c*f_z_two)
e = g*(f_g - f_c) + g*c*(-8*(v - 0.5)*v_c - 8*(v - 0.5)*v_c) + g*z*(-8*(v - 0.5)*v_c - 2*v*v_c) - g*(f_c - f_z_two) + g*(1 - c - g - z)*(8*(v - 0.5)*v_c + 2*(1 - v)*v_c)
f = c*(f_g - f_c) + g*c*(-8*(v - 0.5)*v_g - 8*(v - 0.5)*v_g) + z*(f_g - f_z_one) + g*z*(-8*(v - 0.5)*v_g - 2*v*v_g) + (1 - c - 2*g - z)*(f_c - f_z_two) + (8*(v - 0.5)*v_g + 2*(1 - v)*v_g)
h = g*c*(-8*(v - 0.5)*v_z - 8*(v - 0.5)*v_z) + g*(f_g - f_z_one) + g*z*(-8*(v - 0.5)*v_z - 2*v*v_z) + g*(1 - c - g - z)*(8*(v - 0.5)*v_z + 2*(1 - v)*v_z)
i = z*f_z_one + z*c*2*v*v_c - (z*f_c + z*c*8*(v - 0.5)*v_c) + z*g*2*v*v_c + 8*(v - 0.5)*v_c
j = z*c*(2*v*v_g - 8*(v - 0.5)*v_g) + z*f_z_one + z*g*2*v*v_g - (z*f_g - 8*z*g*(v - 0.5)*v_g)
k = c*f_z_one + z*c*2*v*v_z - (c*f_c + 8*(v - 0.5)*v_z) + g*f_z_one + z*g*2*v*v_z*(g*f_g - 8*z*g*(v - 0.5)*v_z)

# Jacobian matrix
J = [a b d; e f h; i j k]

# Compute eigenvalues
@btime eigenvalues = eigenvals(J)
