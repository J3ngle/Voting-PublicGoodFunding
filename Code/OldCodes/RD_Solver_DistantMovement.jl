using DifferentialEquations, Plots, LinearAlgebra, Roots, Statistics, Sundials, ColorSchemes
@time begin
# Parameters for computations
D_i= 0
M_i = 1e-3
D_c = D_i#1e-0 #1e-3 #Diffusion Coefficient for Consensus makers
D_g = D_i#1e-10 #1e-3 #3 #Diffusion Coefficient for Gridlockers
D_z = D_i#1e-0 #1e-3 #Diffusion Coefficient for Zealots
D_z2 = D_i#1e-10 #1e-3 #3 #Diffusion Coefficient for Zealots Party 2
m_c =  M_i#1e-10 # #Migration rate for Consensus makers
m_g =  M_i #1e-10 # #Migration rate for Gridlockers
m_z =  M_i #1e-10 #1e-0 # #Migration rate for Zealots Party 1
m_z2 =  M_i#1e-10 # #Migration rate for Zealots Party 2
λ=  1 #Economic preference 
s= 0 #Spillovers
L = 10 #Length of domain    
Nx, Ny = 10, 10 #Number of discretization points in either direction
dx = L / (Nx - 1) #Chop up x equally
dy = L / (Ny - 1) #Chop up y equally
x = range(0, L, length=Nx) # X size
y = range(0, L, length=Ny) # y size 
tfinal=25.0 #Final time
X, Y = [xi for xi in x, yi in y], [yi for xi in x, yi in y]

#Initial distribution/ conditions
N=Nx
c₀ = rand(N,N) #gaussian(X, Y, 0.5, 0.5, var) #initial distribution for Consensus makers
#c₀ = clamp.(c₀, 0, .6) #Control the bounds of initial conditions
g₀ = rand(N,N) #gaussian(X, Y, 0.5, 0.4, var) #initial distribution for Gridlockers
#g₀ = clamp.(g₀, 0, 0.15) #Control the bounds of initial conditions, we can change these around to see what will happen
z1₀ = rand(N,N) #gaussian(X, Y, 0.5, 0.5, var) #initial distribution for Zealots Party 1
#z1₀ = clamp.(z1₀, 0, 0.15) #Control the bounds of initial conditions, we can change these around to see what will happen
z2₀ = rand(N,N) #gaussian(X, Y, 0.5, 0.5, var) #initial distribution for Zealots Party 2
#z2₀ = clamp.(z2₀, 0, 0.1) #Control the bounds of initial conditions, we can change these around to see what will happen
τ= c₀ .+ g₀ .+ z1₀ .+ z2₀
c₀=c₀ ./ τ #Normalize initial conditions
g₀=g₀ ./ τ
z1₀=z1₀ ./ τ
z2₀=z2₀ ./ τ
v_0 = c₀ .+ 0.5*g₀ .+ z1₀ #Initial vote on the domain
v_c₀= v_0.^2 ./(2 .* v_0.^2 .- 2 .* v_0 .+ 1) #Initial vote for Consensus makers
v_g₀= (1 .- v_0).^2 ./ (2 .*v_0.^2 .- 2 .* v_0 .+ 1) #Initial vote for Gridlockers

# Pack and unpack functions for the state vector
# This is used to convert the 2D arrays into a 1D vector for the ODE solver
# and to convert it back to the 2D arrays after the ODE solver has computed the solution
function pack(c, g, z, z2, v_c, v_g)
    return vcat(vec(c), vec(g), vec(z), vec(z2), vec(v_c), vec(v_g))
end

function unpack(u)
    N = Nx * Ny
    c   = reshape(u[1:N], Nx, Ny)
    g   = reshape(u[N+1:2N], Nx, Ny)
    z   = reshape(u[2N+1:3N], Nx, Ny)
    z2  = reshape(u[3N+1:4N], Nx, Ny)
    v_c = reshape(u[4N+1:5N], Nx, Ny)
    v_g = reshape(u[5N+1:6N], Nx, Ny)
    return c, g, z, z2, v_c, v_g
end

#Construct laplacian in 2D
function laplacian(U)
    L = similar(U)
    # Wrap around for periodic boundary conditions
    for i in 1:Nx, j in 1:Ny
        ip1 = mod(i, Nx) + 1  
        im1 = mod(i - 2, Nx) + 1
        jp1 = mod(j, Ny) + 1
        jm1 = mod(j - 2, Ny) + 1
        L[i, j] = (U[ip1, j] + U[im1, j] + U[i, jp1] + U[i, jm1] - 4U[i, j]) / dx^2
    end
    return L
end
#Gradient function
function Gradient(u, dx, dy)
    Nx, Ny = size(u)
    T = eltype(u)
    grad_x = zeros(T, Nx, Ny)
    grad_y = zeros(T, Nx, Ny)
    #Boundary conditions are handled by the mod functions
    for i in 1:Nx, j in 1:Ny
        ip1 = mod(i, Nx) + 1
        im1 = mod(i-2, Nx) + 1
        jp1 = mod(j, Ny) + 1
        jm1 = mod(j-2, Ny) + 1
        #Central difference for interior points
        grad_x[i,j] = (u[ip1,j] - u[im1,j])/(2dx)
        grad_y[i,j] = (u[i,jp1] - u[i,jm1])/(2dy)
    end

    return grad_x, grad_y
end

# Compute the sum of votes in the N, S, E, W directions for a focal node (i, j)
# v: 2D array of votes, i: row index, j: column index
# Returns the sum of the four neighbors (with periodic boundary conditions)
function positive_spillover(v, i, j)
    Nx, Ny = size(v)
    ip1 = mod(i, Nx) + 1  #Spillover from the South neighbor
    im1 = mod(i - 2, Nx) + 1  #Spillover from the North neighbor
    jp1 = mod(j, Ny) + 1  #Spillover from the East neighbor
    jm1 = mod(j - 2, Ny) + 1  #Spillover from the West neighbor
    return v[im1, j] + v[ip1, j] + v[i, jm1] + v[i, jp1] #Sum of spillovers from the four neighbors
end

# Fitness functions
Fitness_c(v) =  (1 .+ cos.(2 .* pi .* v)) ./ 2 #Strategy fitness for Consensus makers
Fitness_g(v) = (1 .- cos.(2 .* pi .* v)) ./ 2 #Strategy fitness for Gridlockers
Fitness_z1(v) = (1 .- cos.(pi .* v)) ./2 #Strategy fitness for Zealots party 1
Fitness_z2(v) = (1 .+ cos.(pi .* v)) ./2 #Strategy fitness for Zealots party 2

#Economic Utility functions 
Utility_c(v, positive_spillover) = λ .* ((1-s)*v+(s/4)*(positive_spillover))  .+ (1-λ).*Fitness_c(v)
Utility_g(v, positive_spillover) = λ .* ((1-s)*v+(s/4)*(positive_spillover))  .+ (1-λ).*Fitness_g(v)
Utility_z1(v, positive_spillover) = λ .* ((1-s)*v+(s/4)*(positive_spillover)) .+ (1-λ).*Fitness_z1(v)
Utility_z2(v, positive_spillover) = λ .* ((1-s)*v+(s/4)*(positive_spillover))  .+ (1-λ).*Fitness_z2(v)

# Initial condition
u0 = pack(c₀, g₀, z1₀, z2₀, v_c₀, v_g₀)
du0 = zeros(size(u0))
tspan = (0.0, tfinal)

function rd_system!(du, u, p, t)
    # unpack state
    c, g, z, z2, v_c, v_g = unpack(u)

    # compute v and fitnesses/utilities
    v = c .* v_c .+ g .* v_g .+ z
    F_c  = Fitness_c(v)
    F_g  = Fitness_g(v)
    F_z  = Fitness_z1(v)
    F_z2 = Fitness_z2(v)

    # Compute spillover_value for each node and pass to Utility functions
    spillover_v = [positive_spillover(v, i, j) for i in 1:Nx, j in 1:Ny]
    u_c   = [Utility_c(v[i, j], spillover_v[i, j]) for i in 1:Nx, j in 1:Ny]
    u_g   = [Utility_g(v[i, j], spillover_v[i, j]) for i in 1:Nx, j in 1:Ny]
    u_z1  = [Utility_z1(v[i, j], spillover_v[i, j]) for i in 1:Nx, j in 1:Ny]
    u_z2  = [Utility_z2(v[i, j], spillover_v[i, j]) for i in 1:Nx, j in 1:Ny]

    # PDE, Laplacian + replicator. Distant movement added at the end
    du_c  = D_c .* laplacian(c) .+ (c .* g .* (F_c .- F_g)  .+ c .* z .* (F_c .- F_z)  .+ c .* z2 .* (F_c .- F_z2))
    du_g  = D_g .* laplacian(g) .+ (g .* c .* (F_g .- F_c)  .+ g .* z .* (F_g .- F_z)  .+ g .* z2 .* (F_g .- F_z2))
    du_z  = D_z .* laplacian(z) .+ (z .* c .* (F_z .- F_c)  .+ z .* g .* (F_z .- F_g))
    du_z2 = D_z2 .* laplacian(z2) .+ (z2 .* c .* (F_z2 .- F_c) .+ z2 .* g .* (F_z2 .- F_g))

    # Movement for c: each cell moves to the cell j
    T = eltype(c)
    move_gain = zeros(T, Nx, Ny)
    move_loss = zeros(T, Nx, Ny)
    for i1 in 1:Nx, j1 in 1:Ny
        Fi = F_c[i1, j1]
        maxval = zero(T)
        maxi, maxj = i1, j1
        for i2 in 1:Nx, j2 in 1:Ny
            if i1 == i2 && j1 == j2
                continue
            end
            dx2 = 1#(X[i1,j1] - X[i2,j2])^2 + (Y[i1,j1] - Y[i2,j2])^2
            # avoid division by zero
            if dx2 == 0
                continue
            end
            val = (F_c[i2, j2] - Fi) / dx2
            if val > maxval
                maxval = val
                maxi, maxj = i2, j2
            end
        end
        if maxval > 0
            flux = m_c * maxval * c[i1, j1]
            move_loss[i1, j1] += flux
            move_gain[maxi, maxj] += flux
        end
    end

    # Apply movement to du_c (gain minus loss)
    du_c .= du_c .+ (move_gain .- move_loss)

    # Movement for g 
    move_gain_g = zeros(T, Nx, Ny)
    move_loss_g = zeros(T, Nx, Ny)
    for i1 in 1:Nx, j1 in 1:Ny
        Fi = F_g[i1, j1]
        maxval = zero(T)
        maxi, maxj = i1, j1
        for i2 in 1:Nx, j2 in 1:Ny
            if i1 == i2 && j1 == j2
                continue
            end
            dx2 = 1#(X[i1, j1] - X[i2, j2])^2 + (Y[i1, j1] - Y[i2, j2])^2
            if dx2 == 0
                continue
            end
            val = (F_g[i2, j2] - Fi) / dx2
            if val > maxval
                maxval = val
                maxi, maxj = i2, j2
            end
        end
        if maxval > 0
            flux = m_g * maxval * g[i1, j1]
            move_loss_g[i1, j1] += flux
            move_gain_g[maxi, maxj] += flux
        end
    end
    du_g .= du_g .+ (move_gain_g .- move_loss_g)

    # Movement for z (zealots 1)
    move_gain_z = zeros(T, Nx, Ny)
    move_loss_z = zeros(T, Nx, Ny)
    for i1 in 1:Nx, j1 in 1:Ny
        Fi = F_z[i1, j1]
        maxval = zero(T)
        maxi, maxj = i1, j1
        for i2 in 1:Nx, j2 in 1:Ny
            if i1 == i2 && j1 == j2
                continue
            end
            dx2 = 1#(X[i1, j1] - X[i2, j2])^2 + (Y[i1, j1] - Y[i2, j2])^2
            if dx2 == 0
                continue
            end
            val = (F_z[i2, j2] - Fi) / dx2
            if val > maxval
                maxval = val
                maxi, maxj = i2, j2
            end
        end
        if maxval > 0
            flux = m_z * maxval * z[i1, j1]
            move_loss_z[i1, j1] += flux
            move_gain_z[maxi, maxj] += flux
        end
    end
    du_z .= du_z .+ m_z .* (move_gain_z .- move_loss_z)
    # Movement for z2 (zealots 2)
    move_gain_z2 = zeros(T, Nx, Ny)
    move_loss_z2 = zeros(T, Nx, Ny)
    for i1 in 1:Nx, j1 in 1:Ny
        Fi = F_z2[i1, j1]
        maxval = zero(T)
        maxi, maxj = i1, j1
        for i2 in 1:Nx, j2 in 1:Ny
            if i1 == i2 && j1 == j2
                continue
            end
            dx2 = 1#(X[i1, j1] - X[i2, j2])^2 + (Y[i1, j1] - Y[i2, j2])^2
            if dx2 == 0
                continue
            end
            val = (F_z2[i2, j2] - Fi) / dx2
            if val > maxval
                maxval = val
                maxi, maxj = i2, j2
            end
        end
        if maxval > 0
            flux = m_z2 * maxval * z2[i1, j1]
            move_loss_z2[i1, j1] += flux
            move_gain_z2[maxi, maxj] += flux
        end
    end
    du_z2 .= du_z2 .+ (move_gain_z2 .- move_loss_z2)
    # algebraic dynamics for v_c, v_g
    du_v_c = (1 .- v_c) .* v.^2 .- v_c .* (1 .- v).^2
    du_v_g = (1 .- v_g) .* (1 .- v).^2 .- v_g .* v.^2
    # pack into du (in-place)
    du .= pack(du_c, du_g, du_z, du_z2, du_v_c, du_v_g)
    return nothing
end

problem = ODEProblem(rd_system!, u0, tspan)
sol = solve(problem, RadauIIA5(), saveat=0.01, reltol=1e-12, abstol=1e-12)

## HEATMAPS
fontsize=14
c, g, z, z2, v_c, v_g = unpack(sol[end]) #computations from the end of the simulation, we could pull these at any other times
population = c .+ g .+ z .+ z2 #Compute population, this is a matrix
total_population = sum(c) .+ sum(g) .+ sum(z) .+ sum(z2) #Compute population at the end, this is a scalar
c=c ./ population #Normalize c
g=g ./ population #Normalize g
z=z ./ population #Normalize z
z2=z2 ./ population #Normalize z2
heatmap_population = population  ./ maximum(population) #Normalize population
v = (c .* v_c .+ g .* v_g .+ z) #Compute v at the end
clims = (0, 1) #Color limits for heatmaps
p1 = heatmap(x, y, c',  aspect_ratio=1,colorbar=false, clims=clims)# clims=clims
p2 = heatmap(x, y, g',  aspect_ratio=1,colorbar=false, clims=clims)# clims=clims
p3 = heatmap(x, y, z',  aspect_ratio=1,colorbar=false, clims=clims)# clims=clims
p4 = heatmap(x, y, z2',  aspect_ratio=1,colorbar=false, clims=clims)# clims=clims
p5 = heatmap(x, y, heatmap_population',  aspect_ratio=1,colorbar=false, clims=clims) # clims=clims
p6 = heatmap(x, y, v',  aspect_ratio=1,color=:viridis, colorbar=false, clims=clims) # clims=climscolor=:balance,
heatmap_figure = plot(p1, p2, p3, p4, p5, p6, layout=(3,3), size=(1400, 1500),colorbar=true, titlefontsize=fontsize, guidefontsize=fontsize, tickfontsize=fontsize, plot_title="Solutions at final time $tfinal")
display(plot(p1, axis=false, framestyle=:none,ticks=false, size=(625, 625))) #Consensus makers
savefig("7DistantMovement_ConsensusMakers,Nx=$Nx,Dc=$D_c,M_c=$m_c,lambda=$λ,s=$s,T=$tfinal.pdf")
display(plot(p2, axis=false, framestyle=:none, ticks=false,size=(625, 625))) #Gridlockers
savefig("7DistantMovement_Gridlockers,Nx=$Nx,Dg=$D_g,M_g=$m_g,lambda=$λ,s=$s,T=$tfinal.pdf")
display(plot(p3, axis=false, framestyle=:none, ticks=false,size=(625, 625))) #Zealots of party 1
savefig("7DistantMovement_Zealots1,Nx=$Nx,Dz1=$D_z,M_z1=$m_z,lambda=$λ,s=$s,T=$tfinal.pdf")
display(plot(p4, axis=false, framestyle=:none, ticks=false,size=(625, 625))) #Zealots of party 2
savefig("7DistantMovement_Zealots2,Nx=$Nx,Dz2=$D_z2,M_z2=$m_z2,lambda=$λ,s=$s,T=$tfinal.pdf")
display(plot(p5, axis=false, framestyle=:none, ticks=false,size=(625, 625))) #Population
savefig("7DistantMovement_Population,Nx=$Nx,Dc=$D_c,M_c=$m_c,lambda=$λ,s=$s,T=$tfinal.pdf")
display(plot(p6, axis=false, framestyle=:none, ticks=false, size=(625,625))) #Vote
savefig("7DistantMovement_Vote,Nx=$Nx,Dc=$D_c,M_c=$m_c,lambda=$λ,s=$s,T=$tfinal.pdf")
display(heatmap_figure)#savefig("Heatmap_Clean_DifferentD_EvenIC_Finaltime=$tfinal.pdf")

# TIME SERIES: Compute averages over the domain at each time step
time_steps = sol.t
average_c = [mean(unpack(sol[i])[1]) for i in 1:length(time_steps)] #Average Consensus-makers
average_g = [mean(unpack(sol[i])[2]) for i in 1:length(time_steps)] #Average Gridlockers
average_z = [mean(unpack(sol[i])[3]) for i in 1:length(time_steps)] #Average Zealots of party 1
average_z2 = [mean(unpack(sol[i])[4]) for i in 1:length(time_steps)] #Average Zealots of party 2
average_Fitness_z1 = [mean(Fitness_z1(unpack(sol[i])[5])) for i in 1:length(time_steps)]
average_Fitness_z2 = [mean(Fitness_z2(unpack(sol[i])[5])) for i in 1:length(time_steps)]
average_Fitness_c = [mean(Fitness_c(unpack(sol[i])[5])) for i in 1:length(time_steps)]
average_Fitness_g = [mean(Fitness_g(unpack(sol[i])[5])) for i in 1:length(time_steps)]
#ts_max_pop = [maximum(unpack(sol[i])[1]) + maximum(unpack(sol[i])[2]) + maximum(unpack(sol[i])[3]) + maximum(unpack(sol[i])[4]) for i in 1:length(time_steps)]
average_v = [mean(unpack(sol[i])[5]) .* mean(unpack(sol[i])[1])  .+ mean(unpack(sol[i])[2]) .* mean(unpack(sol[i])[6]) .+ mean(unpack(sol[i])[3]) .+ mean(unpack(sol[i])[4]) for i in 1:length(time_steps)]
# Above computes c*v_c + g*v_g + z at each time step
# Plot averages
time_series = plot(time_steps, average_c, xlabel="Time", ylabel="Mean",lw=8, xlabelfontsize=20, ylabelfontsize=20,
     titlefontsize=12, legendfontsize=12, tickfontsize=16, yticks=0:0.25:1.1, xticks=0:5:tfinal, ylim=(0,1), xlim=(0,tfinal), legend=false) #, label="Mean Consensus Makers"
plot!(time_steps, average_g,lw=8)
plot!(time_steps, average_z,lw=8)
plot!(time_steps, average_z2,lw=8)
plot!(time_steps, average_v,lw=8)
# plot!(time_steps, average_Fitness_z1,lw=8)
# plot!(time_steps, average_Fitness_z2,lw=8)
#plot!(time_steps, ts_max_pop, label="Max Population",lw=3)
display(time_series)
savefig("DistantMovement_TimeSeries_,Dc=$D_c,M_c=$m_c,lambda=$λ,s=$s,T=$tfinal.pdf")
end