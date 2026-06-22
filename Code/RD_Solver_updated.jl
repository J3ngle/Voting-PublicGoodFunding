using DifferentialEquations, Plots, LinearAlgebra, Roots, Statistics, Sundials, ColorSchemes
@time begin
    # Parameters
    κ = 0
    λ = 0 # Economic preference
    s = 0 # Spillovers
    tax = 0 # 0.2 Taxes
    L = 120 # Length of domain
    M = 0.05/L^2
    x = range(0, L, length=L) # X size
    y = range(0, L, length=L) # y size 
    tfinal = 10000.0 # Final time
    X, Y = [xi for xi in x, yi in y], [yi for xi in x, yi in y]

    #Initial conditions
    c₀ = rand(L,L) # Initial distribution for Consensus-makers
    g₀ = rand(L,L) # Initial distribution for Gridlockers
    z1₀ = rand(L,L) # Initial distribution for Party 1 Zealots
    z2₀ = rand(L,L) # Initial distribution for Party 2 Zealots
    total = c₀ .+ g₀ .+ z1₀ .+ z2₀ # Normalize
    c₀ = c₀ ./ total
    g₀ = g₀ ./ total
    z1₀ = z1₀ ./ total
    z2₀ = z2₀ ./ total
    v₀ = 0.5 * c₀ .+ 0.5 * g₀ .+ z1₀ #Initial vote
    vc₀ = (v₀ .^ 2) ./ (2 .* v₀ .^ 2 .- 2 .* v₀ .+ 1) # Initial vote for Consensus-makers
    vg₀ = ((1 .- v₀) .^ 2) ./ (2 .* v₀ .^ 2 .- 2 .* v₀ .+ 1) # Initial vote for Gridlockers

    # Pack and unpack functions for the state vector
    # This is used to convert the 2D arrays into a 1D vector for the ODE solver
    # and to convert it back to the 2D arrays after the ODE solver has computed the solution
    function pack(c, g, z1, z2, vc, vg)
        return vcat(vec(c), vec(g), vec(z1), vec(z2), vec(vc), vec(vg))
    end

    function unpack(u)
        N = L * L
        c = reshape(u[1:N], L, L)
        g = reshape(u[N+1:2N], L, L)
        z1 = reshape(u[2N+1:3N], L, L)
        z2 = reshape(u[3N+1:4N], L, L)
        vc = reshape(u[4N+1:5N], L, L)
        vg = reshape(u[5N+1:6N], L, L)
        return c, g, z1, z2, vc, vg
    end

    # function unpack(u)
    #     N = L * L
    #     c = reshape(view(u[1:N]), L, L)
    #     g = reshape(view(u[N+1:2N]), L, L)
    #     z1 = reshape(view(u[2N+1:3N]), L, L)
    #     z2 = reshape(view(u[3N+1:4N]), L, L)
    #     vc = reshape(view(u[4N+1:5N]), L, L)
    #     vg = reshape(view(u[5N+1:6N]), L, L)
    #     return c, g, z1, z2, vc, vg
    # end

    # Compute the sum of votes in the N, S, E, W directions for a focal node (i, j)
    # v: 2D array of votes, i: row index, j: column index
    # Returns the sum of the four neighbors (with periodic boundary conditions)
    function positive_spillover(v, i, j)
        ip1 = mod(i, L) + 1  #Spillover from the South neighbor
        im1 = mod(i - 2, L) + 1  #Spillover from the North neighbor
        jp1 = mod(j, L) + 1  #Spillover from the East neighbor
        jm1 = mod(j - 2, L) + 1  #Spillover from the West neighbor
        return v[im1, j] + v[ip1, j] + v[i, jm1] + v[i, jp1] #Sum of spillovers from the four neighbors
    end

    # Fitness functions
    f_c(v) = (1 .+ cos.(2 .* pi .* v)) ./ 2 #Strategy fitness for Consensus makers
    f_g(v) = (1 .- cos.(2 .* pi .* v)) ./ 2 #Strategy fitness for Gridlockers
    f_z1(v) = (1 .- cos.(pi .* v)) ./ 2 #Strategy fitness for Zealots party 1
    f_z2(v) = (1 .+ cos.(pi .* v)) ./ 2 #Strategy fitness for Zealots party 2

    # Utility functions 
    u_c(v, positive_spillover) = λ .* ((1 - s - tax) * v + (s / 4) * (positive_spillover)) .+ (1 - λ) .* f_c(v)
    u_g(v, positive_spillover) = λ .* ((1 - s - tax) * v + (s / 4) * (positive_spillover)) .+ (1 - λ) .* f_g(v)
    u_z1(v, positive_spillover) = λ .* ((1 - s - tax) * v + (s / 4) * (positive_spillover)) .+ (1 - λ) .* f_z1(v)
    u_z2(v, positive_spillover) = λ .* ((1 - s - tax) * v + (s / 4) * (positive_spillover)) .+ (1 - λ) .* f_z2(v)

    # Initial condition
    u0 = pack(c₀, g₀, z1₀, z2₀, vc₀, vg₀)
    du0 = zeros(size(u0))
    tspan = (0.0, tfinal)

    function rd_system!(du, u, p, t)
        # unpack state
        c, g, z1, z2, vc, vg = unpack(u)

        # compute v and fitnesses/utilities
        v = c .* vc .+ g .* vg .+ z1
        F_c = f_c(v)
        F_g = f_g(v)
        F_z1 = f_z1(v)
        F_z2 = f_z2(v)

        # Compute spillover_value for each node and pass to Utility functions
        spillover_v = [positive_spillover(v, i, j) for i in 1:L, j in 1:L]
        U_c = [u_c(v[i, j], spillover_v[i, j]) for i in 1:L, j in 1:L]
        U_g = [u_g(v[i, j], spillover_v[i, j]) for i in 1:L, j in 1:L]
        U_z1 = [u_z1(v[i, j], spillover_v[i, j]) for i in 1:L, j in 1:L]
        U_z2 = [u_z2(v[i, j], spillover_v[i, j]) for i in 1:L, j in 1:L]

        # Local replicator dynamics. Distant movement added at the end.
        du_c = c .* g .* (F_c .- F_g) .+ c .* z1 .* (F_c .- F_z1) .+ c .* z2 .* (F_c .- F_z2)
        du_g = g .* c .* (F_g .- F_c) .+ g .* z1 .* (F_g .- F_z1) .+ g .* z2 .* (F_g .- F_z2)
        du_z1 = z1 .* c .* (F_z1 .- F_c) .+ z1 .* g .* (F_z1 .- F_g)
        du_z2 = z2 .* c .* (F_z2 .- F_c) .+ z2 .* g .* (F_z2 .- F_g)

        # # Movement gain and loss
        move_gain_c = zeros(L, L)
        move_loss_c = zeros(L, L)
        move_gain_g = zeros(L, L)
        move_loss_g = zeros(L, L)
        move_gain_z1 = zeros(L, L)
        move_loss_z1 = zeros(L, L)
        move_gain_z2 = zeros(L, L)
        move_loss_z2 = zeros(L, L)

        # for i1 in 1:L, j1 in 1:L
        #     Uci = U_c[i1, j1]
        #     Ugi = U_g[i1, j1]
        #     Uz1i = U_z1[i1, j1]
        #     Uz2i = U_z2[i1, j1]

        #     for i2 in 1:L, j2 in 1:L
        #         if i1 == i2 && j1 == j2
        #             continue
        #         end
        #         dx2 = c[i2, j2]+g[i2, j2]+z1[i2, j2]+z2[i2, j2]#(X[i1, j1] - X[i2, j2])^2 + (Y[i1, j1] - Y[i2, j2])^2
        #         flux_c = M * c[i1, j1] * exp(κ*(U_c[i2, j2] - Uci)) * dx2
        #         flux_g = M * g[i1, j1] * exp(κ*(U_g[i2, j2] - Ugi)) * dx2
        #         flux_z1 = M * z1[i1, j1] * exp(κ*(U_z1[i2, j2] - Uz1i)) * dx2
        #         flux_z2 = M * z2[i1, j1] * exp(κ*(U_z2[i2, j2] - Uz2i)) * dx2

        #         move_loss_c[i1, j1] += flux_c
        #         move_gain_c[i2, j2] += flux_c
        #         move_loss_g[i1, j1] += flux_g
        #         move_gain_g[i2, j2] += flux_g
        #         move_loss_z1[i1, j1] += flux_z1
        #         move_gain_z1[i2, j2] += flux_z1
        #         move_loss_z2[i1, j1] += flux_z2
        #         move_gain_z2[i2, j2] += flux_z2
        #     end
        # end

        # # Apply movement to du_c (gain minus loss)
        # du_c .= du_c .+ move_gain_c .- move_loss_c
        # du_g .= du_g .+ move_gain_g .- move_loss_g
        # du_z1 .= du_z1 .+ move_gain_z1 .- move_loss_z1
        # du_z2 .= du_z2 .+ move_gain_z2 .- move_loss_z2

         # O(L²) movement — factorised mean-field sums
# net[i] = M*(exp(κU[i])·Sw − pop[i]·exp(−κU[i])·Se)
# Sw = Σⱼ pop[j]·exp(−κU[j]),  Se = Σⱼ exp(κU[j])
for (pop, U, du_pop) in (
        (c,  U_c,  du_c),
        (g,  U_g,  du_g),
        (z1, U_z1, du_z1),
        (z2, U_z2, du_z2))
    eU  = exp.(κ .* U)
    emU = inv.(eU)                        # exp.(−κ .* U), no second exp pass
    Sw  = dot(vec(pop), vec(emU))         # scalar
    Se  = sum(eU)                         # scalar
    du_pop .+= M .* (eU .* Sw .- pop .* emU .* Se)
end

# Code for gravity model
# P_total = c + g + z1 + z2   # computed once, reused for all four populations
# for (pop, U, du_pop) in (
#         (c,  U_c,  du_c),
#         (g,  U_g,  du_g),
#         (z1, U_z1, du_z1),
#         (z2, U_z2, du_z2))
#     eU = exp.(κ * U)
#     emU = 1.0 ./ eU
#     Sw  = dot(vec(pop),       vec(emU))   # Σ pop·exp(−κU)
#     Swp = dot(vec(P_total), vec(eU))    # Σ P_total·exp(κU)
#     du_pop .+= (M / L^2) .* (eU .* P_total .* Sw .- pop .* emU .* Swp)
# end

        # algebraic dynamics for v_c, v_g
        du_vc = (1 .- vc) .* v .^ 2 .- vc .* (1 .- v) .^ 2
        du_vg = (1 .- vg) .* (1 .- v) .^ 2 .- vg .* v .^ 2
        # pack into du (in-place)
        du .= pack(du_c, du_g, du_z1, du_z2, du_vc, du_vg)
        return nothing
    end

    problem = ODEProblem(rd_system!, u0, tspan)
    sol = solve(problem, Tsit5(), saveat=1)#, reltol=1e-12, abstol=1e-12)

    ## HEATMAPS
    fontsize = 14
    c, g, z1, z2, v_c, v_g = unpack(sol[end]) #computations from the end of the simulation, we could pull these at any other times
    population = c .+ g .+ z1 .+ z2 #Compute population, this is a matrix
    total_population = sum(c) .+ sum(g) .+ sum(z1) .+ sum(z2) #Compute population at the end, this is a scalar
    c = c ./ population #Normalize c
    g = g ./ population #Normalize g
    z1 = z1 ./ population #Normalize z
    z2 = z2 ./ population #Normalize z2
    heatmap_population = population/ maximum(population) #Normalize population
    v = (c .* v_c .+ g .* v_g .+ z1) #Compute v at the end
    clims = (0, 1) #Color limits for heatmaps
    p1 = heatmap(x, y, c', aspect_ratio=1, colorbar=false, clims=clims)# clims=clims
    p2 = heatmap(x, y, g', aspect_ratio=1, colorbar=false, clims=clims)# clims=clims
    p3 = heatmap(x, y, z1', aspect_ratio=1, colorbar=false, clims=clims)# clims=clims
    p4 = heatmap(x, y, z2', aspect_ratio=1, colorbar=false, clims=clims)# clims=clims
    p5 = heatmap(x, y, heatmap_population', aspect_ratio=1, colorbar=false, clims=clims) # clims=clims
    p6 = heatmap(x, y, v', aspect_ratio=1, color=:viridis, colorbar=false, clims=clims) # clims=climscolor=:balance,
    heatmap_figure = plot(p1, p2, p3, p4, p5, p6, layout=(3, 3), size=(1400, 1500), colorbar=true, titlefontsize=fontsize, guidefontsize=fontsize, tickfontsize=fontsize, plot_title="Solutions at final time $tfinal")
    display(plot(p1, axis=false, framestyle=:none, ticks=false, size=(625, 625))) #Consensus makers
    savefig("HM_c,kappa=$κ,lambda=$λ,s=$s.png")
    display(plot(p2, axis=false, framestyle=:none, ticks=false, size=(625, 625))) #Gridlockers
    savefig("HM_g,kappa=$κ,lambda=$λ,s=$s.png")
    display(plot(p3, axis=false, framestyle=:none, ticks=false, size=(625, 625))) #Zealots of party 1
    savefig("HM_z1,kappa=$κ,lambda=$λ,s=$s.png")
    display(plot(p4, axis=false, framestyle=:none, ticks=false, size=(625, 625))) #Zealots of party 2
    savefig("HM_z2,kappa=$κ,lambda=$λ,s=$s.png")
    display(plot(p5, axis=false, framestyle=:none, ticks=false, size=(625, 625))) #Population
    savefig("HM_p,kappa=$κ,lambda=$λ,s=$s.png")
    display(plot(p6, axis=false, framestyle=:none, ticks=false, size=(625, 625))) #Vote
    savefig("HM_v,kappa=$κ,lambda=$λ,s=$s.png")
    display(heatmap_figure)#savefig("Heatmap_Clean_DifferentD_EvenIC_Finaltime=$tfinal.pdf")

    # TIME SERIES: Compute averages over the domain at each time step
    time_steps = sol.t
    average_c = [mean(unpack(sol[i])[1]) for i in 1:length(time_steps)] #Average Consensus-makers
    average_g = [mean(unpack(sol[i])[2]) for i in 1:length(time_steps)] #Average Gridlockers
    average_z1 = [mean(unpack(sol[i])[3]) for i in 1:length(time_steps)] #Average Zealots of party 1
    average_z2 = [mean(unpack(sol[i])[4]) for i in 1:length(time_steps)] #Average Zealots of party 2
    average_Fitness_z1 = [mean(f_z1(unpack(sol[i])[5])) for i in 1:length(time_steps)]
    average_Fitness_z2 = [mean(f_z2(unpack(sol[i])[5])) for i in 1:length(time_steps)]
    average_Fitness_c = [mean(f_c(unpack(sol[i])[5])) for i in 1:length(time_steps)]
    average_Fitness_g = [mean(f_g(unpack(sol[i])[5])) for i in 1:length(time_steps)]
    #ts_max_pop = [maximum(unpack(sol[i])[1]) + maximum(unpack(sol[i])[2]) + maximum(unpack(sol[i])[3]) + maximum(unpack(sol[i])[4]) for i in 1:length(time_steps)]
    average_v = [mean(unpack(sol[i])[5]) .* mean(unpack(sol[i])[1]) .+ mean(unpack(sol[i])[2]) .* mean(unpack(sol[i])[6]) .+ mean(unpack(sol[i])[3]) .+ mean(unpack(sol[i])[4]) for i in 1:length(time_steps)]
    # Above computes c*v_c + g*v_g + z at each time step
    # Plot averages
    time_series = plot(time_steps[2:end], average_c[2:end], xlabel="Time", ylabel="Mean", lw=8, xlabelfontsize=20, ylabelfontsize=20, xscale=:log10,
        titlefontsize=12, legendfontsize=12, tickfontsize=16, yticks=0:0.25:1.1, ylim=(-0.01, 1.01), xlim=(1, tfinal*1.2), legend=false) #, label="Mean Consensus Makers"
    plot!(time_steps[2:end], average_g[2:end], lw=8)
    plot!(time_steps[2:end], average_z1[2:end], lw=8)
    plot!(time_steps[2:end], average_z2[2:end], lw=8)
    plot!(time_steps[2:end], average_v[2:end], lw=8, 
    xticks = ([1, 10^1, 10^2, 10^3, 10^4], ["1", "10¹", "10²", "10³", "10⁴"]))
    # plot!(time_steps, average_Fitness_z1,lw=8)
    # plot!(time_steps, average_Fitness_z2,lw=8)
    #plot!(time_steps, ts_max_pop, label="Max Population",lw=3)
    display(time_series)
    savefig("TS,kappa=$κ,lambda=$λ,s=$s.png")
end
#xticks=0:1000:tfinal