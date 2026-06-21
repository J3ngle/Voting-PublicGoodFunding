using DifferentialEquations, Plots, LinearAlgebra, Roots, Statistics, Sundials, ColorSchemes
@time begin
    # Parameters
    κ = 1 # 1 for directed, 0 for undirected
    λ = 0 # Economic preference
    s = 0 # Spillovers
    tax = 0 # 0.2 Taxes
    L = 20 # Length of domain
    M = 0.05 #0.05/L^2#, Good 
    x = range(0, L, length=L) # X size
    y = range(0, L, length=L) # y size 
    tfinal = 10000 # Final time
    X, Y = [xi for xi in x, yi in y], [yi for xi in x, yi in y]
    # Precompute Euclidean distance squaredbetween every pair of grid nodes.
    Xv, Yv  = vec(X), vec(Y)                                 
    Dist    = (Xv .- Xv') .^ 2 .+ (Yv .- Yv') .^ 2 
    invDist = inv.(Dist)                                      
    invDist[diagind(invDist)] .= 0  #Distance from a node to itself is set to zero, Good 
    Z =1/sum(invDist)  #Normalization factor for Distance Dependent movement, Good                     
    #Initial conditions
    c₀ = rand(L,L) # Initial distribution for Consensus-makers
    g₀ = rand(L,L) # Initial distribution for Gridlockers
    #g₀ = clamp.(g₀, 0, 0.125) 
    z1₀ = rand(L,L) # Initial distribution for Party 1 Zealots
    z2₀ = rand(L,L) # Initial distribution for Party 2 Zealots
    #z2₀ = clamp.(z2₀, 0, 0.125)
    total = c₀ .+ g₀ .+ z1₀ .+ z2₀ # Normalize
    c₀ = c₀ ./ total
    g₀ = g₀ ./ total
    z1₀ = z1₀ ./ total
    z2₀ = z2₀ ./ total 
    v₀ = 0.5 * c₀ .+ 0.5 * g₀ .+ z1₀ #Initial vote
    vc₀ = (v₀ .^ 2) ./ (2 .* v₀ .^ 2 .- 2 .* v₀ .+ 1) # Initial vote for Consensus-makers, Good
    vg₀ = ((1 .- v₀) .^ 2) ./ (2 .* v₀ .^ 2 .- 2 .* v₀ .+ 1) # Initial vote for Gridlockers, Good

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

    function positive_spillover(v, i, j)
        total = 0.0
        count = 0 #Need to adjust for number of neighbours
        if i > 1;  total += v[i-1, j]; count += 1; end #if south neighbor exists
        if i < L;  total += v[i+1, j]; count += 1; end #if north neighbor exists
        if j > 1;  total += v[i, j-1]; count += 1; end #if west neighbor exists
        if j < L;  total += v[i, j+1]; count += 1; end #if east neighbor exists
        return total / count   # average over existing neighbors instead of raw sum
    end

    # Fitness functions
    f_c(v) = (1 .+ cos.(2 .* pi .* v)) ./ 2 #Strategy fitness for Consensus makers, Good
    f_g(v) = (1 .- cos.(2 .* pi .* v)) ./ 2 #Strategy fitness for Gridlockers, Good
    f_z1(v) = (1 .- cos.(pi .* v)) ./ 2 #Strategy fitness for Zealots party 1, Good
    f_z2(v) = (1 .+ cos.(pi .* v)) ./ 2 #Strategy fitness for Zealots party 2, Good

    # Utility functions 
    u_c(v, positive_spillover) = λ .* ((1 - s) .* v .+ (s) * (positive_spillover)) .+ (1 - λ) .* f_c(v) #Good
    u_g(v, positive_spillover) = λ .* ((1 - s) .* v .+ (s) * (positive_spillover)) .+ (1 - λ) .* f_g(v) #Good
    u_z1(v, positive_spillover) = λ .* ((1 - s) .* v .+ (s) * (positive_spillover)) .+ (1 - λ) .* f_z1(v) #Good
    u_z2(v, positive_spillover) = λ .* ((1 - s) .* v .+ (s) * (positive_spillover)) .+ (1 - λ) .* f_z2(v) #Good

    # Initial condition
    u0 = pack(c₀, g₀, z1₀, z2₀, vc₀, vg₀)
    du0 = zeros(size(u0))
    tspan = (0.0, tfinal)

    function pg_system!(du, u, p, t)
        # unpack state
        c, g, z1, z2, vc, vg = unpack(u)

        # compute v and fitnesses/utilities
        v = c .* vc .+ g .* vg .+ z1 #Good
        P_total = c + g + z1 + z2 #Matrix, Good
        v = (c .* vc .+ g .* vg .+ z1) ./ max.(P_total, 1e-12) #Normalize vote, for gravity model
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

        # Local replicator dynamics.
        du_c = c .* g .* (F_c .- F_g) .+ c .* z1 .* (F_c .- F_z1) .+ c .* z2 .* (F_c .- F_z2)
        du_g = g .* c .* (F_g .- F_c) .+ g .* z1 .* (F_g .- F_z1) .+ g .* z2 .* (F_g .- F_z2)
        du_z1 = z1 .* c .* (F_z1 .- F_c) .+ z1 .* g .* (F_z1 .- F_g)
        du_z2 = z2 .* c .* (F_z2 .- F_c) .+ z2 .* g .* (F_z2 .- F_g)

        # ## 1.) Directed movement — factorised mean-field sums, Voters can move anywhere, κ=0 for undirected, κ=1 for directed
        # for (pop, U, du_pop) in (
        #         (c,  U_c,  du_c),
        #         (g,  U_g,  du_g),
        #         (z1, U_z1, du_z1),
        #         (z2, U_z2, du_z2))
        #     eU  = exp.(κ .* U)
        #     emU = inv.(eU)                       
        #     Sw  = dot(vec(pop), vec(eU))  # Sw = Σ_(j,k) pop(j,k)*eU(j,k)    
        #     Se  = sum(emU)      # Se = Σ_(j,k) emU(j,k)                   
        #     du_pop .+= M .* (emU .* Sw .- pop .* eU .* Se) 
        # end

        # # 2.) Directed movement divided by distance
        #     for (pop, U, du_pop) in (
        #             (c,  U_c,  du_c),
        #             (g,  U_g,  du_g),
        #             (z1, U_z1, du_z1),
        #             (z2, U_z2, du_z2))
        #         eU  = exp.(κ .* U)
        #         emU = inv.(eU)
        #         Sw  = reshape(invDist * vec(pop .* eU), L, L)   # Sw[i,j] = Σ_k pop[k]*eU[k] / dist((i,j),k)
        #         Se  = reshape(invDist * vec(emU), L, L)   # Se[i,j] = Σ_k emU[k] / dist((i,j),k)
        #         du_pop .+= M .* Z .* (emU .* Sw .- pop .* eU .* Se)
        #     end

        #  3.) Directed distant movement (Population dependent) — Gravity model
        P_total = c + g + z1 + z2   # computed once at time step t, reused for all four populations
        P_tot = vec(P_total)
        Z_vec = sum(P_tot) .- P_tot # Vector of total population minus population at each node
        Z_grav = sum(Z_vec)/ length(Z_vec) # reshape(Z_vec, L, L)
         for (pop, U, du_pop) in (
                 (c,  U_c,  du_c),
                 (g,  U_g,  du_g),
                 (z1, U_z1, du_z1),
                 (z2, U_z2, du_z2))

             eU = exp.(κ .* U)
             emU = inv.(eU)
             iv=vec(pop) 
             Sw_in  = dot(iv,vec(eU))   # Σ_{j,k} i(j,k)·exp(+κu(j,k)) -destination attractiveness
             Sw_out = dot(P_tot, vec(emU))   # Σ_{j,k} p(j,k)·exp(-κu(j,k)) - source attractiveness
             inflow = reshape(P_tot .* vec(emU),L,L).* Sw_in
             outflow = reshape(iv .* vec(eU), L, L) .* Sw_out
            #  P_cap = 4.0   # e.g. 4× the initial per-node total
            #  sat = max.(1 .- P_tot ./ P_cap, 0.0)   # logistic damping near capacity
             du_pop .+= M ./Z_grav .*(inflow .- outflow)
         end

        # Algebraic dynamics for v_c, v_g, Good
        du_vc = (1 .- vc) .* v .^ 2 .- vc .* (1 .- v) .^ 2
        du_vg = (1 .- vg) .* (1 .- v) .^ 2 .- vg .* v .^ 2
        # pack into du (in-place)
        du .= pack(du_c, du_g, du_z1, du_z2, du_vc, du_vg)
        return nothing
    end

    problem = ODEProblem(pg_system!, u0, tspan) #Good
    cb = PositiveDomain()
    sol = solve(problem, Rodas5(), callback=cb, saveat=1, reltol=1e-12, abstol=1e-12) #, reltol=1e-12, abstol=1e-12) #Good
    #sol = solve(problem, CVODE_BDF(), saveat=1)

    ## HEATMAPS
    fontsize = 14
    c, g, z1, z2, v_c, v_g = unpack(sol[end]) #computations from the end of the simulation, Good
    population = c .+ g .+ z1 .+ z2 #Compute population, this is a matrix
    total_population = sum(c) .+ sum(g) .+ sum(z1) .+ sum(z2) #Compute population at the end, this is a scalar
    c = c ./ population #Normalize c, Good
    g = g ./ population #Normalize g, Good
    z1 = z1 ./ population #Normalize z, Good
    z2 = z2 ./ population #Normalize z2, Good
    heatmap_population = population/ maximum(population) #Normalize population, Good
    v = (c .* v_c .+ g .* v_g .+ z1) #Compute v at the end, Good
    clims = (0, 1) #Color limits for heatmaps
    p1 = heatmap(x, y, c', aspect_ratio=1, colorbar=false, clims=clims)# clims=clims
    p2 = heatmap(x, y, g', aspect_ratio=1, colorbar=false, clims=clims)# clims=clims
    p3 = heatmap(x, y, z1', aspect_ratio=1, colorbar=false, clims=clims)# clims=clims
    p4 = heatmap(x, y, z2', aspect_ratio=1, colorbar=false, clims=clims)# clims=clims
    p5 = heatmap(x, y, heatmap_population', aspect_ratio=1, colorbar=false, clims=clims) # clims=clims
    p6 = heatmap(x, y, v', aspect_ratio=1, color=:viridis, colorbar=false, clims=clims) # clims=climscolor=:balance,
    heatmap_figure = plot(p1, p2, p3, p4, p5, p6, layout=(3, 3), size=(1400, 1500), colorbar=true, titlefontsize=fontsize, guidefontsize=fontsize, tickfontsize=fontsize, plot_title="Solutions at final time $tfinal")
    display(plot(p1, axis=false, framestyle=:none, ticks=false, size=(625, 625))) #Consensus makers
    savefig("GravityModelHM_c,kappa=$κ,lambda=$λ,s=$s.png")
    display(plot(p2, axis=false, framestyle=:none, ticks=false, size=(625, 625))) #Gridlockers
    savefig("GravityModelHM_g,kappa=$κ,lambda=$λ,s=$s.png")
    display(plot(p3, axis=false, framestyle=:none, ticks=false, size=(625, 625))) #Zealots of party 1
    savefig("GravityModelHM_z1,kappa=$κ,lambda=$λ,s=$s.png")
    display(plot(p4, axis=false, framestyle=:none, ticks=false, size=(625, 625))) #Zealots of party 2
    savefig("GravityModelHM_z2,kappa=$κ,lambda=$λ,s=$s.png")
    display(plot(p5, axis=false, framestyle=:none, ticks=false, size=(625, 625))) #Population
    savefig("GravityModelHM_p,kappa=$κ,lambda=$λ,s=$s.png")
    display(plot(p6, axis=false, framestyle=:none, ticks=false, size=(625, 625))) #Vote
    savefig("GravityModelHM_v,kappa=$κ,lambda=$λ,s=$s.png")
    display(heatmap_figure)#savefig("Heatmap_Clean_DifferentD_EvenIC_Finaltime=$tfinal.pdf")

    ## TIME SERIES: Compute averages over the domain at each time step
    time_steps = sol.t
    #averages
    average_c = [mean(unpack(sol[i])[1]) for i in 1:length(time_steps)] #Average Consensus-makers
    average_g = [mean(unpack(sol[i])[2]) for i in 1:length(time_steps)] #Average Gridlockers
    average_z1 = [mean(unpack(sol[i])[3]) for i in 1:length(time_steps)] #Average Zealots of party 1
    average_z2 = [mean(unpack(sol[i])[4]) for i in 1:length(time_steps)] #Average Zealots of party 2
   
    average_v = [mean(unpack(sol[i])[5].*unpack(sol[i])[1]) .+ mean(unpack(sol[i])[2].*unpack(sol[i])[6]) .+ mean(unpack(sol[i])[3]) for i in 1:length(time_steps)]
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
    savefig("GravityModelTS,kappa=$κ,lambda=$λ,s=$s.png")
end
#xticks=0:1000:tfinal