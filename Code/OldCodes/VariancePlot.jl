using DifferentialEquations, DiffEqCallbacks, Plots, LinearAlgebra, Roots, Statistics, Sundials, ColorSchemes
@time begin
    # Parameters
    κ = 0 # 1 for directed, 0 for undirected
    λ = 0 # Economic preference
    s = 0 # Spillovers
    tax = 0 # 0.2 Taxes
    L = 20 # Length of domain
   # M = 0.05#/L^2 # M/L^2 in the main text , L^2 moved to Directed movement to account for changing M
    x = range(1, L, length=L) # X size
    y = range(1, L, length=L) # y size 
    tfinal = 10000 # Final time
    X, Y = [xi for xi in x, yi in y], [yi for xi in x, yi in y]
    # Precompute Euclidean distance squared between every pair of grid nodes.
    Xv, Yv  = vec(X), vec(Y)                                 
    Dist = (Xv .- Xv') .^ 2 .+ (Yv .- Yv') .^ 2 
    invDist = inv.(Dist)                                      
    invDist[diagind(invDist)] .= 0  #Distance from a node to itself is set to zero, Good 
    A = 1/sum(invDist)  #Normalization factor for Distance Dependent movement, Good                     
    #Initial conditions
    c₀ = rand(L,L) # Initial distribution for Consensus-makers
    g₀ = rand(L,L) # Initial distribution for Gridlockers
    z1₀ = rand(L,L) # Initial distribution for Party 1 Zealots
    z2₀ = rand(L,L) # Initial distribution for Party 2 Zealots
    total = 0.5 * c₀ .+ 0.5 * g₀ .+ z1₀  # Normalize
    c₀ = c₀ ./ total
    g₀ = g₀ ./ total
    z1₀ = z1₀ ./ total
    z2₀ = z2₀ ./ total 
    v₀ = c₀ .+  g₀ .+ z1₀ #Initial vote
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
        return total / count # average over existing neighbors instead of raw sum
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
        M = p  # M passed as ODE parameter, not captured from outer scope
        # unpack state
        c, g, z1, z2, vc, vg = unpack(u)

        # compute v and fitnesses/utilities
        v = c .* vc .+ g .* vg .+ z1 #Good
        # P_total = c + g + z1 + z2 #Matrix, Good
        # v = (c .* vc .+ g .* vg .+ z1) ./ max.(P_total, 1e-12) #Normalize vote, for gravity model
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

        ## 1.) Directed movement — factorised mean-field sums, Voters can move anywhere, κ=0 for undirected, κ=1 for directed
        for (pop, U, du_pop) in (
                (c,  U_c,  du_c),
                (g,  U_g,  du_g),
                (z1, U_z1, du_z1),
                (z2, U_z2, du_z2))
            eU  = exp.(-κ .* U)
            emU = inv.(eU)                       
            Sw  = dot(vec(pop), vec(eU))  # Sw = Σ_(j,k) pop(j,k)*eU(j,k)    
            Se  = sum(emU)      # Se = Σ_(j,k) emU(j,k)                   
            du_pop .+= M ./ (L^2) .* (emU .* Sw .- pop .* eU .* Se) 
        end

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
        # P_total = c + g + z1 + z2   # computed once at time step t, reused for all four populations
        # P_tot = vec(P_total)
        # Z_vec = sum(P_tot) .- P_tot # Vector of total population minus population at each node
        # Z_grav = sum(Z_vec)/ length(Z_vec) # reshape(Z_vec, L, L)
        #  for (pop, U, du_pop) in (
        #          (c,  U_c,  du_c),
        #          (g,  U_g,  du_g),
        #          (z1, U_z1, du_z1),
        #          (z2, U_z2, du_z2))

        #      eU = exp.(κ .* U)
        #      emU = inv.(eU)
        #      iv=vec(pop) 
        #      Sw_in  = dot(iv,vec(eU))   # Σ_{j,k} i(j,k)·exp(+κu(j,k)) -destination attractiveness
        #      Sw_out = dot(P_tot, vec(emU))   # Σ_{j,k} p(j,k)·exp(-κu(j,k)) - source attractiveness
        #      inflow = reshape(P_tot .* vec(emU),L,L).* Sw_in
        #      outflow = reshape(iv .* vec(eU), L, L) .* Sw_out
        #     #  P_cap = 4.0   # e.g. 4× the initial per-node total
        #     #  sat = max.(1 .- P_tot ./ P_cap, 0.0)   # logistic damping near capacity
        #      du_pop .+= M ./Z_grav .*(inflow .- outflow)
        #  end

        # Algebraic dynamics for v_c, v_g, Good
        du_vc = (1 .- vc) .* v .^ 2 .- vc .* (1 .- v) .^ 2
        du_vg = (1 .- vg) .* (1 .- v) .^ 2 .- vg .* v .^ 2
        # pack into du (in-place)
        du .= pack(du_c, du_g, du_z1, du_z2, du_vc, du_vg)
        return nothing
    end


    ## VARIANCE ANALYSIS: Loop over different diffusion coefficients
    iter = 20
    M_i_array = range(0, 0.1, length=iter) # Diffusion coefficient for all types
    P = 5 # Number of iterations per M value (averaged over different random ICs)
    averagevar = zeros(iter)
    maxvar = zeros(iter)

    for i in 1:iter
        M_val = M_i_array[i]
        averagevar_P = zeros(P)
        maxvar_P     = zeros(P)

        for p in 1:P
            # FIX 2: Re-randomize initial conditions for each repetition
            c₀  = rand(L, L)
            g₀  = rand(L, L)
            z1₀ = rand(L, L)
            z2₀ = rand(L, L)
            total_ic = 0.5 .* c₀ .+ 0.5 .* g₀ .+ z1₀
            c₀  = c₀  ./ total_ic
            g₀  = g₀  ./ total_ic
            z1₀ = z1₀ ./ total_ic
            z2₀ = z2₀ ./ total_ic
            v₀_p  = c₀ .+ g₀ .+ z1₀
            vc₀_p = (v₀_p .^ 2) ./ (2 .* v₀_p .^ 2 .- 2 .* v₀_p .+ 1)
            vg₀_p = ((1 .- v₀_p) .^ 2) ./ (2 .* v₀_p .^ 2 .- 2 .* v₀_p .+ 1)
            u0_p  = pack(c₀, g₀, z1₀, z2₀, vc₀_p, vg₀_p)

            # FIX 4: Pass M_val as ODE parameter instead of capturing from outer scope
            problem = ODEProblem(pg_system!, u0_p, tspan, M_val)
            cb  = PositiveDomain()
            sol = solve(problem, Rodas5(), callback=cb, saveat=1)

            # FIX 1: var_timeseries is a vector (one scalar per time step),
            # because var(v) returns a single spatial-variance scalar, not an L×L matrix
            var_timeseries = zeros(length(sol))
            for t in 1:length(sol)
                c, g, z1, z2, v_c, v_g = unpack(sol[t])
                population = c .+ g .+ z1 .+ z2
                c  = c  ./ max.(population, 1e-12)
                g  = g  ./ max.(population, 1e-12)
                z1 = z1 ./ max.(population, 1e-12)
                z2 = z2 ./ max.(population, 1e-12)
                v  = c .* v_c .+ g .* v_g .+ z1
                var_timeseries[t] = var(vec(v))  # scalar: spatial variance across grid at time t
            end

            averagevar_P[p] = mean(var_timeseries)
            maxvar_P[p]     = maximum(var_timeseries)
        end

        # FIX 3: Average over all repetitions *after* the p-loop
        averagevar[i] = mean(averagevar_P)
        maxvar[i]     = mean(maxvar_P)
    end
    
    # Plot variance results
    averagevar_plot = plot(M_i_array, averagevar, xlabel="M",
        ylabel="Mean Variance of Vote",
        lw=8, xlabelfontsize=20, ylabelfontsize=20, titlefontsize=12,
        legendfontsize=12, tickfontsize=16,
        legend=false)
    display(averagevar_plot)
    #savefig("AverageVariance_DifferentM.pdf")
    
    # maxvar_plot = plot(M_i_array, maxvar, xlabel="M",
    #     ylabel="Max Vote Variance",
    #     lw=8, xlabelfontsize=20, ylabelfontsize=20, titlefontsize=12,
    #     legendfontsize=12, tickfontsize=16,
    #     legend=false)
    # display(maxvar_plot)
    # #savefig("MaxVariance_DifferentM.pdf")

end
#xticks=0:1000:tfinal