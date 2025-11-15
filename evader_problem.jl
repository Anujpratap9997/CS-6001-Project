function evader_optimal(x_p, y_p, th_p, Δt, state_eC, params, α)
	(;vm_p, um_p, state_p0, n, vm_e, um_e, N, state_e0, l) = params
    x_e0, y_e0, th_e0 = state_e0[1], state_e0[2], state_e0[3]
    # x_eC, y_eC, th_eT = state_eC[1], state_eC[2], state_eC[3]
    # x_eT, y_eT, th_eT = x_p[n-1], y_p[n-1], th_p[n-1]
    x_eC, y_eC, th_eC = x_p[n], y_p[n], th_p[n]


    evader = Model(Ipopt.Optimizer)
    set_silent(evader)

    # #expression for α
    # denom_first = dot([2*(x_p[n-2]-x_eT) 2*(y_p[n-2]-y_eT)], [vm_p*cos((th_p[n-2])) vm_p*sin((th_p[n-2]))])
    # denom_second = dot(-[2*(x_p[n-2]-x_eT) 2*(y_p[n-2]-y_eT)], [vm_e*cos((th_eT)) vm_e*sin((th_eT))])

    # α = -1/(denom_first+denom_second)

    #quantities required for objective function
    dlbyde = [2*(x_eC-x_p[n-1]) 2*(y_eC-y_p[n-1])]
    first_term = -α*dlbyde

    # Decision variables
    @variables(evader, begin

        ## State variables
        x_e[1:N]            
        y_e[1:N]        
        th_e[1:N]           # Orientation 
        ## Control variables
        -um_e ≤ u_e[1:N] ≤ um_e
    end)

    fix(x_e[1], x_e0; force = true)
    fix(y_e[1], y_e0; force = true)
    fix(th_e[1], th_e0; force = true)

    ## Dynamics

    for j in 2:N

        @NLconstraint(evader, x_e[j] == x_e[j - 1] + Δt *0.5* vm_e*(cos(th_e[j - 1])+cos(th_e[j])))
        ## Trapezoidal integration
        @NLconstraint(evader, y_e[j] == y_e[j - 1] + Δt *0.5* vm_e*(sin(th_e[j - 1])+sin(th_e[j])))
        ## Trapezoidal integration
        @NLconstraint(evader, th_e[j] == th_e[j - 1] + 0.5*Δt *vm_e* (u_e[j]+u_e[j-1]))
        ## Trapezoidal integration

    end

    @objective(evader, Max, (x_e[N]- x_eC)*first_term[1] + (y_e[N]-y_eC)*first_term[2])

    println("Solving Evader...")
    optimize!(evader)

    return value.(x_e), value.(y_e), value.(th_e), value.(u_e)
end