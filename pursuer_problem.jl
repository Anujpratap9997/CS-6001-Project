# Function to solve pursuers problem
function pursuer_optimal(x_e, y_e, δt, params)
	(;vm_p, um_p, state_p0, n, vm_e, um_e, N, state_e0, l) = params

	# tmin and tmax are 
    tmin = (2 * pi * 0.6) / vm_p
    tmax = 10.0
    x_p0, y_p0, th_p0 = state_p0[1], state_p0[2], state_p0[3]

    pursuer = Model(Ipopt.Optimizer)
    set_silent(pursuer)

	timeE = LinRange(0,N*δt, N)

	function xval(Δt)
		#global δt, timeE, x_e, n
		interp = LinearInterpolation(timeE, x_e, extrapolation_bc = x_e[N])
		return interp(n*Δt)
	end
	register(pursuer,:xval, 1, xval; autodiff = true)
	
	function yval(Δt)
		#global δt, timeE, y_e, n
		interp = LinearInterpolation(timeE, y_e, extrapolation_bc = y_e[N])
		return interp(n*Δt)
	end
	register(pursuer,:yval, 1, yval; autodiff = true)

    # Decision variables
    @variables(pursuer, begin
        tmin / n <= Δt <= tmax / n     # Time step                       # HEURISTICS NEEDED HERE

        ## State variables
        x_p[1:n]
        y_p[1:n]
        th_p[1:n]           # Orientation 
        ## Control variables
        -um_p ≤ u_p[1:n] ≤ um_p

    end)

    @objective(pursuer, Min, Δt)

    fix(x_p[1], x_p0; force=true)
    fix(y_p[1], y_p0; force=true)
    fix(th_p[1], th_p0; force=true)


	# Final conditions
	@NLconstraint(pursuer, con,(x_p[n] - xval(Δt))^2 + (y_p[n] - yval(Δt))^2 <= l * l)

    for j in 2:n
        @NLconstraint(pursuer, x_p[j] == x_p[j-1] + Δt * 0.5 * vm_p * (cos(th_p[j-1]) + cos(th_p[j])))
        ## Trapezoidal integration
        @NLconstraint(pursuer, y_p[j] == y_p[j-1] + Δt * 0.5 * vm_p * (sin(th_p[j-1]) + sin(th_p[j])))
        ## Trapezoidal integration
        @NLconstraint(pursuer, th_p[j] == th_p[j-1] + 0.5 * Δt * vm_p * (u_p[j] + u_p[j-1]))
        ## Trapezoidal integration

    end

    #Solve PD using the extended solution to obtain e_(k+1), T_(k+1), α_(k+1)
    println("Solving Pursuer...")
    optimize!(pursuer)
    return value.(x_p), value.(y_p), value.(th_p), value.(u_p), value.(Δt), dual(con)
end