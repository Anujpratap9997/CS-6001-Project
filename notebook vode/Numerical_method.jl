using JuMP
using LinearAlgebra
using Plots
# using CSV, DataFrames
import Ipopt
using LaTeXStrings
using Interpolations

# File which contains function to solve pursuer's problem
include("pursuer_problem.jl")
# File which contains function to solve evader's problem
include("evader_problem.jl")

# Function to simulate initial Dubins trajectory
function dubins_next_state(s, u, δt)
    sn = zeros(3)
    sn[1] = s[1] + cos(s[3]) * δt
    sn[2] = s[2] + sin(s[3]) * δt
    sn[3] = s[3] + u * δt
    return sn
end

struct Params
	# Pursuer parameters
    vm_p::Float64
    um_p::Float64
    state_p0::Vector{Float64}
    n::Int64

	# Evader parameters
    vm_e::Float64
    um_e::Float64
    state_e0::Vector{Float64}
    N::Int64

    l::Float64   # Capture distance
end

vm_p = 2.0
um_p = 1.0
state_p0 = [0.0, 0.0, 0.0]
n = 200
vm_e = 1.0
um_e = 1.0
# state_e0 = [0.0, 7.0, -π/2]
# state_e0 = [7.0, 3.0, π/2]

# state_p0 = [-5.5, -2.1, 2.78]

# stateE0 = [0, 7, -π/2]
state_e0 = [-3,-7, -π/2]

N = 200
l = 0.05
params = Params(vm_p, um_p, state_p0, n,
                vm_e, um_e, state_e0, N, l)


function main(params)
    (; vm_p, um_p, state_p0, n, vm_e, um_e, N, state_e0, l) = params
    x_e0, y_e0, th_e0 = state_e0[1], state_e0[2], state_e0[3]
    x_p0, y_p0, th_p0 = state_p0[1], state_p0[2], state_p0[3]

    threshold = 0.0001   # Capture distance
    condition = true
    iter = 1

    # Variables to store the values
    x_e = zeros(N)
    y_e = zeros(N)
    th_e = zeros(N)
    u_e = zeros(N)

    x_p = zeros(n)
    y_p = zeros(n)
    th_p = zeros(n)
    u_p = zeros(n)

    t = zeros(N)

    # Initial randomly generated evader trajectory
    δt = 0.1
    timeE = LinRange(0.0,δt*N,N)
    
	se = zeros(3, length(timeE))
	se[:,1] = [x_e0, y_e0, th_e0]
    for i = 1:(length(timeE)-1)
        se[:, i+1] = dubins_next_state(se[:, i], 0, δt)
    end
	x_e = se[1,:]
	y_e = se[2,:]
	th_e = se[3,:]

	# -----------------------------Initial pursuer solution-------------------------------------------
	x_p, y_p, th_p, u_p, Δt, α = pursuer_optimal(x_e, y_e, δt, params)
	TE = Δt*n
	NE = Int(floor(TE/δt))

	# Inital evaders terminal position
	x_eC = x_e[NE]
	y_eC = y_e[NE]
	th_eC = th_e[NE]
	δt = Δt

    fig1 = plot(aspect_ratio=1, frame_style=:box, lw=2,  title="Game Simulation using Numerical Method", titlefontsize=10)
    cap_x = 0.0
    cap_y = 0.0


    # Arrow length
    L = 0.6

    # Convert radian angles to dx, dy
    dx_p = L * cos(state_p0[3])
    dy_p = L * sin(state_p0[3])

    dx_e = L * cos(state_e0[3])
    dy_e = L * sin(state_e0[3])



    while condition 
		N = 200
        n = 200

		#-----------------------------------------------------------------------------------------
		# Plotting both of them together after each iteration
		x::Matrix{Float64} = [value.(x_e) value.(x_p)]
		y = [value.(y_e) value.(y_p)]
		fig1 = plot(aspect_ratio=1, frame_style=:box, lw=2,  title="Game Simulation using Numerical Method", titlefontsize=10)
        plot!(fig1, x[:, 1], y[:, 1], color=:red, ls=:dash, lw=2, label=L"$\textrm{Evader \; trajectory}$")
        plot!(fig1, x[:, 2], y[:, 2], color=:blue, ls=:dashdot, lw=2, label=L"$\textrm{Pursuer \; trajectory}$")
        scatter!(fig1, [x_e0], [y_e0], label=nothing)
        scatter!(fig1, [x_p0], [y_p0], label=nothing)
        xlabel!(fig1, L"$x-\textrm{position}$")
        ylabel!(fig1, L"$y-\textrm{position}$")
        plot!(fig1, [x_p0, x_p0 + dx_p], [y_p0, y_p0 + dy_p], color=:black, arrow=true, arrowsize=1, lw=2, label=nothing)
        scatter!(fig1, [x_p0], [y_p0], color=:black, markersize=5, label=nothing)
        plot!(fig1, [x_e0, x_e0 + dx_e ], [y_e0, y_e0 + dy_e], color=:black, arrow=true, arrowsize=1, lw=2, label=nothing)
        scatter!(fig1, [x_e0], [y_e0], color=:black, markersize=5, label=nothing)
        # scatter!(fig1, [0], [0], color=:black, markersize=5, label=nothing)
        scatter!(fig1, [x[end, 2]], [y[end, 2]], shape=:star5, label = L"$\textrm{Capture \; Point}$", markersize=5)
        # scatter!(fig1, [x[end,2]], [y[end,2]], shape=:circle, color=:green)
        annotate!(fig1, x_p0 - 0.5, y_p0 + 0.2, L"$\bf{P}$")
        annotate!(fig1, x_e0 - 0.5, y_e0 + 0.2, L"$\bf{E}$")
        display(fig1)
        cap_x = x[end, 2]
        cap_y = y[end, 2]
		sleep(0.5)  # Pause for 0.5 seconds

		#-----------------------------------------------------------------------------------------
        #------------------------------------- EVADER PART -------------------------------------
        x_e, y_e, th_e, u_e = evader_optimal(x_p, y_p, th_p, Δt, [x_eC, y_eC, th_eC], params, α)
		x_eC_prev, y_eC_prev = x_eC, y_eC
		TE = Δt*n
		# Evaders terminal position
		x_eC = x_e[NE]
		y_eC = y_e[NE]
		th_eC = th_e[NE]
		δt = Δt

		#-----------------------------------------------------------------------------------------
		#--------------------------------------PURSUER PART-------------------------------------------
		x_p, y_p, th_p, u_p, Δt, α = pursuer_optimal(x_e, y_e, δt, params)

        condition = (value.(x_eC_prev - x_eC))^2 + (value.(y_eC_prev - y_eC))^2 > threshold

        t[iter] = Δt * n
        iter = iter + 1
    end
    println("Capture Point: (", cap_x, ", ", cap_y, ")")
    return fig1
    # CSV.write("./twoCarsData.csv", DataFrame(matrix, :auto),header = false, append = true)

end

# Run the function main
fig1 = main(params)
# savefig(fig1, "PENumerical.pdf")
outdir = joinpath(@__DIR__, "Results/Numerical Method/")
# mkpath(outdir) # creates the folder if it doesn't exist
savefig(fig1, joinpath(outdir, "PENumerical_case_IV.pdf"))