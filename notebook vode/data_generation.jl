using JuMP
using LinearAlgebra
using Plots
using CSV, DataFrames
import Ipopt
using LaTeXStrings
using Interpolations
using Random

# -------------------------------------------------------------------
# REQUIRED FILES (Assumed to be in the same directory)
# -------------------------------------------------------------------
# File which contains function to solve pursuer's problem
include("pursuer_problem.jl")
# File which contains function to solve evader's problem
include("evader_problem.jl")

# -------------------------------------------------------------------
# HELPER FUNCTIONS AND STRUCTS
# -------------------------------------------------------------------

# Function to simulate initial Dubins trajectory
function dubins_next_state(s, u, δt)
    sn = zeros(3)
    sn[1] = s[1] + cos(s[3]) * δt
    sn[2] = s[2] + sin(s[3]) * δt
    sn[3] = s[3] + u * δt
    return sn
end

# Struct to hold parameters for a single game
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

# -------------------------------------------------------------------
# FUNCTION TO SOLVE ONE GAME SCENARIO
# -------------------------------------------------------------------

"""
Solves a single pursuit-evasion game for a given set of parameters.
Once converged, it appends the resulting 200 data points to the CSV file.
"""
function solve_and_save_game(params::Params, csv_filename::String)
    # --- Setup ---
    (; vm_p, um_p, state_p0, n, vm_e, um_e, N, state_e0, l) = params
    x_e0, y_e0, th_e0 = state_e0[1], state_e0[2], state_e0[3]
    x_p0, y_p0, th_p0 = state_p0[1], state_p0[2], state_p0[3]

    threshold = 0.0001   # Convergence threshold
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

    # --- Initial Evader Trajectory (Guess) ---
    δt = 0.1
    timeE = LinRange(0.0, δt * N, N)
    
    se = zeros(3, length(timeE))
    se[:,1] = [x_e0, y_e0, th_e0]
    for i = 1:(length(timeE)-1)
        se[:, i+1] = dubins_next_state(se[:, i], 0.0, δt) # Use 0.0 for initial guess
    end
    x_e = se[1,:]
    y_e = se[2,:]
    th_e = se[3,:]

    # --- Initial Pursuer Solution ---
    x_p, y_p, th_p, u_p, Δt, α = pursuer_optimal(x_e, y_e, δt, params)
    TE = Δt * n
    NE = Int(floor(TE / δt))
    if NE > N
        NE = N # Ensure index is within bounds
    elseif NE < 1
        NE = 1
    end

    x_eC = x_e[NE]
    y_eC = y_e[NE]
    th_eC = th_e[NE]
    δt = Δt

    # --- Iterative Solver Loop ---
    while condition 
        N = 200
        n = 200

        # --- EVADER PART ---
        x_e, y_e, th_e, u_e = evader_optimal(x_p, y_p, th_p, Δt, [x_eC, y_eC, th_eC], params, α)
        x_eC_prev, y_eC_prev = x_eC, y_eC
        TE = Δt * n
        
        # Ensure NE is valid for the new trajectory
        NE = Int(floor(TE / δt))
        if NE > N
            NE = N
        elseif NE < 1
            NE = 1
        end

        x_eC = x_e[NE]
        y_eC = y_e[NE]
        th_eC = th_e[NE]
        δt = Δt

        # --- PURSUER PART ---
        x_p, y_p, th_p, u_p, Δt, α = pursuer_optimal(x_e, y_e, δt, params)

        # --- Check Convergence ---
        condition = (value.(x_eC_prev - x_eC))^2 + (value.(y_eC_prev - y_eC))^2 > threshold

        iter = iter + 1
        if iter > 20 # Add a failsafe to prevent infinite loops
            println("Warning: Game did not converge after 20 iterations. Saving last result.")
            condition = false
        end
    end

    # --- Save Data ---
    # The loop has converged. Format and save the 200 data points.
    df = DataFrame(
        xP = x_p,
        yP = y_p,
        thetaP = th_p,
        inputP = u_p,
        xE = x_e,
        yE = y_e,
        thetaE = th_e,
        inputE = u_e
    )

    # Append this DataFrame to the main CSV file (no header)
    CSV.write(csv_filename, df, header = false, append = true)
end


# -------------------------------------------------------------------
# MAIN DRIVER FUNCTION: GENERATES ALL DATA
# -------------------------------------------------------------------

function generate_all_data()
    println("--- Starting Data Generation ---")
    
    # --- Define the grid of evader starting positions ---
    # We need 45 games (45 games * 200 points/game = 9000 points)
    # This 5x3x3 grid creates 45 scenarios.
    
    x_positions = LinRange(-8.0, 8.0, 5)  # 5 different x-values
    y_positions = LinRange(5.0, 15.0, 3)  # 3 different y-values
    thetas = [0.0, π/2, -π/2]             # 3 different angles
    

    csv_filename = "./twoCarsData.csv"
    println("Data will be saved to: $csv_filename")

    # --- Create the CSV file and write the header ONCE ---
    try
        open(csv_filename, "w") do f
            println(f, "xP,yP,thetaP,xE,yE,thetaE,inputP,inputE")
        end
    catch e
        println("Error: Could not write to $csv_filename. Check permissions.")
        return
    end

    game_count = 1
    total_games = length(x_positions) * length(y_positions) * length(thetas)

    # --- Loop through every combination ---
    for x in x_positions
        for y in y_positions
            for th in thetas
                
                println("Running game $game_count / $total_games ... (e0 = [$(round(x, digits=1)), $(round(y, digits=1)), $(round(th, digits=2))])")

                # Your array
                angles = [0, π/4, π/2, 3π/4, π, 5π/4, 3π/2, 7π/4]

                # Randomly select one value
                random_angle = rand(angles)
                
                # 1. Define the evader's state for this game
                state_e0 = [x, y, th]
                
                # 2. Create the params struct for this specific game
                # (Using default values from your code)
                params = Params(
                    2.0, 1.0, [0.0, 0.0, random_angle], 200, # Pursuer params (vm, um, state0, n)
                    1.0, 1.0, state_e0, 200,         # Evader params (vm, um, state0, N)
                    0.05                             # Capture distance (l)
                )

                # 3. Solve the game and save its 200 data points
                try
                    solve_and_save_game(params, csv_filename)
                catch e
                    println("ERROR in game $game_count: $e. Skipping this scenario.")
                end

                game_count += 1
            end
        end
    end

    println("-----------------------------------")
    println("Data generation complete!")
    println("Saved $total_games games (approx. $(total_games * 200) data points) to $csv_filename")
end

# -------------------------------------------------------------------
# RUN THE SCRIPT
# -------------------------------------------------------------------
generate_all_data()