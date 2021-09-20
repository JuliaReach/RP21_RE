# instantiate project
#import Pkg
#Pkg.activate(@__DIR__)
#Pkg.instantiate()

const TARGET_FOLDER = "result"
const RESULTS_FILE = "results.csv"

function main()
    if !isdir(TARGET_FOLDER)
        mkdir(TARGET_FOLDER)
    end
    global io = open(joinpath(TARGET_FOLDER, RESULTS_FILE), "w")

    println("Running evaluation...\n")

    println(">>> Running Logistic model\n")
    include("evaluation/Logistic/Logistic.jl")

    println(">>> Running SEIR model\n")
    include("evaluation/SEIR/SEIR.jl")

    println(">>> Running Burgers model\n")
    include("evaluation/Burgers/Burgers.jl")

    println("Finished running evaluation. Results stored in $TARGET_FOLDER")
    close(io)
    nothing
end

main()
