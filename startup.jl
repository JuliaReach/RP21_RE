# instantiate project
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

const TARGET_FOLDER = "result"
const RESULTS_FILE = "results.csv"

function main()
    if !isdir(TARGET_FOLDER)
        mkdir(TARGET_FOLDER)
    end
    global io = open(joinpath(TARGET_FOLDER, RESULTS_FILE), "w")

    println("Running evaluation...")

    println("###\nRunning Logistic model\n###")
    include("models/Logistic/Logistic.jl")

    println("###\nRunning SEIR model\n###")
    include("models/SEIR/SEIR.jl")

    println("###\nRunning Burguers model\n###")
    include("models/Burguers/Burguers.jl")

    print(io, "\n")
    println("Finished running benchmarks, results stored in $TARGET_FOLDER")
    close(io)
    nothing
end

main()
