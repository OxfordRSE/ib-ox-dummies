"""
    parse_count_spec(s) -> CountSpec

Parse a command-line string into a `CountSpec`.

Supported formats:
- `"5"` → `Int(5)`
- `"1:5"` or `"1,5"` → `Range(1, 5)`
- `"norm(30,7)"` or `"dnorm(30,7)"` → `NormalDist(30.0, 7.0)`
"""
function parse_count_spec(s::AbstractString)::CountSpec
    s = strip(s)

    # NormalDist: norm(μ,σ) or dnorm(μ,σ)
    m = match(r"^d?norm\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)$"i, s)
    if !isnothing(m)
        return NormalDist(parse(Float64, m[1]), parse(Float64, m[2]))
    end

    # Range: "min:max" or "min,max"
    m = match(r"^(\d+)[,:]\s*(\d+)$", s)
    if !isnothing(m)
        return Range(parse(Int, m[1]), parse(Int, m[2]))
    end

    # Plain integer
    m = match(r"^\d+$", s)
    if !isnothing(m)
        return parse(Int, s)
    end

    throw(ArgumentError(
        "Cannot parse count spec: \"$s\". " *
        "Use an integer (e.g. '5'), a range (e.g. '1:5'), " *
        "or a normal distribution (e.g. 'norm(30,7)')."
    ))
end

"""
    print_usage(io)

Print usage information for the `ib_ox_dummies` CLI.
"""
function print_usage(io::IO = stdout)
    println(io, """
ib_ox_dummies — Generate mock longitudinal questionnaire data.

USAGE
  ib_ox_dummies [OPTIONS]

OPTIONS
  --nWaves INT                     Number of data-collection waves (default: 3)
  --nSchools INT                   Number of schools (default: 10)
  --nYeargroupsPerSchool SPEC      Yeargroups per school (default: 5)
  --nClassesPerSchoolYeargroup SPEC Classes per school yeargroup (default: 1,5)
  --nStudentsPerClass SPEC         Students per class (default: norm(30,7))
  --seed INT                       Random seed for reproducibility
  --output FORMAT                  Output format: csv | json | schema (default: csv)
  --schema                         Print JSON Schema and exit
  --help, -h                       Show this help and exit

SPEC formats:
  5             Fixed integer
  1:5           Inclusive range [min, max] (uniform sample)
  norm(30,7)    Normal distribution N(μ=30, σ=7) (rounded to nearest integer)

EXAMPLES
  ib_ox_dummies
  ib_ox_dummies --nWaves 2 --nSchools 3 --seed 42
  ib_ox_dummies --nStudentsPerClass norm(25,5) --output json
  ib_ox_dummies --output schema
""")
end

"""
    parse_cli_args(args) -> SimulationConfig

Parse command-line arguments and return a `SimulationConfig`.
"""
function parse_cli_args(args::Vector{String})::SimulationConfig
    # Defaults
    nWaves                      = 3
    nSchools                    = 10
    nYeargroupsPerSchool        = 5          :: CountSpec
    nClassesPerSchoolYeargroup  = Range(1,5) :: CountSpec
    nStudentsPerClass           = NormalDist(30.0, 7.0) :: CountSpec
    seed::Union{Int,Nothing}    = nothing
    output::Union{String,OutputFn} = "csv"
    print_schema_and_exit       = false

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("--help", "-h")
            print_usage()
            exit(0)
        elseif arg == "--schema"
            print_schema_and_exit = true
        elseif arg == "--nWaves"
            i += 1
            i > length(args) && error("--nWaves requires an argument")
            nWaves = parse(Int, args[i])
        elseif arg == "--nSchools"
            i += 1
            i > length(args) && error("--nSchools requires an argument")
            nSchools = parse(Int, args[i])
        elseif arg == "--nYeargroupsPerSchool"
            i += 1
            i > length(args) && error("--nYeargroupsPerSchool requires an argument")
            nYeargroupsPerSchool = parse_count_spec(args[i])
        elseif arg == "--nClassesPerSchoolYeargroup"
            i += 1
            i > length(args) && error("--nClassesPerSchoolYeargroup requires an argument")
            nClassesPerSchoolYeargroup = parse_count_spec(args[i])
        elseif arg == "--nStudentsPerClass"
            i += 1
            i > length(args) && error("--nStudentsPerClass requires an argument")
            nStudentsPerClass = parse_count_spec(args[i])
        elseif arg == "--seed"
            i += 1
            i > length(args) && error("--seed requires an argument")
            seed = parse(Int, args[i])
        elseif arg == "--output"
            i += 1
            i > length(args) && error("--output requires an argument")
            output = args[i]
        else
            error("Unknown argument: $arg\nRun with --help for usage information.")
        end
        i += 1
    end

    if print_schema_and_exit
        output = "schema"
    end

    return SimulationConfig(
        nWaves                     = nWaves,
        nSchools                   = nSchools,
        nYeargroupsPerSchool       = nYeargroupsPerSchool,
        nClassesPerSchoolYeargroup = nClassesPerSchoolYeargroup,
        nStudentsPerClass          = nStudentsPerClass,
        questionnaires             = Dict{String,Questionnaire}(),
        demographicsUpdateFn       = default_demographics_update,
        naughtyMonkey              = default_naughty_monkey,
        output                     = output,
        seed                       = seed,
    )
end

"""
    ib_ox_dummies_cli(args=ARGS)

Main entry point for the `ib_ox_dummies` command-line tool.
Parses `args`, runs the simulation, and writes output.
"""
function ib_ox_dummies_cli(args::Vector{String} = ARGS)
    config = try
        parse_cli_args(args)
    catch e
        println(stderr, "Error: ", e.msg)
        println(stderr, "Run with --help for usage information.")
        exit(1)
    end

    data, schema = simulate(config)
    write_output(data, schema, config)
end
