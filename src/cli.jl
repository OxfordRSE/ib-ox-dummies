"""
    parse_count_spec(s) -> CountSpec

Parse a command-line string into a `CountSpec`.

Supported formats:
- `"5"` → `Int(5)`
- `"1:5"` or `"1,5"` → `Range(1, 5)`
- `"norm(30,7)"` or `"dnorm(30,7)"` → `Normal(30.0, 7.0)` (from Distributions.jl)
"""
function parse_count_spec(s::AbstractString)::CountSpec
    s = strip(s)

    # Normal distribution: norm(μ,σ) or dnorm(μ,σ)
    m = match(r"^d?norm\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)$"i, s)
    if !isnothing(m)
        return Normal(parse(Float64, m[1]), parse(Float64, m[2]))
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
    parse_cli_args(args) -> SimulationConfig

Parse command-line arguments using `ArgParse.jl` and return a `SimulationConfig`.
"""
function parse_cli_args(args::Vector{String})::SimulationConfig
    s = ArgParseSettings(
        description = "Generate mock longitudinal questionnaire data for schoolchildren.\n\n" *
                      "SPEC formats: integer (e.g. '5'), inclusive range (e.g. '1:5'), " *
                      "or normal distribution (e.g. 'norm(30,7)').",
        prog        = "ib_ox_dummies",
        add_help    = true,
        version     = "0.1.0",
        add_version = true,
    )

    @add_arg_table! s begin
        "--nWaves"
            help     = "Number of data-collection waves"
            arg_type = Int
            default  = 3
        "--nSchools"
            help     = "Number of schools"
            arg_type = Int
            default  = 10
        "--nYeargroupsPerSchool"
            help     = "Yeargroups per school (SPEC)"
            default  = "5"
        "--nClassesPerSchoolYeargroup"
            help     = "Classes per school yeargroup (SPEC)"
            default  = "1:5"
        "--nStudentsPerClass"
            help     = "Students per class (SPEC)"
            default  = "norm(30,7)"
        "--seed"
            help     = "Random seed for reproducibility"
            arg_type = Int
            default  = nothing
        "--output"
            help     = "Output format: csv | json | schema"
            default  = "csv"
        "--schema"
            help     = "Print JSON Schema describing the output columns and exit"
            action   = :store_true
    end

    parsed = parse_args(args, s)
    output = parsed["schema"] ? "schema" : parsed["output"]

    return SimulationConfig(
        nWaves                     = parsed["nWaves"],
        nSchools                   = parsed["nSchools"],
        nYeargroupsPerSchool       = parse_count_spec(parsed["nYeargroupsPerSchool"]),
        nClassesPerSchoolYeargroup = parse_count_spec(parsed["nClassesPerSchoolYeargroup"]),
        nStudentsPerClass          = parse_count_spec(parsed["nStudentsPerClass"]),
        questionnaires             = Dict{String,Questionnaire}(),
        demographicsUpdateFn       = default_demographics_update,
        naughtyMonkey              = default_naughty_monkey,
        output                     = output,
        seed                       = parsed["seed"],
    )
end

"""
    ib_ox_dummies_cli(args=ARGS)

Main entry point for the `ib_ox_dummies` command-line tool.
Parses `args`, runs the simulation, and writes output to stdout.
"""
function ib_ox_dummies_cli(args::Vector{String} = ARGS)
    config = parse_cli_args(args)
    data, schema = simulate(config)
    write_output(data, schema, config)
end
