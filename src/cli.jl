"""
    parse_count_spec(s) -> CountSpec

Parse a command-line string into a `CountSpec`.

Supported formats:
- `"5"` → `Int(5)`
- `"1:5"` or `"1,5"` → `Range(1, 5)`
- `"norm(30,7)"` or `"normal(30,7)"` → `Normal(30.0, 7.0)`
- `"pois(5)"` or `"poisson(5)"` → `Poisson(5.0)`
- `"negbinom(5,0.5)"` → `NegativeBinomial(5, 0.5)`
- `"lognorm(3,0.5)"` or `"lognormal(3,0.5)"` → `LogNormal(3.0, 0.5)`
- `"unif(1,10)"` or `"uniform(1,10)"` → `DiscreteUniform(1, 10)`
- `"exp(0.1)"` or `"exponential(0.1)"` → `Exponential(10.0)` (mean=1/rate)
- `"gamma(2,3)"` → `Gamma(2.0, 3.0)` (shape, scale)
- `"beta(2,5)"` → `Beta(2.0, 5.0)` (applied to interval [0,1], scaled externally)
"""
function parse_count_spec(s::AbstractString)::CountSpec
    s = strip(s)

    # Normal distribution: norm(μ,σ) or normal(μ,σ) or dnorm(μ,σ)
    m = match(r"^d?norm(?:al)?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)$"i, s)
    if !isnothing(m)
        return Normal(parse(Float64, m[1]), parse(Float64, m[2]))
    end

    # Poisson distribution: pois(λ) or poisson(λ)
    m = match(r"^pois(?:son)?\(\s*([\d.]+)\s*\)$"i, s)
    if !isnothing(m)
        return Poisson(parse(Float64, m[1]))
    end

    # Negative Binomial: negbinom(r,p) or negativebinomial(r,p)
    m = match(r"^neg(?:ative)?binom(?:ial)?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)$"i, s)
    if !isnothing(m)
        return NegativeBinomial(parse(Float64, m[1]), parse(Float64, m[2]))
    end

    # Log-Normal distribution: lognorm(μ,σ) or lognormal(μ,σ)
    m = match(r"^log(?:norm(?:al)?)?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)$"i, s)
    if !isnothing(m)
        return LogNormal(parse(Float64, m[1]), parse(Float64, m[2]))
    end

    # Discrete Uniform: unif(a,b) or uniform(a,b)
    m = match(r"^unif(?:orm)?\(\s*(\d+)\s*,\s*(\d+)\s*\)$"i, s)
    if !isnothing(m)
        return DiscreteUniform(parse(Int, m[1]), parse(Int, m[2]))
    end

    # Exponential: exp(rate) or exponential(rate)  [parameterised by rate, mean=1/rate]
    m = match(r"^exp(?:onential)?\(\s*([\d.]+)\s*\)$"i, s)
    if !isnothing(m)
        return Exponential(1.0 / parse(Float64, m[1]))
    end

    # Gamma: gamma(shape,scale)
    m = match(r"^gamma\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)$"i, s)
    if !isnothing(m)
        return Gamma(parse(Float64, m[1]), parse(Float64, m[2]))
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
        "Supported: integer ('5'), range ('1:5'), " *
        "norm(μ,σ), poisson(λ), negbinom(r,p), lognorm(μ,σ), " *
        "uniform(a,b), exponential(rate), gamma(shape,scale)."
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
                      "or a distribution (e.g. 'norm(30,7)', 'poisson(10)', 'negbinom(5,0.5)', " *
                      "'lognorm(3,0.5)', 'uniform(1,10)', 'exponential(0.1)', 'gamma(2,3)').",
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
