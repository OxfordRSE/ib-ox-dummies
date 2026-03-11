"""
    parse_sampler_spec(s) -> SamplerSpec

Parse a command-line string into a `SamplerSpec`.

Supported formats:
- `"5"` → `Int(5)`
- `"1:5"` or `"1,5"` → `Range(1, 5)`
- `"norm(30,7)"` or `"normal(30,7)"` → `Normal(30.0, 7.0)`
- `"halfnorm(0,0.2)"` or `"hnorm(0,0.2)"` → `truncated(Normal(0.0, 0.2), 0.0, Inf)`
- `"pois(5)"` or `"poisson(5)"` → `Poisson(5.0)`
- `"negbinom(5,0.5)"` → `NegativeBinomial(5, 0.5)`
- `"lognorm(3,0.5)"` or `"lognormal(3,0.5)"` → `LogNormal(3.0, 0.5)`
- `"unif(1,10)"` or `"uniform(1,10)"` → `DiscreteUniform(1, 10)`
- `"exp(0.1)"` or `"exponential(0.1)"` → `Exponential(10.0)` (mean=1/rate)
- `"gamma(2,3)"` → `Gamma(2.0, 3.0)` (shape, scale)

For custom callables, use the Julia API directly:
`rng -> rand(rng, MyDist(...))` — any `(rng::AbstractRNG) -> Number` function.
"""
function parse_sampler_spec(s::AbstractString)::SamplerSpec
    s = strip(s)

    # Float: optional sign, digits (with optional decimal), optional scientific notation
    float_pat = raw"[-+]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][+-]?\d+)?"

    # Normal distribution: norm(μ,σ) or normal(μ,σ) or dnorm(μ,σ)
    m = match(Regex("^d?norm(?:al)?\\(\\s*($float_pat)\\s*,\\s*($float_pat)\\s*\\)\$", "i"), s)
    if !isnothing(m)
        return Normal(parse(Float64, m[1]), parse(Float64, m[2]))
    end

    # Half-normal: halfnorm(μ,σ) or hnorm(μ,σ) → truncated(Normal(μ,σ), 0, Inf)
    m = match(Regex("^h(?:alf)?norm\\(\\s*($float_pat)\\s*,\\s*($float_pat)\\s*\\)\$", "i"), s)
    if !isnothing(m)
        return truncated(Normal(parse(Float64, m[1]), parse(Float64, m[2])), 0.0, Inf)
    end

    # Poisson distribution: pois(λ) or poisson(λ)
    m = match(Regex("^pois(?:son)?\\(\\s*($float_pat)\\s*\\)\$", "i"), s)
    if !isnothing(m)
        return Poisson(parse(Float64, m[1]))
    end

    # Negative Binomial: negbinom(r,p) or negativebinomial(r,p)
    m = match(Regex("^neg(?:ative)?binom(?:ial)?\\(\\s*($float_pat)\\s*,\\s*($float_pat)\\s*\\)\$", "i"), s)
    if !isnothing(m)
        return NegativeBinomial(parse(Float64, m[1]), parse(Float64, m[2]))
    end

    # Log-Normal distribution: lognorm(μ,σ) or lognormal(μ,σ)
    m = match(Regex("^log(?:norm(?:al)?)?\\(\\s*($float_pat)\\s*,\\s*($float_pat)\\s*\\)\$", "i"), s)
    if !isnothing(m)
        return LogNormal(parse(Float64, m[1]), parse(Float64, m[2]))
    end

    # Discrete Uniform: unif(a,b) or uniform(a,b)
    m = match(r"^unif(?:orm)?\(\s*(\d+)\s*,\s*(\d+)\s*\)$"i, s)
    if !isnothing(m)
        return DiscreteUniform(parse(Int, m[1]), parse(Int, m[2]))
    end

    # Exponential: exp(rate) or exponential(rate)  [parameterised by rate, mean=1/rate]
    m = match(Regex("^exp(?:onential)?\\(\\s*($float_pat)\\s*\\)\$", "i"), s)
    if !isnothing(m)
        return Exponential(1.0 / parse(Float64, m[1]))
    end

    # Gamma: gamma(shape,scale)
    m = match(Regex("^gamma\\(\\s*($float_pat)\\s*,\\s*($float_pat)\\s*\\)\$", "i"), s)
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
        "Cannot parse sampler spec: \"$s\". " *
        "Supported: integer ('5'), range ('1:5'), " *
        "norm(μ,σ), halfnorm(μ,σ), poisson(λ), negbinom(r,p), lognorm(μ,σ), " *
        "uniform(a,b), exponential(rate), gamma(shape,scale)."
    ))
end

# Backward-compatible alias
const parse_count_spec = parse_sampler_spec

"""
    parse_linear_effect(s) -> LinearEffect

Parse a string of the form `"target:inputs:value"` into a `LinearEffect`.

- `target`: name of the latent variable (e.g. `"depression"`)
- `inputs`: comma-separated list of input column names (can be empty for an intercept)
- `value`: a floating-point coefficient

## Examples

```
"depression:d_age:0.02"          → LinearEffect("depression", ["d_age"], 0.02)
"anxiety:d_age,_sex_fm:0.004"    → LinearEffect("anxiety", ["d_age", "_sex_fm"], 0.004)
"depression::0.1"                → LinearEffect("depression", [], 0.1)  # intercept
```
"""
function parse_linear_effect(s::AbstractString)::LinearEffect
    parts = split(s, ':', limit = 3)
    length(parts) == 3 || throw(ArgumentError(
        "Cannot parse LinearEffect: \"$s\". Expected format: \"target:inputs:value\""))
    target = strip(parts[1])
    inputs = filter(!isempty, strip.(split(parts[2], ',')))
    value  = parse(Float64, strip(parts[3]))
    return LinearEffect(target, inputs, value)
end

"""
    parse_random_effect(s) -> RandomEffect

Parse a string of the form `"target:numInputs:catInputs:spec"` into a `RandomEffect`.

- `target`: name of the latent variable (e.g. `"depression"`)
- `numInputs`: comma-separated list of numerical input column names (can be empty)
- `catInputs`: comma-separated list of categorical group column names (can be empty → residual)
- `spec`: a `SamplerSpec` string (parsed by `parse_sampler_spec`)

## Examples

```
"depression::uid:norm(0,0.2)"             → RandomEffect("depression", [], ["uid"], Normal(0,0.2))
"depression::uid,wave:norm(0,0.15)"       → per-individual-wave draw
"depression:::norm(0,0.1)"               → residual error (fresh draw each row)
"depression::uid:halfnorm(0,0.2)"        → half-normal baseline
```
"""
function parse_random_effect(s::AbstractString)::RandomEffect
    parts = split(s, ':', limit = 4)
    length(parts) == 4 || throw(ArgumentError(
        "Cannot parse RandomEffect: \"$s\". Expected format: \"target:numInputs:catInputs:spec\""))
    target     = strip(parts[1])
    num_inputs = filter(!isempty, strip.(split(parts[2], ',')))
    cat_inputs = filter(!isempty, strip.(split(parts[3], ',')))
    spec       = parse_sampler_spec(strip(parts[4]))
    return RandomEffect(target, num_inputs, cat_inputs, spec)
end

"""
    parse_cli_args(args) -> SimulationConfig

Parse command-line arguments using `ArgParse.jl` and return a `SimulationConfig`.
"""
function parse_cli_args(args::Vector{String})::SimulationConfig
    s = ArgParseSettings(
        description = "Generate mock longitudinal questionnaire data for schoolchildren.\n\n" *
                      "SPEC formats: integer (e.g. '5'), inclusive range (e.g. '1:5'), " *
                      "or a distribution (e.g. 'norm(30,7)', 'halfnorm(0,0.2)', 'poisson(10)', " *
                      "'negbinom(5,0.5)', 'lognorm(3,0.5)', 'uniform(1,10)', 'exponential(0.1)', " *
                      "'gamma(2,3)').\n\n" *
                      "LinearEffect format: \"target:inputs:value\" " *
                      "(e.g. 'depression:d_age:0.02' or 'anxiety:d_age,_sex_fm:0.004').\n\n" *
                      "RandomEffect format: \"target:numInputs:catInputs:spec\" " *
                      "(e.g. 'depression::uid,wave:norm(0,0.15)' or 'anxiety:::norm(0,0.1)').",
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
        "--latentVariables"
            help     = "Comma-separated latent variable names (e.g. 'depression,anxiety'). " *
                       "Empty string uses the default latent variables."
            default  = ""
        "--linearEffect"
            help     = "LinearEffect spec: 'target:inputs:value' (repeatable). " *
                       "Providing any --linearEffect disables the default linear effects."
            action   = :append_arg
            default  = String[]
        "--randomEffect"
            help     = "RandomEffect spec: 'target:numInputs:catInputs:spec' (repeatable). " *
                       "Providing any --randomEffect disables the default random effects."
            action   = :append_arg
            default  = String[]
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

    latent_str   = strip(parsed["latentVariables"])
    latent_vars  = isempty(latent_str) ? String[] : filter(!isempty, strip.(split(latent_str, ',')))
    le_raw       = something(parsed["linearEffect"], String[])
    re_raw       = something(parsed["randomEffect"], String[])
    linear_effs  = LinearEffect[parse_linear_effect(string(e)) for e in le_raw]
    random_effs  = RandomEffect[parse_random_effect(string(e)) for e in re_raw]

    return SimulationConfig(
        nWaves                     = parsed["nWaves"],
        nSchools                   = parsed["nSchools"],
        nYeargroupsPerSchool       = parse_sampler_spec(parsed["nYeargroupsPerSchool"]),
        nClassesPerSchoolYeargroup = parse_sampler_spec(parsed["nClassesPerSchoolYeargroup"]),
        nStudentsPerClass          = parse_sampler_spec(parsed["nStudentsPerClass"]),
        latentVariables            = latent_vars,
        linearEffects              = linear_effs,
        randomEffects              = random_effs,
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
