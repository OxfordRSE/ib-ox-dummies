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

"""
    parse_demographics_weights(s) -> Vector{Tuple{String,Float64}}

Parse a demographics weight string into a vector of `(category, weight)` pairs.

Format: `"Category1:weight1,Category2:weight2,..."` where each weight is a
non-negative float. Weights do not need to sum to 1 (they are renormalised
internally by `perturb_weights`).

The last `:` in each pair separates the category name from the weight, so
category names may contain spaces or slashes (e.g. `"White British:0.75"`).

## Examples

```
"M:0.49,F:0.49,I:0.02"                     → [("M", 0.49), ("F", 0.49), ("I", 0.02)]
"White British:0.75,Asian:0.15,Other:0.10"  → [("White British", 0.75), ...]
```
"""
function parse_demographics_weights(s::AbstractString)::Vector{Tuple{String,Float64}}
    isempty(strip(s)) && return Tuple{String,Float64}[]
    result = Tuple{String,Float64}[]
    for piece in split(s, ',')
        piece = strip(piece)
        isempty(piece) && continue
        idx = findlast(':', piece)
        isnothing(idx) && throw(ArgumentError(
            "Cannot parse demographics weight pair: \"$piece\". " *
            "Expected format: \"category:weight\" (e.g. \"White British:0.75\")"))
        cat = strip(piece[1:idx-1])
        wt  = parse(Float64, strip(piece[idx+1:end]))
        push!(result, (cat, wt))
    end
    return result
end

"""
    parse_custom_field_value(v) -> Function

Convert a custom demographics field value (from TOML or a CLI `name=value` pair) into a
zero-argument `Function` suitable for `DemographicsSpec.customFields`.

Conversion rules:

| Value                     | Generator returned                        |
|---------------------------|-------------------------------------------|
| `"faker.city"`            | calls `Faker.city()`                      |
| `"faker.first_name"`      | calls `Faker.first_name()`                |
| `"faker.last_name"`       | calls `Faker.last_name()`                 |
| `"faker.email"`           | calls `Faker.email()`                     |
| `"faker.phone_number"`    | calls `Faker.phone_number()`              |
| `"faker.company"`         | calls `Faker.company()`                   |
| `"faker.address"`         | calls `Faker.address()`                   |
| Any other string `s`      | returns constant `s` on every call        |

Faker method names are case-insensitive (normalised to lowercase with underscores).
"""
function parse_custom_field_value(v::AbstractString)::Function
    s = strip(v)
    if startswith(lowercase(s), "faker.")
        method = lowercase(s[7:end])           # strip "faker." prefix
        if method == "city"
            return () -> Faker.city()
        elseif method == "first_name" || method == "firstname"
            return () -> Faker.first_name()
        elseif method == "last_name" || method == "lastname"
            return () -> Faker.last_name()
        elseif method == "email"
            return () -> Faker.email()
        elseif method == "phone_number" || method == "phone"
            return () -> Faker.phone_number()
        elseif method == "company"
            return () -> Faker.company()
        elseif method == "address"
            return () -> Faker.address()
        elseif method == "country"
            return () -> Faker.country()
        elseif method == "state"
            return () -> Faker.state()
        elseif method == "postcode" || method == "zip_code" || method == "zipcode"
            return () -> Faker.postcode()
        elseif method == "street_address"
            return () -> Faker.street_address()
        elseif method == "name"
            return () -> Faker.name()
        elseif method == "user_name" || method == "username"
            return () -> Faker.user_name()
        else
            throw(ArgumentError(
                "Unknown Faker method: \"$s\". " *
                "Supported: faker.city, faker.first_name, faker.last_name, faker.email, " *
                "faker.phone_number, faker.company, faker.address, faker.country, " *
                "faker.state, faker.postcode, faker.street_address, faker.name, faker.user_name"))
        end
    else
        constant = String(s)
        return () -> constant
    end
end

"""
    parse_custom_fields(d) -> Dict{String,Function}

Parse a `Dict{String,Any}` (e.g. from a TOML `[demographics.customFields]` table) into
a `Dict{String,Function}` suitable for `DemographicsSpec.customFields`.

Each key becomes an output column name (should be prefixed with `d_` by convention);
each string value is converted by [`parse_custom_field_value`](@ref).

## TOML example

```toml
[demographics.customFields]
d_city    = "faker.city"
d_country = "United Kingdom"   # constant — same value on every row
```
"""
function parse_custom_fields(d::AbstractDict)::Dict{String,Function}
    result = Dict{String,Function}()
    for (k, v) in d
        result[string(k)] = parse_custom_field_value(string(v))
    end
    return result
end

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
    parse_questionnaire_spec_from_dict(d) -> QuestionnaireSpec

Parse a `QuestionnaireSpec` from a Dict (e.g. from a TOML `[[questionnaire]]` table).

Required keys:
- `name`: questionnaire name, e.g. `"PHQ_9"`
- `nItems`: number of items

Optional keys (with defaults):
- `prefix`: column prefix (default: `name` with underscores and spaces removed and lowercased)
- `nLevels`: Likert levels per item (default: `4`)
- `noiseSD`: item-level noise standard deviation (default: `0.6`)
- `spoilRate`: fraction of responses that are random/spoiled (default: `0.01`)
- `loadings`: array of loading tables (default: `[]`). Each table must have `latentName`
  and either:
  - `scale`: a single `Float64` applied uniformly to all items, or
  - `itemScales`: a table mapping item index strings (`"1"`, `"2"`, …) to `Float64`
    scale factors for per-item loading.

## TOML example

```toml
[[questionnaire]]
name     = "PHQ_9"
prefix   = "phq9"
nItems   = 9
nLevels  = 4
noiseSD  = 0.6
spoilRate = 0.01
# Uniform loading for all items:
loadings = [{latentName = "depression", scale = 2.5}]

[[questionnaire]]
name   = "custom_q"
nItems = 3
# Per-item loading (items not listed receive scale 0):
loadings = [{latentName = "depression", itemScales = {"1" = 3.0, "2" = 2.0, "3" = 1.0}}]
```
"""
function parse_questionnaire_spec_from_dict(d::AbstractDict)::QuestionnaireSpec
    name      = d["name"]
    prefix    = get(d, "prefix", lowercase(replace(name, r"[_ ]+" => "")))
    nItems    = d["nItems"]
    nLevels   = Int(get(d, "nLevels", 4))
    noiseSD   = Float64(get(d, "noiseSD", 0.6))
    spoilRate = Float64(get(d, "spoilRate", 0.01))
    loadings  = LatentLoading[
        if haskey(l, "itemScales")
            LatentLoading(
                string(l["latentName"]),
                Dict{String,Float64}(string(k) => Float64(v) for (k, v) in l["itemScales"]),
            )
        else
            LatentLoading(string(l["latentName"]), Float64(l["scale"]))
        end
        for l in get(d, "loadings", [])
    ]
    return QuestionnaireSpec(name, prefix, nItems, nLevels, loadings, noiseSD, spoilRate)
end

"""
    parse_linear_effect_from_dict(d) -> LinearEffect

Parse a `LinearEffect` from a Dict (e.g. from a TOML `[[linearEffect]]` table).

Required keys:
- `target`: name of the latent variable
- `value`: float coefficient

Optional keys:
- `inputs`: array of numeric column names (default: `[]`)

## TOML example

```toml
[[linearEffect]]
target = "depression"
inputs = ["d_age", "_sex_fm"]
value  = 0.005
```
"""
function parse_linear_effect_from_dict(d::AbstractDict)::LinearEffect
    target = string(d["target"])
    inputs = String[string(s) for s in get(d, "inputs", String[])]
    value  = Float64(d["value"])
    return LinearEffect(target, inputs, value)
end

"""
    parse_random_effect_from_dict(d) -> RandomEffect

Parse a `RandomEffect` from a Dict (e.g. from a TOML `[[randomEffect]]` table).

Required keys:
- `target`: name of the latent variable
- `value`: a `SamplerSpec` string (e.g. `"norm(0,0.1)"`) or a number

Optional keys:
- `numericalInputs`: array of numeric column names that scale the draw (default: `[]`)
- `categoricalInputs`: array of column names defining groups (default: `[]`)

## TOML example

```toml
[[randomEffect]]
target           = "depression"
numericalInputs  = []
categoricalInputs = ["uid", "wave"]
value            = "norm(0,0.15)"
```
"""
function parse_random_effect_from_dict(d::AbstractDict)::RandomEffect
    target     = string(d["target"])
    num_inputs = String[string(s) for s in get(d, "numericalInputs", String[])]
    cat_inputs = String[string(s) for s in get(d, "categoricalInputs", String[])]
    value_raw  = d["value"]
    value      = value_raw isa Number ? Float64(value_raw) : parse_sampler_spec(string(value_raw))
    return RandomEffect(target, num_inputs, cat_inputs, value)
end

"""
    load_toml_config(path) -> Dict{String,Any}

Load a TOML configuration file and return its contents as a nested `Dict`.

The following top-level sections are recognised by `parse_cli_args`:

- `[simulation]` — scalar simulation parameters:
  `nWaves`, `nSchools`, `nYeargroupsPerSchool`, `nClassesPerSchoolYeargroup`,
  `nStudentsPerClass`, `latentVariables` (array of strings), `seed`
- `[demographics]` — demographic weight strings:
  `ethnicity`, `sex`, `genderIdentity`, `sexualOrientation`
  (each formatted as `"Category1:weight1,Category2:weight2,..."}`)
  and `[demographics.customFields]` — arbitrary extra columns mapped to
  Faker method names (e.g. `"faker.city"`) or constant string values
- `[[linearEffect]]` — array of linear effect tables
  (see `parse_linear_effect_from_dict`)
- `[[randomEffect]]` — array of random effect tables
  (see `parse_random_effect_from_dict`)
- `[[questionnaire]]` — array of questionnaire specification tables
  (see `parse_questionnaire_spec_from_dict`)

CLI arguments override values in the TOML file. See `parse_cli_args` for details.
"""
function load_toml_config(path::AbstractString)::Dict{String,Any}
    isfile(path) || throw(ArgumentError("Config file not found: \"$path\""))
    return TOML.parsefile(path)
end

"""
    parse_cli_args(args) -> SimulationConfig

Parse command-line arguments using `ArgParse.jl` and return a `SimulationConfig`.

All optional arguments default to `nothing`; explicitly provided CLI values override
any values found in a `--config` TOML file, which in turn override built-in defaults.

## Precedence (highest to lowest)

1. Explicit CLI argument
2. `--config` TOML file value
3. Built-in default

## TOML config file

Use `--config path/to/config.toml` to load base settings from a TOML file.  See
`load_toml_config` and `examples/default_model.toml` for the full supported schema.
The TOML file is the primary way to specify questionnaires and latent variable loadings,
which have no equivalent CLI-only representation.

## Format references

- **SPEC** (for count args): integer `'5'`, range `'1:5'`, or distribution
  `'norm(30,7)'`, `'halfnorm(0,0.2)'`, `'poisson(10)'`, `'negbinom(5,0.5)'`,
  `'lognorm(3,0.5)'`, `'uniform(1,10)'`, `'exponential(0.1)'`, `'gamma(2,3)'`
- **LinearEffect**: `"target:inputs:value"` e.g. `'depression:d_age:0.02'`
- **RandomEffect**: `"target:numInputs:catInputs:spec"` e.g. `'depression::uid,wave:norm(0,0.15)'`
- **Demographics**: `"Category1:weight1,Category2:weight2,..."` e.g. `'M:0.49,F:0.49,I:0.02'`
"""
function parse_cli_args(args::Vector{String})::SimulationConfig
    s = ArgParseSettings(
        description = "Generate mock longitudinal questionnaire data for schoolchildren.\n\n" *
                      "Use --config to load a TOML file specifying the full model " *
                      "(questionnaires, latent variable loadings, effects, demographics).\n" *
                      "CLI arguments override TOML values.\n\n" *
                      "SPEC formats: integer (e.g. '5'), inclusive range (e.g. '1:5'), " *
                      "or a distribution (e.g. 'norm(30,7)', 'halfnorm(0,0.2)', 'poisson(10)', " *
                      "'negbinom(5,0.5)', 'lognorm(3,0.5)', 'uniform(1,10)', 'exponential(0.1)', " *
                      "'gamma(2,3)').\n\n" *
                      "LinearEffect format: \"target:inputs:value\" " *
                      "(e.g. 'depression:d_age:0.02' or 'anxiety:d_age,_sex_fm:0.004').\n\n" *
                      "RandomEffect format: \"target:numInputs:catInputs:spec\" " *
                      "(e.g. 'depression::uid,wave:norm(0,0.15)' or 'anxiety:::norm(0,0.1)').\n\n" *
                      "Demographics weight format: \"Category1:weight1,Category2:weight2,...\" " *
                      "(e.g. 'M:0.49,F:0.49,I:0.02').",
        prog        = "ib_ox_dummies",
        add_help    = true,
        version     = "0.1.0",
        add_version = true,
    )

    @add_arg_table! s begin
        "--config"
            help     = "Path to a TOML configuration file. CLI arguments override TOML values. " *
                       "See examples/default_model.toml for the full supported schema."
            default  = nothing
        "--nWaves"
            help     = "Number of data-collection waves"
            arg_type = Int
            default  = nothing
        "--nSchools"
            help     = "Number of schools"
            arg_type = Int
            default  = nothing
        "--nYeargroupsPerSchool"
            help     = "Yeargroups per school (SPEC)"
            default  = nothing
        "--nClassesPerSchoolYeargroup"
            help     = "Classes per school yeargroup (SPEC)"
            default  = nothing
        "--nStudentsPerClass"
            help     = "Students per class (SPEC)"
            default  = nothing
        "--latentVariables"
            help     = "Comma-separated latent variable names (e.g. 'depression,anxiety'). " *
                       "Overrides the TOML [simulation] latentVariables field."
            default  = nothing
        "--linearEffect"
            help     = "LinearEffect spec: 'target:inputs:value' (repeatable). " *
                       "If any --linearEffect is given, all TOML linearEffects are replaced."
            action   = :append_arg
            default  = nothing
        "--randomEffect"
            help     = "RandomEffect spec: 'target:numInputs:catInputs:spec' (repeatable). " *
                       "If any --randomEffect is given, all TOML randomEffects are replaced."
            action   = :append_arg
            default  = nothing
        "--ethnicity"
            help     = "Ethnicity weight distribution: 'Category1:weight1,Category2:weight2,...'. " *
                       "Overrides the TOML [demographics] ethnicity field."
            default  = nothing
        "--sex"
            help     = "Sex weight distribution: 'M:weight,F:weight,I:weight'. " *
                       "Overrides the TOML [demographics] sex field."
            default  = nothing
        "--genderIdentity"
            help     = "Gender identity weight distribution: 'Category1:weight1,...'. " *
                       "Overrides the TOML [demographics] genderIdentity field."
            default  = nothing
        "--sexualOrientation"
            help     = "Sexual orientation weight distribution: 'Category1:weight1,...'. " *
                       "Overrides the TOML [demographics] sexualOrientation field."
            default  = nothing
        "--customField"
            help     = "Custom demographic field: 'columnName=value' (repeatable). " *
                       "Value is a Faker method name (e.g. 'faker.city') or a constant string. " *
                       "Overrides matching TOML [demographics.customFields] entries. " *
                       "Column names conventionally start with 'd_'."
            action   = :append_arg
            default  = nothing
        "--seed"
            help     = "Random seed for reproducibility. Overrides the TOML [simulation] seed field."
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

    # Load TOML config file if --config was provided
    toml     = isnothing(parsed["config"]) ? Dict{String,Any}() : load_toml_config(parsed["config"])
    sim_toml = get(toml, "simulation", Dict{String,Any}())
    dem_toml = get(toml, "demographics", Dict{String,Any}())

    # --- Scalar simulation parameters: CLI > TOML > built-in default ---
    nWaves   = something(parsed["nWaves"],   get(sim_toml, "nWaves",   nothing), 3)
    nSchools = something(parsed["nSchools"], get(sim_toml, "nSchools", nothing), 10)
    seed     = !isnothing(parsed["seed"]) ? parsed["seed"] : get(sim_toml, "seed", nothing)

    nyps_str  = something(parsed["nYeargroupsPerSchool"],
                          _toml_str(get(sim_toml, "nYeargroupsPerSchool", nothing)),
                          "5")
    ncpsy_str = something(parsed["nClassesPerSchoolYeargroup"],
                          _toml_str(get(sim_toml, "nClassesPerSchoolYeargroup", nothing)),
                          "1:5")
    nspc_str  = something(parsed["nStudentsPerClass"],
                          _toml_str(get(sim_toml, "nStudentsPerClass", nothing)),
                          "norm(30,7)")

    # --- Latent variables: CLI > TOML > empty (simulate() fills in defaults) ---
    latent_vars = if !isnothing(parsed["latentVariables"])
        filter(!isempty, strip.(split(parsed["latentVariables"], ',')))
    elseif haskey(sim_toml, "latentVariables")
        String[string(v) for v in sim_toml["latentVariables"]]
    else
        String[]
    end

    # --- Linear effects: CLI (non-empty) > TOML > empty (simulate() fills in defaults) ---
    linear_effs = if !isempty(parsed["linearEffect"])
        LinearEffect[parse_linear_effect(string(e)) for e in parsed["linearEffect"]]
    elseif haskey(toml, "linearEffect")
        LinearEffect[parse_linear_effect_from_dict(d) for d in toml["linearEffect"]]
    else
        LinearEffect[]
    end

    # --- Random effects: CLI (non-empty) > TOML > empty (simulate() fills in defaults) ---
    random_effs = if !isempty(parsed["randomEffect"])
        RandomEffect[parse_random_effect(string(e)) for e in parsed["randomEffect"]]
    elseif haskey(toml, "randomEffect")
        RandomEffect[parse_random_effect_from_dict(d) for d in toml["randomEffect"]]
    else
        RandomEffect[]
    end

    # --- Questionnaires: TOML only (complex spec; no CLI equivalent) ---
    questionnaires = if haskey(toml, "questionnaire")
        QuestionnaireSpec[parse_questionnaire_spec_from_dict(d) for d in toml["questionnaire"]]
    else
        QuestionnaireSpec[]
    end

    # --- Demographics: CLI > TOML > empty (simulate() fills in defaults) ---
    eth_wts  = _resolve_demo_weights(parsed["ethnicity"],        get(dem_toml, "ethnicity",        nothing))
    sex_wts  = _resolve_demo_weights(parsed["sex"],              get(dem_toml, "sex",              nothing))
    gend_wts = _resolve_demo_weights(parsed["genderIdentity"],   get(dem_toml, "genderIdentity",   nothing))
    ori_wts  = _resolve_demo_weights(parsed["sexualOrientation"], get(dem_toml, "sexualOrientation", nothing))

    # Custom fields: start from TOML, then overlay CLI overrides
    toml_custom_fields = parse_custom_fields(get(dem_toml, "customFields", Dict{String,Any}()))
    cli_custom_fields  = _parse_cli_custom_fields(parsed["customField"])
    custom_fields      = merge(toml_custom_fields, cli_custom_fields)  # CLI wins on conflict

    demo_spec = if any(!isempty, [eth_wts, sex_wts, gend_wts, ori_wts]) || !isempty(custom_fields)
        DemographicsSpec(
            ethnicity         = eth_wts,
            sex               = sex_wts,
            genderIdentity    = gend_wts,
            sexualOrientation = ori_wts,
            customFields      = custom_fields,
        )
    else
        nothing
    end

    output = parsed["schema"] ? "schema" : parsed["output"]

    return SimulationConfig(
        nWaves                     = nWaves,
        nSchools                   = nSchools,
        nYeargroupsPerSchool       = parse_sampler_spec(nyps_str),
        nClassesPerSchoolYeargroup = parse_sampler_spec(ncpsy_str),
        nStudentsPerClass          = parse_sampler_spec(nspc_str),
        latentVariables            = latent_vars,
        linearEffects              = linear_effs,
        randomEffects              = random_effs,
        questionnaires             = questionnaires,
        demographicsSpec           = demo_spec,
        demographicsUpdateFn       = default_demographics_update,
        naughtyMonkey              = default_naughty_monkey,
        output                     = output,
        seed                       = seed,
    )
end

# Internal: convert a TOML numeric or string SamplerSpec value to a String for parsing.
_toml_str(::Nothing)         = nothing
_toml_str(v::Integer)        = string(v)
_toml_str(v::AbstractString) = v

# Internal: resolve a demographics weight vector from CLI string > TOML string > empty.
function _resolve_demo_weights(
    cli_str::Union{AbstractString,Nothing},
    toml_str::Union{AbstractString,Nothing},
)::Vector{Tuple{String,Float64}}
    if !isnothing(cli_str)
        return parse_demographics_weights(cli_str)
    elseif !isnothing(toml_str)
        return parse_demographics_weights(toml_str)
    else
        return Tuple{String,Float64}[]
    end
end

# Internal: parse --customField "name=value" CLI args into a Dict{String,Function}.
function _parse_cli_custom_fields(
    cli_args::Union{Vector,Nothing},
)::Dict{String,Function}
    isnothing(cli_args) && return Dict{String,Function}()
    result = Dict{String,Function}()
    for item in cli_args
        s = strip(string(item))
        idx = findfirst('=', s)
        isnothing(idx) && throw(ArgumentError(
            "Cannot parse --customField: \"$s\". Expected format: \"columnName=value\" " *
            "(e.g. \"d_city=faker.city\" or \"d_country=United Kingdom\")"))
        name  = strip(s[1:idx-1])
        value = strip(s[idx+1:end])
        result[name] = parse_custom_field_value(value)
    end
    return result
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
