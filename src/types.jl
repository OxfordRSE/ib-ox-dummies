"""
    Response

Answer data: the value of a single field for a student in a given wave.
Can be an integer, float, string, or missing (unanswered/redacted).
"""
const Response = Union{Int,Float64,String,Missing}

"""
    DataRow

A named record of answer data representing one row (one student × one wave).
Keys are column names; values are `Response` entries.
"""
const DataRow = Dict{String,Response}

"""
    Schema

Metadata describing the columns of the output dataset.

- `demographicsColumns`: column names that are demographic fields.
- `questionnaireColumns`: mapping from column name to the name of the
  questionnaire it belongs to.
- `latentColumns`: column names for latent variable values (non-empty only
  when `SimulationConfig.includeLatents = true`).
"""
struct Schema
    demographicsColumns::Vector{String}
    questionnaireColumns::Dict{String,String}
    latentColumns::Vector{String}
    # 3-arg constructor (all fields explicit)
    Schema(d::Vector{String}, q::Dict{String,String}, l::Vector{String}) = new(d, q, l)
    # 2-arg convenience constructor (no latent columns)
    Schema(d::Vector{String}, q::Dict{String,String}) = new(d, q, String[])
end

"""
    Range

Inclusive integer range [min, max] used as a count specification.
"""
struct Range
    min::Int
    max::Int
    function Range(min::Int, max::Int)
        min <= max || throw(ArgumentError("Range min ($min) must be ≤ max ($max)"))
        new(min, max)
    end
end

"""
    SamplerSpec

A flexible sampler specification accepted anywhere a random value is needed.
Can be:
- A fixed `Int` (always returns that value)
- A `Range` (samples uniformly from `[min, max]`)
- A `UnivariateDistribution` from `Distributions.jl` (e.g. `Normal(30, 7)`)
- A callable `(rng::AbstractRNG) -> Number` (custom sampler)

Used for:
- Count fields in `SimulationConfig` (values are rounded to the nearest integer and clamped ≥ 1 via `sample_count`)
- `RandomEffect.value` (drawn as `Float64` via `draw_sampler`)

## Examples

```julia
5                              # fixed value
Range(1, 5)                    # uniform sample from [1, 5]
Normal(30.0, 7.0)              # sample from Normal
truncated(Normal(0.0, 0.2), 0, Inf)          # half-normal
rng -> rand(rng) < 0.01 ? rand(rng, Normal(0.75, 0.1)) : 0.0  # custom: MDE
```
"""
const SamplerSpec = Union{Int,Range,UnivariateDistribution,Function}


"""
    LinearEffect

A fixed linear effect on a latent variable.

Adds `value × ∏(inputs)` to the named `target` latent variable for each row.
All `inputs` must name numeric columns present in the row (e.g. `"d_age"`,
`"_sex_fm"`). If any input is non-numeric the coefficient contributes 0.
"""
struct LinearEffect
    target::String          # latent variable name
    inputs::Vector{String}  # numeric column names to multiply together
    value::Float64          # scaling factor
end

"""
    RandomEffect

A random effect on a latent variable.

One value is pre-drawn from `value` for each unique combination of
`categoricalInputs` (e.g. one draw per school, one per uid × wave pair).
That draw is multiplied by the product of `numericalInputs` column values
and added to the `target` latent variable.

If `categoricalInputs` is empty, a fresh draw is taken on every evaluation,
modelling residual / error variance.

`value` is a `SamplerSpec`: a `UnivariateDistribution`, a `Function`
`(rng::AbstractRNG) -> Number`, a fixed `Int`, or a `Range`.
"""
struct RandomEffect
    target::String
    numericalInputs::Vector{String}    # numeric columns that scale the draw
    categoricalInputs::Vector{String}  # columns that define groups (one draw per group)
    value::SamplerSpec                 # distribution or callable to draw from
end

"""
    LatentLoading

Specifies how one latent variable contributes to questionnaire item mean scores.
`scale` maps the latent value onto the item's Likert scale
(e.g. `2.5` maps a latent value of 1.0 to a mean item score of 2.5 on a 0–3 scale).
"""
struct LatentLoading
    latentName::String
    scale::Float64
end

"""
    QuestionnaireSpec

Declarative specification for a Likert-scale questionnaire.

All items share the same `nLevels` (e.g. 4 for a 0–3 scale), `loadings` from
latent variables, and `noiseSD`. A `spoilRate` fraction of responses are
entirely random (simulating students who do not engage with the questionnaire).
"""
struct QuestionnaireSpec
    name::String                    # e.g. "PHQ_9"
    prefix::String                  # column prefix, e.g. "phq9"
    nItems::Int                     # number of items
    nLevels::Int                    # Likert levels per item (e.g. 4 for 0-3)
    loadings::Vector{LatentLoading} # which latent variables drive item means
    noiseSD::Float64                # noise around the latent-derived mean
    spoilRate::Float64              # probability of a random/spoiled response
end

"""
    DemographicsSpec

Specifies the demographic weight distributions used when generating student demographics.
Each field is a vector of `(category, weight)` pairs whose weights sum to 1.

The optional `customFields` dictionary maps output column names to zero-argument
functions that produce a value (e.g. `Dict("d_city" => () -> Faker.city())`).
These fields are included in every generated `DataRow` and in the `Schema`.

Use `default_demographics_spec()` from `demographics.jl` to get the UK-census-derived defaults.
"""
Base.@kwdef struct DemographicsSpec
    ethnicity::Vector{Tuple{String,Float64}}         = []
    sex::Vector{Tuple{String,Float64}}               = []
    genderIdentity::Vector{Tuple{String,Float64}}    = []
    sexualOrientation::Vector{Tuple{String,Float64}} = []
    customFields::Dict{String,Function}              = Dict{String,Function}()
end

"""
    SimulationConfig

Holds all configuration parameters for a simulation run.

Fields whose defaults are empty collections (`questionnaires`, `latentVariables`,
`linearEffects`, `randomEffects`) are filled with sensible defaults by `simulate()` when
left at their defaults.
"""
Base.@kwdef struct SimulationConfig
    nWaves::Int = 3
    nSchools::Int = 10
    nYeargroupsPerSchool::SamplerSpec = 5
    nClassesPerSchoolYeargroup::SamplerSpec = Range(1, 5)
    nStudentsPerClass::SamplerSpec = Normal(30.0, 7.0)
    questionnaires::Vector{QuestionnaireSpec} = QuestionnaireSpec[]
    latentVariables::Vector{String} = String[]
    linearEffects::Vector{LinearEffect} = LinearEffect[]
    randomEffects::Vector{RandomEffect} = RandomEffect[]
    includeLatents::Bool = false
    demographicPerturbationSD::Float64 = 0.05
    demographicsSpec::Union{DemographicsSpec,Nothing} = nothing
    demographicsUpdateFn::Function = default_demographics_update
    naughtyMonkey::Function = default_naughty_monkey
    output::Union{String,Function} = "csv"
    seed::Union{Int,Nothing} = nothing
end
