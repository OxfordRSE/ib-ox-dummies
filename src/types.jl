"""
    AData

Answer data: the value of a single field for a student in a given wave.
Can be an integer, float, string, or missing (unanswered/redacted).
"""
const AData = Union{Int,Float64,String,Missing}

"""
    QData

Questionnaire data: a named record of answer data representing one row
(one student × one wave).
"""
const QData = Dict{String,AData}

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
    # Backward-compatible 2-arg constructor
    Schema(d::Vector{String}, q::Dict{String,String}) = new(d, q, String[])
    Schema(d::Vector{String}, q::Dict{String,String}, l::Vector{String}) = new(d, q, l)
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
    CountSpec

A count specification: either a fixed integer, an inclusive range, or a
`UnivariateDistribution` from `Distributions.jl` to sample from
(values are rounded to the nearest integer and clamped to ≥ 1).

## Examples

```julia
5                      # fixed count
Range(1, 5)            # uniform sample from [1, 5]
Normal(30.0, 7.0)      # round(Normal(30, 7)) ≥ 1
truncated(Normal(30.0, 7.0), 1, Inf)  # truncated to avoid non-positive values
```
"""
const CountSpec = Union{Int,Range,UnivariateDistribution}

"""
    Coefficient

A fixed linear effect on a latent variable.

Adds `value × ∏(inputs)` to the named `target` latent variable for each row.
All `inputs` must name numeric columns present in the row (e.g. `"d_age"`,
`"_sex_fm"`). If any input is non-numeric the coefficient contributes 0.
"""
struct Coefficient
    target::String          # latent variable name
    inputs::Vector{String}  # numeric column names to multiply together
    value::Float64          # scaling factor
end

"""
    Effect

A random effect on a latent variable.

One value is pre-drawn from `value` for each unique combination of
`categoricalInputs` (e.g. one draw per school, one per uid × wave pair).
That draw is multiplied by the product of `numericalInputs` column values
and added to the `target` latent variable.

If `categoricalInputs` is empty, a fresh draw is taken on every evaluation,
modelling residual / error variance.
"""
struct Effect
    target::String
    numericalInputs::Vector{String}    # numeric columns that scale the draw
    categoricalInputs::Vector{String}  # columns that define groups (one draw per group)
    value::UnivariateDistribution      # distribution to draw from
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
    NaughtyMonkeyFn

Last-call function to randomly corrupt or remap the simulated output.

Signature:
    (rng, output::Vector{QData}, schema::Schema) -> Vector{QData}
"""
const NaughtyMonkeyFn = Function

"""
    OutputFn

Custom output function.

Signature:
    (data::DataFrame, schema::Schema) -> Any
"""
const OutputFn = Function

"""
    DemographicsUpdateFn

Function to update demographics between waves.

Signature:
    (rng, prevData::Vector{QData}) -> QData

`prevData` contains all data for this student across all previous waves.
Unreturned keys are copied from the most recent wave's demographics.
"""
const DemographicsUpdateFn = Function

"""
    SimulationConfig

Holds all configuration parameters for a simulation run.

Fields whose defaults are empty collections (`questionnaires`, `latentVariables`,
`coefficients`, `effects`) are filled with sensible defaults by `simulate()` when
left at their defaults.
"""
Base.@kwdef struct SimulationConfig
    nWaves::Int = 3
    nSchools::Int = 10
    nYeargroupsPerSchool::CountSpec = 5
    nClassesPerSchoolYeargroup::CountSpec = Range(1, 5)
    nStudentsPerClass::CountSpec = Normal(30.0, 7.0)
    questionnaires::Vector{QuestionnaireSpec} = QuestionnaireSpec[]
    latentVariables::Vector{String} = String[]
    coefficients::Vector{Coefficient} = Coefficient[]
    effects::Vector{Effect} = Effect[]
    includeLatents::Bool = false
    demographicPerturbationSD::Float64 = 0.05
    demographicsUpdateFn::DemographicsUpdateFn = default_demographics_update
    naughtyMonkey::NaughtyMonkeyFn = default_naughty_monkey
    output::Union{String,OutputFn} = "csv"
    seed::Union{Int,Nothing} = nothing
end
