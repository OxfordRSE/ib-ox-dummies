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
"""
struct Schema
    demographicsColumns::Vector{String}
    questionnaireColumns::Dict{String,String}
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
    Questionnaire

A questionnaire is a callable that simulates responses for one student in
one wave.

Signature:
    (rng, studentData::Vector{QData}, schema::Schema) -> QData

- `rng`: random number generator (passed for reproducibility).
- `studentData`: all data collected for this student so far, including
  demographics and all previous wave responses. May be empty on the first wave.
- `schema`: the current schema.

Returns a `QData` dict containing only the questionnaire column responses
for this questionnaire.
"""
const Questionnaire = Function

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
"""
Base.@kwdef struct SimulationConfig
    nWaves::Int = 3
    nSchools::Int = 10
    nYeargroupsPerSchool::CountSpec = 5
    nClassesPerSchoolYeargroup::CountSpec = Range(1, 5)
    nStudentsPerClass::CountSpec = Normal(30.0, 7.0)
    questionnaires::Dict{String,Questionnaire} = Dict{String,Questionnaire}()
    demographicsUpdateFn::DemographicsUpdateFn = default_demographics_update
    naughtyMonkey::NaughtyMonkeyFn = default_naughty_monkey
    output::Union{String,OutputFn} = "csv"
    seed::Union{Int,Nothing} = nothing
end
