"""
    IbOxDummies

A simulator for generating realistic-ish longitudinal questionnaire data,
designed for schoolchildren completing annual surveys.

## Quick start

```julia
using IbOxDummies

# Run with defaults (3 waves, 10 schools, PHQ-9 + GAD-7)
data, schema = simulate(SimulationConfig(seed = 42))

# Write as CSV (uses CSV.jl)
to_csv(data, schema)

# Write as JSON (uses JSON3.jl)
to_json(data, schema)

# Export JSON Schema describing the output
println(to_json_schema(schema))

# Include latent variable values in the output DataFrame
data, schema = simulate(SimulationConfig(seed = 42, includeLatents = true))
```

## CLI

The package provides an `ib_ox_dummies` executable.  Run it with:

    ib_ox_dummies --help
"""
module IbOxDummies

using Random
using Distributions
using StatsBase: sample, Weights
using DataFrames
using Tables
using CSV
using JSON3
using ArgParse

include("types.jl")
include("demographics.jl")
include("latents.jl")
include("questionnaires.jl")
include("simulator.jl")
include("output.jl")
include("cli.jl")

export
    # Core types
    AData,
    QData,
    Schema,
    Range,
    CountSpec,
    Coefficient,
    Effect,
    LatentLoading,
    QuestionnaireSpec,
    NaughtyMonkeyFn,
    OutputFn,
    DemographicsUpdateFn,
    SimulationConfig,

    # Questionnaire factories (return QuestionnaireSpec)
    make_phq9,
    make_gad7,
    default_questionnaires,
    default_questionnaire_specs,
    generate_questionnaire_responses,

    # Latent variable system
    default_latent_variables,
    default_coefficients,
    default_effects,
    add_numeric_encodings!,
    precompute_effect_draws,
    compute_row_latents,

    # Demographics helpers
    default_demographics_update,
    default_naughty_monkey,
    generate_demographics,
    perturb_weights,
    weighted_sample,
    sample_count,

    # Simulation
    simulate,
    build_schema,
    qdata_to_dataframe,

    # Output
    to_csv,
    to_json,
    to_json_schema,
    write_output,
    column_order,

    # CLI
    ib_ox_dummies_cli,
    parse_cli_args,
    parse_count_spec

end
