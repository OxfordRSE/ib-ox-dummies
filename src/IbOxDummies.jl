"""
    IbOxDummies

A simulator for generating realistic-ish longitudinal questionnaire data,
designed for schoolchildren completing annual surveys.

Simulation uses an agent-based approach driven by latent variables (e.g.
`"depression"`, `"anxiety"`). Each student's questionnaire responses are
derived from their individual latent values at each wave, composed from:
- Fixed-effect `LinearEffect`s (e.g. age or sex effects)
- Random-effect `RandomEffect`s (e.g. school cluster, individual baseline, residual error)

## Quick start

```julia
using IbOxDummies

# Questionnaires and latent variables must be specified explicitly.
# Use default_questionnaires() for the built-in PHQ-9 + GAD-7 model:
data, schema = simulate(SimulationConfig(
    seed           = 42,
    latentVariables = default_latent_variables(),
    linearEffects  = default_linear_effects(),
    randomEffects  = default_random_effects(),
    questionnaires  = default_questionnaires(),
))

# Or load a model from a TOML config file (CLI):
#   ib_ox_dummies --config examples/minimal_model.toml --seed 42

# Write as CSV (uses CSV.jl)
to_csv(data, schema)

# Write as JSON (uses JSON3.jl)
to_json(data, schema)

# Include latent variable values in the output for ground-truth comparison
data, schema = simulate(SimulationConfig(seed = 42, includeLatents = true))
# data now has l_depression and l_anxiety columns

# Custom latent model
data, schema = simulate(SimulationConfig(
    seed          = 42,
    latentVariables = ["depression"],
    linearEffects = [LinearEffect("depression", ["d_age"], 0.03)],
    randomEffects = [
        RandomEffect("depression", [], ["uid"], truncated(Normal(0, 0.2), 0, Inf)),
        RandomEffect("depression", [], [],      Normal(0, 0.1)),  # residual error
    ],
    questionnaires  = [make_phq9()],
))
```

## CLI

The package provides an `ib_ox_dummies` executable.  Run it with:

    ib_ox_dummies --help

Use `--config examples/default_model.toml` to specify questionnaires, latent variable
loadings, effects, and demographics from a TOML file (see `examples/default_model.toml`).
CLI arguments override any TOML values.
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
using Faker
import TOML

include("types.jl")
include("demographics.jl")
include("latents.jl")
include("questionnaires.jl")
include("simulator.jl")
include("output.jl")
include("cli.jl")

export
    # Core types
    Response,
    DataRow,
    Schema,
    Range,
    SamplerSpec,
    LinearEffect,
    RandomEffect,
    LatentLoading,
    QuestionnaireSpec,
    DemographicsSpec,
    SimulationConfig,

    # Questionnaire factories (return QuestionnaireSpec)
    make_phq9,
    make_gad7,
    default_questionnaires,
    beewell_questionnaires,
    beewell_latent_variables,
    beewell_linear_effects,
    beewell_random_effects,
    generate_questionnaire_responses,

    # Latent variable system
    default_latent_variables,
    default_linear_effects,
    default_random_effects,
    add_numeric_encodings!,
    precompute_effect_draws,
    compute_row_latents,

    # Demographics helpers
    default_demographics_spec,
    default_demographics_update,
    default_naughty_monkey,
    generate_demographics,
    perturb_weights,
    weighted_sample,
    sample_count,
    draw_sampler,

    # Simulation
    simulate,
    build_schema,
    rows_to_dataframe,

    # Output
    to_csv,
    to_json,
    to_json_schema,
    write_output,
    column_order,

    # CLI
    ib_ox_dummies_cli,
    parse_cli_args,
    parse_sampler_spec,
    parse_linear_effect,
    parse_random_effect,
    parse_demographics_weights,
    parse_custom_field_value,
    parse_custom_fields,
    parse_questionnaire_spec_from_dict,
    parse_linear_effect_from_dict,
    parse_random_effect_from_dict,
    load_toml_config

end
