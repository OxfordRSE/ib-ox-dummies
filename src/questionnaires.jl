"""
    default_questionnaires() -> Vector{QuestionnaireSpec}

Return the default set of questionnaire specifications: PHQ-9 and GAD-7.

PHQ-9 measures depression (9 items, 0–3 scale, loads on `"depression"`).
GAD-7 measures anxiety (7 items, 0–3 scale, loads on `"anxiety"` and secondarily `"depression"`).
"""
function default_questionnaires()::Vector{QuestionnaireSpec}
    return [
        QuestionnaireSpec(
            "PHQ_9", "phq9", 9, 4,
            [LatentLoading("depression", 2.5)],
            0.6, 0.01,
        ),
        QuestionnaireSpec(
            "GAD_7", "gad7", 7, 4,
            [LatentLoading("anxiety", 2.5), LatentLoading("depression", 0.8)],
            0.6, 0.01,
        ),
    ]
end

"""
    make_phq9() -> QuestionnaireSpec

Return a PHQ-9 questionnaire specification.
The PHQ-9 has 9 items scored 0–3 and loads primarily on the `"depression"` latent variable.
"""
function make_phq9()::QuestionnaireSpec
    return QuestionnaireSpec(
        "PHQ_9", "phq9", 9, 4,
        [LatentLoading("depression", 2.5)],
        0.6, 0.01,
    )
end

"""
    make_gad7() -> QuestionnaireSpec

Return a GAD-7 questionnaire specification.
The GAD-7 has 7 items scored 0–3 and loads on `"anxiety"` (and secondarily `"depression"`).
"""
function make_gad7()::QuestionnaireSpec
    return QuestionnaireSpec(
        "GAD_7", "gad7", 7, 4,
        [LatentLoading("anxiety", 2.5), LatentLoading("depression", 0.8)],
        0.6, 0.01,
    )
end

"""
    generate_questionnaire_responses(rng, spec, latents, prev_responses) -> QData

Generate item responses for one student at one wave.

The mean item score is derived from the weighted sum of latent variable contributions
(`loadings`). A noise sample from a truncated Normal is added; the result is rounded
and clamped to `[0, nLevels - 1]`.

Longitudinal continuity: each item has a 75% chance (given a previous response exists)
of blending the latent-derived mean with the previous wave's score, producing realistic
stability over time.

`spoilRate` fraction of responses are entirely random (simulating disengaged students).
"""
function generate_questionnaire_responses(
    rng::AbstractRNG,
    spec::QuestionnaireSpec,
    latents::Dict{String,Float64},
    prev_responses::Union{QData,Nothing},
)::QData
    result = QData()

    # Spoiled / random response
    if rand(rng) < spec.spoilRate
        for i in 1:spec.nItems
            result["$(spec.prefix)_$i"] = rand(rng, 0:(spec.nLevels - 1))
        end
        return result
    end

    # Latent-derived mean score, clamped to valid item range
    latent_mean = clamp(
        sum(l.scale * get(latents, l.latentName, 0.0) for l in spec.loadings; init = 0.0),
        0.0,
        Float64(spec.nLevels - 1),
    )

    lo = -0.5
    hi = Float64(spec.nLevels) - 0.5

    for i in 1:spec.nItems
        key = "$(spec.prefix)_$i"

        # 5% chance of a random item
        if rand(rng) < 0.05
            result[key] = rand(rng, 0:(spec.nLevels - 1))
            continue
        end

        # Previous value for longitudinal continuity
        prev_val = if !isnothing(prev_responses)
            v = get(prev_responses, key, nothing)
            v isa Int ? Float64(v) : nothing
        else
            nothing
        end

        # Blend with previous response 75% of the time
        item_mean = if !isnothing(prev_val) && rand(rng) < 0.75
            0.5 * latent_mean + 0.5 * prev_val
        else
            latent_mean
        end

        # Sample from truncated Normal, round, and clamp
        raw = rand(rng, truncated(Normal(item_mean, spec.noiseSD), lo, hi))
        result[key] = clamp(round(Int, raw), 0, spec.nLevels - 1)
    end

    return result
end
