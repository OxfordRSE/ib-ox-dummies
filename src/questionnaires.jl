# Depression risk category weights (PHQ-9 based, approximate population estimates)
# None/Minimal: ~50%, Mild: ~25%, Moderate: ~15%, Severe: ~10%
const PHQ9_RISK_WEIGHTS = [
    (:none,     0.50, 0.5, 0.5),   # (category, pop_weight, item_mean, item_sd)
    (:mild,     0.25, 1.0, 1.0),
    (:moderate, 0.15, 1.5, 1.5),
    (:severe,   0.10, 2.5, 2.0),
]

# Anxiety risk category weights (GAD-7 based)
const GAD7_RISK_WEIGHTS = [
    (:none,     0.55, 0.4, 0.5),
    (:mild,     0.25, 1.0, 1.0),
    (:moderate, 0.12, 1.6, 1.4),
    (:severe,   0.08, 2.5, 2.0),
]

"""
    sample_risk_category(rng, weights) -> (Symbol, Float64, Float64)

Sample a risk category and return `(category, item_mean, item_sd)`.
Uses a `Categorical` distribution over the provided population weights.
"""
function sample_risk_category(rng::AbstractRNG, weights)
    wts = [w for (_, w, _, _) in weights]
    idx = rand(rng, Categorical(wts ./ sum(wts)))
    entry = weights[idx]
    return (entry[1], entry[3], entry[4])
end

"""
    phq9_score_from_items(row, prefix) -> Union{Int, Nothing}

Calculate total PHQ-9 score from individual item responses.
Returns `nothing` if no items are present.
"""
function phq9_score_from_items(row::QData, prefix::String)::Union{Int,Nothing}
    total = 0
    found = false
    for i in 1:9
        key = "$(prefix)_$(i)"
        if haskey(row, key) && row[key] isa Int
            total += row[key]::Int
            found = true
        end
    end
    found ? total : nothing
end

"""
    score_to_risk_category(score, thresholds) -> Symbol

Map a total score to a risk category using `thresholds = [(category, max_score)]`
in ascending order.
"""
function score_to_risk_category(score::Int, thresholds)::Symbol
    for (cat, threshold) in thresholds
        score <= threshold && return cat
    end
    return thresholds[end][1]
end

const PHQ9_THRESHOLDS = [(:none, 4), (:mild, 9), (:moderate, 14), (:severe, 27)]
const GAD7_THRESHOLDS = [(:none, 4), (:mild, 9), (:moderate, 14), (:severe, 21)]

"""
    simulate_likert_item(rng, n_levels, item_mean, item_sd, prev_val) -> Int

Simulate a single Likert-scale item response (0 to n_levels-1).

- 5% chance of a uniformly random response.
- If `prev_val` is provided: 75% chance of sampling from `Normal(prev_val, item_sd)`.
- Otherwise: sample from `truncated(Normal(item_mean, item_sd), 0, Inf)` (half-normal
  starting at 0 to reflect mild baseline severity).
"""
function simulate_likert_item(
    rng::AbstractRNG,
    n_levels::Int,
    item_mean::Float64,
    item_sd::Float64,
    prev_val::Union{Int,Nothing},
)::Int
    # 5% chance of a completely random response
    if rand(rng) < 0.05
        return rand(rng, 0:(n_levels - 1))
    end

    raw = if !isnothing(prev_val) && rand(rng) < 0.75
        # Continue from previous answer using Normal centred on previous value
        rand(rng, Normal(Float64(prev_val), item_sd))
    else
        # First response or no continuation: truncated normal starting at 0
        rand(rng, truncated(Normal(item_mean, item_sd), 0.0, Inf))
    end

    return clamp(round(Int, raw), 0, n_levels - 1)
end

"""
    make_phq9(; prefix="phq9") -> Questionnaire

Return a PHQ-9 questionnaire simulation function.
The PHQ-9 has 9 items, each scored 0-3.
"""
function make_phq9(; prefix::String = "phq9")::Questionnaire
    function phq9(rng::AbstractRNG, studentData::Vector{QData}, schema::Schema)::QData
        result = QData()

        # 1% chance of entirely spoiled/random response
        if rand(rng) < 0.01
            for i in 1:9
                result["$(prefix)_$(i)"] = rand(rng, 0:3)
            end
            return result
        end

        # Determine risk category
        cat, item_mean, item_sd = if isempty(studentData)
            sample_risk_category(rng, PHQ9_RISK_WEIGHTS)
        else
            # Find the most recent wave that has PHQ-9 data
            prev_total = nothing
            for row in reverse(studentData)
                t = phq9_score_from_items(row, prefix)
                if !isnothing(t)
                    prev_total = t
                    break
                end
            end
            if !isnothing(prev_total)
                cat_sym = score_to_risk_category(prev_total, PHQ9_THRESHOLDS)
                idx = findfirst(x -> x[1] == cat_sym, PHQ9_RISK_WEIGHTS)
                isnothing(idx) ? sample_risk_category(rng, PHQ9_RISK_WEIGHTS) :
                    (PHQ9_RISK_WEIGHTS[idx][1], PHQ9_RISK_WEIGHTS[idx][3],
                     PHQ9_RISK_WEIGHTS[idx][4])
            else
                sample_risk_category(rng, PHQ9_RISK_WEIGHTS)
            end
        end

        for i in 1:9
            key = "$(prefix)_$(i)"
            prev_val::Union{Int,Nothing} = nothing
            for row in reverse(studentData)
                if haskey(row, key) && row[key] isa Int
                    prev_val = row[key]::Int
                    break
                end
            end
            result[key] = simulate_likert_item(rng, 4, item_mean, item_sd, prev_val)
        end

        return result
    end
    return phq9
end

"""
    make_gad7(; prefix="gad7") -> Questionnaire

Return a GAD-7 questionnaire simulation function.
The GAD-7 has 7 items, each scored 0-3.
"""
function make_gad7(; prefix::String = "gad7")::Questionnaire
    function gad7(rng::AbstractRNG, studentData::Vector{QData}, schema::Schema)::QData
        result = QData()

        # 1% chance of entirely spoiled/random response
        if rand(rng) < 0.01
            for i in 1:7
                result["$(prefix)_$(i)"] = rand(rng, 0:3)
            end
            return result
        end

        # Determine risk category
        cat, item_mean, item_sd = if isempty(studentData)
            sample_risk_category(rng, GAD7_RISK_WEIGHTS)
        else
            prev_total = nothing
            for row in reverse(studentData)
                total = 0
                found = false
                for i in 1:7
                    key = "$(prefix)_$(i)"
                    if haskey(row, key) && row[key] isa Int
                        total += row[key]::Int
                        found = true
                    end
                end
                if found
                    prev_total = total
                    break
                end
            end
            if !isnothing(prev_total)
                cat_sym = score_to_risk_category(prev_total, GAD7_THRESHOLDS)
                idx = findfirst(x -> x[1] == cat_sym, GAD7_RISK_WEIGHTS)
                isnothing(idx) ? sample_risk_category(rng, GAD7_RISK_WEIGHTS) :
                    (GAD7_RISK_WEIGHTS[idx][1], GAD7_RISK_WEIGHTS[idx][3],
                     GAD7_RISK_WEIGHTS[idx][4])
            else
                sample_risk_category(rng, GAD7_RISK_WEIGHTS)
            end
        end

        for i in 1:7
            key = "$(prefix)_$(i)"
            prev_val::Union{Int,Nothing} = nothing
            for row in reverse(studentData)
                if haskey(row, key) && row[key] isa Int
                    prev_val = row[key]::Int
                    break
                end
            end
            result[key] = simulate_likert_item(rng, 4, item_mean, item_sd, prev_val)
        end

        return result
    end
    return gad7
end

"""
    default_questionnaires() -> Dict{String, Questionnaire}

Return the default set of questionnaires: PHQ-9 and GAD-7.
"""
function default_questionnaires()::Dict{String,Questionnaire}
    return Dict{String,Questionnaire}(
        "PHQ_9" => make_phq9(prefix = "phq9"),
        "GAD_7" => make_gad7(prefix = "gad7"),
    )
end

"""
    questionnaire_columns(name, questionnaire_fn) -> Vector{String}

Determine the column names produced by a questionnaire function.
We run the questionnaire once with an empty RNG and empty data to discover
which columns it produces.
"""
function questionnaire_columns(
    name::String,
    questionnaire_fn::Questionnaire,
    schema::Schema,
)::Vector{String}
    rng = MersenneTwister(42)
    dummy_schema = Schema(String[], Dict{String,String}())
    result = questionnaire_fn(rng, QData[], dummy_schema)
    return sort(collect(keys(result)))
end
