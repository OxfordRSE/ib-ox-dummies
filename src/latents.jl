"""
    default_latent_variables() -> Vector{String}

Return the default latent variable names used by the built-in questionnaires.
"""
function default_latent_variables()::Vector{String}
    return ["depression", "anxiety"]
end

"""
    default_linear_effects() -> Vector{LinearEffect}

Return the default fixed-effect linear effects for the built-in latent variables.

Numeric column encodings used:
- `"d_age"`   — student age in years (≈ 10–16 for secondary school)
- `"_sex_fm"` — effect-coded sex: Female = 1.0, Male = -1.0, other = 0.0
"""
function default_linear_effects()::Vector{LinearEffect}
    return [
        # Moderate positive age effect (older students report worse mental health on average)
        LinearEffect("depression", ["d_age"], 0.02),
        LinearEffect("anxiety",    ["d_age"], 0.015),

        # Small sex effect (female > male on depression and anxiety scales)
        LinearEffect("depression", ["_sex_fm"], 0.05),
        LinearEffect("anxiety",    ["_sex_fm"], 0.05),

        # Small age × sex interaction (female disadvantage grows with age)
        LinearEffect("depression", ["d_age", "_sex_fm"], 0.005),
        LinearEffect("anxiety",    ["d_age", "_sex_fm"], 0.004),
    ]
end

"""
    default_random_effects() -> Vector{RandomEffect}

Return the default random effects for the built-in latent variables.

The list captures:
1. Cohort (yearGroup) cluster effect — small shared experience per school year.
2. Ethnicity × class × school cluster effect — small intersectional cluster variation.
3. Individual × wave effect — moderate fluctuation in mental health across waves.
4. Individual baseline effect — major inter-individual variation (half-normal, always ≥ 0).
5. Residual / error term — independent noise drawn fresh per observation.
"""
function default_random_effects()::Vector{RandomEffect}
    return [
        # 1. Small cohort (yearGroup) effect
        RandomEffect("depression", [], ["yearGroup"], Normal(0.0, 0.05)),
        RandomEffect("anxiety",    [], ["yearGroup"], Normal(0.0, 0.05)),

        # 2. Small ethnicity × class × school cluster effect
        RandomEffect("depression", [], ["d_ethnicity", "class", "school"], Normal(0.0, 0.03)),
        RandomEffect("anxiety",    [], ["d_ethnicity", "class", "school"], Normal(0.0, 0.03)),

        # 3. Individual × wave effect (fluctuating trajectories over time)
        RandomEffect("depression", [], ["uid", "wave"], Normal(0.0, 0.15)),
        RandomEffect("anxiety",    [], ["uid", "wave"], Normal(0.0, 0.12)),

        # 4. Major individual baseline (half-normal: always >= 0)
        RandomEffect("depression", [], ["uid"], truncated(Normal(0.0, 0.2), 0.0, Inf)),
        RandomEffect("anxiety",    [], ["uid"], truncated(Normal(0.0, 0.15), 0.0, Inf)),

        # 5. Residual error (fresh draw per observation)
        RandomEffect("depression", [], [], Normal(0.0, 0.1)),
        RandomEffect("anxiety",    [], [], Normal(0.0, 0.1)),
    ]
end

"""
    add_numeric_encodings!(row::StudentDataRow) -> StudentDataRow

Compute and insert synthetic numeric columns used by `LinearEffect` inputs:
- `_sex_fm`: effect-coded sex (F = 1.0, M = -1.0, other/intersex = 0.0).

Modifies `row` in-place and returns it.
"""
function add_numeric_encodings!(row::StudentDataRow)::StudentDataRow
    sex = get(row, "d_sex", "")
    row["_sex_fm"] = sex == "F" ? 1.0 : sex == "M" ? -1.0 : 0.0
    return row
end

"""
    precompute_effect_draws(rng, effects, rows) -> Vector{Dict{Any,Float64}}

Pre-draw one random value per unique combination of `categoricalInputs` for each RandomEffect.

Returns a vector (parallel to `effects`) of dicts mapping group-key tuples to
drawn `Float64` values. Effects with empty `categoricalInputs` return an empty
dict because they draw a fresh value on every evaluation (residual error).
"""
function precompute_effect_draws(
    rng::AbstractRNG,
    effects::Vector{RandomEffect},
    rows::Vector{StudentDataRow},
)::Vector{Dict{Any,Float64}}
    draws = Vector{Dict{Any,Float64}}(undef, length(effects))
    for (i, eff) in enumerate(effects)
        if isempty(eff.categoricalInputs)
            draws[i] = Dict{Any,Float64}()  # fresh draw every time
        else
            group_vals = Set(
                Tuple(get(row, col, missing) for col in eff.categoricalInputs)
                for row in rows
            )
            draws[i] = Dict{Any,Float64}(
                grp => rand(rng, eff.value) for grp in group_vals
            )
        end
    end
    return draws
end

"""
    compute_row_latents(rng, row, latent_names, linearEffects, randomEffects, effect_draws) -> Dict{String,Float64}

Compute latent variable values for a single row.

Each `LinearEffect` contributes `value × ∏(inputs)` to the target latent.
Each `RandomEffect` contributes `draw × ∏(numericalInputs)` where `draw` is looked
up from `effect_draws` by the row's categorical group key, or drawn fresh for
error terms (empty `categoricalInputs`).
"""
function compute_row_latents(
    rng::AbstractRNG,
    row::StudentDataRow,
    latent_names::Vector{String},
    linearEffects::Vector{LinearEffect},
    randomEffects::Vector{RandomEffect},
    effect_draws::Vector{Dict{Any,Float64}},
)::Dict{String,Float64}
    lv = Dict{String,Float64}(name => 0.0 for name in latent_names)

    # Fixed effects (LinearEffects)
    for coef in linearEffects
        coef.target ∈ latent_names || continue
        product = coef.value
        valid = true
        for inp in coef.inputs
            v = get(row, inp, nothing)
            if v isa Number
                product *= Float64(v)
            else
                valid = false
                break
            end
        end
        valid && (lv[coef.target] += product)
    end

    # Random effects (RandomEffects)
    for (i, eff) in enumerate(randomEffects)
        eff.target ∈ latent_names || continue

        draw = if isempty(eff.categoricalInputs)
            rand(rng, eff.value)  # fresh residual draw
        else
            grp_key = Tuple(get(row, col, missing) for col in eff.categoricalInputs)
            get(effect_draws[i], grp_key, 0.0)
        end

        num_scale = 1.0
        for inp in eff.numericalInputs
            v = get(row, inp, nothing)
            if v isa Number
                num_scale *= Float64(v)
            else
                num_scale = 0.0
                break
            end
        end

        lv[eff.target] += draw * num_scale
    end

    return lv
end
