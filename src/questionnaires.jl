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
    beewell_latent_variables() -> Vector{String}

Return the latent variable names used by the `#BeeWell` GM Survey model.

The thirteen latents capture positive wellbeing, three negative-affect
dimensions, physical health, social context, future orientation, and three
low-prevalence spike constructs:

| Name               | Construct                                             |
|--------------------|-------------------------------------------------------|
| `wellbeing`        | Positive psychological wellbeing                      |
| `depression`       | Depressive affect                                     |
| `anxiety`          | Anxious / worried affect                              |
| `behaviour`        | Behavioural difficulties (anger, aggression)          |
| `physical_health`  | Physical health and activity level                    |
| `unhealthy_diet`   | Frequency of unhealthy food/drink consumption         |
| `social_connection`| Quality of relationships and social belonging         |
| `future_optimism`  | Hope and readiness for the future                     |
| `socioeconomic`    | Socioeconomic advantage                               |
| `migration`        | Migrant family background (spike: ~15 % non-zero)     |
| `discrimination`   | Exposure to discrimination (spike: ~10 % non-zero)    |
| `victimization`    | Exposure to bullying (spike: ~10 % non-zero)          |
| `screen_time`      | Daily screen / social-media engagement                |
"""
function beewell_latent_variables()::Vector{String}
    return [
        "wellbeing", "depression", "anxiety", "behaviour",
        "physical_health", "unhealthy_diet", "social_connection",
        "future_optimism", "socioeconomic",
        "migration", "discrimination", "victimization", "screen_time",
    ]
end

"""
    beewell_linear_effects() -> Vector{LinearEffect}

Return the fixed linear effects for the `#BeeWell` GM Survey model.

Includes constant (intercept) offsets that centre the latent values in a
realistic range for secondary school pupils, plus age and sex effects on the
mental-health latents.
"""
function beewell_linear_effects()::Vector{LinearEffect}
    return [
        # Intercept offsets (empty inputs ⟹ constant addition)
        LinearEffect("wellbeing",         [], 0.45),
        LinearEffect("social_connection", [], 0.50),
        LinearEffect("physical_health",   [], 0.50),
        LinearEffect("future_optimism",   [], 0.45),
        LinearEffect("socioeconomic",     [], 0.45),
        LinearEffect("unhealthy_diet",    [], 0.30),
        LinearEffect("screen_time",       [], 0.25),

        # Age effects (older pupils report slightly worse mental health)
        LinearEffect("depression", ["d_age"], 0.012),
        LinearEffect("anxiety",    ["d_age"], 0.010),
        LinearEffect("wellbeing",  ["d_age"], -0.008),

        # Sex effects (_sex_fm: F = +1, M = −1, other/intersex = 0)
        LinearEffect("depression", ["_sex_fm"], 0.05),
        LinearEffect("anxiety",    ["_sex_fm"], 0.05),
        LinearEffect("wellbeing",  ["_sex_fm"], -0.03),
    ]
end

"""
    beewell_random_effects() -> Vector{RandomEffect}

Return the random effects for the `#BeeWell` GM Survey model.

Covers individual baselines, individual×wave fluctuations, school cluster
effects, and residual error terms for each latent variable.
Spike constructs (migration, discrimination, victimization) use `mde`-style
Bernoulli draws to reflect their low population prevalence.
"""
function beewell_random_effects()::Vector{RandomEffect}
    return [
        # --- Individual baselines (half-normal: always ≥ 0) ---
        RandomEffect("wellbeing",         [], ["uid"], truncated(Normal(0.0, 0.25), 0.0, Inf)),
        RandomEffect("depression",        [], ["uid"], truncated(Normal(0.0, 0.15), 0.0, Inf)),
        RandomEffect("anxiety",           [], ["uid"], truncated(Normal(0.0, 0.15), 0.0, Inf)),
        RandomEffect("behaviour",         [], ["uid"], truncated(Normal(0.0, 0.12), 0.0, Inf)),
        RandomEffect("physical_health",   [], ["uid"], truncated(Normal(0.0, 0.20), 0.0, Inf)),
        RandomEffect("unhealthy_diet",    [], ["uid"], truncated(Normal(0.0, 0.20), 0.0, Inf)),
        RandomEffect("social_connection", [], ["uid"], truncated(Normal(0.0, 0.20), 0.0, Inf)),
        RandomEffect("future_optimism",   [], ["uid"], truncated(Normal(0.0, 0.20), 0.0, Inf)),
        RandomEffect("socioeconomic",     [], ["uid"], truncated(Normal(0.0, 0.20), 0.0, Inf)),
        RandomEffect("screen_time",       [], ["uid"], truncated(Normal(0.0, 0.25), 0.0, Inf)),

        # Spike latents: Bernoulli(p) × Normal(μ,σ) — low prevalence constructs
        RandomEffect("migration",      [], ["uid"], rng -> rand(rng) < 0.15 ? rand(rng, Normal(0.70, 0.10)) : 0.0),
        RandomEffect("discrimination", [], ["uid"], rng -> rand(rng) < 0.10 ? rand(rng, Normal(0.50, 0.15)) : 0.0),
        RandomEffect("victimization",  [], ["uid"], rng -> rand(rng) < 0.10 ? rand(rng, Normal(0.50, 0.15)) : 0.0),

        # --- Individual × wave fluctuations ---
        RandomEffect("wellbeing",         [], ["uid", "wave"], Normal(0.0, 0.08)),
        RandomEffect("depression",        [], ["uid", "wave"], Normal(0.0, 0.08)),
        RandomEffect("anxiety",           [], ["uid", "wave"], Normal(0.0, 0.08)),
        RandomEffect("social_connection", [], ["uid", "wave"], Normal(0.0, 0.06)),
        RandomEffect("future_optimism",   [], ["uid", "wave"], Normal(0.0, 0.06)),

        # --- School cluster effects ---
        RandomEffect("wellbeing",         [], ["_school_id"], Normal(0.0, 0.04)),
        RandomEffect("socioeconomic",     [], ["_school_id"], Normal(0.0, 0.05)),
        RandomEffect("social_connection", [], ["_school_id"], Normal(0.0, 0.04)),

        # --- Residual error (fresh draw per observation) ---
        RandomEffect("wellbeing",         [], [], Normal(0.0, 0.05)),
        RandomEffect("depression",        [], [], Normal(0.0, 0.05)),
        RandomEffect("anxiety",           [], [], Normal(0.0, 0.05)),
        RandomEffect("physical_health",   [], [], Normal(0.0, 0.04)),
        RandomEffect("social_connection", [], [], Normal(0.0, 0.04)),
    ]
end

"""
    beewell_questionnaires() -> Vector{QuestionnaireSpec}

Return all `QuestionnaireSpec`s for the `#BeeWell` Greater Manchester Survey
(updated August 2025).

The 49 specifications cover every item group in the survey booklet, from
demographic background items (Q4–5) through the wellbeing domains (Q6–42)
and drivers of wellbeing (Q43–124).  Together they produce 136 output columns
prefixed `bw_*`.

Latent variable names referenced in the loadings:

    $(join(beewell_latent_variables(), ", "))

Use `beewell_latent_variables()`, `beewell_linear_effects()`, and
`beewell_random_effects()` to build the matching `SimulationConfig`.

# Example

```julia
using IbOxDummies

data, schema = simulate(SimulationConfig(
    seed            = 42,
    latentVariables = beewell_latent_variables(),
    linearEffects   = beewell_linear_effects(),
    randomEffects   = beewell_random_effects(),
    questionnaires  = beewell_questionnaires(),
))
```
"""
function beewell_questionnaires()::Vector{QuestionnaireSpec}
    return [
        # Q4: Migration status (3 items: birth parent 1, 2, and you)
        # 0 = No, 1 = Don't know, 2 = Yes
        QuestionnaireSpec("BW_Migration", "bw_migration", 3, 3,
            [LatentLoading("migration", 2.0)], 0.8, 0.01),

        # Q5: Age of arrival in the UK (0–15 years; conditional on Q4c = Yes)
        QuestionnaireSpec("BW_ArrivalAge", "bw_arrival", 1, 16,
            [LatentLoading("migration", 7.5)], 2.0, 0.05),

        # Q6: Life satisfaction (0–10; 0 = not at all, 10 = completely)
        QuestionnaireSpec("BW_LifeSatisfaction", "bw_life_sat", 1, 11,
            [LatentLoading("wellbeing", 9.0), LatentLoading("depression", -4.0)], 1.2, 0.01),

        # Q7–13: SWEMWBS psychological wellbeing (7 items, 0–4)
        QuestionnaireSpec("BW_Wellbeing_SWEMWBS", "bw_wbeing", 7, 5,
            [LatentLoading("wellbeing", 4.0), LatentLoading("depression", -2.0),
             LatentLoading("social_connection", 0.5)], 0.6, 0.01),

        # Q14–18: Rosenberg self-esteem (5 items; 0 = strongly disagree … 3 = strongly agree)
        QuestionnaireSpec("BW_SelfEsteem", "bw_selfest", 5, 4,
            [LatentLoading("wellbeing", 3.0), LatentLoading("depression", -1.5)], 0.6, 0.01),

        # Q19–21: Emotion regulation / CWMS coping subscale (3 items, 0–2)
        QuestionnaireSpec("BW_EmoRegulation", "bw_emoreg", 3, 3,
            [LatentLoading("wellbeing", 2.0), LatentLoading("anxiety", -1.0)], 0.5, 0.01),

        # Q22: Appearance happiness (0–10)
        QuestionnaireSpec("BW_Appearance", "bw_appear", 1, 11,
            [LatentLoading("wellbeing", 9.0), LatentLoading("depression", -4.0)], 1.5, 0.01),

        # Q23–24: PSS-4 stress (2 items, 0–4; not asked of Year 7)
        QuestionnaireSpec("BW_Stress", "bw_stress", 2, 5,
            [LatentLoading("anxiety", 4.0), LatentLoading("depression", 1.5)], 0.7, 0.01),

        # Q25–26: PSS-4 coping / positive items (2 items, 0–4; not asked of Year 7)
        QuestionnaireSpec("BW_Coping", "bw_coping", 2, 5,
            [LatentLoading("wellbeing", 3.0), LatentLoading("anxiety", -2.0)], 0.7, 0.01),

        # Q27–36: Emotional difficulties / Me & My Feelings (10 items, 0–2)
        QuestionnaireSpec("BW_EmoDifficulties", "bw_emodies", 10, 3,
            [LatentLoading("depression", 1.5), LatentLoading("anxiety", 1.0),
             LatentLoading("social_connection", -0.5)], 0.5, 0.01),

        # Q37–42: Behavioural difficulties / Me & My Feelings (6 items, 0–2)
        # Item 4 "I am calm" is reverse-scored (driven by wellbeing, not behaviour)
        QuestionnaireSpec("BW_BehavDifficulties", "bw_behav", 6, 3,
            [LatentLoading("behaviour", Dict("1" => 2.0, "2" => 2.0, "3" => 2.0, "5" => 2.0, "6" => 2.0)),
             LatentLoading("wellbeing", Dict("4" => 2.0))], 0.5, 0.01),

        # Q43: Physical health rating (0 = Poor … 4 = Excellent)
        QuestionnaireSpec("BW_PhysHealth", "bw_physh", 1, 5,
            [LatentLoading("physical_health", 4.0)], 0.7, 0.01),

        # Q44: Sleep adequacy (0 = No, 1 = Yes)
        QuestionnaireSpec("BW_Sleep", "bw_sleep", 1, 2,
            [LatentLoading("physical_health", 1.0)], 0.4, 0.01),

        # Q45: Physical activity days/week (0–7)
        QuestionnaireSpec("BW_PhysActDays", "bw_physact", 1, 8,
            [LatentLoading("physical_health", 6.0)], 1.0, 0.01),

        # Q46: Physical activity duration (0 = ~0.5 h … 3 = ~2 h+)
        QuestionnaireSpec("BW_PhysActDur", "bw_physdur", 1, 4,
            [LatentLoading("physical_health", 3.0)], 0.6, 0.01),

        # Q47: Fruit & veg portions/day (0–5)
        QuestionnaireSpec("BW_FruitVeg", "bw_fruitveg", 1, 6,
            [LatentLoading("physical_health", 4.0)], 0.8, 0.01),

        # Q48–51: Unhealthy food/drink frequency (4 items, 0–7 days)
        QuestionnaireSpec("BW_UnhealthyFood", "bw_unhealthy", 4, 8,
            [LatentLoading("unhealthy_diet", 7.0)], 1.0, 0.01),

        # Q52: Free time adequacy (0 = almost never … 4 = almost always)
        QuestionnaireSpec("BW_FreeTime", "bw_freetime", 1, 5,
            [LatentLoading("wellbeing", 3.5), LatentLoading("socioeconomic", 1.0)], 0.7, 0.01),

        # Q53: Social media time/day (0 = none … 7 = 7 h+)
        QuestionnaireSpec("BW_SocialMediaTime", "bw_socmedia", 1, 8,
            [LatentLoading("screen_time", 7.0)], 1.2, 0.01),

        # Q54–55: Social media type — active (item 1) vs passive (item 2)
        # 0 = none of the time … 4 = all of the time
        QuestionnaireSpec("BW_SocialMediaType", "bw_socmtype", 2, 5,
            [LatentLoading("social_connection", Dict("1" => 3.0)),
             LatentLoading("screen_time",       Dict("2" => 3.0))], 1.0, 0.02),

        # Q56: Volunteering frequency (0 = never … 5 = most days)
        QuestionnaireSpec("BW_Volunteering", "bw_volunteer", 1, 6,
            [LatentLoading("social_connection", 3.0), LatentLoading("future_optimism", 1.5)], 0.8, 0.01),

        # Q57–67: Activity participation — 11 activities (0 = never … 5 = most days)
        QuestionnaireSpec("BW_Activities", "bw_activ", 11, 6,
            [LatentLoading("wellbeing", 3.0), LatentLoading("physical_health", 2.0)], 0.8, 0.01),

        # Q68: School belonging/connection (0 = not at all … 4 = a lot)
        QuestionnaireSpec("BW_SchoolConnection", "bw_schoolconn", 1, 5,
            [LatentLoading("social_connection", 3.5), LatentLoading("wellbeing", 1.5)], 0.7, 0.01),

        # Q69: Attainment happiness (0–10)
        QuestionnaireSpec("BW_Attainment", "bw_attain", 1, 11,
            [LatentLoading("wellbeing", 6.0), LatentLoading("social_connection", 3.0),
             LatentLoading("depression", -3.0)], 1.5, 0.01),

        # Q70–73: Relationships with school staff (4 items, 0 = never … 4 = always)
        QuestionnaireSpec("BW_StaffRelationships", "bw_staffrel", 4, 5,
            [LatentLoading("social_connection", 4.0)], 0.7, 0.01),

        # Q74: Placed in school isolation (0 = No, 1 = Yes)
        QuestionnaireSpec("BW_Isolation", "bw_iso", 1, 2,
            [LatentLoading("behaviour", 1.0)], 0.4, 0.01),

        # Q75: Days/week in isolation (0 = 1 day … 4 = 5 days; conditional on Q74)
        QuestionnaireSpec("BW_IsolationDays", "bw_isodays", 1, 5,
            [LatentLoading("behaviour", 4.0)], 0.8, 0.01),

        # Q76: Isolation duration (0 = 0.5 h … 12 = 6.5 h; conditional on Q74)
        QuestionnaireSpec("BW_IsolationDur", "bw_isodur", 1, 13,
            [LatentLoading("behaviour", 8.0)], 1.5, 0.01),

        # Q77: School pressure (0 = not at all … 3 = a lot)
        QuestionnaireSpec("BW_SchoolPressure", "bw_schpress", 1, 4,
            [LatentLoading("anxiety", 3.0)], 0.6, 0.01),

        # Q78: Home environment happiness (0–10)
        QuestionnaireSpec("BW_HomeEnvironment", "bw_homeenv", 1, 11,
            [LatentLoading("socioeconomic", 9.0), LatentLoading("wellbeing", 2.0)], 1.5, 0.01),

        # Q79: Safety in local area (0 = very unsafe … 3 = very safe, 4 = don't know)
        QuestionnaireSpec("BW_Safety", "bw_safety", 1, 5,
            [LatentLoading("socioeconomic", 3.0)], 0.8, 0.02),

        # Q80–83: Local environment quality (4 items; 0 = strongly disagree … 4 = strongly agree)
        QuestionnaireSpec("BW_LocalEnvironment", "bw_localenv", 4, 5,
            [LatentLoading("socioeconomic", 3.5), LatentLoading("social_connection", 1.5)], 0.7, 0.01),

        # Q84: Being heard outside school/home (0 = never … 4 = always)
        QuestionnaireSpec("BW_BeingHeard", "bw_beinheard", 1, 5,
            [LatentLoading("social_connection", 3.5)], 0.8, 0.01),

        # Q85: Food security — higher = MORE insecure (0 = most days … 5 = never)
        # Note: loaded negatively on socioeconomic so low SES → high insecurity score
        QuestionnaireSpec("BW_FoodSecurity", "bw_foodsec", 1, 6,
            [LatentLoading("socioeconomic", -5.0)], 0.6, 0.01),

        # Q86: Material/possessions happiness (0–10)
        QuestionnaireSpec("BW_Material", "bw_material", 1, 11,
            [LatentLoading("socioeconomic", 9.0), LatentLoading("wellbeing", 1.5)], 1.5, 0.01),

        # Q87–93: Future readiness (7 items; 0 = strongly disagree … 3 = strongly agree)
        # Q88–93 are not shown to Year 7 pupils; modelled for all years here.
        QuestionnaireSpec("BW_Future", "bw_future", 7, 4,
            [LatentLoading("future_optimism", 3.0), LatentLoading("wellbeing", 0.5),
             LatentLoading("depression", -1.0)], 0.6, 0.01),

        # Q94: Careers education received — count of types (0–8; Year 10 only)
        QuestionnaireSpec("BW_CareersEdReceived", "bw_careersed", 1, 9,
            [LatentLoading("future_optimism", 5.0)], 1.0, 0.05),

        # Q95: Careers education helpfulness (0 = not at all … 3 = very helpful; Year 10)
        QuestionnaireSpec("BW_CareersEdHelpfulness", "bw_careershlp", 1, 4,
            [LatentLoading("future_optimism", 3.0)], 0.6, 0.05),

        # Q96a–h: Post-Year-11 plans (8 binary items; Year 10 only)
        QuestionnaireSpec("BW_Plans", "bw_plans", 8, 2,
            [LatentLoading("future_optimism", 1.0)], 0.4, 0.05),

        # Q97–98: GMACS awareness and usage (2 binary items; Year 10 only)
        QuestionnaireSpec("BW_GMACS", "bw_gmacs", 2, 2,
            [LatentLoading("future_optimism", 0.6)], 0.4, 0.05),

        # Q99–102: Relationships with parents/carers (4 items, 0 = never … 4 = always)
        QuestionnaireSpec("BW_ParentRelationships", "bw_parentsrel", 4, 5,
            [LatentLoading("social_connection", 4.0)], 0.7, 0.01),

        # Q103–106: Friendships and social support (4 items, 0 = not at all … 4 = a lot)
        QuestionnaireSpec("BW_Friendships", "bw_friends", 4, 5,
            [LatentLoading("social_connection", 3.5), LatentLoading("depression", -1.5)], 0.7, 0.01),

        # Q107: Loneliness (0 = often/always lonely … 4 = never lonely)
        QuestionnaireSpec("BW_Loneliness", "bw_lonely", 1, 5,
            [LatentLoading("social_connection", 4.0), LatentLoading("depression", -2.0)], 0.7, 0.01),

        # Q108–112: Discrimination by 5 characteristics (5 items; 0 = never … 4 = often/always)
        QuestionnaireSpec("BW_Discrimination", "bw_discrim", 5, 5,
            [LatentLoading("discrimination", 4.0)], 0.6, 0.01),

        # Q113a–g: Where discrimination occurred (7 binary items; 0 = No, 1 = Yes)
        QuestionnaireSpec("BW_DiscrimLocation", "bw_discloc", 7, 2,
            [LatentLoading("discrimination", 1.0)], 0.4, 0.01),

        # Q114–116: Bullying — physical, social, cyber (3 items; 0 = not bullied … 3 = a lot)
        QuestionnaireSpec("BW_Bullying", "bw_bullying", 3, 4,
            [LatentLoading("victimization", 3.0), LatentLoading("social_connection", -0.5)], 0.5, 0.01),

        # Q117: Access to wellbeing support (0 = disagree a lot … 4 = agree a lot)
        QuestionnaireSpec("BW_SupportAccess", "bw_support", 1, 5,
            [LatentLoading("social_connection", 3.5), LatentLoading("socioeconomic", 1.0)], 0.7, 0.01),

        # Q118–123: Mental health contact by 6 source types (0 = never … 4 = always)
        QuestionnaireSpec("BW_MentalHealthContact", "bw_mhcontact", 6, 5,
            [LatentLoading("depression", 2.0), LatentLoading("anxiety", 1.5),
             LatentLoading("social_connection", 0.5)], 0.7, 0.01),

        # Q124: Kooth usage (0 = never heard … 3 = used and helpful)
        QuestionnaireSpec("BW_Kooth", "bw_kooth", 1, 4,
            [LatentLoading("depression", 1.5), LatentLoading("anxiety", 1.0),
             LatentLoading("social_connection", 0.5)], 0.6, 0.01),
    ]
end

"""
    generate_questionnaire_responses(rng, spec, latents, prev_responses) -> DataRow

Generate item responses for one student at one wave.

The mean item score is derived from the weighted sum of latent variable contributions
(`loadings`). A noise sample from a truncated Normal is added; the result is rounded
and clamped to `[0, nLevels - 1]`.

Each `LatentLoading` may specify a uniform scale (applied to all items) or a per-item
scale dict (keyed by item index strings `"1"`, `"2"`, …).

Longitudinal continuity: each item has a 75% chance (given a previous response exists)
of blending the latent-derived mean with the previous wave's score, producing realistic
stability over time.

`spoilRate` fraction of responses are entirely random (simulating disengaged students).
"""
function generate_questionnaire_responses(
    rng::AbstractRNG,
    spec::QuestionnaireSpec,
    latents::Dict{String,Float64},
    prev_responses::Union{DataRow,Nothing},
)::DataRow
    result = DataRow()

    # Spoiled / random response
    if rand(rng) < spec.spoilRate
        for i in 1:spec.nItems
            result["$(spec.prefix)_$i"] = rand(rng, 0:(spec.nLevels - 1))
        end
        return result
    end

    lo = -0.5
    hi = Float64(spec.nLevels) - 0.5

    # Pre-compute per-item scale vectors to avoid repeated type checks inside the loop.
    # Each entry is a Vector{Float64} of length nItems.
    item_scales = [
        if l.scale isa Float64
            fill(l.scale, spec.nItems)
        else
            Float64[get(l.scale, string(i), 0.0) for i in 1:spec.nItems]
        end
        for l in spec.loadings
    ]
    latent_values = Float64[get(latents, l.latentName, 0.0) for l in spec.loadings]

    for i in 1:spec.nItems
        key = "$(spec.prefix)_$i"

        # 5% chance of a random item
        if rand(rng) < 0.05
            result[key] = rand(rng, 0:(spec.nLevels - 1))
            continue
        end

        # Latent-derived mean score for this item
        latent_mean = clamp(
            sum(item_scales[j][i] * latent_values[j] for j in eachindex(spec.loadings); init = 0.0),
            0.0,
            Float64(spec.nLevels - 1),
        )

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
