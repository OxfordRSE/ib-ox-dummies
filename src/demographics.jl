# UK 2021 Census approximate ethnicity distribution (simplified)
const ETHNICITY_WEIGHTS = [
    ("White British",          0.748),
    ("White Irish",            0.010),
    ("White Other",            0.053),
    ("Mixed White/Black Caribbean", 0.008),
    ("Mixed White/Black African",   0.004),
    ("Mixed White/Asian",      0.009),
    ("Mixed Other",            0.006),
    ("Asian British Indian",   0.026),
    ("Asian British Pakistani",0.021),
    ("Asian British Bangladeshi", 0.009),
    ("Asian British Chinese",  0.007),
    ("Asian Other",            0.016),
    ("Black British African",  0.024),
    ("Black British Caribbean",0.010),
    ("Black Other",            0.004),
    ("Arab",                   0.006),
    ("Other",                  0.006),
    ("Prefer not to say",      0.013),
]

const SEXUAL_ORIENTATION_WEIGHTS = [
    ("Heterosexual/Straight", 0.920),
    ("Bisexual",              0.030),
    ("Gay/Lesbian",           0.020),
    ("Pansexual",             0.010),
    ("Asexual",               0.005),
    ("Other",                 0.005),
    ("Prefer not to say",     0.010),
]

const GENDER_IDENTITY_WEIGHTS = [
    ("Cis",              0.960),
    ("Trans",            0.010),
    ("Non-binary",       0.015),
    ("Gender fluid",     0.005),
    ("Other",            0.003),
    ("Prefer not to say",0.007),
]

const SEX_WEIGHTS = [
    ("M", 0.498),
    ("F", 0.498),
    ("I", 0.004),  # Intersex
]

"""
    default_demographics_spec() -> DemographicsSpec

Return a `DemographicsSpec` built from UK 2021 Census approximate distributions.
"""
function default_demographics_spec()::DemographicsSpec
    return DemographicsSpec(
        ethnicity          = ETHNICITY_WEIGHTS,
        sex                = SEX_WEIGHTS,
        genderIdentity     = GENDER_IDENTITY_WEIGHTS,
        sexualOrientation  = SEXUAL_ORIENTATION_WEIGHTS,
    )
end

"""
    weighted_sample(rng, options)

Sample one item from a weighted list of `(value, weight)` pairs using
`StatsBase.sample` with `Weights`.
"""
function weighted_sample(rng::AbstractRNG, options::Vector{Tuple{String,Float64}})
    return sample(rng, first.(options), Weights(last.(options)))
end

"""
    sample_count(rng, spec) -> Int

Sample a positive integer count from a `SamplerSpec`.
- `Int`: return the value directly (must be ≥ 1).
- `Range`: sample uniformly from [min, max].
- `UnivariateDistribution`: draw a sample, round to nearest integer, clamp to ≥ 1.
- `Function`: call `spec(rng)` where `spec` has signature `(rng::AbstractRNG) -> Number`,
  round to nearest integer, clamp to ≥ 1.
"""
function sample_count(rng::AbstractRNG, spec::SamplerSpec)::Int
    if spec isa Int
        spec >= 1 || throw(ArgumentError("Count must be ≥ 1, got $spec"))
        return spec
    elseif spec isa Range
        return rand(rng, spec.min:spec.max)
    elseif spec isa Function
        return max(1, round(Int, spec(rng)))
    else  # UnivariateDistribution
        return max(1, round(Int, rand(rng, spec)))
    end
end

"""
    draw_sampler(rng, spec) -> Float64

Draw a `Float64` sample from a `SamplerSpec`.
- `Int`: return as `Float64` (fixed value).
- `Range`: sample uniformly from [min, max] and return as `Float64`.
- `UnivariateDistribution`: draw a sample via `rand(rng, spec)`.
- `Function`: call `spec(rng)` where `spec` has signature `(rng::AbstractRNG) -> Number`.
"""
function draw_sampler(rng::AbstractRNG, spec::SamplerSpec)::Float64
    if spec isa Int
        return Float64(spec)
    elseif spec isa Range
        return Float64(rand(rng, spec.min:spec.max))
    elseif spec isa Function
        return Float64(spec(rng))
    else  # UnivariateDistribution
        return Float64(rand(rng, spec))
    end
end

"""
    generate_uid(rng) -> String

Generate a short URL-safe unique identifier for a student.
"""
function generate_uid(rng::AbstractRNG)::String
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    return String([chars[rand(rng, 1:length(chars))] for _ in 1:9])
end

"""
    generate_name(sex) -> String

Generate a plausible full name given the sex code ("M", "F", or other) using Faker.jl.
"""
function generate_name(sex::String)::String
    first = if sex == "M"
        Faker.first_name("M")
    elseif sex == "F"
        Faker.first_name("F")
    else
        Faker.first_name()
    end
    return "$first $(Faker.last_name())"
end

"""
    generate_school_name(idx) -> String

Generate a school name using a Faker city name combined with a deterministic
school type (cycled by `idx` so consecutive schools vary).
"""
function generate_school_name(idx::Int)::String
    types = [
        "High School", "Grammar School", "Academy", "College",
        "Community School", "Free School", "Academy of Arts",
        "Secondary School", "Comprehensive", "Studio School",
    ]
    city = Faker.city()                     # display name — not used as a group key
    typ  = types[mod1(idx, length(types))]  # deterministic type cycle
    return "$city $typ"
end

"""
    generate_class_label(rng, yeargroup, class_idx) -> String

Generate a class label such as "2b".
"""
function generate_class_label(rng::AbstractRNG, yeargroup::Int, class_idx::Int)::String
    return "$(yeargroup)$(('a' + class_idx - 1))"
end

"""
    perturb_weights(rng, weights, sd) -> Vector{Tuple{String,Float64}}

Return a perturbed copy of a categorical weight vector.

Each weight `w` is sampled from `Normal(w, sd * max(w, 0.01))`, clamped to ≥ 0,
and the result is renormalised to sum to 1. When `sd == 0` the original weights
are returned unchanged. Falls back to the original weights on degenerate inputs.

Used to give each school a slightly different demographic composition.
"""
function perturb_weights(
    rng::AbstractRNG,
    weights::Vector{Tuple{String,Float64}},
    sd::Float64,
)::Vector{Tuple{String,Float64}}
    sd == 0.0 && return weights
    raw = [max(0.0, rand(rng, Normal(w, sd * max(w, 0.01)))) for (_, w) in weights]
    total = sum(raw)
    total <= 0.0 && return weights  # fallback on degenerate case
    return [(cat, raw[i] / total) for (i, (cat, _)) in enumerate(weights)]
end

"""
    generate_demographics(rng, school_name, yeargroup, school_year, class_label, uid; ...) -> DataRow

Generate initial demographics for one student.

Keyword arguments allow passing school-specific perturbed weight distributions
(from `perturb_weights`) to introduce realistic inter-school variation.
`custom_fields` is a `Dict{String,Function}` of zero-argument field generators
(e.g. from `DemographicsSpec.customFields`) added verbatim to the row.
"""
function generate_demographics(
    rng::AbstractRNG,
    school_name::String,
    yeargroup::Int,
    school_year::Int,
    class_label::String,
    uid::String;
    ethnicity_weights::Vector{Tuple{String,Float64}}   = ETHNICITY_WEIGHTS,
    sex_weights::Vector{Tuple{String,Float64}}          = SEX_WEIGHTS,
    gender_weights::Vector{Tuple{String,Float64}}       = GENDER_IDENTITY_WEIGHTS,
    orientation_weights::Vector{Tuple{String,Float64}}  = SEXUAL_ORIENTATION_WEIGHTS,
    custom_fields::Dict{String,Function}                = Dict{String,Function}(),
)::DataRow
    sex = weighted_sample(rng, sex_weights)
    name = generate_name(sex)
    age = 9 + school_year  # approximate age from school year
    ethnicity = weighted_sample(rng, ethnicity_weights)
    sexual_orientation = weighted_sample(rng, orientation_weights)
    gender_identity = weighted_sample(rng, gender_weights)

    row = DataRow(
        "uid"          => uid,
        "name"         => name,
        "school"       => school_name,
        "yearGroup"    => yeargroup,
        "schoolYear"   => school_year,
        "class"        => class_label,
        "d_age"        => age,
        "d_sex"        => sex,
        "d_ethnicity"  => ethnicity,
        "d_sexualOrientation" => sexual_orientation,
        "d_genderIdentity"    => gender_identity,
    )

    for (col, fn) in custom_fields
        row[col] = fn()
    end

    return row
end

"""
    default_demographics_update(rng, prevData) -> DataRow

Default demographics update function. Increments age by 1 and otherwise
copies forward all demographics from the most recent wave.
"""
function default_demographics_update(rng::AbstractRNG, prevData::Vector{DataRow})::DataRow
    isempty(prevData) && return DataRow()
    latest = prevData[end]
    updated = copy(latest)

    # Increment age
    if haskey(latest, "d_age") && latest["d_age"] isa Int
        updated["d_age"] = (latest["d_age"]::Int) + 1
    elseif haskey(latest, "schoolYear") && latest["schoolYear"] isa Int
        updated["d_age"] = 9 + (latest["schoolYear"]::Int) + length(prevData)
    end

    return updated
end

"""
    default_naughty_monkey(rng, output, schema) -> Vector{DataRow}

Default naughty-monkey function: removes ~0.25% of questionnaire data cells
and ~5% of demographics data cells at random by replacing them with `missing`.
"""
function default_naughty_monkey(
    rng::AbstractRNG,
    output::Vector{DataRow},
    schema::Schema,
)::Vector{DataRow}
    result = deepcopy(output)
    q_cols = collect(keys(schema.questionnaireColumns))
    d_cols = schema.demographicsColumns

    for row in result
        for col in q_cols
            if haskey(row, col) && !ismissing(row[col]) && rand(rng) < 0.0025
                row[col] = missing
            end
        end
        for col in d_cols
            if haskey(row, col) && !ismissing(row[col]) && rand(rng) < 0.05
                row[col] = missing
            end
        end
    end
    return result
end
