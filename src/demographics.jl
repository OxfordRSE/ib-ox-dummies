# UK-typical name lists used for generating realistic student names
const FIRST_NAMES_MALE = [
    "James", "Oliver", "Harry", "Jack", "George", "Noah", "Charlie", "Jacob",
    "Alfie", "Freddie", "Oscar", "Archie", "Henry", "Leo", "William", "Thomas",
    "Ethan", "Joshua", "Mohammed", "Liam", "Lucas", "Mason", "Elijah", "Daniel",
    "Rayan", "Adam", "Max", "Samuel", "Logan", "Dylan",
]

const FIRST_NAMES_FEMALE = [
    "Olivia", "Amelia", "Isla", "Ava", "Emily", "Isabella", "Mia", "Poppy",
    "Ella", "Lily", "Jessica", "Sophie", "Grace", "Evie", "Florence", "Alice",
    "Freya", "Charlotte", "Daisy", "Sophia", "Layla", "Ruby", "Sienna", "Zara",
    "Ellie", "Millie", "Phoebe", "Evelyn", "Hannah", "Rosie",
]

const FIRST_NAMES_NB = [
    "Alex", "Jordan", "Morgan", "Robin", "Taylor", "Casey", "Riley", "Avery",
    "Quinn", "Cameron", "Jesse", "Skyler", "Drew", "Frankie", "Jamie",
]

const LAST_NAMES = [
    "Smith", "Jones", "Williams", "Taylor", "Brown", "Davies", "Evans", "Wilson",
    "Thomas", "Roberts", "Johnson", "Lewis", "Walker", "Robinson", "Wood",
    "Thompson", "White", "Watson", "Jackson", "Wright", "Green", "Harris",
    "Cooper", "King", "Lee", "Martin", "Clarke", "James", "Morgan", "Hughes",
    "Edwards", "Hill", "Moore", "Clark", "Harrison", "Scott", "Young", "Morris",
    "Hall", "Ward", "Turner", "Carter", "Phillips", "Mitchell", "Patel", "Khan",
    "Ali", "Ahmed", "Singh", "Kumar",
]

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
    weighted_sample(rng, options)

Sample one item from a weighted list of `(value, weight)` pairs using
`StatsBase.sample` with `Weights`.
"""
function weighted_sample(rng::AbstractRNG, options::Vector{Tuple{String,Float64}})
    return sample(rng, first.(options), Weights(last.(options)))
end

"""
    sample_count(rng, spec) -> Int

Sample a positive integer count from a `CountSpec`.
- `Int`: return the value directly (must be ≥ 1).
- `Range`: sample uniformly from [min, max].
- `UnivariateDistribution`: draw a sample, round to nearest integer, clamp to ≥ 1.
"""
function sample_count(rng::AbstractRNG, spec::CountSpec)::Int
    if spec isa Int
        spec >= 1 || throw(ArgumentError("Count must be ≥ 1, got $spec"))
        return spec
    elseif spec isa Range
        return rand(rng, spec.min:spec.max)
    else  # UnivariateDistribution
        return max(1, round(Int, rand(rng, spec)))
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
    generate_name(rng, sex) -> String

Generate a plausible full name given the sex code ("M", "F", or other).
"""
function generate_name(rng::AbstractRNG, sex::String)::String
    first = if sex == "M"
        FIRST_NAMES_MALE[rand(rng, 1:length(FIRST_NAMES_MALE))]
    elseif sex == "F"
        FIRST_NAMES_FEMALE[rand(rng, 1:length(FIRST_NAMES_FEMALE))]
    else
        FIRST_NAMES_NB[rand(rng, 1:length(FIRST_NAMES_NB))]
    end
    last = LAST_NAMES[rand(rng, 1:length(LAST_NAMES))]
    return "$first $last"
end

"""
    generate_school_name(rng, idx) -> String

Generate a plausible UK school name.
"""
function generate_school_name(rng::AbstractRNG, idx::Int)::String
    places = [
        "Islington", "Hackney", "Newham", "Southwark", "Lambeth",
        "Camden", "Greenwich", "Lewisham", "Haringey", "Wandsworth",
        "Oxfordshire", "Bristol", "Manchester", "Leeds", "Sheffield",
        "Birmingham", "Liverpool", "Nottingham", "Leicester", "Newcastle",
    ]
    types = [
        "High School", "Grammar School", "Academy", "College",
        "Community School", "Free School", "Academy of Arts",
        "Secondary School", "Comprehensive", "Studio School",
    ]
    place = places[mod1(idx, length(places))]
    typ   = types[rand(rng, 1:length(types))]
    return "$place $typ"
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
    generate_demographics(rng, school_name, yeargroup, school_year, class_label, uid; ...) -> QData

Generate initial demographics for one student.

Keyword arguments allow passing school-specific perturbed weight distributions
(from `perturb_weights`) to introduce realistic inter-school variation.
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
)::QData
    sex = weighted_sample(rng, sex_weights)
    name = generate_name(rng, sex)
    age = 9 + school_year  # approximate age from school year
    ethnicity = weighted_sample(rng, ethnicity_weights)
    sexual_orientation = weighted_sample(rng, orientation_weights)
    gender_identity = weighted_sample(rng, gender_weights)

    return QData(
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
end

"""
    default_demographics_update(rng, prevData) -> QData

Default demographics update function. Increments age by 1 and otherwise
copies forward all demographics from the most recent wave.
"""
function default_demographics_update(rng::AbstractRNG, prevData::Vector{QData})::QData
    isempty(prevData) && return QData()
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
    default_naughty_monkey(rng, output, schema) -> Vector{QData}

Default naughty-monkey function: removes ~0.25% of questionnaire data cells
and ~5% of demographics data cells at random by replacing them with `missing`.
"""
function default_naughty_monkey(
    rng::AbstractRNG,
    output::Vector{QData},
    schema::Schema,
)::Vector{QData}
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
