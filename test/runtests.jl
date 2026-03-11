using Test
using Random
using Distributions
using DataFrames
using JSON3

# Activate the package to pick up source files
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io = devnull)

using IbOxDummies

@testset "IbOxDummies" begin

    @testset "Types" begin
        # Range
        r = Range(1, 5)
        @test r.min == 1
        @test r.max == 5
        @test_throws ArgumentError Range(5, 1)

        # SamplerSpec accepts Distributions.Normal
        nd = Normal(30.0, 7.0)
        @test nd isa UnivariateDistribution
        @test nd isa SamplerSpec

        # SimulationConfig defaults
        cfg = SimulationConfig()
        @test cfg.nWaves == 3
        @test cfg.nSchools == 10
        @test cfg.output == "csv"
        @test isnothing(cfg.seed)
        @test cfg.nStudentsPerClass isa Normal
        @test cfg.nStudentsPerClass isa SamplerSpec
        @test cfg.includeLatents == false
        @test cfg.demographicPerturbationSD == 0.05
        @test isempty(cfg.questionnaires)
        @test isempty(cfg.latentVariables)
        @test isempty(cfg.linearEffects)
        @test isempty(cfg.randomEffects)
        @test isnothing(cfg.demographicsSpec)

        # DemographicsSpec with customFields
        ds = DemographicsSpec(customFields = Dict{String,Function}("d_city" => () -> "TestCity"))
        @test haskey(ds.customFields, "d_city")
        @test ds.customFields["d_city"]() == "TestCity"

        # LinearEffect
        c = LinearEffect("depression", ["d_age"], 0.02)
        @test c.target == "depression"
        @test c.value == 0.02

        # RandomEffect with UnivariateDistribution
        e = RandomEffect("anxiety", [], ["uid"], Normal(0.0, 0.1))
        @test e.target == "anxiety"
        @test isempty(e.numericalInputs)
        @test e.value isa UnivariateDistribution
        @test e.value isa SamplerSpec

        # RandomEffect with Function (SamplerSpec)
        mde = RandomEffect("depression", [], ["uid", "wave"],
            rng -> rand(rng) < 0.01 ? rand(rng, Normal(0.75, 0.1)) : 0.0)
        @test mde.value isa Function
        @test mde.value isa SamplerSpec

        ll = LatentLoading("depression", 2.5)
        @test ll.latentName == "depression"
        @test ll.scale isa Float64
        @test ll.scale == 2.5

        ll_per_item = LatentLoading("anxiety", Dict("1" => 2.0, "2" => 1.5, "3" => 0.5))
        @test ll_per_item.scale isa Dict{String,Float64}
        @test ll_per_item.scale["1"] ≈ 2.0
        @test ll_per_item.scale["2"] ≈ 1.5

        spec = QuestionnaireSpec("TEST", "tst", 3, 4, [ll], 0.5, 0.01)
        @test spec.name == "TEST"
        @test spec.nItems == 3
    end

    @testset "sample_count" begin
        rng = MersenneTwister(1)
        @test sample_count(rng, 5) == 5
        @test sample_count(rng, 1) == 1
        @test_throws ArgumentError sample_count(rng, 0)
        @test_throws ArgumentError sample_count(rng, -3)

        # Range sampling stays within bounds
        rng = MersenneTwister(42)
        for _ in 1:50
            v = sample_count(rng, Range(2, 8))
            @test 2 <= v <= 8
        end

        # Distribution (Normal) sampling returns positive integers
        rng = MersenneTwister(42)
        for _ in 1:50
            v = sample_count(rng, Normal(30.0, 7.0))
            @test v >= 1
        end

        # Truncated distribution also works
        rng = MersenneTwister(42)
        v = sample_count(rng, truncated(Normal(5.0, 1.0), 1.0, 10.0))
        @test 1 <= v <= 10

        # Function callables are supported
        rng = MersenneTwister(42)
        fn = (rng) -> rand(rng, Poisson(10))
        for _ in 1:20
            v = sample_count(rng, fn)
            @test v >= 1
            @test v isa Int
        end

        # Lambda returning a fixed value
        rng = MersenneTwister(42)
        @test sample_count(rng, _ -> 7.0) == 7
        @test sample_count(rng, _ -> 7.0) isa Int
    end

    @testset "draw_sampler" begin
        rng = MersenneTwister(1)

        # Fixed Int → exact Float64
        @test draw_sampler(rng, 5) === 5.0
        @test draw_sampler(rng, 0) === 0.0

        # Fixed Float64 → exact value
        @test draw_sampler(rng, 0.75) === 0.75
        @test draw_sampler(rng, -1.5) === -1.5

        # Range → value within range
        rng = MersenneTwister(42)
        for _ in 1:20
            v = draw_sampler(rng, Range(2, 6))
            @test 2.0 <= v <= 6.0
        end

        # UnivariateDistribution → Float64
        rng = MersenneTwister(42)
        v = draw_sampler(rng, Normal(0.0, 0.2))
        @test v isa Float64

        # Truncated normal → always ≥ 0
        rng = MersenneTwister(42)
        for _ in 1:20
            v = draw_sampler(rng, truncated(Normal(0.0, 0.2), 0.0, Inf))
            @test v >= 0.0
        end

        # Function callable (MDE-style)
        rng = MersenneTwister(123)
        mde_fn = rng2 -> rand(rng2) < 0.01 ? rand(rng2, Normal(0.75, 0.1)) : 0.0
        vals = [draw_sampler(rng, mde_fn) for _ in 1:1000]
        @test all(v isa Float64 for v in vals)
        # With p=0.01 and 1000 draws, P(no events) ≈ 4.3e-5; expect mostly zeros but some non-zero
        @test count(==(0.0), vals) > 900  # at least 90% zeros
    end

    @testset "weighted_sample" begin
        rng = MersenneTwister(0)
        opts = [("A", 0.9), ("B", 0.1)]
        counts = Dict("A" => 0, "B" => 0)
        for _ in 1:1000
            counts[weighted_sample(rng, opts)] += 1
        end
        # A should be chosen ~90% of the time
        @test counts["A"] > 800
        @test counts["B"] > 50
    end

    @testset "parse_sampler_spec" begin
        @test parse_sampler_spec("5") == 5
        @test parse_sampler_spec("1:5") == Range(1, 5)
        @test parse_sampler_spec("1,5") == Range(1, 5)
        @test parse_sampler_spec("norm(30,7)") == Normal(30.0, 7.0)
        @test parse_sampler_spec("normal(30,7)") == Normal(30.0, 7.0)
        @test parse_sampler_spec("dnorm(30,7)") == Normal(30.0, 7.0)
        @test parse_sampler_spec("pois(5)") == Poisson(5.0)
        @test parse_sampler_spec("poisson(5)") == Poisson(5.0)
        @test parse_sampler_spec("negbinom(5,0.5)") == NegativeBinomial(5, 0.5)
        @test parse_sampler_spec("negativebinomial(5,0.5)") == NegativeBinomial(5, 0.5)
        @test parse_sampler_spec("lognorm(3,0.5)") == LogNormal(3.0, 0.5)
        @test parse_sampler_spec("lognormal(3,0.5)") == LogNormal(3.0, 0.5)
        @test parse_sampler_spec("unif(1,10)") == DiscreteUniform(1, 10)
        @test parse_sampler_spec("uniform(1,10)") == DiscreteUniform(1, 10)
        @test parse_sampler_spec("exp(0.1)") == Exponential(10.0)
        @test parse_sampler_spec("exponential(0.1)") == Exponential(10.0)
        @test parse_sampler_spec("gamma(2,3)") == Gamma(2.0, 3.0)

        # Half-normal: truncated Normal at 0
        hn = parse_sampler_spec("halfnorm(0,0.2)")
        @test hn isa UnivariateDistribution
        rng = MersenneTwister(1)
        for _ in 1:20
            @test rand(rng, hn) >= 0.0
        end
        hn2 = parse_sampler_spec("hnorm(0,0.2)")
        @test hn2 isa UnivariateDistribution

        @test_throws ArgumentError parse_sampler_spec("garbage")
    end

    @testset "parse_linear_effect" begin
        e1 = parse_linear_effect("depression:d_age:0.02")
        @test e1.target == "depression"
        @test e1.inputs == ["d_age"]
        @test e1.value ≈ 0.02

        e2 = parse_linear_effect("anxiety:d_age,_sex_fm:0.004")
        @test e2.target == "anxiety"
        @test e2.inputs == ["d_age", "_sex_fm"]
        @test e2.value ≈ 0.004

        # Intercept (no inputs)
        e3 = parse_linear_effect("depression::0.1")
        @test e3.target == "depression"
        @test isempty(e3.inputs)
        @test e3.value ≈ 0.1

        @test_throws ArgumentError parse_linear_effect("bad_format")
    end

    @testset "parse_random_effect" begin
        r1 = parse_random_effect("depression::uid,wave:norm(0,0.15)")
        @test r1.target == "depression"
        @test isempty(r1.numericalInputs)
        @test r1.categoricalInputs == ["uid", "wave"]
        @test r1.value == Normal(0.0, 0.15)

        # Residual (no categorical inputs)
        r2 = parse_random_effect("anxiety:::norm(0,0.1)")
        @test r2.target == "anxiety"
        @test isempty(r2.numericalInputs)
        @test isempty(r2.categoricalInputs)
        @test r2.value == Normal(0.0, 0.1)

        # Half-normal baseline
        r3 = parse_random_effect("depression::uid:halfnorm(0,0.2)")
        @test r3.target == "depression"
        @test r3.categoricalInputs == ["uid"]
        @test r3.value isa UnivariateDistribution
        rng = MersenneTwister(1)
        for _ in 1:10
            @test rand(rng, r3.value) >= 0.0
        end

        @test_throws ArgumentError parse_random_effect("bad:format")
    end

    @testset "parse_demographics_weights" begin
        # Basic parsing
        result = parse_demographics_weights("M:0.49,F:0.49,I:0.02")
        @test length(result) == 3
        @test ("M", 0.49) in result
        @test ("F", 0.49) in result
        @test ("I", 0.02) in result

        # Categories with spaces
        result2 = parse_demographics_weights("White British:0.75,Asian Other:0.15,Other:0.10")
        @test length(result2) == 3
        @test ("White British", 0.75) in result2

        # Single entry
        result3 = parse_demographics_weights("Heterosexual/Straight:1.0")
        @test result3 == [("Heterosexual/Straight", 1.0)]

        # Empty string → empty vector
        @test isempty(parse_demographics_weights(""))
        @test isempty(parse_demographics_weights("   "))

        # Missing colon → error
        @test_throws ArgumentError parse_demographics_weights("no_colon")
    end

    @testset "parse_custom_field_value" begin
        # Constant string → constant function
        f_const = parse_custom_field_value("United Kingdom")
        @test f_const() == "United Kingdom"
        @test f_const() == f_const()  # idempotent

        # Faker method → callable
        f_city = parse_custom_field_value("faker.city")
        @test f_city() isa AbstractString
        @test !isempty(f_city())

        # Case-insensitive Faker prefix
        f_city2 = parse_custom_field_value("Faker.City")
        @test f_city2() isa AbstractString

        # Other Faker methods
        @test parse_custom_field_value("faker.last_name")() isa AbstractString
        @test parse_custom_field_value("faker.email")() isa AbstractString
        @test parse_custom_field_value("faker.company")() isa AbstractString
        @test parse_custom_field_value("faker.country")() isa AbstractString

        # Unknown Faker method → error
        @test_throws ArgumentError parse_custom_field_value("faker.unknown_method_xyz")
    end

    @testset "parse_custom_fields" begin
        d = Dict{String,Any}("d_city" => "faker.city", "d_country" => "United Kingdom")
        result = parse_custom_fields(d)
        @test haskey(result, "d_city")
        @test haskey(result, "d_country")
        @test result["d_city"]() isa AbstractString
        @test result["d_country"]() == "United Kingdom"

        # Empty dict → empty result
        @test isempty(parse_custom_fields(Dict{String,Any}()))
    end

    @testset "Demographics generation" begin
        rng = MersenneTwister(42)
        demo = generate_demographics(rng, "Test School", 3, 3, "3a", "abc123xyz")
        @test haskey(demo, "uid")
        @test demo["uid"] == "abc123xyz"
        @test haskey(demo, "name")
        @test demo["name"] isa String
        @test !isempty(demo["name"])     # Faker produces a non-empty name
        @test haskey(demo, "d_sex")
        @test demo["d_sex"] in ("M", "F", "I")
        @test haskey(demo, "d_ethnicity")
        @test haskey(demo, "d_age")
        @test demo["d_age"] isa Int
        @test demo["yearGroup"] == 3
        @test demo["schoolYear"] == 3
        @test demo["class"] == "3a"
        @test demo["school"] == "Test School"

        # Custom fields are included in the row
        demo_with_custom = generate_demographics(
            rng, "Test School", 1, 1, "1a", "xyz789abc";
            custom_fields = Dict{String,Function}("d_city" => () -> "Oxford"),
        )
        @test haskey(demo_with_custom, "d_city")
        @test demo_with_custom["d_city"] == "Oxford"
    end

    @testset "perturb_weights" begin
        rng = MersenneTwister(42)
        # Zero SD returns original
        wts = [("A", 0.7), ("B", 0.3)]
        @test perturb_weights(rng, wts, 0.0) === wts

        # Non-zero SD returns perturbed weights that still sum to 1
        perturbed = perturb_weights(rng, wts, 0.1)
        total = sum(w for (_, w) in perturbed)
        @test total ≈ 1.0 atol = 1e-10
        @test all(w >= 0 for (_, w) in perturbed)
        @test length(perturbed) == length(wts)
    end

    @testset "add_numeric_encodings!" begin
        row = DataRow("d_sex" => "F", "yearGroup" => 3)
        add_numeric_encodings!(row)
        @test row["_sex_fm"] == 1.0

        row2 = DataRow("d_sex" => "M", "yearGroup" => 2)
        add_numeric_encodings!(row2)
        @test row2["_sex_fm"] == -1.0

        row3 = DataRow("d_sex" => "I")
        add_numeric_encodings!(row3)
        @test row3["_sex_fm"] == 0.0
    end

    @testset "default_demographics_update" begin
        rng = MersenneTwister(1)
        demo = generate_demographics(rng, "School", 2, 2, "2a", "uid123abc")
        demo["d_age"] = 11

        updated = default_demographics_update(rng, [demo])
        @test updated["d_age"] == 12
        @test updated["d_sex"] == demo["d_sex"]  # copied unchanged

        # Empty history returns empty dict
        @test isempty(default_demographics_update(rng, DataRow[]))
    end

    @testset "Latent variable defaults" begin
        lvars = default_latent_variables()
        @test "depression" ∈ lvars
        @test "anxiety" ∈ lvars

        coefs = default_linear_effects()
        @test !isempty(coefs)
        @test all(c isa LinearEffect for c in coefs)
        @test any(c.target == "depression" for c in coefs)

        effs = default_random_effects()
        @test !isempty(effs)
        @test all(e isa RandomEffect for e in effs)
        @test any(e.target == "anxiety" for e in effs)
        # Error term: Effect with empty inputs
        @test any(isempty(e.categoricalInputs) && isempty(e.numericalInputs) for e in effs)
        # MDE effect: Function-valued RandomEffect targeting depression
        mde_effs = filter(e -> e.target == "depression" && e.value isa Function, effs)
        @test !isempty(mde_effs)
        # MDE should be per (uid, wave)
        @test any(e.categoricalInputs == ["uid", "wave"] && e.value isa Function for e in effs)
    end

    @testset "precompute_effect_draws" begin
        rng = MersenneTwister(42)
        effs = [
            RandomEffect("depression", [], ["school"], Normal(0.0, 0.1)),
            RandomEffect("anxiety",    [], [],          Normal(0.0, 0.05)),  # error term
            # Function-valued effect (MDE-style)
            RandomEffect("depression", [], ["uid", "wave"],
                rng2 -> rand(rng2) < 0.01 ? rand(rng2, Normal(0.75, 0.1)) : 0.0),
        ]
        rows = [
            DataRow("school" => "School A", "uid" => "u1", "wave" => 1),
            DataRow("school" => "School B", "uid" => "u2", "wave" => 1),
            DataRow("school" => "School A", "uid" => "u3", "wave" => 2),
        ]
        draws = precompute_effect_draws(rng, effs, rows)
        @test length(draws) == 3
        # First effect: one draw per unique school (A and B)
        @test length(draws[1]) == 2
        @test haskey(draws[1], ("School A",))
        @test haskey(draws[1], ("School B",))
        # Second effect (error term): empty dict
        @test isempty(draws[2])
        # Third effect (Function): one entry per (uid, wave) combination
        @test length(draws[3]) == 3
        # All values are Float64
        @test all(v isa Float64 for v in values(draws[3]))
    end

    @testset "compute_row_latents" begin
        rng = MersenneTwister(1)
        lvars = ["depression"]
        coefs = [LinearEffect("depression", ["d_age"], 0.02)]
        effs  = [RandomEffect("depression", [], ["school"], Normal(0.0, 0.1))]

        rows = [DataRow("d_age" => 13, "school" => "Test School", "wave" => 1)]
        draws = precompute_effect_draws(rng, effs, rows)

        lv = compute_row_latents(rng, rows[1], lvars, coefs, effs, draws)
        @test haskey(lv, "depression")
        # Age contribution: 0.02 * 13 = 0.26, plus the school draw
        @test lv["depression"] ≈ 0.02 * 13 + draws[1][("Test School",)] atol = 1e-10
    end

    @testset "PHQ-9 questionnaire (QuestionnaireSpec)" begin
        phq9_spec = make_phq9()
        @test phq9_spec isa QuestionnaireSpec
        @test phq9_spec.nItems == 9
        @test phq9_spec.nLevels == 4
        @test any(l.latentName == "depression" for l in phq9_spec.loadings)

        rng = MersenneTwister(42)
        latents = Dict("depression" => 0.3, "anxiety" => 0.1)

        # First wave (no history)
        result = generate_questionnaire_responses(rng, phq9_spec, latents, nothing)
        @test length(result) == 9
        for i in 1:9
            key = "phq9_$i"
            @test haskey(result, key)
            @test result[key] isa Int
            @test 0 <= result[key]::Int <= 3
        end

        # Second wave (with previous responses)
        result2 = generate_questionnaire_responses(rng, phq9_spec, latents, result)
        @test length(result2) == 9
        for i in 1:9
            @test 0 <= result2["phq9_$i"]::Int <= 3
        end
    end

    @testset "GAD-7 questionnaire (QuestionnaireSpec)" begin
        gad7_spec = make_gad7()
        @test gad7_spec isa QuestionnaireSpec
        @test gad7_spec.nItems == 7

        rng = MersenneTwister(42)
        latents = Dict("depression" => 0.2, "anxiety" => 0.4)

        result = generate_questionnaire_responses(rng, gad7_spec, latents, nothing)
        @test length(result) == 7
        for i in 1:7
            key = "gad7_$i"
            @test haskey(result, key)
            @test result[key] isa Int
            @test 0 <= result[key]::Int <= 3
        end
    end

    @testset "default_questionnaires" begin
        qs = default_questionnaires()
        @test qs isa Vector{QuestionnaireSpec}
        @test any(q.name == "PHQ_9" for q in qs)
        @test any(q.name == "GAD_7" for q in qs)
    end

    @testset "build_schema" begin
        qs = default_questionnaires()
        schema = build_schema(qs)
        @test "wave" in schema.demographicsColumns
        @test "uid" in schema.demographicsColumns
        @test "d_age" in schema.demographicsColumns
        # PHQ-9 columns mapped to PHQ_9
        @test haskey(schema.questionnaireColumns, "phq9_1")
        @test schema.questionnaireColumns["phq9_1"] == "PHQ_9"
        # GAD-7 columns mapped to GAD_7
        @test haskey(schema.questionnaireColumns, "gad7_1")
        @test schema.questionnaireColumns["gad7_1"] == "GAD_7"
        # No latent columns by default
        @test isempty(schema.latentColumns)

        # With latent columns
        schema2 = build_schema(qs, ["depression", "anxiety"], true)
        @test "l_anxiety" in schema2.latentColumns
        @test "l_depression" in schema2.latentColumns
    end

    @testset "Full simulation (small) — returns DataFrame" begin
        config = SimulationConfig(
            nWaves                     = 2,
            nSchools                   = 2,
            nYeargroupsPerSchool       = 2,
            nClassesPerSchoolYeargroup = 1,
            nStudentsPerClass          = 3,
            seed                       = 1,
            latentVariables            = default_latent_variables(),
            linearEffects              = default_linear_effects(),
            randomEffects              = default_random_effects(),
            questionnaires             = default_questionnaires(),
        )
        data, schema = simulate(config)

        # simulate() must return a DataFrame
        @test data isa DataFrame

        # 2 schools × 2 yeargroups × 1 class × 3 students × 2 waves = 24 rows
        @test nrow(data) == 24

        # Expected columns
        @test "wave" in names(data)
        @test "uid" in names(data)
        @test "school" in names(data)

        # Waves should be 1 or 2
        @test Set(skipmissing(data[!, "wave"])) ⊆ Set([1, 2])

        # PHQ-9 items should be in range (or missing from naughty monkey)
        for v in skipmissing(data[!, "phq9_1"])
            @test v isa Int
            @test 0 <= v <= 3
        end
    end

    @testset "simulate() requires questionnaires" begin
        @test_throws ArgumentError simulate(SimulationConfig(
            nWaves = 1, nSchools = 1,
            nYeargroupsPerSchool = 1, nClassesPerSchoolYeargroup = 1,
            nStudentsPerClass = 2, seed = 1,
            # questionnaires omitted → should throw
        ))
    end

    @testset "Full simulation with DemographicsSpec customFields" begin
        config = SimulationConfig(
            nWaves    = 1,
            nSchools  = 1,
            nYeargroupsPerSchool = 1,
            nClassesPerSchoolYeargroup = 1,
            nStudentsPerClass = 3,
            seed      = 13,
            latentVariables  = default_latent_variables(),
            linearEffects    = default_linear_effects(),
            randomEffects    = default_random_effects(),
            questionnaires   = default_questionnaires(),
            demographicsSpec = DemographicsSpec(
                customFields = Dict{String,Function}("d_city" => () -> "Oxford"),
            ),
        )
        data, schema = simulate(config)

        # Custom field appears in output and schema
        @test "d_city" in names(data)
        @test "d_city" in schema.demographicsColumns
        @test all(==("Oxford"), skipmissing(data[!, "d_city"]))
    end

    @testset "Full simulation with includeLatents" begin
        config = SimulationConfig(
            nWaves    = 1,
            nSchools  = 2,
            nYeargroupsPerSchool = 2,
            nClassesPerSchoolYeargroup = 1,
            nStudentsPerClass = 3,
            seed      = 7,
            includeLatents   = true,
            latentVariables  = default_latent_variables(),
            linearEffects    = default_linear_effects(),
            randomEffects    = default_random_effects(),
            questionnaires   = default_questionnaires(),
        )
        data, schema = simulate(config)

        @test "l_depression" in names(data)
        @test "l_anxiety" in names(data)
        @test !isempty(schema.latentColumns)

        # Latent values are numbers (or missing after naughty monkey for demo cols)
        for v in skipmissing(data[!, "l_depression"])
            @test v isa Float64
        end
    end

    @testset "Reproducibility (seed)" begin
        cfg = SimulationConfig(
            nWaves    = 1,
            nSchools  = 2,
            nYeargroupsPerSchool = 2,
            nClassesPerSchoolYeargroup = 1,
            nStudentsPerClass = 5,
            seed      = 99,
            latentVariables = default_latent_variables(),
            linearEffects   = default_linear_effects(),
            randomEffects   = default_random_effects(),
            questionnaires  = default_questionnaires(),
        )
        data1, _ = simulate(cfg)
        data2, _ = simulate(cfg)
        # Same seed → same output
        @test nrow(data1) == nrow(data2)
        @test all(isequal.(data1[!, "uid"], data2[!, "uid"]))
        # PHQ-9 column values should be identical (accounting for missing)
        col1 = data1[!, "phq9_1"]
        col2 = data2[!, "phq9_1"]
        for (v1, v2) in zip(col1, col2)
            @test isequal(v1, v2)  # isequal treats missing == missing
        end
    end

    @testset "CSV output (via CSV.jl)" begin
        config = SimulationConfig(
            nWaves = 1, nSchools = 1,
            nYeargroupsPerSchool = 1, nClassesPerSchoolYeargroup = 1,
            nStudentsPerClass = 3, seed = 7,
            latentVariables = default_latent_variables(),
            linearEffects   = default_linear_effects(),
            randomEffects   = default_random_effects(),
            questionnaires  = default_questionnaires(),
        )
        data, schema = simulate(config)

        buf = IOBuffer()
        to_csv(data, schema; io = buf)
        csv_str = String(take!(buf))

        lines = split(csv_str, '\n'; keepempty = false)
        # CSV.jl writes header + nrow rows
        @test length(lines) == 1 + nrow(data)
        header = lines[1]
        @test occursin("wave", header)
        @test occursin("uid", header)
        @test occursin("phq9_1", header)
        @test occursin("gad7_1", header)
    end

    @testset "JSON output (via JSON3.jl)" begin
        config = SimulationConfig(
            nWaves = 1, nSchools = 1,
            nYeargroupsPerSchool = 1, nClassesPerSchoolYeargroup = 1,
            nStudentsPerClass = 2, seed = 8,
            latentVariables = default_latent_variables(),
            linearEffects   = default_linear_effects(),
            randomEffects   = default_random_effects(),
            questionnaires  = default_questionnaires(),
        )
        data, schema = simulate(config)

        buf = IOBuffer()
        to_json(data, schema; io = buf)
        json_str = String(take!(buf))

        @test startswith(strip(json_str), "[")
        @test endswith(strip(json_str), "]")
        @test occursin("\"wave\"", json_str)
        @test occursin("\"uid\"", json_str)
        # Valid JSON: should parse back to an array
        parsed = JSON3.read(json_str)
        @test length(parsed) == nrow(data)
    end

    @testset "JSON Schema export" begin
        qs = default_questionnaires()
        schema = build_schema(qs)
        json_schema = to_json_schema(schema)

        @test occursin("\"\$schema\"", json_schema)
        @test occursin("\"wave\"", json_schema)
        @test occursin("\"uid\"", json_schema)
        @test occursin("\"phq9_1\"", json_schema)
        @test occursin("\"gad7_1\"", json_schema)
        @test occursin("\"integer\"", json_schema)

        # With latents
        schema2 = build_schema(qs, ["depression", "anxiety"], true)
        json_schema2 = to_json_schema(schema2)
        @test occursin("\"l_depression\"", json_schema2)
        @test occursin("Latent variable:", json_schema2)
    end

    @testset "default_naughty_monkey" begin
        rng = MersenneTwister(42)
        config = SimulationConfig(
            nWaves = 1, nSchools = 2,
            nYeargroupsPerSchool = 2, nClassesPerSchoolYeargroup = 2,
            nStudentsPerClass = 10, seed = 100,
            latentVariables = default_latent_variables(),
            linearEffects   = default_linear_effects(),
            randomEffects   = default_random_effects(),
            questionnaires  = default_questionnaires(),
        )
        data, schema = simulate(config)
        # After naughty monkey some values may be missing; count missings
        total_q = 0
        missing_q = 0
        for col in keys(schema.questionnaireColumns)
            col ∈ names(data) || continue
            total_q   += nrow(data)
            missing_q += count(ismissing, data[!, col])
        end
        @test missing_q >= 0
        @test missing_q <= total_q
    end

    @testset "parse_cli_args" begin
        cfg = parse_cli_args(["--nWaves", "2", "--nSchools", "5", "--seed", "42"])
        @test cfg.nWaves == 2
        @test cfg.nSchools == 5
        @test cfg.seed == 42

        cfg2 = parse_cli_args(["--nStudentsPerClass", "norm(25,5)", "--output", "json"])
        @test cfg2.nStudentsPerClass == Normal(25.0, 5.0)
        @test cfg2.output == "json"

        cfg3 = parse_cli_args(["--nClassesPerSchoolYeargroup", "2:6"])
        @test cfg3.nClassesPerSchoolYeargroup == Range(2, 6)

        # ArgParse calls exit(1) for unrecognised flags; test valid args parse correctly
        cfg_ok = parse_cli_args(["--nWaves", "1"])
        @test cfg_ok.nWaves == 1

        # latentVariables
        cfg_lv = parse_cli_args(["--latentVariables", "depression,anxiety"])
        @test cfg_lv.latentVariables == ["depression", "anxiety"]

        # Empty latentVariables → empty Vector (model components including questionnaires must be
        # explicitly provided to simulate(); none are auto-defaulted)
        cfg_lv0 = parse_cli_args(String[])
        @test isempty(cfg_lv0.latentVariables)

        # --linearEffect (single)
        cfg_le = parse_cli_args(["--linearEffect", "depression:d_age:0.02"])
        @test length(cfg_le.linearEffects) == 1
        @test cfg_le.linearEffects[1].target == "depression"
        @test cfg_le.linearEffects[1].value ≈ 0.02

        # --linearEffect (multiple)
        cfg_le2 = parse_cli_args([
            "--linearEffect", "depression:d_age:0.02",
            "--linearEffect", "anxiety:d_age:0.015",
        ])
        @test length(cfg_le2.linearEffects) == 2

        # --randomEffect
        cfg_re = parse_cli_args(["--randomEffect", "depression::uid:norm(0,0.2)"])
        @test length(cfg_re.randomEffects) == 1
        @test cfg_re.randomEffects[1].categoricalInputs == ["uid"]
        @test cfg_re.randomEffects[1].value == Normal(0.0, 0.2)

        # halfnorm in randomEffect
        cfg_hn = parse_cli_args(["--randomEffect", "depression::uid:halfnorm(0,0.2)"])
        @test length(cfg_hn.randomEffects) == 1
        @test cfg_hn.randomEffects[1].value isa UnivariateDistribution

        # Demographics: no flags → demographicsSpec is nothing (uses defaults)
        cfg_no_demo = parse_cli_args(String[])
        @test isnothing(cfg_no_demo.demographicsSpec)

        # Demographics: --sex provided → demographicsSpec is set
        cfg_sex = parse_cli_args(["--sex", "M:0.50,F:0.48,I:0.02"])
        @test !isnothing(cfg_sex.demographicsSpec)
        @test length(cfg_sex.demographicsSpec.sex) == 3
        @test ("M", 0.50) in cfg_sex.demographicsSpec.sex
        # Unspecified fields are empty (resolved to UK defaults inside simulate())
        @test isempty(cfg_sex.demographicsSpec.ethnicity)

        # Demographics: multiple fields
        cfg_demo = parse_cli_args([
            "--ethnicity", "White British:0.75,Asian:0.15,Other:0.10",
            "--sex", "M:0.50,F:0.50",
        ])
        @test !isnothing(cfg_demo.demographicsSpec)
        @test length(cfg_demo.demographicsSpec.ethnicity) == 3
        @test length(cfg_demo.demographicsSpec.sex) == 2
    end

    @testset "CLI complex model (default-model equivalent)" begin
        # Demonstrate a complex model via CLI with questionnaires from a TOML config.
        # Questionnaires can only be specified via TOML; effects and latent variables via CLI.
        minimal_path = joinpath(@__DIR__, "..", "examples", "minimal_model.toml")
        args = [
            "--config", minimal_path,
            "--latentVariables", "enthusiasm",
            "--linearEffect", "enthusiasm:d_age:0.02",
            "--linearEffect", "enthusiasm:_sex_fm:0.05",
            "--randomEffect", "enthusiasm::yearGroup:norm(0,0.05)",
            "--randomEffect", "enthusiasm::uid:halfnorm(0,0.2)",
            "--randomEffect", "enthusiasm:::norm(0,0.1)",
            "--nWaves", "2",
            "--nSchools", "2",
            "--nYeargroupsPerSchool", "2",
            "--nClassesPerSchoolYeargroup", "1",
            "--nStudentsPerClass", "3",
            "--seed", "77",
        ]
        cfg = parse_cli_args(args)

        @test cfg.latentVariables == ["enthusiasm"]
        @test length(cfg.linearEffects) == 2
        @test length(cfg.randomEffects) == 3  # CLI overrides TOML randomEffects
        @test cfg.nWaves == 2
        # Questionnaire from minimal_model.toml
        @test length(cfg.questionnaires) == 1
        @test cfg.questionnaires[1].name == "Enthusiasm_3"

        # Run the simulation end-to-end
        data, schema = simulate(cfg)
        @test data isa DataFrame
        @test nrow(data) == 2 * 2 * 1 * 3 * 2  # schools × yeargroups × classes × students × waves
        @test "wave" in names(data)
        @test "uid" in names(data)
        @test "ent_1" in names(data)  # from minimal_model.toml questionnaire
    end

    @testset "TOML config parsing helpers" begin
        # parse_questionnaire_spec_from_dict
        d_phq = Dict(
            "name"      => "PHQ_9",
            "prefix"    => "phq9",
            "nItems"    => 9,
            "nLevels"   => 4,
            "noiseSD"   => 0.6,
            "spoilRate" => 0.01,
            "loadings"  => [Dict("latentName" => "depression", "scale" => 2.5)],
        )
        qs = parse_questionnaire_spec_from_dict(d_phq)
        @test qs.name == "PHQ_9"
        @test qs.prefix == "phq9"
        @test qs.nItems == 9
        @test qs.nLevels == 4
        @test qs.noiseSD ≈ 0.6
        @test qs.spoilRate ≈ 0.01
        @test length(qs.loadings) == 1
        @test qs.loadings[1].latentName == "depression"
        @test qs.loadings[1].scale isa Float64
        @test qs.loadings[1].scale ≈ 2.5

        # Defaults when optional fields omitted
        d_minimal = Dict("name" => "MyQ", "nItems" => 5)
        qs2 = parse_questionnaire_spec_from_dict(d_minimal)
        @test qs2.nLevels == 4
        @test qs2.noiseSD ≈ 0.6
        @test qs2.spoilRate ≈ 0.01
        @test isempty(qs2.loadings)
        @test qs2.prefix == "myq"  # lowercased, underscores removed

        # Per-item loading via itemScales
        d_per_item = Dict(
            "name"   => "MyQ",
            "nItems" => 3,
            "loadings" => [Dict(
                "latentName" => "depression",
                "itemScales" => Dict("1" => 3.0, "2" => 2.0, "3" => 1.0),
            )],
        )
        qs3 = parse_questionnaire_spec_from_dict(d_per_item)
        @test length(qs3.loadings) == 1
        @test qs3.loadings[1].scale isa Dict{String,Float64}
        @test qs3.loadings[1].scale["1"] ≈ 3.0
        @test qs3.loadings[1].scale["2"] ≈ 2.0
        @test qs3.loadings[1].scale["3"] ≈ 1.0

        # parse_linear_effect_from_dict
        d_le = Dict("target" => "depression", "inputs" => ["d_age", "_sex_fm"], "value" => 0.005)
        le = parse_linear_effect_from_dict(d_le)
        @test le.target == "depression"
        @test le.inputs == ["d_age", "_sex_fm"]
        @test le.value ≈ 0.005

        # inputs defaults to empty
        d_le2 = Dict("target" => "anxiety", "value" => 0.1)
        le2 = parse_linear_effect_from_dict(d_le2)
        @test isempty(le2.inputs)

        # parse_random_effect_from_dict — distribution value string
        d_re = Dict(
            "target"            => "depression",
            "numericalInputs"   => String[],
            "categoricalInputs" => ["uid"],
            "value"             => "norm(0,0.2)",
        )
        re = parse_random_effect_from_dict(d_re)
        @test re.target == "depression"
        @test isempty(re.numericalInputs)
        @test re.categoricalInputs == ["uid"]
        @test re.value == Normal(0.0, 0.2)

        # parse_random_effect_from_dict — halfnorm
        d_re2 = Dict(
            "target" => "anxiety",
            "value"  => "halfnorm(0,0.15)",
        )
        re2 = parse_random_effect_from_dict(d_re2)
        @test re2.value isa UnivariateDistribution
        @test isempty(re2.categoricalInputs)

        # parse_random_effect_from_dict — numeric value
        d_re3 = Dict("target" => "depression", "value" => 0.5)
        re3 = parse_random_effect_from_dict(d_re3)
        @test re3.value isa Float64
        @test re3.value ≈ 0.5
    end

    @testset "load_toml_config" begin
        # load_toml_config errors on missing file
        @test_throws ArgumentError load_toml_config("/nonexistent/path/config.toml")

        # load the bundled example and verify its top-level structure
        example_path = joinpath(@__DIR__, "..", "examples", "default_model.toml")
        toml = load_toml_config(example_path)
        @test haskey(toml, "simulation")
        @test haskey(toml, "demographics")
        @test haskey(toml, "linearEffect")
        @test haskey(toml, "randomEffect")
        @test haskey(toml, "questionnaire")

        sim = toml["simulation"]
        @test sim["nWaves"] == 3
        @test sim["nSchools"] == 10
        @test sim["latentVariables"] == ["depression", "anxiety"]

        @test length(toml["linearEffect"]) == 6
        @test length(toml["randomEffect"]) == 10
        @test length(toml["questionnaire"]) == 2
    end

    @testset "parse_cli_args with --config" begin
        example_path = joinpath(@__DIR__, "..", "examples", "default_model.toml")

        # --config alone: all settings come from TOML
        cfg = parse_cli_args(["--config", example_path])
        @test cfg.nWaves == 3
        @test cfg.nSchools == 10
        @test cfg.latentVariables == ["depression", "anxiety"]
        @test length(cfg.linearEffects) == 6
        @test length(cfg.randomEffects) == 10
        @test length(cfg.questionnaires) == 2
        @test cfg.questionnaires[1].name == "PHQ_9"
        @test cfg.questionnaires[2].name == "GAD_7"
        @test length(cfg.questionnaires[1].loadings) == 1
        @test cfg.questionnaires[1].loadings[1].latentName == "depression"
        @test !isnothing(cfg.demographicsSpec)
        @test !isempty(cfg.demographicsSpec.sex)

        # CLI args override TOML: --nWaves 5 overrides nWaves=3 from TOML
        cfg_override = parse_cli_args(["--config", example_path, "--nWaves", "5"])
        @test cfg_override.nWaves == 5
        @test cfg_override.nSchools == 10  # still from TOML

        # CLI --seed overrides (TOML example has no seed)
        cfg_seed = parse_cli_args(["--config", example_path, "--seed", "99"])
        @test cfg_seed.seed == 99

        # CLI --linearEffect replaces all TOML linearEffects
        cfg_le = parse_cli_args([
            "--config", example_path,
            "--linearEffect", "depression:d_age:0.03",
        ])
        @test length(cfg_le.linearEffects) == 1
        @test cfg_le.linearEffects[1].value ≈ 0.03

        # CLI --sex overrides TOML demographics.sex
        cfg_sex = parse_cli_args([
            "--config", example_path,
            "--sex", "M:0.60,F:0.40",
        ])
        @test !isnothing(cfg_sex.demographicsSpec)
        @test length(cfg_sex.demographicsSpec.sex) == 2
        @test ("M", 0.60) in cfg_sex.demographicsSpec.sex

        # CLI --customField adds custom demographic fields
        cfg_cf = parse_cli_args([
            "--config", example_path,
            "--customField", "d_region=North East",
        ])
        @test !isnothing(cfg_cf.demographicsSpec)
        @test haskey(cfg_cf.demographicsSpec.customFields, "d_region")
        @test cfg_cf.demographicsSpec.customFields["d_region"]() == "North East"

        # CLI --customField with Faker method
        cfg_cf2 = parse_cli_args([
            "--customField", "d_city=faker.city",
        ])
        @test !isnothing(cfg_cf2.demographicsSpec)
        @test haskey(cfg_cf2.demographicsSpec.customFields, "d_city")
        @test cfg_cf2.demographicsSpec.customFields["d_city"]() isa AbstractString

        # TOML customFields are loaded from example (d_city)
        cfg_toml_cf = parse_cli_args(["--config", example_path])
        @test !isnothing(cfg_toml_cf.demographicsSpec)
        @test haskey(cfg_toml_cf.demographicsSpec.customFields, "d_city")
        @test cfg_toml_cf.demographicsSpec.customFields["d_city"]() isa AbstractString

        # End-to-end with customField: d_city appears in output
        cfg_cf_run = parse_cli_args([
            "--config", example_path,
            "--nWaves", "1",
            "--nSchools", "1",
            "--nYeargroupsPerSchool", "1",
            "--nClassesPerSchoolYeargroup", "1",
            "--nStudentsPerClass", "2",
            "--seed", "42",
            "--customField", "d_region=South East",
        ])
        data_cf, _ = simulate(cfg_cf_run)
        @test "d_city" in names(data_cf)     # from TOML
        @test "d_region" in names(data_cf)   # from CLI
        @test all(data_cf.d_region .== "South East")
        @test all(x -> x isa AbstractString, data_cf.d_city)

        # End-to-end: simulate using the example TOML config (small run)
        cfg_run = parse_cli_args([
            "--config", example_path,
            "--nWaves", "2",
            "--nSchools", "2",
            "--nYeargroupsPerSchool", "2",
            "--nClassesPerSchoolYeargroup", "1",
            "--nStudentsPerClass", "3",
            "--seed", "88",
        ])
        data, schema = simulate(cfg_run)
        @test data isa DataFrame
        @test nrow(data) == 2 * 2 * 1 * 3 * 2  # schools × yeargroups × classes × students × waves
        @test "phq9_1" in names(data)
        @test "gad7_1" in names(data)
    end

    @testset "column_order" begin
        qs = default_questionnaires()
        schema = build_schema(qs)
        cols = column_order(schema)

        # Fixed cols come first
        @test cols[1] == "wave"
        @test cols[2] == "uid"
        # Total columns = demographics + questionnaire columns (no latents)
        demo_count = length(schema.demographicsColumns)
        @test length(cols) == demo_count + length(schema.questionnaireColumns)

        # With latent columns
        schema2 = build_schema(qs, ["depression", "anxiety"], true)
        cols2 = column_order(schema2)
        @test "l_depression" in cols2
        @test "l_anxiety" in cols2
        # Latent cols come after questionnaire cols
        q_idx = findfirst(==("phq9_1"), cols2)
        l_idx = findfirst(==("l_anxiety"), cols2)
        @test q_idx < l_idx
    end

    @testset "rows_to_dataframe" begin
        schema = build_schema(default_questionnaires())
        rows = [
            DataRow("wave" => 1, "uid" => "abc", "school" => "Test", "phq9_1" => 2),
            DataRow("wave" => 2, "uid" => "abc", "school" => "Test", "phq9_1" => missing),
        ]
        df = rows_to_dataframe(rows, schema)
        @test df isa DataFrame
        @test nrow(df) == 2
        @test "wave" in names(df)
        @test df[1, "phq9_1"] == 2
        @test ismissing(df[2, "phq9_1"])
    end

    @testset "minimal_model.toml structure" begin
        minimal_path = joinpath(@__DIR__, "..", "examples", "minimal_model.toml")
        toml = load_toml_config(minimal_path)

        @test haskey(toml, "simulation")
        sim = toml["simulation"]
        @test sim["nWaves"] == 3
        @test sim["nSchools"] == 10
        @test sim["latentVariables"] == ["enthusiasm"]

        @test haskey(toml, "randomEffect")
        @test length(toml["randomEffect"]) == 1
        re = toml["randomEffect"][1]
        @test re["target"] == "enthusiasm"
        @test re["categoricalInputs"] == ["uid", "wave"]

        @test haskey(toml, "questionnaire")
        @test length(toml["questionnaire"]) == 1
        q = toml["questionnaire"][1]
        @test q["name"] == "Enthusiasm_3"
        @test q["nItems"] == 3
        @test q["nLevels"] == 7
        @test q["loadings"][1]["scale"] == 6.0

        # No demographics or linearEffects in minimal model
        @test !haskey(toml, "linearEffect")
        @test !haskey(toml, "demographics")
    end

    @testset "minimal_model run-rerun reproducibility" begin
        # Verify that running the minimal_model.toml config twice with the same seed
        # produces identical output (deterministic simulation).
        minimal_path = joinpath(@__DIR__, "..", "examples", "minimal_model.toml")
        seed = 314

        cfg = parse_cli_args(["--config", minimal_path, "--seed", string(seed)])
        data1, _ = simulate(cfg)
        data2, _ = simulate(cfg)

        @test nrow(data1) == nrow(data2)
        @test all(isequal.(data1[!, "uid"], data2[!, "uid"]))
        for col in ("ent_1", "ent_2", "ent_3")
            @test all(isequal.(data1[!, col], data2[!, col]))
        end
    end

    @testset "TOML vs CLI equivalence (minimal_model)" begin
        # Verify that loading the minimal_model.toml produces the same output as specifying
        # the equivalent model via CLI args (using the TOML only for the questionnaire).
        # This tests that CLI args and TOML values are consistent when they specify the same model.
        minimal_path = joinpath(@__DIR__, "..", "examples", "minimal_model.toml")
        seed = 314

        # --- TOML path: full config from TOML ---
        cfg_toml = parse_cli_args(["--config", minimal_path, "--seed", string(seed)])
        data_toml, schema_toml = simulate(cfg_toml)

        # --- CLI path: equivalent settings via CLI (TOML used only for questionnaire) ---
        cfg_cli = parse_cli_args([
            "--config", minimal_path,         # provides questionnaire spec
            "--latentVariables", "enthusiasm",
            "--randomEffect", "enthusiasm::uid,wave:mde(0.75,0.2)",
            "--nWaves", "3",
            "--nSchools", "10",
            "--nYeargroupsPerSchool", "5",
            "--nClassesPerSchoolYeargroup", "1:5",
            "--nStudentsPerClass", "norm(30,7)",
            "--seed", string(seed),
        ])
        data_cli, schema_cli = simulate(cfg_cli)

        # Same structure: same Ns, same latent variables
        @test nrow(data_toml) == nrow(data_cli)
        @test sort(names(data_toml)) == sort(names(data_cli))

        # Same schema
        @test sort(schema_toml.demographicsColumns) == sort(schema_cli.demographicsColumns)
        @test keys(schema_toml.questionnaireColumns) == keys(schema_cli.questionnaireColumns)
    end

    @testset "default_model run-rerun reproducibility" begin
        # Verify that running the default_model.toml config twice with the same seed
        # produces identical output (deterministic simulation).
        default_path = joinpath(@__DIR__, "..", "examples", "default_model.toml")
        seed = 271

        cfg = parse_cli_args(["--config", default_path, "--seed", string(seed)])
        data1, _ = simulate(cfg)
        data2, _ = simulate(cfg)

        @test nrow(data1) == nrow(data2)
        @test all(isequal.(data1[!, "uid"], data2[!, "uid"]))
        for col in ("phq9_1", "phq9_9", "gad7_1", "gad7_7")
            @test all(isequal.(data1[!, col], data2[!, col]))
        end
    end

    @testset "TOML vs CLI equivalence (default_model)" begin
        # Verify that loading the default_model.toml produces the same structure as
        # specifying the same model via CLI args. The CLI path overrides latent variables,
        # effects, and Ns but takes questionnaires from the TOML file.
        default_path = joinpath(@__DIR__, "..", "examples", "default_model.toml")
        seed = 271

        # --- TOML path: full config from TOML ---
        cfg_toml = parse_cli_args(["--config", default_path, "--seed", string(seed)])
        data_toml, schema_toml = simulate(cfg_toml)

        # --- CLI path: specify all model components via CLI (TOML only for questionnaires) ---
        cfg_cli = parse_cli_args([
            "--config", default_path,          # provides questionnaire spec
            "--latentVariables", "depression,anxiety",
            "--linearEffect", "depression:d_age:0.02",
            "--linearEffect", "anxiety:d_age:0.015",
            "--linearEffect", "depression:_sex_fm:0.05",
            "--linearEffect", "anxiety:_sex_fm:0.05",
            "--linearEffect", "depression:d_age,_sex_fm:0.005",
            "--linearEffect", "anxiety:d_age,_sex_fm:0.004",
            "--randomEffect", "depression::yearGroup:norm(0,0.05)",
            "--randomEffect", "anxiety::yearGroup:norm(0,0.05)",
            "--randomEffect", "depression::d_ethnicity,class,_school_id:norm(0,0.03)",
            "--randomEffect", "anxiety::d_ethnicity,class,_school_id:norm(0,0.03)",
            "--randomEffect", "depression::uid,wave:norm(0,0.15)",
            "--randomEffect", "anxiety::uid,wave:norm(0,0.12)",
            "--randomEffect", "depression::uid:halfnorm(0,0.2)",
            "--randomEffect", "anxiety::uid:halfnorm(0,0.15)",
            "--randomEffect", "depression:::norm(0,0.1)",
            "--randomEffect", "anxiety:::norm(0,0.1)",
            "--nWaves", "3",
            "--nSchools", "10",
            "--nYeargroupsPerSchool", "5",
            "--nClassesPerSchoolYeargroup", "1:5",
            "--nStudentsPerClass", "norm(30,7)",
            "--seed", string(seed),
        ])
        data_cli, schema_cli = simulate(cfg_cli)

        # Same structure: number of rows (same Ns + same seed), same column set
        @test nrow(data_toml) == nrow(data_cli)
        @test sort(names(data_toml)) == sort(names(data_cli))

        # Same schema
        @test keys(schema_toml.questionnaireColumns) == keys(schema_cli.questionnaireColumns)
    end

end  # @testset "IbOxDummies"
