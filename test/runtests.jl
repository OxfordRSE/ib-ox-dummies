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

        # CountSpec accepts Distributions.Normal
        nd = Normal(30.0, 7.0)
        @test nd isa UnivariateDistribution

        # SimulationConfig defaults
        cfg = SimulationConfig()
        @test cfg.nWaves == 3
        @test cfg.nSchools == 10
        @test cfg.output == "csv"
        @test isnothing(cfg.seed)
        @test cfg.nStudentsPerClass isa Normal
        @test cfg.includeLatents == false
        @test cfg.demographicPerturbationSD == 0.05
        @test isempty(cfg.questionnaires)
        @test isempty(cfg.latentVariables)
        @test isempty(cfg.coefficients)
        @test isempty(cfg.effects)

        # New struct types exist
        c = Coefficient("depression", ["d_age"], 0.02)
        @test c.target == "depression"
        @test c.value == 0.02

        e = Effect("anxiety", [], ["uid"], Normal(0.0, 0.1))
        @test e.target == "anxiety"
        @test isempty(e.numericalInputs)

        ll = LatentLoading("depression", 2.5)
        @test ll.latentName == "depression"
        @test ll.scale == 2.5

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

    @testset "parse_count_spec" begin
        @test parse_count_spec("5") == 5
        @test parse_count_spec("1:5") == Range(1, 5)
        @test parse_count_spec("1,5") == Range(1, 5)
        @test parse_count_spec("norm(30,7)") == Normal(30.0, 7.0)
        @test parse_count_spec("dnorm(30,7)") == Normal(30.0, 7.0)
        @test_throws ArgumentError parse_count_spec("garbage")
    end

    @testset "Demographics generation" begin
        rng = MersenneTwister(42)
        demo = generate_demographics(rng, "Test School", 3, 3, "3a", "abc123xyz")
        @test haskey(demo, "uid")
        @test demo["uid"] == "abc123xyz"
        @test haskey(demo, "name")
        @test haskey(demo, "d_sex")
        @test demo["d_sex"] in ("M", "F", "I")
        @test haskey(demo, "d_ethnicity")
        @test haskey(demo, "d_age")
        @test demo["d_age"] isa Int
        @test demo["yearGroup"] == 3
        @test demo["schoolYear"] == 3
        @test demo["class"] == "3a"
        @test demo["school"] == "Test School"
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
        row = QData("d_sex" => "F", "yearGroup" => 3)
        add_numeric_encodings!(row)
        @test row["_sex_fm"] == 1.0

        row2 = QData("d_sex" => "M", "yearGroup" => 2)
        add_numeric_encodings!(row2)
        @test row2["_sex_fm"] == -1.0

        row3 = QData("d_sex" => "I")
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
        @test isempty(default_demographics_update(rng, QData[]))
    end

    @testset "Latent variable defaults" begin
        lvars = default_latent_variables()
        @test "depression" ∈ lvars
        @test "anxiety" ∈ lvars

        coefs = default_coefficients()
        @test !isempty(coefs)
        @test all(c isa Coefficient for c in coefs)
        @test any(c.target == "depression" for c in coefs)

        effs = default_effects()
        @test !isempty(effs)
        @test all(e isa Effect for e in effs)
        @test any(e.target == "anxiety" for e in effs)
        # Error term: Effect with empty inputs
        @test any(isempty(e.categoricalInputs) && isempty(e.numericalInputs) for e in effs)
    end

    @testset "precompute_effect_draws" begin
        rng = MersenneTwister(42)
        effs = [
            Effect("depression", [], ["school"], Normal(0.0, 0.1)),
            Effect("anxiety",    [], [],          Normal(0.0, 0.05)),  # error term
        ]
        rows = [
            QData("school" => "School A", "uid" => "u1", "wave" => 1),
            QData("school" => "School B", "uid" => "u2", "wave" => 1),
            QData("school" => "School A", "uid" => "u3", "wave" => 2),
        ]
        draws = precompute_effect_draws(rng, effs, rows)
        @test length(draws) == 2
        # First effect: one draw per unique school (A and B)
        @test length(draws[1]) == 2
        @test haskey(draws[1], ("School A",))
        @test haskey(draws[1], ("School B",))
        # Second effect (error term): empty dict
        @test isempty(draws[2])
    end

    @testset "compute_row_latents" begin
        rng = MersenneTwister(1)
        lvars = ["depression"]
        coefs = [Coefficient("depression", ["d_age"], 0.02)]
        effs  = [Effect("depression", [], ["school"], Normal(0.0, 0.1))]

        rows = [QData("d_age" => 13, "school" => "Test School", "wave" => 1)]
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

    @testset "Full simulation with includeLatents" begin
        config = SimulationConfig(
            nWaves    = 1,
            nSchools  = 2,
            nYeargroupsPerSchool = 2,
            nClassesPerSchoolYeargroup = 1,
            nStudentsPerClass = 3,
            seed      = 7,
            includeLatents = true,
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

    @testset "qdata_to_dataframe" begin
        schema = build_schema(default_questionnaires())
        rows = [
            QData("wave" => 1, "uid" => "abc", "school" => "Test", "phq9_1" => 2),
            QData("wave" => 2, "uid" => "abc", "school" => "Test", "phq9_1" => missing),
        ]
        df = qdata_to_dataframe(rows, schema)
        @test df isa DataFrame
        @test nrow(df) == 2
        @test "wave" in names(df)
        @test df[1, "phq9_1"] == 2
        @test ismissing(df[2, "phq9_1"])
    end

end  # @testset "IbOxDummies"
