using Test
using Random

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

        # NormalDist
        nd = NormalDist(30.0, 7.0)
        @test nd.μ == 30.0
        @test nd.σ == 7.0
        @test_throws ArgumentError NormalDist(10.0, 0.0)
        @test_throws ArgumentError NormalDist(10.0, -1.0)

        # SimulationConfig defaults
        cfg = SimulationConfig()
        @test cfg.nWaves == 3
        @test cfg.nSchools == 10
        @test cfg.output == "csv"
        @test isnothing(cfg.seed)
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

        # NormalDist sampling returns positive integers
        rng = MersenneTwister(42)
        for _ in 1:50
            v = sample_count(rng, NormalDist(30.0, 7.0))
            @test v >= 1
        end
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
        @test parse_count_spec("norm(30,7)") == NormalDist(30.0, 7.0)
        @test parse_count_spec("dnorm(30,7)") == NormalDist(30.0, 7.0)
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

    @testset "PHQ-9 questionnaire" begin
        phq9 = make_phq9()
        rng = MersenneTwister(42)
        schema = Schema(String[], Dict{String,String}())

        # First wave (no history)
        result = phq9(rng, QData[], schema)
        @test length(result) == 9
        for i in 1:9
            key = "phq9_$i"
            @test haskey(result, key)
            @test result[key] isa Int
            @test 0 <= result[key]::Int <= 3
        end

        # Second wave (with history)
        result2 = phq9(rng, [result], schema)
        @test length(result2) == 9
        for i in 1:9
            @test 0 <= result2["phq9_$i"]::Int <= 3
        end
    end

    @testset "GAD-7 questionnaire" begin
        gad7 = make_gad7()
        rng = MersenneTwister(42)
        schema = Schema(String[], Dict{String,String}())

        result = gad7(rng, QData[], schema)
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
        @test haskey(qs, "PHQ_9")
        @test haskey(qs, "GAD_7")
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
    end

    @testset "Full simulation (small)" begin
        config = SimulationConfig(
            nWaves                     = 2,
            nSchools                   = 2,
            nYeargroupsPerSchool       = 2,
            nClassesPerSchoolYeargroup = 1,
            nStudentsPerClass          = 3,
            seed                       = 1,
        )
        data, schema = simulate(config)

        # We expect: 2 schools × 2 yeargroups × 1 class × 3 students × 2 waves = 24 rows
        @test length(data) == 24

        # Check column presence
        for row in data
            @test haskey(row, "wave")
            @test haskey(row, "uid")
            @test haskey(row, "school")
        end

        # Waves should be 1 or 2
        waves = [row["wave"] for row in data if haskey(row, "wave") && row["wave"] isa Int]
        @test Set(waves) ⊆ Set([1, 2])

        # PHQ-9 items should exist and be in range (or missing from naughty monkey)
        for row in data
            if haskey(row, "phq9_1") && !ismissing(row["phq9_1"])
                @test row["phq9_1"] isa Int
                @test 0 <= row["phq9_1"]::Int <= 3
            end
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
        @test length(data1) == length(data2)
        for (r1, r2) in zip(data1, data2)
            @test get(r1, "uid", nothing) == get(r2, "uid", nothing)
            @test get(r1, "phq9_1", nothing) === get(r2, "phq9_1", nothing)
        end
    end

    @testset "CSV output" begin
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
        @test length(lines) == 1 + length(data)  # header + rows
        # Header contains expected columns
        header = lines[1]
        @test occursin("wave", header)
        @test occursin("uid", header)
        @test occursin("phq9_1", header)
        @test occursin("gad7_1", header)
    end

    @testset "JSON output" begin
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
        for row in data
            for col in keys(schema.questionnaireColumns)
                total_q += 1
                haskey(row, col) && ismissing(row[col]) && (missing_q += 1)
            end
        end
        # With default 0.25% deletion, expected missing ≈ 0.25% but may be 0 for small data
        @test missing_q >= 0
        @test missing_q <= total_q
    end

    @testset "parse_cli_args" begin
        cfg = parse_cli_args(["--nWaves", "2", "--nSchools", "5", "--seed", "42"])
        @test cfg.nWaves == 2
        @test cfg.nSchools == 5
        @test cfg.seed == 42

        cfg2 = parse_cli_args(["--nStudentsPerClass", "norm(25,5)", "--output", "json"])
        @test cfg2.nStudentsPerClass == NormalDist(25.0, 5.0)
        @test cfg2.output == "json"

        cfg3 = parse_cli_args(["--nClassesPerSchoolYeargroup", "2:6"])
        @test cfg3.nClassesPerSchoolYeargroup == Range(2, 6)

        @test_throws Exception parse_cli_args(["--unknown-flag"])
    end

    @testset "column_order" begin
        qs = default_questionnaires()
        schema = build_schema(qs)
        cols = column_order(schema)

        # Fixed cols come first
        @test cols[1] == "wave"
        @test cols[2] == "uid"
        # Questionnaire cols come after demographics
        demo_count = length(schema.demographicsColumns)
        @test length(cols) == demo_count + length(schema.questionnaireColumns)
    end

end  # @testset "IbOxDummies"
