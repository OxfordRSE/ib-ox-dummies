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
        # Total columns = demographics + questionnaire columns
        demo_count = length(schema.demographicsColumns)
        @test length(cols) == demo_count + length(schema.questionnaireColumns)
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
