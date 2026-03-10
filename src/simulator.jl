"""
    build_schema(questionnaires) -> Schema

Build a `Schema` from the given questionnaire mapping.
Demographics columns are the fixed set of structural fields.
"""
function build_schema(questionnaires::Dict{String,Questionnaire})::Schema
    demo_cols = [
        "wave", "uid", "name", "school", "yearGroup", "schoolYear", "class",
        "d_age", "d_sex", "d_ethnicity", "d_sexualOrientation", "d_genderIdentity",
    ]

    q_cols = Dict{String,String}()
    # Discover columns by running each questionnaire once with empty data
    for (q_name, q_fn) in questionnaires
        rng = MersenneTwister(0)
        dummy_schema = Schema(String[], Dict{String,String}())
        sample_result = q_fn(rng, QData[], dummy_schema)
        for col in keys(sample_result)
            q_cols[col] = q_name
        end
    end

    return Schema(demo_cols, q_cols)
end

"""
    qdata_to_dataframe(rows, schema) -> DataFrame

Convert a `Vector{QData}` to a `DataFrame` with columns in canonical order.
Missing values (absent keys) are represented as `missing`.
"""
function qdata_to_dataframe(rows::Vector{QData}, schema::Schema)::DataFrame
    cols = column_order(schema)
    return DataFrame(
        [col => [get(row, col, missing) for row in rows] for col in cols]
    )
end

"""
    simulate(config::SimulationConfig) -> (DataFrame, Schema)

Run the full simulation and return the long-format output as a `DataFrame`
and the associated `Schema`.
"""
function simulate(config::SimulationConfig)::Tuple{DataFrame,Schema}
    qs = isempty(config.questionnaires) ? default_questionnaires() :
        config.questionnaires

    # Seed the RNG
    rng = if isnothing(config.seed)
        MersenneTwister()
    else
        MersenneTwister(config.seed)
    end

    schema = build_schema(qs)

    # --- Build school/yeargroup/class/student structure ---
    schools = Dict{Int,String}()
    for s in 1:config.nSchools
        schools[s] = generate_school_name(rng, s)
    end

    # First pass: create all students
    struct_students = []

    for s_id in 1:config.nSchools
        n_yg = sample_count(rng, config.nYeargroupsPerSchool)
        school_name = schools[s_id]
        for yg in 1:n_yg
            school_year = yg
            n_cls = sample_count(rng, config.nClassesPerSchoolYeargroup)
            for cls in 1:n_cls
                class_label = generate_class_label(rng, yg, cls)
                n_stu = sample_count(rng, config.nStudentsPerClass)
                for _ in 1:n_stu
                    uid = generate_uid(rng)
                    demo = generate_demographics(
                        rng, school_name, yg, school_year, class_label, uid
                    )
                    push!(struct_students, (
                        school_id    = s_id,
                        yeargroup    = yg,
                        school_year  = school_year,
                        class_label  = class_label,
                        uid          = uid,
                        demographics = demo,
                    ))
                end
            end
        end
    end

    # all_output collects every row across all waves
    all_output = QData[]

    # Per-student history and current demographics
    student_history      = Dict{String,Vector{QData}}()
    student_demographics = Dict{String,QData}()

    for stu in struct_students
        student_history[stu.uid]      = QData[]
        student_demographics[stu.uid] = stu.demographics
    end

    # --- Run waves ---
    for wave in 1:config.nWaves
        for stu in struct_students
            uid     = stu.uid
            history = student_history[uid]

            # Update demographics after the first wave
            current_demo = if wave == 1
                student_demographics[uid]
            else
                updated   = config.demographicsUpdateFn(rng, history)
                prev_demo = student_demographics[uid]
                merged    = copy(prev_demo)
                for (k, v) in updated
                    merged[k] = v
                end
                student_demographics[uid] = merged
                merged
            end

            # Build the row: wave + demographics + questionnaire responses
            row = QData()
            row["wave"] = wave
            for (k, v) in current_demo
                row[k] = v
            end

            for (_, q_fn) in qs
                q_result = q_fn(rng, history, schema)
                for (k, v) in q_result
                    row[k] = v
                end
            end

            push!(all_output, row)
            push!(history, row)
        end
    end

    # Apply naughty monkey
    all_output = config.naughtyMonkey(rng, all_output, schema)

    return qdata_to_dataframe(all_output, schema), schema
end
