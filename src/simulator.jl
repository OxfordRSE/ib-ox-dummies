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
    simulate(config::SimulationConfig) -> (Vector{QData}, Schema)

Run the full simulation and return the long-format output rows and schema.
"""
function simulate(config::SimulationConfig)::Tuple{Vector{QData},Schema}
    qs = isempty(config.questionnaires) ? default_questionnaires() :
        config.questionnaires

    # Seed the RNG
    rng = if isnothing(config.seed)
        MersenneTwister()
    else
        MersenneTwister(config.seed)
    end

    schema = build_schema(qs)

    # Column ordering for output consistency
    demo_cols  = schema.demographicsColumns
    q_col_order = sort(collect(keys(schema.questionnaireColumns)))

    # --- Build school/yeargroup/class/student structure ---
    # school_id => school_name
    schools = Dict{Int,String}()
    for s in 1:config.nSchools
        schools[s] = generate_school_name(rng, s)
    end

    # Build list of (school_id, yeargroup, school_year, class_label) tuples
    # and for each, a list of student UIDs with their initial demographics.
    # Structure: student_key => [wave_data...]
    #   student_key = (school_id, yeargroup, class_idx, student_idx)
    #   wave_data[w] = QData for that wave (grows as waves are simulated)

    # First pass: create all students
    # students: list of (school_id, yeargroup, school_year, class_label, uid, demographics)
    struct_students = []  # NamedTuple list

    for s_id in 1:config.nSchools
        n_yg = sample_count(rng, config.nYeargroupsPerSchool)
        school_name = schools[s_id]
        for yg in 1:n_yg
            school_year = yg  # yeargroup index used directly as school year
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
                        school_id   = s_id,
                        yeargroup   = yg,
                        school_year = school_year,
                        class_label = class_label,
                        uid         = uid,
                        demographics = demo,
                    ))
                end
            end
        end
    end

    # all_output: will collect every row across all waves
    all_output = QData[]

    # Per-student history: uid => Vector{QData} (cumulative across waves)
    student_history = Dict{String,Vector{QData}}()
    # Per-student current demographics: uid => QData
    student_demographics = Dict{String,QData}()

    for stu in struct_students
        student_history[stu.uid] = QData[]
        student_demographics[stu.uid] = stu.demographics
    end

    # --- Run waves ---
    for wave in 1:config.nWaves
        for stu in struct_students
            uid = stu.uid
            history = student_history[uid]

            # Update demographics after the first wave
            current_demo = if wave == 1
                student_demographics[uid]
            else
                updated = config.demographicsUpdateFn(rng, history)
                # Merge: keep existing keys not returned by the update fn
                prev_demo = student_demographics[uid]
                merged = copy(prev_demo)
                for (k, v) in updated
                    merged[k] = v
                end
                student_demographics[uid] = merged
                merged
            end

            # Build the row: start with wave + demographics
            row = QData()
            row["wave"] = wave
            for (k, v) in current_demo
                row[k] = v
            end

            # Run each questionnaire
            for (q_name, q_fn) in qs
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

    return all_output, schema
end
