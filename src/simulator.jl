"""
    build_schema(questionnaires; latent_variables, include_latents) -> Schema

Build a `Schema` from the given questionnaire specifications.
"""
function build_schema(
    questionnaires::Vector{QuestionnaireSpec},
    latent_variables::Vector{String} = String[],
    include_latents::Bool = false,
)::Schema
    demo_cols = [
        "wave", "uid", "name", "school", "yearGroup", "schoolYear", "class",
        "d_age", "d_sex", "d_ethnicity", "d_sexualOrientation", "d_genderIdentity",
    ]

    q_cols = Dict{String,String}()
    for spec in questionnaires
        for i in 1:spec.nItems
            q_cols["$(spec.prefix)_$i"] = spec.name
        end
    end

    latent_cols = include_latents ? sort(["l_$n" for n in latent_variables]) : String[]

    return Schema(demo_cols, q_cols, latent_cols)
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

The simulation proceeds in five stages:
1. Generate school → yeargroup → class → student hierarchy with per-school
   perturbed demographic weight distributions.
2. Build all (student × wave) row templates and pre-compute random effect draws.
3. Run waves: for each student compute latent variable values then generate
   questionnaire responses driven by those latents.
4. Optionally attach latent variable values as `l_*` columns (within stage 3).
5. Apply the naughty-monkey corruption function.
"""
function simulate(config::SimulationConfig)::Tuple{DataFrame,Schema}
    qs    = isempty(config.questionnaires)   ? default_questionnaire_specs() : config.questionnaires
    lvars = isempty(config.latentVariables)  ? default_latent_variables()    : config.latentVariables
    coefs = isempty(config.coefficients)     ? default_coefficients()        : config.coefficients
    effs  = isempty(config.effects)          ? default_effects()             : config.effects

    rng = isnothing(config.seed) ? MersenneTwister() : MersenneTwister(config.seed)

    schema = build_schema(qs, lvars, config.includeLatents)

    # --- Stage 1: Generate school/yeargroup/class/student structure ---
    struct_students = []

    for s_id in 1:config.nSchools
        school_name = generate_school_name(rng, s_id)
        n_yg = sample_count(rng, config.nYeargroupsPerSchool)

        # Per-school perturbed demographic weight distributions
        sd = config.demographicPerturbationSD
        eth_wts  = perturb_weights(rng, ETHNICITY_WEIGHTS,           sd)
        sex_wts  = perturb_weights(rng, SEX_WEIGHTS,                 sd)
        gend_wts = perturb_weights(rng, GENDER_IDENTITY_WEIGHTS,     sd)
        ori_wts  = perturb_weights(rng, SEXUAL_ORIENTATION_WEIGHTS,  sd)

        for yg in 1:n_yg
            n_cls = sample_count(rng, config.nClassesPerSchoolYeargroup)
            for cls in 1:n_cls
                class_label = generate_class_label(rng, yg, cls)
                n_stu = sample_count(rng, config.nStudentsPerClass)
                for _ in 1:n_stu
                    uid = generate_uid(rng)
                    demo = generate_demographics(
                        rng, school_name, yg, yg, class_label, uid;
                        ethnicity_weights   = eth_wts,
                        sex_weights         = sex_wts,
                        gender_weights      = gend_wts,
                        orientation_weights = ori_wts,
                    )
                    add_numeric_encodings!(demo)
                    push!(struct_students, (uid = uid, demographics = demo))
                end
            end
        end
    end

    # --- Stage 2: Build all (student × wave) templates and pre-compute effects ---
    all_templates = QData[]
    for wave in 1:config.nWaves, stu in struct_students
        tmpl = copy(stu.demographics)
        tmpl["wave"] = wave
        push!(all_templates, tmpl)
    end

    effect_draws = precompute_effect_draws(rng, effs, all_templates)

    # --- Initialise per-student state ---
    student_history      = Dict{String,Vector{QData}}()
    student_demographics = Dict{String,QData}()
    for stu in struct_students
        student_history[stu.uid]      = QData[]
        student_demographics[stu.uid] = stu.demographics
    end

    all_output = QData[]

    # --- Stage 3: Run waves ---
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
                add_numeric_encodings!(merged)
                student_demographics[uid] = merged
                merged
            end

            # Build row: demographics + wave
            row = copy(current_demo)
            row["wave"] = wave

            # Compute latent variables for this row
            latents = compute_row_latents(rng, row, lvars, coefs, effs, effect_draws)

            # Generate questionnaire responses driven by latent values
            prev_resp = isempty(history) ? nothing : history[end]
            for spec in qs
                q_result = generate_questionnaire_responses(rng, spec, latents, prev_resp)
                for (k, v) in q_result
                    row[k] = v
                end
            end

            # Stage 4: Optionally attach latent variable values as l_* columns
            if config.includeLatents
                for (name, val) in latents
                    row["l_$name"] = val
                end
            end

            push!(all_output, row)
            push!(history, row)
        end
    end

    # --- Stage 5: Apply naughty monkey ---
    all_output = config.naughtyMonkey(rng, all_output, schema)

    return qdata_to_dataframe(all_output, schema), schema
end
