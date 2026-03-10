"""
    column_order(schema) -> Vector{String}

Return the canonical column ordering for the output dataset:
  wave, uid, name, school, yearGroup, schoolYear, class,
  demographics columns, then questionnaire columns (sorted by questionnaire name, then column).
"""
function column_order(schema::Schema)::Vector{String}
    fixed = ["wave", "uid", "name", "school", "yearGroup", "schoolYear", "class"]
    demo = [
        c for c in schema.demographicsColumns
        if c ∉ fixed
    ]
    q_cols = sort(
        collect(keys(schema.questionnaireColumns));
        by = c -> (schema.questionnaireColumns[c], c),
    )
    return vcat(fixed, demo, q_cols)
end

"""
    to_csv(df, schema; io=stdout)

Write the output `DataFrame` as CSV to `io` using `CSV.jl`.
Columns are written in canonical order defined by `column_order(schema)`.
"""
function to_csv(df::DataFrame, schema::Schema; io::IO = stdout)
    cols = [c for c in column_order(schema) if c ∈ names(df)]
    CSV.write(io, df[:, cols])
end

"""
    to_json(df, schema; io=stdout)

Write the output `DataFrame` as a JSON array of objects to `io` using `JSON3.jl`.
Missing values are serialised as `null`. Columns appear in canonical order.
"""
function to_json(df::DataFrame, schema::Schema; io::IO = stdout)
    cols = [c for c in column_order(schema) if c ∈ names(df)]
    JSON3.write(io, Tables.rowtable(df[:, cols]))
    println(io)
end

"""
    to_json_schema(schema) -> String

Export the schema as a JSON Schema (Draft 7) document describing the output
row type. Suitable for downstream validation or code generation.
"""
function to_json_schema(schema::Schema)::String
    cols = column_order(schema)

    io = IOBuffer()
    println(io, """{
  "\$schema": "http://json-schema.org/draft-07/schema#",
  "title": "IbOxDummies output row",
  "description": "One row of simulated student questionnaire data.",
  "type": "object",
  "properties": {""")

    function col_schema(c)
        qname = get(schema.questionnaireColumns, c, nothing)
        is_q  = !isnothing(qname)

        if c == "uid"
            return """    "$c": {"type": "string", "description": "Unique student identifier"}"""
        elseif c == "name"
            return """    "$c": {"type": ["string", "null"], "description": "Student full name"}"""
        elseif c == "school"
            return """    "$c": {"type": ["string", "null"], "description": "School name"}"""
        elseif c in ("wave", "yearGroup", "schoolYear", "d_age")
            return """    "$c": {"type": ["integer", "null"]}"""
        elseif c == "class"
            return """    "$c": {"type": ["string", "null"], "description": "Class label"}"""
        elseif c == "d_sex"
            return """    "$c": {"type": ["string", "null"], "enum": ["M", "F", "I", null]}"""
        elseif c in ("d_ethnicity", "d_sexualOrientation", "d_genderIdentity")
            return """    "$c": {"type": ["string", "null"]}"""
        elseif is_q
            return """    "$c": {"type": ["integer", "null"], "minimum": 0, "maximum": 3, """ *
                   """"description": "Item from questionnaire $(qname)"}"""
        else
            return """    "$c": {"type": ["string", "integer", "number", "null"]}"""
        end
    end

    prop_strs = [col_schema(c) for c in cols]
    print(io, join(prop_strs, ",\n"))
    println(io, "\n  },")
    println(io, """  "required": ["wave", "uid"]""")
    print(io, "}")

    return String(take!(io))
end

"""
    write_output(df, schema, config)

Produce the final output according to `config.output`.
- `"csv"` → write CSV to stdout (via `CSV.jl`).
- `"json"` → write JSON to stdout (via `JSON3.jl`).
- `"schema"` → write JSON Schema to stdout.
- A `Function` → call `config.output(df, schema)` and print the result.
"""
function write_output(df::DataFrame, schema::Schema, config::SimulationConfig)
    if config.output isa String
        if config.output == "csv"
            to_csv(df, schema)
        elseif config.output == "json"
            to_json(df, schema)
        elseif config.output == "schema"
            println(to_json_schema(schema))
        else
            error("Unknown output format: $(config.output). Use 'csv', 'json', or 'schema'.")
        end
    else
        result = config.output(df, schema)
        println(result)
    end
end
