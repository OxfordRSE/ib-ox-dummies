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
    adata_to_string(v::AData) -> String

Convert a single answer datum to its string representation.
`missing` becomes the empty string (for CSV) or `null` (for JSON).
"""
function adata_to_csv_string(v::AData)::String
    ismissing(v) && return ""
    v isa String && return v  # no quoting for internal use; handled by write_csv
    return string(v)
end

"""
    escape_csv_field(s) -> String

Wrap a CSV field in quotes if it contains commas, quotes, or newlines.
"""
function escape_csv_field(s::String)::String
    if occursin(',', s) || occursin('"', s) || occursin('\n', s)
        return '"' * replace(s, '"' => "\"\"") * '"'
    end
    return s
end

"""
    to_csv(data, schema; io=stdout)

Write the output data as CSV to `io`.
"""
function to_csv(data::Vector{QData}, schema::Schema; io::IO = stdout)
    cols = column_order(schema)

    # Header
    println(io, join(cols, ","))

    # Rows
    for row in data
        fields = [
            escape_csv_field(adata_to_csv_string(get(row, c, missing)))
            for c in cols
        ]
        println(io, join(fields, ","))
    end
end

"""
    adata_to_json(v::AData) -> String

Serialise a single `AData` value to its JSON representation.
"""
function adata_to_json(v::AData)::String
    ismissing(v) && return "null"
    v isa String && return '"' * replace(v, '\\' => "\\\\", '"' => "\\\"") * '"'
    v isa Int    && return string(v)
    v isa Float64 && isnan(v) && return "null"
    v isa Float64 && isinf(v) && return "null"
    return string(v)
end

"""
    to_json(data, schema; io=stdout)

Write the output data as a JSON array of objects to `io`.
"""
function to_json(data::Vector{QData}, schema::Schema; io::IO = stdout)
    cols = column_order(schema)
    println(io, "[")
    for (i, row) in enumerate(data)
        print(io, "  {")
        pairs_strs = [
            '"' * c * '"' * ": " * adata_to_json(get(row, c, missing))
            for c in cols
        ]
        print(io, join(pairs_strs, ", "))
        print(io, "}")
        i < length(data) && print(io, ",")
        println(io)
    end
    println(io, "]")
end

"""
    to_json_schema(schema) -> String

Export the schema as a JSON Schema (Draft 7) document describing the output
row type.
"""
function to_json_schema(schema::Schema)::String
    cols = column_order(schema)

    # Build property definitions
    io = IOBuffer()
    println(io, """{
  "\$schema": "http://json-schema.org/draft-07/schema#",
  "title": "IbOxDummies output row",
  "description": "One row of simulated student questionnaire data.",
  "type": "object",
  "properties": {""")

    int_cols = Set([
        "wave", "yearGroup", "schoolYear", "d_age",
        # PHQ-9 and GAD-7 items are integers
    ])

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
    write_output(data, schema, config)

Produce the final output according to `config.output`.
- `"csv"` → write CSV to stdout.
- `"json"` → write JSON to stdout.
- `"schema"` → write JSON Schema to stdout.
- A `Function` → call `config.output(data, schema)` and pretty-print the result.
"""
function write_output(data::Vector{QData}, schema::Schema, config::SimulationConfig)
    if config.output isa String
        if config.output == "csv"
            to_csv(data, schema)
        elseif config.output == "json"
            to_json(data, schema)
        elseif config.output == "schema"
            println(to_json_schema(schema))
        else
            error("Unknown output format: $(config.output). Use 'csv', 'json', or 'schema'.")
        end
    else
        result = config.output(data, schema)
        println(result)
    end
end
