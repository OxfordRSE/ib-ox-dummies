# ib-ox-dummies

A simulator for generating realistic mock longitudinal questionnaire data,
designed to represent annual survey responses from schoolchildren.

Built in [Julia](https://julialang.org/) using only the standard library.

## Requirements

- Julia ≥ 1.0

## Installation

Clone the repository and activate the package:

```bash
git clone https://github.com/OxfordRSE/ib-ox-dummies.git
cd ib-ox-dummies
julia --project=. -e 'import Pkg; Pkg.instantiate()'
```

To run the CLI directly:

```bash
julia bin/ib_ox_dummies --help
```

Or add the `bin/` directory to your `PATH`:

```bash
export PATH="$PATH:/path/to/ib-ox-dummies/bin"
ib_ox_dummies --help
```

## Usage

```
ib_ox_dummies — Generate mock longitudinal questionnaire data.

USAGE
  ib_ox_dummies [OPTIONS]

OPTIONS
  --nWaves INT                     Number of data-collection waves (default: 3)
  --nSchools INT                   Number of schools (default: 10)
  --nYeargroupsPerSchool SPEC      Yeargroups per school (default: 5)
  --nClassesPerSchoolYeargroup SPEC Classes per school yeargroup (default: 1,5)
  --nStudentsPerClass SPEC         Students per class (default: norm(30,7))
  --seed INT                       Random seed for reproducibility
  --output FORMAT                  Output format: csv | json | schema (default: csv)
  --schema                         Print JSON Schema and exit
  --help, -h                       Show this help and exit

SPEC formats:
  5             Fixed integer
  1:5           Inclusive range [min, max] (uniform sample)
  norm(30,7)    Normal distribution N(μ=30, σ=7) (rounded to nearest integer ≥ 1)
```

### Examples

```bash
# Default run (3 waves, 10 schools, ~30 students/class)
ib_ox_dummies

# Small reproducible run → CSV
ib_ox_dummies --nWaves 2 --nSchools 3 --seed 42

# JSON output
ib_ox_dummies --nStudentsPerClass norm(25,5) --output json

# Print JSON Schema describing the output columns
ib_ox_dummies --output schema
```

## Output

Long-format tabular data with one row per student per wave:

| wave | uid | name | school | yearGroup | schoolYear | class | d_age | d_sex | d_ethnicity | d_sexualOrientation | d_genderIdentity | phq9_1 | … | gad7_1 | … |
|------|-----|------|--------|-----------|------------|-------|-------|-------|-------------|---------------------|------------------|--------|---|--------|---|
| 1 | s5agas99p | Michelle Roux | Islington Academy | 2 | 2 | 2b | 11 | F | White British | NA | Cis | 2 | … | 5 | … |

- **Demographics** (`d_*`) are generated once and updated each wave (age increments by 1).
- **Questionnaire items** (PHQ-9, GAD-7) use Likert 0–3 scales.
- A configurable *naughty monkey* randomly removes ≈0.25 % of questionnaire cells
  and ≈5 % of demographics cells to simulate real-world data quality.
- `missing` values appear as empty strings in CSV and `null` in JSON.

### JSON Schema

The `--output schema` flag (or `--schema`) prints a
[JSON Schema (Draft 7)](https://json-schema.org/) document describing the
output row type, suitable for validation or code generation.

## Package API

The package can also be used programmatically from Julia:

```julia
using IbOxDummies

# Run with all defaults
data, schema = simulate(SimulationConfig(seed = 42))

# Write CSV to stdout
to_csv(data, schema)

# Write JSON to stdout
to_json(data, schema)

# Get JSON Schema string
json_schema = to_json_schema(schema)

# Custom questionnaires
my_qs = Dict{String, Questionnaire}(
    "PHQ_9" => make_phq9(),
    "GAD_7" => make_gad7(),
)
config = SimulationConfig(
    nWaves                     = 2,
    nSchools                   = 5,
    nYeargroupsPerSchool       = Range(3, 6),
    nClassesPerSchoolYeargroup = Range(1, 4),
    nStudentsPerClass          = NormalDist(28.0, 5.0),
    questionnaires             = my_qs,
    seed                       = 123,
    output                     = "csv",
)
data, schema = simulate(config)
```

### Key types

| Type | Description |
|------|-------------|
| `AData` | `Union{Int, Float64, String, Missing}` — a single answer |
| `QData` | `Dict{String, AData}` — one student-wave row |
| `Schema` | Column metadata (demographics vs questionnaire columns) |
| `Range` | Inclusive integer range `[min, max]` |
| `NormalDist` | Normal distribution `N(μ, σ)` |
| `SimulationConfig` | All simulation parameters |
| `Questionnaire` | `(rng, studentData, schema) -> QData` |

## Questionnaires

### PHQ-9 (Patient Health Questionnaire-9)

Nine items scored 0–3, measuring depression severity. Population distribution
follows approximate UK estimates: ~50 % none, 25 % mild, 15 % moderate,
10 % severe. Longitudinal continuity is modelled by sampling from previous
responses with category-specific standard deviations.

### GAD-7 (Generalised Anxiety Disorder-7)

Seven items scored 0–3, measuring anxiety severity. Similar longitudinal
modelling approach to PHQ-9.

## Development

Run the test suite:

```bash
julia --project=. test/runtests.jl
```

## License

See [LICENSE](LICENSE).
