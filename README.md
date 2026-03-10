# ib-ox-dummies

A simulator for generating realistic mock longitudinal questionnaire data,
designed to represent annual survey responses from schoolchildren.

Built in [Julia](https://julialang.org/) using:
- **[ArgParse.jl](https://github.com/carlobaldassi/ArgParse.jl)** — CLI argument parsing
- **[Distributions.jl](https://github.com/JuliaStats/Distributions.jl)** — statistical distributions for realistic count and item sampling
- **[StatsBase.jl](https://github.com/JuliaStats/StatsBase.jl)** — weighted categorical sampling for demographics
- **[DataFrames.jl](https://github.com/JuliaData/DataFrames.jl)** — tabular output representation
- **[CSV.jl](https://github.com/JuliaData/CSV.jl)** — CSV serialisation
- **[JSON3.jl](https://github.com/quinnj/JSON3.jl)** — JSON serialisation

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
usage: ib_ox_dummies [--nWaves NWAVES] [--nSchools NSCHOOLS]
                     [--nYeargroupsPerSchool NYEARGROUPSPERSCHOOL]
                     [--nClassesPerSchoolYeargroup NCLASSESPERSCHOOLYEARGROUP]
                     [--nStudentsPerClass NSTUDENTSPERCLASS]
                     [--seed SEED] [--output OUTPUT] [--schema]
                     [--version] [-h]

Generate mock longitudinal questionnaire data for schoolchildren.
SPEC formats: integer (e.g. '5'), inclusive range (e.g. '1:5'), or
normal distribution (e.g. 'norm(30,7)').

optional arguments:
  --nWaves NWAVES       Number of data-collection waves (type: Int64, default: 3)
  --nSchools NSCHOOLS   Number of schools (type: Int64, default: 10)
  --nYeargroupsPerSchool NYEARGROUPSPERSCHOOL
                        Yeargroups per school (SPEC) (default: "5")
  --nClassesPerSchoolYeargroup NCLASSESPERSCHOOLYEARGROUP
                        Classes per school yeargroup (SPEC) (default: "1:5")
  --nStudentsPerClass NSTUDENTSPERCLASS
                        Students per class (SPEC) (default: "norm(30,7)")
  --seed SEED           Random seed for reproducibility (type: Int64)
  --output OUTPUT       Output format: csv | json | schema (default: "csv")
  --schema              Print JSON Schema describing the output columns and exit
  --version             show version information and exit
  -h, --help            show this help message and exit
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
using Distributions  # for Normal, truncated, etc.

# Run with all defaults — returns a DataFrame
data, schema = simulate(SimulationConfig(seed = 42))

# Write CSV to stdout (uses CSV.jl)
to_csv(data, schema)

# Write JSON to stdout (uses JSON3.jl)
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
    nStudentsPerClass          = Normal(28.0, 5.0),    # Distributions.Normal
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
| `QData` | `Dict{String, AData}` — internal per-student-wave record |
| `Schema` | Column metadata (demographics vs questionnaire columns) |
| `Range` | Inclusive integer range `[min, max]` |
| `CountSpec` | `Union{Int, Range, UnivariateDistribution}` — count specification |
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
