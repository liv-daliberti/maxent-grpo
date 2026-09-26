# PantryPlan: verified ingredient-formulation ModeBench

Status: deterministic admission passed; both frozen 0.5B viability gates failed; stopped before training  
Date: 2026-07-29

## Terminal viability outcome

The prospective three-way v2 split passed deterministic admission with 384
train, 64 development, and 128 evaluation prompts. The frozen base-model gate
then failed with zero verified completions across the 64 development prompts.
Its one allowed assistant-prefix prompt repair also returned zero verified
completions. PantryPlan is therefore ineligible for 0.5B training under this
protocol. No threshold, prompt row, verifier rule, or support identity was
changed after observing the failures.

## Admission outcome

The April 2026 USDA Foundation Foods archive and 16-row reviewed ingredient
table are frozen. The generated dataset has 96 train and 32 evaluation prompts,
balanced across four families, with zero prompt overlap. Exhaustive exact
enumeration replayed 1,819 train and 604 evaluation supports; every prompt has
between 8 and 44 valid support modes. The source, manual-review, support,
adversarial, and split audits pass. No model completion or training job has
been sampled.

## Decision

Develop `PantryPlan` as a gated ModeBench domain for structured, constraint-valid ingredient formulations. A prompt supplies a frozen pantry, integer quantity increments, nutrient bounds, dietary exclusions, and a serving-mass range. The response allocates grams to a subset of the available ingredients. The validator checks inventory, exclusions, and every numeric constraint with exact decimal arithmetic.

This is deliberately not a claim about taste, cooking success, food safety, or whether two textual recipes produce the same dish. Those properties are not executable from text in this repository. PantryPlan tests whether online verified MaxEnt discovers multiple materially different feasible ingredient sets under ordinary, legible household constraints.

The verifier is implemented in `src/oat_drgrpo/pantry_plan.py` and connected
to the same reward and canonical-identity boundary as the existing ModeBench
domains. The frozen dataset and admission audit now authorize only the next
preregistered 0.5B development-set viability sample. They do not authorize a
paper claim, experiment identifier, or confirmatory training cohort.

## Source snapshot

Use USDA FoodData Central Foundation Foods rather than a mutable recipe website or an LLM-generated nutrition table.

- release: Foundation Foods, April 2026;
- archive: `FoodData_Central_foundation_food_json_2026-04-30.zip`;
- observed archive SHA-256: `186e988ec542e913f51ef62b86a47758e8cdd0d1dc3889e7b055581f3c09c77a`;
- physical source rows: 395;
- license: CC0 1.0 / U.S. public-domain data;
- official download page: <https://fdc.nal.usda.gov/download-datasets/>;
- official API and licensing guide: <https://fdc.nal.usda.gov/api-guide/>.

Freeze a small reviewed table of ready-to-use ingredients by FDC ID. Retain the source description, publication date, nutrient-number mapping, unit, reported amount, extraction rule, source-archive hash, and row hash. Do not silently substitute a branded food, an older release, a different cooking state, or an imputed nutrient value.

The first table uses only attributes explicitly present for every selected food: energy in kcal, protein in g, dietary fiber in g, and sodium in mg. USDA Foundation rows use more than one documented energy or fiber nutrient number across vintages. The curation artifact must freeze the precedence rule and reject ambiguous duplicates. Cost is excluded until a separately licensed, date- and region-pinned price source exists.

## Executable contract

The candidate syntax is:

```text
ingredient_id=grams;ingredient_id=grams
```

The validator:

1. parses a bounded list of unique ingredient IDs and positive integer grams;
2. requires every ingredient to exist in the prompt-local pantry;
3. enforces its minimum serving, available quantity, and quantity increment;
4. rejects every used ingredient carrying a forbidden dietary tag;
5. computes mass and nutrition totals with exact decimal arithmetic;
6. enforces ingredient-count, mass, energy, protein, fiber, and sodium bounds; and
7. emits a mode key only after all checks pass.

Malformed text, duplicate ingredients, unknown IDs, missing nutrient values, fractional grams, quantity-step violations, forbidden tags, and failed bounds receive reward zero and no key.

## Semantic identity

The mode key is the sorted set of ingredients used:

```text
pantry_plan:pantry-v1:brown_rice+lentils
```

Quantities affect correctness but not identity. Thus `lentils=200;brown_rice=100` and `brown_rice=125;lentils=200` are one formulation mode, while replacing lentils with chickpeas is a different mode. Word order, whitespace, delimiters, explanations, and gram-level perturbations cannot inflate support.

This identity says “different ingredient formulation,” not “different tasting dish.” Preparation prose and recipe names are ineligible mode keys.

## Four prompt families

1. **Plant-protein bowl:** minimum protein and fiber under energy, mass, and sodium ceilings.
2. **Breakfast formulation:** grain/fruit/nut-or-seed alternatives with an allergen-exclusion variant.
3. **Low-sodium pantry meal:** a stricter sodium ceiling with several legume/vegetable support combinations.
4. **High-fiber snack plate:** a smaller mass range and substitutable fruit, vegetable, seed, and spread supports.

Family labels affect generation and stratified reporting, not validation. Every row uses the same generic executable contract.

## Source and support audit

Before policy sampling:

- manually review every selected FDC row and dietary tag;
- reject foods whose named state requires an unstated preparation assumption;
- freeze the ingredient-table and source hashes;
- enumerate all prompt-local allocations over the registered gram increments;
- canonicalize each accepted allocation to its ingredient-support key;
- require at least eight and at most 256 valid support modes per prompt;
- independently replay at least two certified allocations for every row;
- adversarially test duplicate IDs, unknown foods, boundary quantities, decimal aliases, allergen violations, and missing nutrient values;
- freeze disjoint train/development/evaluation prompt identities; and
- verify that no evaluation constraint set is used for curation or viability.

Exhaustive enumeration is an audit tool only. Valid mode counts, target support, and certified answers may not enter the policy prompt or training objective.

## Model viability gate

Use Qwen2.5-0.5B-Instruct, matching the non-code ModeBench rows.

1. Freeze 64 development prompts across the four families.
2. Sample 16 responses per prompt at the intended response budget.
3. Require at least 50% of prompts to have one verified formulation.
4. At 64 samples per prompt, require at least 25% to expose two distinct verified ingredient-support modes.

If the gate fails, make one prospective prompt-format repair based only on parser/instruction failure categories, restart the full development sample, and stop if it fails again. Do not curate prompts around observed policy successes.

## Experimental ladder

### Gate A: deterministic smoke

- 16 development prompts, four per family;
- Dr.GRPO and frozen E58-style online verified MaxEnt;
- seed 43, one pass;
- matched prompts, sampling, verifier calls, replay capacity, optimizer updates, and evaluation draws;
- zero parser, exact-arithmetic, cadence, resume, or information-firewall violations.

### Gate B: compute-matched screen

- 128 train / 64 development prompts;
- seed 43, six passes;
- terminal and fixed-checkpoint AUC for `pass@8` and `distinct@8`;
- advance only if MaxEnt improves `distinct@8` in at least three of four families and no family loses more than .03 terminal `pass@8`.

### Gate C: confirmatory extension

- at least 384 train / 128 evaluation prompts;
- seeds 43, 44, 45, 46, and 47;
- 12 passes and the paper's fixed checkpoint policy;
- report all seed trajectories, parse failures, constraint-failure categories, verified support, discoveries, replay activity, and controller state.

No peak checkpoint, result-dependent family replacement, quantity-resolution change, or post-outcome nutrient-bound change is evidence.

## Integration with the clean cohort

PantryPlan is one of the eight visible rows in the superseding MATH-free design
in `docs/clean_05b_maxent_vs_drgrpo_eight_environment_plan.md`. It passed its
deterministic admission gate, but the whole 80-job cohort stopped before any
viability sample because ConstructiveCode v1 and AntMaze were ineligible.
A separately versioned viability study was later authorized. Both its initial
sample and one allowed prompt-only repair failed with zero verified
completions, so PantryPlan remains an explicit ineligible row and did not join
E70 Stage A.

## Recorded stopping point

1. Completed: verifier, exact arithmetic, support identity, grader integration,
   and focused tests.
2. Completed: pinned USDA extraction and reviewed 16-ingredient table.
3. Completed: exact support enumeration and balanced four-family generator.
4. Passed: deterministic source, support, adversarial, and split audit.
5. Failed: the frozen 0.5B development-set viability gate, with zero
   verified completions across 64 prompts.
6. Failed: the one allowed assistant-prefix prompt repair, also with zero
   verified completions; stopped before any training job.
