#!/usr/bin/env python3
"""Compose the treatment-candidate Level-2 r3 from admitted immutable revisions."""
from __future__ import annotations
from collections import Counter
import json,shutil,tempfile
from pathlib import Path
from datasets import Dataset,DatasetDict
from materialize_modebench_harder_v2 import (
    ROOT,LEVEL1,SPLITS,SPLIT_SEED_OFFSETS,SEEDS,build_countdown,existing_ids,
    identity_set,load_rows,modes,row_hash,
)

R1=ROOT/"var/data/modebench_harder_v2_matched_r1"
R2=ROOT/"var/data/modebench_harder_v2_matched_r2"
OUTPUT=ROOT/"var/data/modebench_harder_v2_matched_r3"

def main(output:Path=OUTPUT)->dict:
    output=output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    r1=json.loads((R1/"identity.json").read_text())
    manifest=json.loads((R2/"identity.json").read_text())
    staging=Path(tempfile.mkdtemp(prefix=".modebench_harder_v2_r3.",dir=output.parent))
    try:
        shutil.rmtree(staging)
        shutil.copytree(R2,staging)
        for domain in ("graph_coloring","python_factors","pantry"):
            shutil.rmtree(staging/domain)
            shutil.copytree(R1/domain,staging/domain)
            manifest["domains"][domain]=r1["domains"][domain]
        shutil.rmtree(staging/"countdown")
        blocked=existing_ids("countdown")
        records={}
        for split in ("dev","eval","train"):
            expected,dataset_split=SPLITS[split]
            easy=load_rows(LEVEL1["countdown"][split],dataset_split)
            target=Counter(int(x["answer_mode_count"]) for x in easy)
            seed=SEEDS["countdown"]+SPLIT_SEED_OFFSETS[split]
            rows=build_countdown(target,seed,f"level2_{split}",blocked)
            ids=identity_set("countdown",rows)
            verifier=lambda row:(str(row.get("modebench_task","")),str(json.loads(row["answer"]).get("verifier","")))
            checks={
                "row_count":len(rows)==expected,
                "unique_identities":len(ids)==expected,
                "disjoint_from_all_level1_and_prior_level2":not(ids&blocked),
                "exact_support_histogram":modes(rows)==target,
                "verifier_and_canonicalization_contract":{verifier(x) for x in easy}=={verifier(x) for x in rows},
            }
            if not all(checks.values()):
                raise RuntimeError(f"countdown/{split} structural failure: {checks}")
            if not all(len(json.loads(x["answer"])["numbers"])==4 for x in rows):
                raise RuntimeError(f"countdown/{split}: expected four operands")
            DatasetDict({dataset_split:Dataset.from_list(rows)}).save_to_disk(str(staging/"countdown"/split))
            records[split]={
                "seed":seed,"level1_reference_rows":len(easy),"histogram_scale_factor":1,
                "rows":len(rows),"rows_sha256":row_hash(rows),"checks":checks,
                "answer_mode_count_histogram":dict(sorted(target.items())),
            }
            blocked|=ids
        manifest["domains"]["countdown"]=records
        manifest["difficulty"]["countdown"]="four operands versus three; operands up to 14; within each exact support cell prefer shallow paired-product or one-multiplication targets, then smaller operands"
        manifest["revision_provenance"]={
            "countdown":"regenerated development-first by compose_modebench_level2_r3.py from Level-1 support targets",
            "mathir":"byte-identical datasets and records from modebench_harder_v2_matched_r2",
            "graph_coloring":"byte-identical datasets and records from modebench_harder_v2_matched_r1",
            "python_factors":"byte-identical datasets and records from modebench_harder_v2_matched_r1",
            "pantry":"byte-identical datasets and records from modebench_harder_v2_matched_r1",
        }
        for name in ("identity.json","admission_fairness_report.json"):
            (staging/name).write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
        staging.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(staging,ignore_errors=True)
        raise

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root",type=Path,default=OUTPUT)
    args=parser.parse_args()
    print(json.dumps(main(args.output_root),indent=2,sort_keys=True))
