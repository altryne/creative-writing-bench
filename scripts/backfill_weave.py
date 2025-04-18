from weave.flow.eval_imperative import ImperativeEvaluationLogger
from utils.file_io import load_json_file

runs = load_json_file("creative_bench_runs.json")
ev = ImperativeEvaluationLogger(
    model={"backfill": "true"},
    dataset="eq-bench-creative"
)
for run in runs.values():
    for item in run["items"]:
        pred = ev.log_prediction(
            inputs={"prompt": item["prompt"]},
            output=item["output"]
        )
        pred.log_score("rubric_0_20", item["scores"]["rubric"])
        pred.finish()
ev.log_summary({"backfilled_runs": len(runs)})
