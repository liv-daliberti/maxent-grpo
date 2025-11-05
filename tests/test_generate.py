import generate


def test_build_distilabel_pipeline_smoke():
    pipe = generate.build_distilabel_pipeline(
        model="x",
        base_url="http://localhost:8000/v1",
        prompt_column="prompt",
        prompt_template="{{ instruction }}",
        max_new_tokens=16,
        num_generations=1,
        input_batch_size=2,
    )
    assert pipe is not None

