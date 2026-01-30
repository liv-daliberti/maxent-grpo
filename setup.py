"""
Copyright 2025 Liv d'Aliberti

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Adapted from huggingface/transformers: https://github.com/huggingface/transformers/blob/21a2d900eceeded7be9edc445b56877b95eda4ca/setup.py


import importlib
import importlib.metadata as importlib_metadata
import re
import shutil
import sys
from pathlib import Path

from setuptools import find_packages, setup

_EXPECTED_TRL_VERSION_PREFIX = "0.18."


def _trl_version_hint() -> str:
    """Return the installed TRL version if available."""
    try:
        return importlib_metadata.version("trl")
    except Exception:
        return "unknown"


def _warn(msg: str) -> None:
    """Emit a loud setup-time warning to stderr."""
    print(f"[open-r1 setup] {msg}", file=sys.stderr)


def _patch_trl_vllm_serve():
    """
    Ensure TRL's vllm_serve.py passes use_tqdm_on_load=True and returns logprobs.

    We locate trl.scripts.vllm_serve, and if the LLM(...) call does not
    already include use_tqdm_on_load, we insert it after enforce_eager. We
    also augment the /generate handler to accept return_logprobs/logprobs and
    emit token-level logprob metadata so downstream callers can reconstruct
    reference logprobs from vLLM responses.
    """
    trl_version = _trl_version_hint()
    if trl_version not in ("unknown",) and not str(trl_version).startswith(
        _EXPECTED_TRL_VERSION_PREFIX
    ):
        _warn(
            f"trl=={trl_version} differs from expected {_EXPECTED_TRL_VERSION_PREFIX}*; "
            "vllm_serve patch may fail."
        )
    try:
        trl_mod = importlib.import_module("trl.scripts.vllm_serve")
        path = Path(trl_mod.__file__)
    except Exception as e:
        _warn(f"Skipping TRL patch (import failed; trl=={trl_version}): {e}")
        return False

    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        _warn(f"Skipping TRL patch (read failed at {path}): {e}")
        return False

    patched = False
    tqdm_anchor = "enforce_eager=script_args.enforce_eager,"
    tqdm_insert = "\n        use_tqdm_on_load=True,"
    if "use_tqdm_on_load=True" not in text:
        new_text = None
        if tqdm_anchor in text:
            new_text = text.replace(tqdm_anchor, tqdm_anchor + tqdm_insert)
        else:
            alt_anchor = "dtype=script_args.dtype,"
            if alt_anchor in text:
                new_text = text.replace(alt_anchor, alt_anchor + tqdm_insert)
            else:
                llm_call = "llm = LLM("
                if llm_call in text:
                    new_text = text.replace(llm_call, llm_call + tqdm_insert + "\n")
        if new_text is None:
            raise RuntimeError(
                f"[open-r1 setup] Could not find insertion point in TRL vllm_serve "
                f"(trl=={trl_version}, path={path}). Update the patch to match the "
                "current TRL release."
            )
        text = new_text
        patched = True
    else:
        print("[open-r1 setup] TRL vllm_serve already patched (use_tqdm_on_load=True).")

    needs_logprob_patch = "output_token_logprobs" not in text and "return_logprobs" not in text
    if needs_logprob_patch:
        req_anchor = (
            "        max_tokens: int = 16\n"
            "        guided_decoding_regex: Optional[str] = None\n"
        )
        req_replacement = (
            req_anchor
            + "        return_logprobs: bool = False\n"
            "        logprobs: Optional[int] = None\n"
        )
        if req_anchor not in text:
            raise RuntimeError(
                f"[open-r1 setup] Could not find GenerateRequest anchor in TRL vllm_serve "
                f"(trl=={trl_version}, path={path}) to inject logprob fields."
            )
        text = text.replace(req_anchor, req_replacement)

        resp_anchor = (
            '    class GenerateResponse(BaseModel):\n'
            '        completion_ids: list[list[int]]\n'
            "\n"
            '    @app.post("/generate/", response_model=GenerateResponse)\n'
            "    async def generate(request: GenerateRequest):\n"
        )
        resp_replacement = (
            '    class GenerateResponse(BaseModel):\n'
            "        text: list[str]\n"
            "        token_ids: list[list[int]]\n"
            "        output_token_logprobs: Optional[list[Optional[list[float]]]] = None\n"
            "        cumulative_logprob: Optional[list[Optional[float]]] = None\n"
            "        completion_ids: Optional[list[list[int]]] = None\n"
            "\n"
            '    @app.post("/generate/", response_model=GenerateResponse)\n'
            "    async def generate(request: GenerateRequest):\n"
        )
        if resp_anchor not in text:
            raise RuntimeError(
                f"[open-r1 setup] Could not update GenerateResponse in TRL vllm_serve "
                f"(trl=={trl_version}, path={path}) to include logprob metadata."
            )
        text = text.replace(resp_anchor, resp_replacement)

        body_anchor = (
            "        # Guided decoding, if enabled\n"
            "        if request.guided_decoding_regex is not None:\n"
            '            guided_decoding = GuidedDecodingParams(backend="outlines", regex=request.guided_decoding_regex)\n'
            "        else:\n"
            "            guided_decoding = None\n"
            "\n"
            "        # Sampling parameters\n"
            "        sampling_params = SamplingParams(\n"
            "            n=request.n,\n"
            "            repetition_penalty=request.repetition_penalty,\n"
            "            temperature=request.temperature,\n"
            "            top_p=request.top_p,\n"
            "            top_k=request.top_k,\n"
            "            min_p=request.min_p,\n"
            "            max_tokens=request.max_tokens,\n"
            "            guided_decoding=guided_decoding,\n"
            "        )\n"
            "        # Evenly distribute prompts across DP ranks\n"
            "        chunked_prompts = chunk_list(request.prompts, script_args.data_parallel_size)\n"
            "\n"
            "        # Send the prompts to each worker\n"
            "        for connection, prompts in zip(connections, chunked_prompts):\n"
            "            # When the number of prompts is less than data_parallel_size, some workers will receive empty prompts.\n"
            "            # However, vLLM requires that we always send at least one prompt. So we send a placeholder prompt to comply\n"
            "            # with vLLM's requirement, and we later ignore the result.\n"
            "            if not prompts:\n"
            '                prompts = ["<placeholder>"]\n'
            '            kwargs = {"prompts": prompts, "sampling_params": sampling_params}\n'
            '            connection.send({"type": "call", "method": "generate", "kwargs": kwargs})\n'
            "\n"
            "        # Receive results\n"
            "        all_outputs = [connection.recv() for connection in connections]\n"
            "\n"
            "        # Handle empty prompts (see above)\n"
            "        all_outputs = [output for output, prompts in zip(all_outputs, chunked_prompts) if prompts]\n"
            "\n"
            "        # Flatten and combine all results\n"
            "        all_outputs = list(chain.from_iterable(all_outputs))  # from list of list to single list\n"
            "        completion_ids = [list(output.token_ids) for outputs in all_outputs for output in outputs.outputs]\n"
            '        return {"completion_ids": completion_ids}\n'
        )
        body_replacement = (
            "        logprobs = request.logprobs if request.return_logprobs else None\n"
            "\n"
            "        # Guided decoding, if enabled\n"
            "        if request.guided_decoding_regex is not None:\n"
            '            guided_decoding = GuidedDecodingParams(backend="outlines", regex=request.guided_decoding_regex)\n'
            "        else:\n"
            "            guided_decoding = None\n"
            "\n"
            "        # Sampling parameters\n"
            "        sampling_params = SamplingParams(\n"
            "            n=request.n,\n"
            "            repetition_penalty=request.repetition_penalty,\n"
            "            temperature=request.temperature,\n"
            "            top_p=request.top_p,\n"
            "            top_k=request.top_k,\n"
            "            min_p=request.min_p,\n"
            "            max_tokens=request.max_tokens,\n"
            "            guided_decoding=guided_decoding,\n"
            "            logprobs=logprobs,\n"
            "        )\n"
            "        # Evenly distribute prompts across DP ranks\n"
            "        chunked_prompts = chunk_list(request.prompts, script_args.data_parallel_size)\n"
            "\n"
            "        # Send the prompts to each worker\n"
            "        for connection, prompts in zip(connections, chunked_prompts):\n"
            "            # When the number of prompts is less than data_parallel_size, some workers will receive empty prompts.\n"
            "            # However, vLLM requires that we always send at least one prompt. So we send a placeholder prompt to comply\n"
            "            # with vLLM's requirement, and we later ignore the result.\n"
            "            if not prompts:\n"
            '                prompts = ["<placeholder>"]\n'
            '            kwargs = {"prompts": prompts, "sampling_params": sampling_params}\n'
            '            connection.send({"type": "call", "method": "generate", "kwargs": kwargs})\n'
            "\n"
            "        # Receive results\n"
            "        all_outputs = [connection.recv() for connection in connections]\n"
            "\n"
            "        # Handle empty prompts (see above)\n"
            "        all_outputs = [output for output, prompts in zip(all_outputs, chunked_prompts) if prompts]\n"
            "\n"
            "        # Flatten and combine all results\n"
            "        all_outputs = list(chain.from_iterable(all_outputs))  # from list of list to single list\n"
            "        texts = []\n"
            "        token_ids = []\n"
            "        token_logprobs = []\n"
            "        cumulative_logprobs = []\n"
            "        for outputs in all_outputs:\n"
            "            for output in outputs.outputs:\n"
            "                ids = list(output.token_ids)\n"
            "                token_ids.append(ids)\n"
            "                if logprobs is not None and getattr(output, \"logprobs\", None):\n"
            "                    per_token = []\n"
            "                    for tok_id, lp_dict in zip(ids, output.logprobs):\n"
            "                        lp_val = None\n"
            "                        try:\n"
            "                            lp = lp_dict.get(tok_id)\n"
            "                            if lp is not None:\n"
            "                                lp_val = float(lp.logprob)\n"
            "                        except Exception:\n"
            "                            lp_val = None\n"
            "                        per_token.append(lp_val)\n"
            "                    token_logprobs.append(per_token)\n"
            "                else:\n"
            "                    token_logprobs.append(None)\n"
            "                cumulative_logprobs.append(output.cumulative_logprob)\n"
            "                texts.append(output.text)\n"
            "        response = {\n"
            '            "text": texts,\n'
            '            "token_ids": token_ids,\n'
            '            "completion_ids": token_ids,\n'
            "        }\n"
            "        if logprobs is not None:\n"
            '            response["output_token_logprobs"] = token_logprobs\n'
            '            response["cumulative_logprob"] = cumulative_logprobs\n'
            "        return response\n"
        )
        if body_anchor not in text:
            raise RuntimeError(
                f"[open-r1 setup] Could not replace generate() body in TRL vllm_serve "
                f"(trl=={trl_version}, path={path}) to return logprobs."
            )
        text = text.replace(body_anchor, body_replacement)
        patched = True
    else:
        print("[open-r1 setup] TRL vllm_serve already patched for logprobs.")

    if not patched:
        return True

    try:
        path.write_text(text, encoding="utf-8")
        print(f"[open-r1 setup] Patched TRL vllm_serve at: {path}")
    except Exception as e:
        _warn(f"Failed to write TRL patch at {path}: {e}")
        return False
    return True

# NOTE: install-time patching is disabled. Run ops/scripts/patch_trl_vllm_serve.py
# explicitly after installing/upgrading TRL to apply this patch.


# Remove stale open_r1.egg-info directories to avoid https://github.com/pypa/pip/issues/5466
repo_root = Path(__file__).parent
stale_egg_infos = [
    repo_root / "open_r1.egg-info",
    repo_root / "src" / "open_r1.egg-info",
]
for stale_egg_info in stale_egg_infos:
    if stale_egg_info.exists():
        print(
            (
                "Warning: {} exists.\n\n"
                "If you recently updated open_r1, this is expected,\n"
                "but it may prevent open_r1 from installing in editable mode.\n\n"
                "This directory is automatically generated by Python's packaging tools.\n"
                "I will remove it now.\n\n"
                "See https://github.com/pypa/pip/issues/5466 for details.\n"
            ).format(stale_egg_info)
        )
        shutil.rmtree(stale_egg_info)


# IMPORTANT: all dependencies should be listed here with their version requirements, if any.
#   * If a dependency is fast-moving (e.g. trl), pin to the exact version
_deps = [
    "accelerate==1.4.0",
    "antlr4-python3-runtime>=4.13.2",
    "bitsandbytes>=0.43.0",
    "datasets>=3.2.0",
    "deepspeed==0.16.8",
    "distilabel[vllm,ray,openai]>=1.5.2",
    "e2b-code-interpreter>=1.0.5",
    "einops>=0.8.0",
    "flake8>=6.0.0",
    "pylint>=3.2.0",
    "pre-commit>=3.8.0",
    "hf_transfer>=0.1.4",
    "huggingface-hub[cli,hf_xet]>=0.30.2,<1.0",
    "isort>=5.12.0",
    "jieba",  # Needed for Chinese language support
    "langdetect",  # Needed for LightEval's extended tasks
    "liger-kernel>=0.5.10",
    "lighteval @ git+https://github.com/huggingface/lighteval.git@d3da6b9bbf38104c8b5e1acc86f83541f9a502d1",  # Critical bug fix for tokenizer revisions
    "morphcloud==0.1.67",
    "more-itertools>=10.0.0",
    "numpy>=1.26.4",
    "packaging>=23.0",
    "parameterized>=0.9.0",
    "pydantic>=2.7",
    "peft>=0.14.0",
    "pytest",
    "python-dotenv",
    "ruff>=0.9.0",
    "safetensors>=0.3.3",
    "sentencepiece>=0.1.99",
    "torch==2.8.0",
    "transformers==4.53.0",
    "trl[vllm]==0.18.0",
    "wandb>=0.19.1",
    "async-lru>=2.0.5",
    "aiofiles>=24.1.0",
    "pandas>=2.2.3",
    "sphinx>=7.2",
    "sphinx-rtd-theme>=1.3",
    "myst-parser",
    "sphinx-copybutton",
    "sphinx-design",
    "linkify-it-py",
    "mdurl",
    "omegaconf>=2.3.0",
    "hydra-core>=1.3.2",
    "typer>=0.12.5",
]

# this is a lookup table with items like:
#
# tokenizers: "tokenizers==0.9.4"
# packaging: "packaging"
#
# some of the values are versioned whereas others aren't.
deps = {
    b: a
    for a, b in (
        re.findall(r"^(([^!=<>~ \[\]]+)(?:\[[^\]]+\])?(?:[!=<>~ ].*)?$)", x)[0]
        for x in _deps
    )
}


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]


extras = {}
extras["tests"] = deps_list("pytest", "parameterized", "jieba")
extras["torch"] = deps_list("torch")
extras["quality"] = deps_list("ruff", "isort", "flake8", "pylint", "pre-commit")
extras["docs"] = deps_list(
    "sphinx",
    "sphinx-rtd-theme",
    "myst-parser",
    "sphinx-copybutton",
    "sphinx-design",
    "linkify-it-py",
    "mdurl",
)
extras["code"] = deps_list(
    "e2b-code-interpreter", "python-dotenv", "morphcloud", "jieba", "pandas", "aiofiles"
)
extras["eval"] = deps_list("lighteval")
extras["dev"] = (
    extras["quality"]
    + extras["tests"]
    + extras["eval"]
    + extras["code"]
    + extras["docs"]
)

# core dependencies shared across the whole project - keep this to a bare minimum :)
install_requires = [
    deps["accelerate"],
    deps["bitsandbytes"],
    deps["einops"],
    deps["datasets"],
    deps["deepspeed"],
    deps["hf_transfer"],
    deps["huggingface-hub"],
    deps["langdetect"],
    deps["liger-kernel"],
    deps["numpy"],
    deps["packaging"],  # utilities from PyPA to e.g., compare versions
    deps["safetensors"],
    deps["sentencepiece"],
    deps["transformers"],
    deps["trl"],
    deps["wandb"],
    deps["async-lru"],
    deps["typer"],
    deps["omegaconf"],
    deps["hydra-core"],
]


# Long description from README if present
readme_path = repo_root / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")
else:
    long_description = "Open R1"


setup(
    name="open-r1",
    version="0.1.0.dev0",  # expected format is one of x.y.z.dev0, or x.y.z.rc1 or x.y.z (no to dashes, yes to dots)
    author="The Hugging Face team (past and future)",
    author_email="lewis@huggingface.co",
    description="Open R1",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="llm inference-time compute reasoning",
    license="Apache-2.0",
    url="https://github.com/huggingface/open-r1",
    package_dir={"": "src"},
    packages=find_packages("src"),
    zip_safe=False,
    cmdclass={},
    extras_require=extras,
    python_requires=">=3.10.9",
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "maxent-grpo=maxent_grpo.cli.hydra_cli:hydra_entry",
            "maxent-grpo-baseline=maxent_grpo.cli.hydra_cli:baseline_entry",
            "maxent-grpo-maxent=maxent_grpo.cli.hydra_cli:maxent_entry",
            "maxent-grpo-infoseed=maxent_grpo.cli.hydra_cli:infoseed_entry",
            "maxent-grpo-generate=maxent_grpo.cli.hydra_cli:generate_entry",
            "maxent-grpo-inference=maxent_grpo.cli.hydra_cli:inference_entry",
            "maxent-grpo-math-eval=maxent_grpo.cli.hydra_cli:math_eval_entry",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
