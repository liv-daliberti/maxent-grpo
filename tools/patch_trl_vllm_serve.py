#!/usr/bin/env python3
"""Patch TRL's vllm_serve to return logprobs and expose tqdm-on-load.

Run explicitly after installing/upgrading TRL. This avoids silent drift during
`pip install -e` and provides clearer failures when the upstream file changes.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


def _warn(msg: str) -> None:
    print(f"[maxent-grpo patch] {msg}", file=sys.stderr)


def _trl_version_hint() -> str:
    try:
        import importlib.metadata as importlib_metadata

        return importlib_metadata.version("trl")
    except Exception:
        return "unknown"


def patch_trl_vllm_serve(*, dry_run: bool = False) -> bool:
    """Patch trl.scripts.vllm_serve in-place.

    Returns True when the file is already patched or patched successfully.
    Raises RuntimeError on unrecognized source layouts.
    """

    trl_version = _trl_version_hint()
    try:
        trl_mod = importlib.import_module("trl.scripts.vllm_serve")
        path = Path(trl_mod.__file__)
    except Exception as exc:
        raise RuntimeError(
            f"Could not import trl.scripts.vllm_serve (trl=={trl_version}): {exc}"
        ) from exc

    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc

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
                "Could not find insertion point for use_tqdm_on_load=True. "
                "Update the patch to match the installed TRL version."
            )
        text = new_text
        patched = True

    needs_logprob_patch = (
        "output_token_logprobs" not in text and "return_logprobs" not in text
    )
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
                "Could not find GenerateRequest anchor to inject logprob fields."
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
                "Could not find GenerateResponse anchor to inject logprob outputs."
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
            "                    for logprob_item in output.logprobs:\n"
            "                        if logprob_item is None:\n"
            "                            per_token.append(None)\n"
            "                            continue\n"
            "                        token_logprob = getattr(logprob_item, \"logprob\", None)\n"
            "                        if token_logprob is None:\n"
            "                            token_logprob = getattr(logprob_item, \"token_logprob\", None)\n"
            "                        if token_logprob is None:\n"
            "                            per_token.append(None)\n"
            "                        else:\n"
            "                            per_token.append(float(token_logprob))\n"
            "                    token_logprobs.append(per_token)\n"
            "                    cumulative_logprob = getattr(output, \"cumulative_logprob\", None)\n"
            "                    cumulative_logprobs.append(cumulative_logprob)\n"
            "                texts.append(output.text)\n"
            "        response = {\"text\": texts, \"token_ids\": token_ids}\n"
            "        if logprobs is not None:\n"
            "            response[\"output_token_logprobs\"] = token_logprobs\n"
            "            response[\"cumulative_logprob\"] = cumulative_logprobs\n"
            "        return response\n"
        )
        if body_anchor not in text:
            raise RuntimeError(
                "Could not find generate() body anchor to inject logprob outputs."
            )
        text = text.replace(body_anchor, body_replacement)
        patched = True

    if not patched:
        print("[maxent-grpo patch] TRL vllm_serve already patched.")
        return True

    if dry_run:
        print(f"[maxent-grpo patch] Dry run: would patch {path}")
        return True

    try:
        path.write_text(text, encoding="utf-8")
    except Exception as exc:
        raise RuntimeError(f"Failed to write patched file {path}: {exc}") from exc

    print(f"[maxent-grpo patch] Patched TRL vllm_serve at: {path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Patch TRL vllm_serve to return logprobs for MaxEnt-GRPO."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Do not write changes, just check."
    )
    args = parser.parse_args()
    try:
        patch_trl_vllm_serve(dry_run=args.dry_run)
    except RuntimeError as exc:
        _warn(str(exc))
        _warn(
            "Patch failed. This usually means TRL changed upstream. "
            "Install trl==0.18.* or update the patch script."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
