# custom_tasks/math500_passk.py
import re
import numpy as np

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.metrics.metrics_sample import SampleLevelComputation

# Optional: SymPy-based symbolic equivalence (best-effort)
try:
    import sympy as sp
    from latex2sympy2_extended.latex2sympy2 import latex2sympy
except Exception:
    sp = None
    latex2sympy = None

# Optional: Math-Verify (more permissive, robust verifier)
try:
    from math_verify import parse as mv_parse
    from math_verify import verify as mv_verify
except Exception:
    mv_parse = None
    mv_verify = None


# ---------------------------
# Regex + basic helpers
# ---------------------------
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
TEXT_RE = re.compile(r"\\text\{([^}]*)\}")
BOX_RE = re.compile(r"\\boxed\{([^}]*)\}")
FRAC_SHORT_RE = re.compile(r"\\frac(\d+)(\d+)")  # \frac14 -> \frac{1}{4}
MC_RE = re.compile(r"^\(?\s*([A-Ea-e])\s*\)?$")


def extract_answer(text: str) -> str:
    """Extract answer span from <answer>...</answer>; fallback to raw text."""
    m = ANSWER_RE.search(text or "")
    return m.group(1).strip() if m else (text or "").strip()


# ---------------------------
# Normalization
# ---------------------------
def _strip_tex_noise(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("\\!", "")
    s = s.replace("\\,", "").replace("\\;", "").replace("\\:", "")
    s = s.replace("\\$", "").replace("$", "")
    s = s.replace(",", "")
    return s


def normalize_latex(s: str) -> str:
    """
    Conservative LaTeX normalization:
    - unboxes
    - unwraps \text{...}
    - normalizes frac macros and \left/\right
    - canonicalizes whitespace
    """
    s = _strip_tex_noise(s)
    s = BOX_RE.sub(r"\1", s)
    s = TEXT_RE.sub(r"\1", s)

    # frac macro aliases
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")

    # bracket sizing
    s = s.replace("\\left", "").replace("\\right", "")

    # \frac14 -> \frac{1}{4}
    s = FRAC_SHORT_RE.sub(r"\\frac{\1}{\2}", s)

    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_bracketed(s: str) -> str:
    """Extra cleanup for tuples/intervals/sets: remove spaces aggressively."""
    s = normalize_latex(s)
    s = s.replace(" ", "")
    s = s.replace("\\infty", "infty").replace("∞", "infty")
    return s


def normalize_texty(s: str) -> str:
    """For answers that are plain words/labels: casefold + strip punctuation."""
    s = (s or "").strip()
    s = BOX_RE.sub(r"\1", s)
    s = TEXT_RE.sub(r"\1", s)
    s = s.casefold()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def looks_texty(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return True
    if "\\" in s:
        return False
    return bool(re.fullmatch(r"[A-Za-z\s\(\)]+", s))


def looks_mathy(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return False
    return (("\\" in s) or any(ch in s for ch in "^_*/+-=") or any(c.isdigit() for c in s))


def mc_variants(s: str) -> list[str]:
    """Map (C), \\text{(C)}, C -> C and (C)."""
    s0 = normalize_latex(s).strip()
    m = MC_RE.match(s0)
    if not m:
        return []
    letter = m.group(1).upper()
    return [letter, f"({letter})"]


def gold_variants(gold: str) -> list[str]:
    """
    Create a small, safe set of equivalent gold forms to reduce format tax.
    """
    g0 = (gold or "").strip()
    vs = []
    vs.append(g0)
    vs.append(normalize_latex(g0))
    vs.append(normalize_bracketed(g0))
    vs.append(normalize_texty(g0))
    vs.extend(mc_variants(g0))

    out = []
    for v in vs:
        v = (v or "").strip()
        if v and v not in out:
            out.append(v)
    return out


# ---------------------------
# Equivalence (MORE permissive)
# ---------------------------
def equiv(pred: str, gold: str) -> bool:
    """
    Matching strategy (in order):
    1) normalized exact match
    2) bracket/space-insensitive match for tuples/intervals/sets
    3) multiple-choice normalization
    4) texty normalization
    5) Math-Verify (robust parse+verify; verify(gold, answer))
    6) SymPy fallback (latex2sympy2_extended + simplify diff)
    """
    p_raw = (pred or "").strip()
    g_raw = (gold or "").strip()

    # (1) normalized exact
    p = normalize_latex(p_raw)
    g = normalize_latex(g_raw)
    if p == g:
        return True

    # (2) bracketed exact
    if normalize_bracketed(p_raw) == normalize_bracketed(g_raw):
        return True

    # (3) multiple choice
    pv = mc_variants(p_raw)
    gv = mc_variants(g_raw)
    if pv and gv and any(x == y for x in pv for y in gv):
        return True

    # (4) texty
    if looks_texty(g_raw) or looks_texty(g):
        return normalize_texty(p_raw) == normalize_texty(g_raw)

    # (5) Math-Verify (very permissive, handles lots of weird math forms)
    # README emphasizes verify(gold, answer) order. :contentReference[oaicite:3]{index=3}
    if mv_parse is not None and mv_verify is not None:
        try:
            gold_parsed = mv_parse(g)
            ans_parsed = mv_parse(p)
            if mv_verify(gold_parsed, ans_parsed):
                return True
        except Exception:
            pass

    # (6) SymPy fallback (gated)
    if latex2sympy is not None and sp is not None and looks_mathy(p) and looks_mathy(g):
        try:
            pp = latex2sympy(p)
            gg = latex2sympy(g)
            return sp.simplify(pp - gg) == 0
        except Exception:
            return False

    return False


# ---------------------------
# Metric: pass@k from <answer> tag
# ---------------------------
class PassAtKAnswerTag(SampleLevelComputation):
    """1.0 if ANY of first k generations matches ANY gold variant, else 0.0."""
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def compute(self, doc: Doc, model_response, **kwargs) -> float:
        golds = doc.get_golds()
        preds = list(getattr(model_response, "final_text", []) or [])[: self.k]

        for raw in preds:
            ans = extract_answer(raw)
            for g in golds:
                if equiv(ans, g):
                    return 1.0
        return 0.0


PASS1_TAG = SampleLevelMetric(
    metric_name="pass@1_answer_tag",
    higher_is_better=True,
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=PassAtKAnswerTag(k=1),  # must be SampleLevelComputation :contentReference[oaicite:4]{index=4}
    corpus_level_fn=np.mean,
)

PASS8_TAG = SampleLevelMetric(
    metric_name="pass@8_answer_tag",
    higher_is_better=True,
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=PassAtKAnswerTag(k=8),
    corpus_level_fn=np.mean,
)


# ---------------------------
# Prompt -> Doc
# ---------------------------
def math500_prompt(row: dict, task_name: str) -> Doc:
    problem = (row.get("problem") or row.get("question") or "").strip()
    answer = row.get("answer") or row.get("final") or row.get("solution") or ""

    # multiple gold variants to reduce format tax
    choices = gold_variants(answer)

    return Doc(
        query=problem + "\n\nReturn ONLY:\n<answer>\n...\n</answer>\n",
        choices=choices,
        gold_index=list(range(len(choices))),  # any variant counts
    )


# ---------------------------
# Tasks
# ---------------------------
TASKS_TABLE = [
    # Deterministic pass@1: 1 sample
    LightevalTaskConfig(
        name="math_500_pass1_det",
        prompt_function=math500_prompt,
        hf_repo="HuggingFaceH4/MATH-500",
        hf_subset="default",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        generation_size=4096,
        num_samples=[1],
        stop_sequence=["</answer>"],
        few_shots_split=None,
        few_shots_select=None,
        num_fewshots=0,
        metrics=[PASS1_TAG],
        version=1,
    ),

    # pass@8: 8 samples
    LightevalTaskConfig(
        name="math_500_pass8",
        prompt_function=math500_prompt,
        hf_repo="HuggingFaceH4/MATH-500",
        hf_subset="default",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        generation_size=4096,
        num_samples=[8],
        stop_sequence=["</answer>"],
        few_shots_split=None,
        few_shots_select=None,
        num_fewshots=0,
        metrics=[PASS8_TAG],
        version=1,
    ),
]
