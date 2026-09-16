# Worked examples for the response, optimizer, and admission sections

The three staging files are complete current subsections extracted from `paper/main.tex`. Replace the matching subsections in each manuscript version with these files. Original formal statements, proofs, labels, and citations are preserved; only conceptual openings and worked illustrations were added. The additions total approximately 643 whitespace-delimited TeX words.

## What the examples explain

- **A9, `exemplar_bridge_section.tex`:** Event containment bridges response scores to execution-key probability. A four-token response (including termination), each token with conditional probability 1/2, has mean log score -log 2 but complete probability 1/16. A second distinct completion with probability 1/32 raises the key's combined lower bound to 3/32. A separate finite-visibility illustration shows a probability-1/16 key appears in eight draws with probability only 0.4033; four such floors give a sufficient 95% all-key observation budget of 71 draws.
- **A10, `optimizer_section.tex`:** The example is an actual mathematical toy objective satisfying the assumptions, not an arbitrary plug-in of unverified constants. The one-response Bernoulli log loss has global smoothness 1/4 and gradient magnitude at most 1. Centered independent noise of +/-0.1, H=1, and step 0.1 give d_t=1/80000. The constant-step budget is finite at N=100 and divergent at infinity. Steps 0.1/(t+1) instead give B_infinity=pi^2/480000. The basic 95% finite-horizon floor is 9.30128e-7; bounded noise permits the sharper 0.390951 floor. These are toy guarantees and are explicitly not estimates for the language-model learner.
- **A11, `discovery_section.tex`:** Generation probability 0.04 and conditional insertion probability 1/2 yield an actual admission hazard floor 0.02. Four possible correct keys require at most 220 opportunities under the exponential 95% bound; two non-evicting slots make full admission impossible. A bank-growth example with rho=0.3, two old probabilities 1/4, and new probability 1/16 has an objective jump 0.1 log 4 = 0.138629. Old coefficients change from 0.15 to 0.10, illustrating why both the switch cost and coefficient floor matter.

## Validation

`verify_optimizer_examples.py` uses 50-digit Decimal arithmetic for probability and confidence calculations, with standard double precision only for the displayed pi-squared series constant and integer budget checks. Every reported number and rounded example passes; the script checks both sufficient budgets and their preceding integers under the stated exponential bounds. The original theorem/proof blocks and citation sequences are also compared with the current manuscript: 2 blocks in A9, 4 in A10, and 6 in A11, all unchanged.

A standalone wrapper using the actual ICLR style, Times, microtype, and the manuscript's paragraph-fill setting compiles with no warnings, overfull boxes, undefined citations, or unresolved references. Its disposable files are in `/tmp/theory_optimizer_examples_check/`. The earlier generic article wrapper had one overfull line in the unchanged switching-energy proof; this is absent under the actual manuscript style.

The root still performs the full manuscript and rendered paragraph-fill audit after integration. No empirical results, training code, references, original supplements, or manuscripts were edited by this subtask.

## Staging hashes

```json
{
  "exemplar_bridge_section.tex": "d76992649c56d6aa3cd7f74713d0b4cdd88ea42157e534e35c99467de050c344",
  "optimizer_section.tex": "f99eb7b4427b946147379937c75d29944339fedbec1edad383bd0744fb47629b",
  "discovery_section.tex": "0452ebc03aec0b33befb8c90bcbac83242ce0b513dfd9a3e5afbfd827a617ac4"
}
```

Final integration note: root subsequently aligned example links, clarified complete-response wording, and adjusted headings/prose for the manuscript layout. Any handoff hashes above describe that review snapshot; current integrated file hashes are recorded in `validation.json`. All formal statements/proofs and citations remain unchanged.
