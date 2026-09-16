# Reasoning-off paper integration

Updated the ICLR main paper to report five complete, validated reasoning-disabled deployments, averaging empirical pass@8 and distinct@8 across the five domains at each level. Each deployment uses 32 original prompts per domain/level and eight samples per prompt (3,840 responses). The matched medium/off comparison uses the same 480 prompts per model. Earlier seven-deployment, 128-prompt medium results and the domain figure remain in Appendix J. The existing broader opening about explanations, plans, and programs is preserved.

Opus 5 is excluded because a receipt reported thinking despite a disabled-thinking request. DeepSeek is excluded at 3,839/3,840 receipts; no missing outcome is imputed. No API calls were made during this paper integration.

Validation: all five complete cohorts and source receipts were authenticated; 20 new renderer/admission tests, 60 preserved/editorial tests, and 19 collector-status/summary tests passed. Main-length and line-fill checks pass (9 main pages; 380 prose blocks), all references resolve, no overfull boxes, and qpdf reports no syntax or stream errors. Both review PDFs were refreshed and verified against the full PDF.

The hosted gates passed during the standard contract invocation. That invocation later encountered a stale captured main.tex after another publication task switched campaign includes to September 12. The remaining non-hosted contract was rerun against the updated manuscript and passed, preserving the 18 formal statement/proof blocks and current figure/evidence checks. This phased recheck did not change the checker or skip any unvalidated condition. The retained logs distinguish both invocations.

Concurrent base-model, campaign, and completed discovery updates were preserved. Several captions were rephrased to satisfy the existing line-fill check; estimates were unchanged. The discovery revision retains byte-identical plot PDFs/PNGs and original frozen evidence; full discovery and inference-followup reconstruction passed.

See pdf_verification.json for the final source/PDF identities and final/ for the corresponding manuscript and review copies.
