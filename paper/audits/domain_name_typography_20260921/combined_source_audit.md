# Independent compiled-source typography audit

All **54 compiled TeX inputs** pass the current domain-formatting scan. There are no missing domain wrappers, nested typewriter wrappers, or unhandled compound domain names. The two legacy PythonFactors occurrences are typewriter formatted. Existing code and prompt literal blocks remain unchanged.

The seven ordinary-Python exclusions were read in context and correctly describe language or runtime behavior: executable code, an isolated interpreter process, invalid expressions, escaped source, a lambda expression, a lambda token, and the modulo operator. Their surrounding benchmark-domain mentions are formatted separately. Citation keys, macro names, reference labels, and paths are protected by the formatter.

Across the **32 result fragments** changed by the two typography reviewers, **25 contain tabular rows**. The **826 checked tabular rows preserve their measured fields**. Concurrent work relabeled eight checkpoint rows from seed IDs 43/46 to paired replicate labels (a)/(b); their measurement columns remain identical. This relabeling is separate from domain typography.

Concurrent prose cleanup subsequently changed twelve result fragments and the main source, including removal of provenance references and a revision table. Those edits were preserved, and the live main source is therefore not claimed to be typography-only against the saved baseline. The JSON audit identifies the affected files, records the seven language exceptions, and hashes the inspected input sources. The earlier renderer checks document the typography changes before these concurrent edits. No additional numerical analysis, experiment, or bootstrap was run.
