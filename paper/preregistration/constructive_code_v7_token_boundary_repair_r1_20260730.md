# ConstructiveCode v7 tokenizer-boundary repair r1

The first v7 SFT/gate job, 30203592, terminated before model loading,
optimization, or development-set sampling. Four of the 64 frozen training
programs begin with a newline. For those four rows, the frozen Qwen tokenizer
merges the completion-leading newline with the prompt-final newline, so
tokenizing the concatenated text does not preserve the separately tokenized
prompt as an exact token-ID prefix.

This is an execution assertion defect, not a failed model gate. The repair
tokenizes each complete prompt-plus-program string once with character offsets.
It masks every token whose ending offset is at or before the prompt boundary
and applies loss to every token containing completion text. A token straddling
the boundary therefore remains a target. This preserves assistant-only loss
without assuming byte-pair tokenization is prefix-stable under concatenation.

The following remain frozen and unchanged:

- all 64 materialized train-only examples and their order;
- the four training problem families and the prohibition on loading
  development or evaluation rows during SFT;
- the Qwen2.5-Coder-0.5B-Instruct base snapshot;
- SFT seed 77201, four epochs, batch accumulation eight, 32 optimizer updates,
  learning rate `1e-5`, BF16, and maximum sequence length 4096;
- the post-SFT development gate, including seed 77102, 64 samples per task,
  prefix 16, decoding parameters, checker boundary, and pass criteria.

The failed job identity and scheduler receipt are retained under job-specific
artifact names. The repaired attempt receives a new immutable execution
snapshot and job identity.
