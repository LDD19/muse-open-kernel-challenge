# Notes

- Kernel boards: one H200 per problem; the overall kernel rank follows decode_layer only.
- Serving: SGLang `297e54fd`, 8 x H200 (TP8 = EP8), 1 scored run + 3 checks per day; the best scored run counts.
- Serving rules: lossless (KL and agreement gates), readable source (no renamed identifiers), GPU memory <= stock + 2 GB, launch flags limited to stock options.
- Serving time split: ~63% first-turn prefill, ~10-14% MoE in verify steps, so most of the patch targets prefill.
- Serving board on 11-20: #1 = 4.494x, #10 = 3.703x.
- Dev GPUs: Modal H200 for kernels and 1-2 GPU rigs, Nebius spot H200 for 8-GPU serving sessions.
- Most iterations ran as long autonomous agent loops (Claude Code) against a local copy of the judge; serving checks and scored runs approved by hand. Setup: [AUTORESEARCH.md](AUTORESEARCH.md).
- `git apply` needs the diff with LF line endings: on Windows, clone with `git -c core.autocrlf=false clone ...`.
