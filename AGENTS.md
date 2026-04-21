# Repo Working Principles

These rules apply to future agent work in this repository.

## 1. Reproduce With Real Code

- Do not stop at static code reading when the user is asking about a bug, regression, or missing output.
- Run the actual repo code path whenever feasible.
- For pose / triangulation / TRC issues, prefer calling the real pipeline entrypoints instead of only inspecting helper functions.

## 2. Verify Before Fixing

- Reproduce the issue first and save the failing evidence.
- Use the same input before and after the change so the comparison is valid.
- If the real user trial is not available in the repo, say that explicitly and build a minimal reproducible synthetic case rather than guessing.

## 3. Fix The Root Cause

- Do not paper over output mismatches in the report only.
- Trace the exact stage where information is dropped: input JSON, skeleton mapping, triangulation selection, TRC writing, or downstream conversion.
- Prefer the smallest code change that restores the expected behavior at the correct stage.

## 4. Show Before / After Evidence

- After a fix, rerun the same code path and show the corrected output.
- For output-format bugs, include concrete before/after artifacts such as TRC headers, marker lists, counts, or diffs.
- Keep the raw evidence files in the repo workspace when practical so the user can inspect them directly.

## 5. Document With Figures

- When the user asks for a report, generate a Markdown document under `docs/`.
- Include figures or screenshot-like visuals under `figures/`.
- The document should contain:
  - what was reproduced
  - what root cause was found
  - what code was changed
  - what the output looked like before
  - what the output looks like after
  - where the raw evidence files are stored

## 6. Be Explicit About Limits

- If a conclusion is based on synthetic reproducible input rather than the user's real trial, say so clearly.
- If a downstream stage was not updated on purpose, say that explicitly rather than implying a full end-to-end fix.
- Separate "this is how the current code behaves" from "this cannot be changed". Do not present the current implementation as the only possible design if a local code change could alter it.

## 7. Keep Collaboration Friction Low

- When the user repeats a working preference, add it here if it is likely to matter again in this repo.
- Treat these rules as default expectations for future debugging and reporting tasks in this workspace.
