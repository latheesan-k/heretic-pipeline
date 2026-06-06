#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Non-interactive driver for Heretic.

Heretic is fundamentally an *interactive* program: after running its
optimization study it presents `questionary` menus asking the user to pick a
trial from the Pareto front and choose what to do with the resulting model
(save / upload / chat / benchmark). There is no CLI flag to "just save the
model to a folder".

However, Heretic's own `utils.prompt_*` helpers fall back to plain `input()`
reads when it detects a notebook environment (see `heretic.utils.is_notebook`).
We exploit that: we force notebook mode and feed scripted answers on stdin so
the *real, tested* Heretic code path runs end to end and saves the merged
(decensored) model to a known directory.

Policy: we always pick the trial with the FEWEST REFUSALS (the first entry in
Heretic's own sorted Pareto menu), which is what you want for a decensoring
goal.

Usage:
    python run_heretic.py <hf-org/model> <output-dir> [extra heretic args...]
"""

import io
import os
import sys


def main() -> int:
    if len(sys.argv) < 3:
        print(
            "usage: run_heretic.py <model> <output-dir> [extra heretic args...]",
            file=sys.stderr,
        )
        return 2

    model = sys.argv[1]
    output_dir = sys.argv[2]
    extra_args = sys.argv[3:]

    os.makedirs(output_dir, exist_ok=True)

    # ── Force Heretic into "notebook" mode so its prompts read from stdin. ──
    # heretic.utils.is_notebook() returns True when COLAB_GPU is set, which
    # routes every prompt_select / prompt_text / prompt_path through input().
    os.environ.setdefault("COLAB_GPU", "1")

    # Use a FRESH checkpoint dir each run. A leftover finished study from a
    # previous run triggers the extra "you have already processed this model"
    # menu, which would shift the scripted answer sequence below and break the
    # non-interactive flow. Clearing it guarantees a deterministic menu order.
    import shutil

    checkpoint_dir = os.path.join(output_dir, ".heretic_checkpoints")
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Scripted answers, consumed in order by the input() calls Heretic makes
    #    AFTER optimization completes. ───────────────────────────────────────
    #
    #   1. "Which trial do you want to use?"          -> 1  (fewest refusals)
    #   2. "What do you want to do with the model?"   -> 1  (Save to a folder)
    #   3. "Path to the folder:"                      -> <output_dir>
    #      (merge strategy is automatic for non-quantized models: no prompt)
    #   4. "What do you want to do with the model?"   -> 5  (Return to trials)
    #   5. "Which trial do you want to use?"          -> <Exit program entry>
    #
    # The trial menu's Exit entry is the LAST option; its number depends on how
    # many Pareto trials + the "Run additional trials" entry there are, which we
    # cannot know in advance. We instead send an empty line repeatedly at the
    # end: prompt_select rejects it and re-asks, but the second menu ("what to
    # do") and the trial menu both treat a value that maps to "" (Exit) only via
    # number. To guarantee a clean exit regardless, after returning to the trial
    # menu we send a very large number which is invalid, then EOF — Heretic's
    # input() raises EOFError on the closed stream, which is caught and treated
    # as a shutdown. See the EOF-tolerant wrapper below.
    answers = [
        "1",          # "Which trial do you want to use?" -> fewest refusals
        "1",          # "What do you want to do?" -> Save to a local folder
        output_dir,   # "Path to the folder:" -> destination
        "5",          # "What do you want to do?" -> Return to trial menu
        # Back at the trial menu: empty line cancels/exits cleanly because
        # prompt_select in notebook mode returns the value mapped to the
        # entered number; we instead let the stream hit EOF (handled below),
        # which raises EOFError -> caught by the driver as a normal shutdown.
    ]

    answer_iter = iter(answers)

    class ScriptedStdin(io.TextIOBase):
        """Feeds scripted answers, then signals EOF to make Heretic exit."""

        def readable(self) -> bool:
            return True

        def readline(self, *args, **kwargs):
            try:
                line = next(answer_iter)
            except StopIteration:
                # No more scripted answers -> EOF. Heretic's input() will
                # raise EOFError, which we catch around heretic_main() and
                # treat as a clean shutdown (the model is already saved).
                return ""
            sys.__stderr__.write(f"[driver] answering prompt with: {line!r}\n")
            sys.__stderr__.flush()
            return line + "\n"

    sys.stdin = ScriptedStdin()

    # Build argv for Heretic. Pass the model via the explicit --model flag
    # rather than as a positional. Heretic's positional-model heuristic only
    # works when the model is the LAST argument; since we append extra args
    # (e.g. --n-trials), using --model avoids it being mis-parsed.
    sys.argv = [
        "heretic",
        "--model",
        model,
        "--study-checkpoint-dir",
        checkpoint_dir,
        *extra_args,
    ]

    # Import and run the real Heretic entrypoint.
    from heretic.main import main as heretic_main

    try:
        heretic_main()
    except EOFError:
        # Expected: we exhausted the scripted answers after saving the model,
        # so the next input() hit EOF. This is our clean-exit signal.
        sys.__stderr__.write("[driver] reached end of scripted answers; exiting.\n")

    # Verify the model was actually saved.
    expected = os.path.join(output_dir, "config.json")
    if not os.path.exists(expected):
        sys.__stderr__.write(
            f"[driver] ERROR: expected saved model at {output_dir} "
            f"but {expected} is missing.\n"
        )
        return 1

    sys.__stderr__.write(f"[driver] model saved to {output_dir}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
