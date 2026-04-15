#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


GITHUB_ROOT = Path(__file__).resolve().parent
MANUSCRIPT_ROOT = GITHUB_ROOT / "manuscript"


def run_python(script_name: str) -> None:
    subprocess.run(
        [sys.executable, str(MANUSCRIPT_ROOT / script_name)],
        cwd=GITHUB_ROOT,
        check=True,
    )


def run_matlab(entry_name: str) -> None:
    github = str(GITHUB_ROOT).replace("'", "''")
    manuscript = str(MANUSCRIPT_ROOT).replace("'", "''")
    expr = (
        f"addpath('{github}'); "
        f"addpath('{manuscript}'); "
        f"{entry_name}"
    )
    shell_cmd = f"module load matlab/r2023b && matlab -batch {subprocess.list2cmdline([expr])}"
    subprocess.run(
        ["/usr/bin/bash", "-lc", shell_cmd],
        cwd=GITHUB_ROOT,
        check=True,
    )


def require_env(name: str, reason: str) -> None:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable {name}: {reason}")


def main() -> None:
    require_env("PSICONNECT_MEQ30_TSV", "needed for PsiConnect manuscript questionnaire joins")
    require_env("PSICONNECT_ASC11_TSV", "needed for PsiConnect ASC11 manuscript analyses")
    require_env("CHARM_MEQ_CSV", "needed for CHARM manuscript questionnaire joins")
    require_env("CHARM_MEANFD_CSV", "needed for exact CHARM afterglow reproduction")
    steps = [
        ("Refresh LME inputs", lambda: subprocess.run([sys.executable, str(GITHUB_ROOT / "prepare_mixed_effects_inputs.py")], cwd=GITHUB_ROOT, check=True)),
        ("Geometry LMEs", lambda: run_matlab("run_harmonized_psychedelics_geometry_lme")),
        ("Afterglow omnibus", lambda: run_matlab("run_charm_afterglow_omnibus_geometry")),
        ("Afterglow posthoc", lambda: run_matlab("run_charm_afterglow_network_posthoc")),
        ("Pharmacological specificity", lambda: run_matlab("run_pharmacological_specificity_interactions")),
        ("LSD core tests", lambda: run_matlab("run_lsd_core_tests")),
        ("LSD postprocess", lambda: run_python("postprocess_lsd_core_tests.py")),
        ("DMT metric family", lambda: run_python("analyze_dmt_schaefer100_cortical_full_metric_family.py")),
        ("Whole-brain FC change", lambda: run_python("compute_wholebrain_fc_change_condition_pvalues.py")),
        ("CHARM FC change", lambda: run_python("compute_charm_wholebrain_fc_change_meq.py")),
        ("Attractor concentration", lambda: run_python("analyze_attractor_concentration.py")),
        ("Missing MEQ panels", lambda: run_python("analyze_missing_meq_panels.py")),
        ("PsiConnect ASC11", lambda: run_python("analyze_psiconnect_asc11.py")),
        ("Geometry behavior", lambda: run_python("analyze_geometry_behavior_focus.py")),
        ("Restricted behavior", lambda: run_python("analyze_restricted_brain_experience_panel.py")),
        ("DMT vs LSD comparison", lambda: run_python("build_dmt_vs_lsd_interaction_like.py")),
        ("Vertex sensitivity", lambda: run_python("summarize_vertex_manifold_sensitivity.py")),
    ]

    for label, step in steps:
        print(f"[manuscript] {label}")
        step()

    print("Regenerated manuscript-supporting tables for Results 2.1-2.6 into github/outputs.")


if __name__ == "__main__":
    main()
