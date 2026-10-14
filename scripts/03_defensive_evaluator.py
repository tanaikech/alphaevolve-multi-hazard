#!/usr/bin/env python3
"""
scripts/03_defensive_evaluator.py
===============================================================================
3-Tier Defensive Evaluator Harness for AlphaEvolve
===============================================================================

Evaluates candidate multi-hazard algorithms across three strict defensive layers:
  - Tier 1 (AST Security & Anti-Synthetic Static Gate):
      Parses the candidate Python code into an Abstract Syntax Tree (AST) and
      immediately rejects any program attempting to import forbidden modules
      (`random`, `os`, `sys`, `subprocess`, `pickle`), invoke dynamic execution
      (`eval`, `exec`, `open`), use pseudo-random number generators (`np.random`),
      or hardcode historical disaster dates / sensor MAC prefixes.
  - Tier 2 (Physics Contract & Double-Run Determinism Gate):
      Executes the candidate functions twice on identical physical inputs inside
      an isolated namespace, verifying floating-point determinism (|y1 - y2| < 1e-12)
      and valid physical bounds.
  - Tier 3 (Popperian Falsification Benchmark):
      Scores the candidate across all 23 multi-hazard events (15 Rain, 4 Earthquake,
      2 Volcano, 2 Control). Any false alarm triggered on the non-disaster control
      baselines incurs a -5,000 point Popperian falsification penalty.

Usage (from the repository root):
    python3 scripts/03_defensive_evaluator.py --test-harness
"""

import ast
import time
import json
import pickle
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Any
import numpy as np

SEED_PROGRAM = """# ==============================================================================
# AlphaEvolve Seed Program: Baseline Prior Heuristic Thresholds
# programmingLanguage: "python"
# ==============================================================================
import numpy as np

# >>>>>>>>>>>>>>>>>>>>>>>> # EVOLVE-BLOCK START <<<<<<<<<<<<<<<<<<<<<<<<<<<<

def compute_rain_convective_score(theta_e_k: float, dtheta_e_dt_k_h: float,
                                 pressure_drop_10min_hpa: float, gust_ratio: float) -> float:
    \"\"\"Baseline convective downdraft score (lacks cold-season airmass rejection).\"\"\"
    z = (-0.15 * dtheta_e_dt_k_h) + (1.2 * pressure_drop_10min_hpa) + (0.8 * (gust_ratio - 1.0))
    if theta_e_k > 340.0:
        z += 1.5
    score = 1.0 / (1.0 + np.exp(-np.clip(z, -10.0, 10.0)))
    return float(score)

def evaluate_laic_and_gate(delta_theta_e_k: float, lpar: float,
                          dp_dt_abs_hpa_h: float, regional_std_k: float) -> bool:
    \"\"\"Baseline earthquake LAIC gate (vulnerable to urban/cyclonic false alarms).\"\"\"
    if dp_dt_abs_hpa_h >= 15.0 and lpar < 2.5:
        return False
    if regional_std_k > 8.0 and lpar < 3.0:
        return False
    return bool(delta_theta_e_k >= 8.0 and lpar >= 2.5)

def compute_volcanic_semblance_threshold(num_active_stations: int, snr_est: float) -> float:
    \"\"\"Dynamic Lamb wave beamforming coherence threshold.\"\"\"
    base_semblance = 0.85
    if num_active_stations >= 50:
        base_semblance = 0.80
    elif num_active_stations < 15:
        base_semblance = 0.92
    return float(base_semblance)

def dispatch_hazard_decision(rain_score: float, eq_gate_open: bool, delta_theta_e_k: float,
                            volc_semblance: float, volc_threshold: float,
                            tdoa_sound_speed_m_s: float) -> dict:
    \"\"\"Baseline tri-hazard routing engine.\"\"\"
    alert_type = "NORMAL"
    confidence = 0.0

    if 300.0 <= tdoa_sound_speed_m_s <= 330.0 and volc_semblance >= volc_threshold:
        alert_type = "VOLCANO_LAMB_WAVE"
        confidence = float(volc_semblance)
    elif eq_gate_open and delta_theta_e_k >= 8.0:
        alert_type = "EARTHQUAKE_LAIC_PRECURSOR"
        confidence = float(min(1.0, delta_theta_e_k / 15.0))
    elif rain_score >= 0.75:
        alert_type = "CONVECTIVE_TORRENTIAL_RAIN"
        confidence = float(rain_score)

    return {"alert": alert_type, "confidence": confidence}

# >>>>>>>>>>>>>>>>>>>>>>>>> # EVOLVE-BLOCK END <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
"""


class SecurityError(Exception):
    pass


class PhysicsContractError(Exception):
    pass


class SecurityASTVisitor(ast.NodeVisitor):
    """Tier 1 AST static analyzer blocking synthetic generators, I/O, and hardcoded cheats."""

    FORBIDDEN_CALLS = {"eval", "exec", "open", "__import__", "compile"}
    FORBIDDEN_MODULES = {
        "os",
        "sys",
        "subprocess",
        "requests",
        "urllib",
        "socket",
        "random",
        "torch",
        "unittest",
        "pickle",
        "shutil",
        "time",
    }

    def visit_Import(self, node):
        for alias in node.names:
            root_mod = alias.name.split(".")[0]
            if root_mod in self.FORBIDDEN_MODULES:
                raise SecurityError(f"Forbidden module import detected: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            root_mod = node.module.split(".")[0]
            if root_mod in self.FORBIDDEN_MODULES:
                raise SecurityError(f"Forbidden from-import detected: {node.module}")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        attr_name = node.attr
        if (
            attr_name == "random"
            or attr_name.startswith("rand")
            or "random" in attr_name
            or attr_name in {"choice", "shuffle", "sample", "normal", "uniform"}
        ):
            raise SecurityError(f"Forbidden randomized generator/attribute detected: {attr_name}")
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in self.FORBIDDEN_CALLS:
            raise SecurityError(f"Forbidden builtin function call detected: {node.func.id}")
        self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            val_lower = node.value.lower()
            if any(
                date_str in val_lower
                for date_str in ["2018-01-23", "2018-03-06", "2016-11-22", "2018-06-18"]
            ):
                raise SecurityError(f"Cheating detected: Hardcoded historical event date '{node.value}'")
            if "70:ee:50" in val_lower:
                raise SecurityError("Cheating detected: Hardcoded Netatmo MAC prefix")
        self.generic_visit(node)


def tier1_ast_scan(code_str: str):
    """Parses candidate code AST and enforces zero synthetic generation & anti-cheating."""
    tree = ast.parse(code_str)
    visitor = SecurityASTVisitor()
    visitor.visit(tree)


def tier2_contract_and_determinism_test(module_dict: dict):
    """Verifies double-run determinism (|y1 - y2| < 1e-12) and physical output bounds."""
    func_rain = module_dict.get("compute_rain_convective_score")
    func_eq = module_dict.get("evaluate_laic_and_gate")
    func_volc = module_dict.get("compute_volcanic_semblance_threshold")
    func_disp = module_dict.get("dispatch_hazard_decision")

    if not (func_rain and func_eq and func_volc and func_disp):
        raise PhysicsContractError("Missing required function implementations in evolve block.")

    # 1. Determinism & bound check for convective rain score
    r1 = func_rain(345.0, -12.0, 3.5, 2.2)
    r2 = func_rain(345.0, -12.0, 3.5, 2.2)
    if abs(r1 - r2) > 1e-12 or np.isnan(r1) or np.isinf(r1) or not (0.0 <= r1 <= 1.0):
        raise PhysicsContractError(f"Non-deterministic or unbounded rain score: r1={r1}, r2={r2}")

    # 2. Determinism check for earthquake LAIC gate
    g1 = func_eq(35.0, 5.2, 0.8, 4.5)
    g2 = func_eq(35.0, 5.2, 0.8, 4.5)
    if g1 != g2:
        raise PhysicsContractError("Non-deterministic EQ gate evaluation.")

    # 3. Determinism & bound check for volcanic semblance threshold
    v1 = func_volc(58, 12.0)
    v2 = func_volc(58, 12.0)
    if abs(v1 - v2) > 1e-12 or not (0.0 <= v1 <= 1.0):
        raise PhysicsContractError(f"Non-deterministic or unbounded volcano semblance: v1={v1}, v2={v2}")

    # 4. Determinism check for tri-hazard dispatcher
    d1 = func_disp(r1, g1, 35.0, 0.92, v1, 312.0)
    d2 = func_disp(r2, g2, 35.0, 0.92, v2, 312.0)
    if d1 != d2 or not isinstance(d1, dict) or "alert" not in d1 or "confidence" not in d1:
        raise PhysicsContractError("Non-deterministic or malformed dispatch decision dictionary.")


class DefensiveEvaluatorHarness:
    """3-Tier Evaluator Harness executing fast in-memory physics verification."""

    def __init__(self, cache_dir: str = "sample_data/processed_cache"):
        self.cache_dir = Path(cache_dir)
        self.datasets = self._load_cached_features()

    def _load_cached_features(self) -> Dict[str, Any]:
        manifest_file = self.cache_dir / "processed_cache_manifest.json"
        if not manifest_file.exists():
            raise FileNotFoundError(
                f"Missing cache manifest at {manifest_file}. Run `python3 scripts/02_extract_physics_features.py` first."
            )
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        data = {}
        for ev_id, info in manifest.items():
            cache_file = Path(info["cache_file"])
            with open(cache_file, "rb") as f_in:
                content = pickle.load(f_in)
            # Ensure read-only memory lock on all feature arrays
            for key in ("feature_matrix", "rain_features", "eq_features", "volc_features"):
                if key in content and isinstance(content[key], np.ndarray):
                    content[key].flags.writeable = False
            data[ev_id] = {
                "category": info["category"],
                "content": content,
            }
        return data

    def evaluate(self, code_str: str) -> Tuple[float, List[Dict[str, str]]]:
        """Evaluates a candidate program across Tier 1, Tier 2, and Tier 3."""
        start_t = time.time()
        insights = []

        # Tier 1: Static AST Security & Anti-Synthetic Scan
        try:
            tier1_ast_scan(code_str)
        except SecurityError as se:
            return -1e12, [{"label": "SECURITY_VIOLATION", "text": str(se)}]
        except Exception as e:
            return -1e12, [{"label": "SYNTAX_ERROR", "text": f"AST Parse Error: {e}"}]

        # Tier 2: Execution & Double-Run Determinism Test
        sandbox = {}
        try:
            exec(code_str, sandbox)
            tier2_contract_and_determinism_test(sandbox)
        except PhysicsContractError as pce:
            return -1e6, [{"label": "PHYSICS_CONTRACT_VIOLATION", "text": str(pce)}]
        except Exception as e:
            return -1e6, [{"label": "RUNTIME_ERROR", "text": f"Execution failed: {e}"}]

        func_rain = sandbox["compute_rain_convective_score"]
        func_eq = sandbox["evaluate_laic_and_gate"]
        func_volc = sandbox["compute_volcanic_semblance_threshold"]
        func_disp = sandbox["dispatch_hazard_decision"]

        # Tier 3: Quantitative Multi-Hazard Benchmark & Popperian Falsification
        rain_detected = 0
        eq_detected = 0
        volc_detected = 0
        control_fps = 0

        for ev_id, ev_data in self.datasets.items():
            cat = ev_data["category"]
            content = ev_data["content"]

            if cat == "RAIN":
                feats = content["feature_matrix"]
                event_flag = False
                for row in feats:
                    th_e, dth_e, dp10, gust = row
                    score = func_rain(th_e, dth_e, dp10, gust)
                    dec = func_disp(score, False, 0.0, 0.0, 1.0, 0.0)
                    if dec.get("alert") == "CONVECTIVE_TORRENTIAL_RAIN":
                        event_flag = True
                        break
                if event_flag:
                    rain_detected += 1
                else:
                    insights.append({
                        "label": "RAIN_MISS",
                        "text": f"Failed to trigger alert for convective rain event: {ev_id}",
                    })

            elif cat == "EARTHQUAKE":
                feats = content["feature_matrix"]
                event_flag = False
                for row in feats:
                    dth_e, lpar, dp_dt, reg_std = row
                    gate_open = func_eq(dth_e, lpar, dp_dt, reg_std)
                    dec = func_disp(0.0, gate_open, dth_e, 0.0, 1.0, 0.0)
                    if dec.get("alert") == "EARTHQUAKE_LAIC_PRECURSOR":
                        event_flag = True
                        break
                if event_flag:
                    eq_detected += 1
                else:
                    insights.append({
                        "label": "EARTHQUAKE_MISS",
                        "text": f"LAIC precursor missed for earthquake event: {ev_id}",
                    })

            elif cat == "VOLCANO":
                feats = content["feature_matrix"]
                event_flag = False
                for row in feats:
                    num_st, snr, sembl, sound_speed = row
                    v_th = func_volc(int(num_st), snr)
                    dec = func_disp(0.0, False, 0.0, sembl, v_th, sound_speed)
                    if dec.get("alert") == "VOLCANO_LAMB_WAVE":
                        event_flag = True
                        break
                if event_flag:
                    volc_detected += 1
                else:
                    insights.append({
                        "label": "VOLCANO_MISS",
                        "text": f"Lamb wave signature missed for volcano event: {ev_id}",
                    })

            elif cat == "CONTROL":
                rain_feats = content["rain_features"]
                eq_feats = content["eq_features"]
                volc_feats = content["volc_features"]

                for row in rain_feats:
                    th_e, dth_e, dp10, gust = row
                    score = func_rain(th_e, dth_e, dp10, gust)
                    dec = func_disp(score, False, 0.0, 0.0, 1.0, 0.0)
                    if dec.get("alert") != "NORMAL":
                        control_fps += 1
                        insights.append({
                            "label": "CONTROL_FALSE_POSITIVE_RAIN",
                            "text": f"Control day {ev_id} falsely triggered {dec.get('alert')}",
                        })
                        break

                for row in eq_feats:
                    dth_e, lpar, dp_dt, reg_std = row
                    gate_open = func_eq(dth_e, lpar, dp_dt, reg_std)
                    dec = func_disp(0.0, gate_open, dth_e, 0.0, 1.0, 0.0)
                    if dec.get("alert") != "NORMAL":
                        control_fps += 1
                        insights.append({
                            "label": "CONTROL_FALSE_POSITIVE_EQ",
                            "text": f"Control day {ev_id} falsely opened EQ gate: {dec.get('alert')}",
                        })
                        break

                for row in volc_feats:
                    num_st, snr, sembl, sound_speed = row
                    v_th = func_volc(int(num_st), snr)
                    dec = func_disp(0.0, False, 0.0, sembl, v_th, sound_speed)
                    if dec.get("alert") != "NORMAL":
                        control_fps += 1
                        insights.append({
                            "label": "CONTROL_FALSE_POSITIVE_VOLCANO",
                            "text": f"Control day {ev_id} falsely triggered Lamb wave: {dec.get('alert')}",
                        })
                        break

        tpr_rain = rain_detected / 15.0
        tpr_eq = eq_detected / 4.0
        tpr_volc = volc_detected / 2.0
        latency_ms = (time.time() - start_t) * 1000.0

        # Popperian Fitness Objective (Eq. 4 in paper):
        # Max positive = 400.0; False Positive Penalty = -5000.0 per control violation
        fitness = (
            (tpr_rain * 100.0)
            + (tpr_eq * 150.0)
            + (tpr_volc * 150.0)
            - (control_fps * 5000.0)
            - (latency_ms * 0.01)
        )
        return float(fitness), insights


def run_test_harness(cache_dir: str):
    print("=" * 80)
    print("🛡️ Step 3: 3-Tier Defensive Evaluator Harness Self-Test")
    print("=" * 80)

    evaluator = DefensiveEvaluatorHarness(cache_dir=cache_dir)

    # Test 1: Seed Program vs Evolved Program
    print("\n--- [Test 1/4] Evaluating Baseline Seed Program vs. Evolved Program ---")
    seed_fit, seed_insights = evaluator.evaluate(SEED_PROGRAM)
    print(f"Baseline Seed Program Fitness: {seed_fit:.2f} (Control False Alarm Penalties Applied)")
    print(f"Diagnostic Insights Generated: {len(seed_insights)}")

    evolved_path = Path(__file__).resolve().parent / "evolved_hazard_dispatcher.py"
    with open(evolved_path, "r", encoding="utf-8") as f:
        evolved_code = f.read()
    ev_fit, ev_insights = evaluator.evaluate(evolved_code)
    print(f"AlphaEvolve Discovered Program Fitness: {ev_fit:.2f} / 400.0 (Issues: {len(ev_insights)})")
    assert ev_fit > 395.0 and len(ev_insights) == 0, "Evolved program must achieve 100% TPR and 0% FPR!"
    print("✅ Evolved Program Passed All 3 Tiers with 100% Sensitivity & 0.0% FPR!")

    # Test 2: Malicious Random Generator Injection
    print("\n--- [Test 2/4] Injecting Forbidden np.random (Anti-Synthetic Gate) ---")
    malicious_code = SEED_PROGRAM.replace("z += 1.5", "z += np.random.normal(0, 1)")
    m_fit, m_insights = evaluator.evaluate(malicious_code)
    print(f"Malicious Code Fitness: {m_fit:.2e} | Insight: {m_insights[0]['text']}")
    assert m_fit == -1e12, "Malicious code was not blocked by Tier 1 AST!"
    print("✅ Tier 1 AST Gate successfully blocked np.random.normal!")

    # Test 3: Historical Event Date Hardcoding Cheat
    print("\n--- [Test 3/4] Injecting Hardcoded Historical Date Cheat ---")
    cheat_code = SEED_PROGRAM.replace(
        'alert_type = "NORMAL"', 'if "2018-01-23" in "": pass\n    alert_type = "NORMAL"'
    )
    c_fit, c_insights = evaluator.evaluate(cheat_code)
    print(f"Cheat Code Fitness: {c_fit:.2e} | Insight: {c_insights[0]['text']}")
    assert c_fit == -1e12, "Hardcoded date cheat was not blocked by Tier 1 AST!"
    print("✅ Tier 1 AST Gate successfully blocked hardcoded event date cheating!")

    # Test 4: Forbidden Module Import
    print("\n--- [Test 4/4] Injecting Forbidden Module Import (import time) ---")
    import_code = SEED_PROGRAM.replace("import numpy as np", "import numpy as np\nimport time")
    i_fit, i_insights = evaluator.evaluate(import_code)
    print(f"Forbidden Import Fitness: {i_fit:.2e} | Insight: {i_insights[0]['text']}")
    assert i_fit == -1e12, "Forbidden module import was not blocked!"
    print("✅ Tier 1 AST Gate successfully blocked forbidden module import!")

    print("\n" + "=" * 80)
    print("🎉 All 3-Tier Defensive Evaluator Tests Passed 100%!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="AlphaEvolve 3-Tier Defensive Evaluator Harness")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="sample_data/processed_cache",
        help="Path to processed feature cache directory (default: sample_data/processed_cache)",
    )
    parser.add_argument(
        "--test-harness",
        action="store_true",
        help="Run self-test verification suite on the 3-Tier Evaluator",
    )
    args = parser.parse_args()

    if args.test_harness:
        run_test_harness(args.cache_dir)
    else:
        print("Run with `--test-harness` to execute the 3-Tier Evaluator verification suite.")


if __name__ == "__main__":
    main()
