import os
import zipfile
import json
import logging
from typing import Dict, List, Tuple, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RGTSDataLoader:
    def __init__(self, zip_path: str):
        self.zip_path = zip_path
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Archive not found: {zip_path}")
        self.zip_file = zipfile.ZipFile(zip_path, 'r')
        self._namelist = set(self.zip_file.namelist())

    def close(self):
        if self.zip_file:
            self.zip_file.close()

    def get_reference_corpus(self, difficulty: str, problem_name: str, lang: str = "py") -> Dict[str, str]:
        """Loads the 256 reference input->output mapping."""
        # Try lang specific or fallbacks
        candidates = [
            f"data/{difficulty}/{problem_name}/output_corpus_{lang}_256.json",
            f"data/{difficulty}/{problem_name}/output_corpus_py_256.json",
            f"data/{difficulty}/{problem_name}/output_corpus_cpp_256.json",
            f"data/{difficulty}/{problem_name}/output_corpus_java_256.json"
        ]
        for c in candidates:
            if c in self._namelist:
                with self.zip_file.open(c) as f:
                    return json.load(f)
        raise FileNotFoundError(f"No 256 corpus found for {difficulty}/{problem_name}")

    def load_candidate_behaviors(self, difficulty: str, problem_name: str, 
                                 temperature: str, model: str, task: str) -> Dict[str, Dict[str, str]]:
        """Loads all candidate behavior.json files for a given configuration."""
        prefix = f"data/{difficulty}/{problem_name}/{temperature}/{model}/{task}/repeated/preprocessed/"
        results = {}
        
        # Look for Solution1..20/behavior.json
        for sol_idx in range(1, 21):
            target = f"{prefix}Solution{sol_idx}/behavior.json"
            if target in self._namelist:
                with self.zip_file.open(target) as f:
                    try:
                        data = json.load(f)
                        if data and not data.get("compiling_error"):
                            results[f"Solution{sol_idx}"] = data
                        else:
                            # Compilation or runtime error (or empty dictionary)
                            results[f"Solution{sol_idx}"] = {"__ERROR__": True}
                    except Exception:
                        results[f"Solution{sol_idx}"] = {"__ERROR__": True}
            else:
                # Missing behavior.json: explicitly assign error sentinel to ensure constant cohort size of 20
                results[f"Solution{sol_idx}"] = {"__ERROR__": True}
        return results

