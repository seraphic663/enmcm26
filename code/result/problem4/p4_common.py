"""P4 output locations and the shared paper constants."""
from ...common.paths import project_root, results_root
from ...common.dataset import INTERVALS_PER_DAY as N, SOC_INITIAL, SOC_MAX, SOC_MIN, OUTPUT_START
from ..problem3.problem3_data import ISSUE_HOURS

PROJECT_DIR = project_root()
ATTACHMENT_DIR = PROJECT_DIR / "附件"
FORMAL_DIR = results_root() / "problem4" / "formal"
AUDIT_DIR = results_root() / "problem4" / "audit"
