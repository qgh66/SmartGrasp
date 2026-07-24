from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


RSR_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = RSR_ROOT.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
RSR_DATA_ROOT = RSR_ROOT / "data"
DEFAULT_INPUT_ROOT = RSR_DATA_ROOT / "input"
DEFAULT_OUTPUT_ROOT = RSR_DATA_ROOT / "output"

REASON_MODELS = ("gpt-5.5", "gpt-4o")


@dataclass(frozen=True)
class TestCase:
    order: int
    slug: str
    difficulty: str
    ambiguous: bool

    @property
    def directory_name(self) -> str:
        return f"{self.order:02d}_{self.slug}"


TEST_CASES = (
    TestCase(1, "hard_ambiguous", "Hard", True),
    TestCase(2, "medium_ambiguous", "Medium", True),
    TestCase(3, "easy_ambiguous", "Easy", True),
    TestCase(4, "easy_unambiguous", "Easy", False),
    TestCase(5, "medium_unambiguous", "Medium", False),
    TestCase(6, "hard_unambiguous", "Hard", False),
)

TEST_CASE_BY_DIRECTORY = {case.directory_name: case for case in TEST_CASES}
TEST_CASE_BY_SLUG = {case.slug: case for case in TEST_CASES}


@dataclass(frozen=True)
class Algorithm:
    slug: str
    ranking_score: str
    prior_prompt: str = "graspability"


ALGORITHMS = (
    Algorithm("information_gain_original", "ig", "original"),
    Algorithm("information_gain_graspability", "ig_graspability", "graspability"),
    Algorithm("theory_original", "theory", "original"),
    Algorithm("theory_graspability", "theory", "graspability"),
)

# Keep the former names readable for existing cached results and old commands,
# but do not include them in the new four-method default matrix.
LEGACY_ALGORITHM_ALIASES = (
    Algorithm("information_gain", "ig_graspability", "graspability"),
    Algorithm("theory", "theory", "graspability"),
)
ALGORITHM_BY_SLUG = {
    algorithm.slug: algorithm
    for algorithm in (*ALGORITHMS, *LEGACY_ALGORITHM_ALIASES)
}


def parse_ground_truth_ids(value: object) -> list[int]:
    """Parse FreeGrasp's comma-separated, zero-based ground-truth IDs."""
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return sorted({int(part.strip()) for part in text.split(",") if part.strip()})


def safe_model_name(name: str) -> str:
    return name.replace("/", "_").replace(":", "_").replace(" ", "_")
