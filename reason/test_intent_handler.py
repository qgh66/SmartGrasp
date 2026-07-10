from pathlib import Path
import unittest

from reason.intent_handle.intent_handler import (
    BRANCH_FULLY_VISIBLE,
    BRANCH_INVISIBLE,
    BRANCH_PARTIALLY_VISIBLE,
    resolve_intent,
)


ROOT = Path(__file__).resolve().parent
PERCEPTION_SUMMARIES = sorted(ROOT.glob("sample_data/scene_*/perception/summary.json"))


class FakeVLMClient:
    def __init__(self, decision):
        self.decision = decision
        self.calls = []

    def choose_target(self, instruction, scene_context, image_paths):
        self.calls.append((instruction, scene_context, list(image_paths)))
        return self.decision


class IntentHandlerTest(unittest.TestCase):
    def test_vlm_candidates_choose_least_occluded_duplicate(self):
        summary = ROOT / "sample_data" / "scene_564" / "perception" / "summary.json"
        client = FakeVLMClient(
            {
                "target_present": True,
                "target_object_id": 2,
                "target_category": "red round lid",
                "candidate_object_ids": [1, 2],
                "reason": "Both red lids can satisfy the request.",
            }
        )

        result = resolve_intent("帮我拿一下红色盖子", summary, client=client)

        self.assertEqual(result.branch, BRANCH_FULLY_VISIBLE)
        self.assertIsNotNone(result.target_object)
        self.assertEqual(result.target_object.object_id, 1)
        self.assertEqual(result.target_object.category, "red round lid")
        self.assertTrue(client.calls[0][2])

    def test_partially_visible_branch_when_vlm_selects_occluded_object(self):
        summary = ROOT / "sample_data" / "scene_365" / "perception" / "summary.json"
        client = FakeVLMClient(
            {
                "target_present": True,
                "target_object_id": 4,
                "target_category": "white cylindrical stick",
                "candidate_object_ids": [4],
                "reason": "The user explicitly refers to object 4.",
            }
        )

        result = resolve_intent("拿一下4号物体", summary, client=client)

        self.assertEqual(result.branch, BRANCH_PARTIALLY_VISIBLE)
        self.assertIsNotNone(result.target_object)
        self.assertEqual(result.target_object.object_id, 4)
        self.assertEqual(result.occluded_by, (3,))

    def test_invisible_when_vlm_reports_no_target(self):
        summary = ROOT / "sample_data" / "scene_365" / "perception" / "summary.json"
        client = FakeVLMClient(
            {
                "target_present": False,
                "target_object_id": None,
                "target_category": None,
                "candidate_object_ids": [],
                "reason": "There is no key in the scene.",
            }
        )

        result = resolve_intent("帮我拿一下钥匙", summary, client=client)

        self.assertEqual(result.branch, BRANCH_INVISIBLE)
        self.assertIsNone(result.target_object)

    def test_invalid_vlm_object_id_is_invisible(self):
        summary = ROOT / "sample_data" / "scene_365" / "perception" / "summary.json"
        client = FakeVLMClient(
            {
                "target_present": True,
                "target_object_id": 99,
                "target_category": "unknown",
                "candidate_object_ids": [99],
                "reason": "Bad id.",
            }
        )

        result = resolve_intent("拿一下不存在的99号物体", summary, client=client)

        self.assertEqual(result.branch, BRANCH_INVISIBLE)
        self.assertIsNone(result.target_object)

    def test_all_perception_summaries_reject_invalid_target_id(self):
        self.assertTrue(PERCEPTION_SUMMARIES, "No perception summaries found in sample_data.")

        for summary in PERCEPTION_SUMMARIES:
            with self.subTest(summary=str(summary)):
                client = FakeVLMClient(
                    {
                        "target_present": True,
                        "target_object_id": 999999,
                        "target_category": "unknown",
                        "candidate_object_ids": [999999],
                        "reason": "Invalid id for regression test.",
                    }
                )
                result = resolve_intent("拿一下不存在的物体", summary, client=client)
                self.assertEqual(result.branch, BRANCH_INVISIBLE)
                self.assertIsNone(result.target_object)

    def test_all_perception_summaries_accept_a_valid_summary_object_id(self):
        self.assertTrue(PERCEPTION_SUMMARIES, "No perception summaries found in sample_data.")

        for summary in PERCEPTION_SUMMARIES:
            with self.subTest(summary=str(summary)):
                scene_context = FakeVLMClient({}).calls  # no-op placeholder to keep local style simple
                del scene_context
                object_ids = []
                for item in resolve_summary_labels(summary):
                    if item not in object_ids:
                        object_ids.append(item)
                self.assertTrue(object_ids, f"No object ids parsed from {summary}")

                first_id = object_ids[0]
                client = FakeVLMClient(
                    {
                        "target_present": True,
                        "target_object_id": first_id,
                        "target_category": "valid target",
                        "candidate_object_ids": [first_id],
                        "reason": "Valid id for regression test.",
                    }
                )
                result = resolve_intent("拿一下这个物体", summary, client=client)
                self.assertIsNotNone(result.target_object)
                self.assertEqual(result.target_object.object_id, first_id)
                self.assertIn(result.branch, {BRANCH_FULLY_VISIBLE, BRANCH_PARTIALLY_VISIBLE})

    def test_all_perception_summaries_return_invisible_when_vlm_reports_absent(self):
        self.assertTrue(PERCEPTION_SUMMARIES, "No perception summaries found in sample_data.")

        for summary in PERCEPTION_SUMMARIES:
            with self.subTest(summary=str(summary)):
                client = FakeVLMClient(
                    {
                        "target_present": False,
                        "target_object_id": None,
                        "target_category": None,
                        "candidate_object_ids": [],
                        "reason": "Absent target for regression test.",
                    }
                )
                result = resolve_intent("帮我拿一个场景里没有的东西", summary, client=client)
                self.assertEqual(result.branch, BRANCH_INVISIBLE)
                self.assertIsNone(result.target_object)


def resolve_summary_labels(summary_path: Path) -> list[int]:
    import json

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    object_ids = []
    for item in payload.get("matrix_labels", []):
        text = str(item)
        if ":" not in text:
            continue
        raw_id = text.split(":", 1)[0].strip()
        try:
            object_ids.append(int(raw_id))
        except ValueError:
            continue
    return object_ids


if __name__ == "__main__":
    unittest.main()
