import sys
import unittest
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))


class FakeEncoding(dict):
    def word_ids(self):
        return self["_word_ids"]


class FakeFastTokenizer:
    cls_token_id = 101
    sep_token_id = 102
    pad_token_id = 0

    def __call__(
        self,
        text,
        return_offsets_mapping=False,
        return_attention_mask=True,
        add_special_tokens=True,
        truncation=True,
    ):
        self.last_text = text
        return FakeEncoding(
            {
                "input_ids": [101, 201, 202, 300, 102],
                "attention_mask": [1, 1, 1, 1, 1],
                "offset_mapping": [(0, 0), (0, 5), (5, 12), (13, 18), (0, 0)],
                "_word_ids": [None, 0, 0, 1, None],
            }
        )


class DiagnosticPrimitiveTests(unittest.TestCase):
    def test_alignment_groups_all_subwords_and_skips_special_tokens(self):
        from src.tokenizer_alignment import align_text_to_tokens

        result = align_text_to_tokens(FakeFastTokenizer(), "unbelievable movie")

        self.assertEqual([unit.text for unit in result.words], ["unbelievable", "movie"])
        self.assertEqual(result.words[0].subword_indices, [1, 2])
        self.assertEqual(result.words[1].subword_indices, [3])
        self.assertNotIn(0, result.candidate_token_indices)
        self.assertNotIn(4, result.candidate_token_indices)

    def test_candidate_graph_deduplicates_and_merges_edge_types(self):
        from src.candidate_graph import build_candidate_edges
        from src.word_units import WordUnit

        words = [
            WordUnit(index=0, text="not", subword_indices=[1]),
            WordUnit(index=1, text="very", subword_indices=[2]),
            WordUnit(index=2, text="good", subword_indices=[3]),
        ]

        edges = build_candidate_edges(words, dependency_edges=[(0, 1), (0, 2)])
        edge_types = {(edge.i, edge.j): edge.edge_type for edge in edges}

        self.assertEqual(edge_types[(0, 1)], "adjacent_dependency")
        self.assertEqual(edge_types[(1, 2)], "adjacent")
        self.assertEqual(edge_types[(0, 2)], "dependency")

    def test_masking_word_group_removes_all_subwords_only_for_that_word(self):
        from src.masking_adapter import keep_word_groups, mask_word_groups
        from src.word_units import WordUnit

        words = [
            WordUnit(index=0, text="unbelievable", subword_indices=[1, 2]),
            WordUnit(index=1, text="movie", subword_indices=[3]),
        ]
        input_ids = [101, 201, 202, 300, 102]
        attention_mask = [1, 1, 1, 1, 1]

        masked = mask_word_groups(input_ids, attention_mask, words, masked_word_indices={0})
        self.assertEqual(masked.input_ids, [101, 300, 102])
        self.assertEqual(masked.kept_token_indices, [0, 3, 4])

        kept = keep_word_groups(
            input_ids,
            attention_mask,
            words,
            kept_word_indices={1},
            required_token_indices={0, 4},
        )
        self.assertEqual(kept.input_ids, [101, 300, 102])
        self.assertEqual(kept.kept_token_indices, [0, 3, 4])

    def test_interaction_teacher_uses_finite_difference_and_caches_singletons(self):
        from src.candidate_graph import CandidateEdge
        from src.interaction_teacher import compute_interactions

        calls = []
        scores = {
            frozenset(): 10.0,
            frozenset({0}): 4.0,
            frozenset({1}): 4.0,
            frozenset({0, 1}): 1.0,
        }

        def scorer(masked_words):
            key = frozenset(masked_words)
            calls.append(key)
            return scores[key]

        result = compute_interactions([CandidateEdge(0, 1, "adjacent")], scorer)

        self.assertEqual(result.edge_scores[0].interaction_score, 3.0)
        self.assertEqual(result.edge_scores[0].absolute_interaction_score, 3.0)
        self.assertEqual(calls.count(frozenset({0})), 1)
        self.assertEqual(calls.count(frozenset({1})), 1)

    def test_metric_direction_matches_aml_protocol(self):
        from src.faithfulness_metrics import (
            aopc,
            comprehensiveness_score,
            sufficiency_error,
        )

        self.assertEqual(comprehensiveness_score(original_score=0.9, masked_score=0.2), 0.7)
        self.assertEqual(sufficiency_error(original_score=0.9, kept_score=0.85), 0.05)
        self.assertAlmostEqual(aopc([0.1, 0.2, 0.4]), (0.1 + 0.2 + 0.4) / 4)


if __name__ == "__main__":
    unittest.main()
