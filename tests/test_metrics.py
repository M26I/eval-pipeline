"""Tests for evaluation metrics."""

import pytest
from src.evals.metrics import (
    retrieval_hit_rate,
    abstention_correct,
    faithfulness,
    aggregate_results,
    is_soft_abstention,
)


class TestRetrievalHitRate:
    """Tests for retrieval_hit_rate."""
    
    def test_hit_found(self):
        """Test when relevant doc is retrieved."""
        assert retrieval_hit_rate(
            ["doc1", "doc2", "doc3"],
            ["doc2"]
        ) is True
    
    def test_hit_multiple_relevant(self):
        """Test with multiple relevant docs."""
        assert retrieval_hit_rate(
            ["doc1", "doc2"],
            ["doc2", "doc3"]
        ) is True
    
    def test_no_hit(self):
        """Test when no relevant doc is retrieved."""
        assert retrieval_hit_rate(
            ["doc1", "doc3"],
            ["doc2"]
        ) is False
    
    def test_empty_retrieved(self):
        """Test with no retrieved docs."""
        assert retrieval_hit_rate(
            [],
            ["doc1"]
        ) is False
    
    def test_empty_relevant(self):
        """Edge case: empty relevant list returns False."""
        assert retrieval_hit_rate(
            ["doc1", "doc2"],
            []
        ) is False
    
    def test_empty_both(self):
        """Edge case: both empty."""
        assert retrieval_hit_rate([], []) is False


class TestAbstentionCorrect:
    """Tests for abstention_correct."""
    
    def test_both_true(self):
        """Both predicted and should abstain."""
        assert abstention_correct(True, True) is True
    
    def test_both_false(self):
        """Neither predicted nor should abstain."""
        assert abstention_correct(False, False) is True
    
    def test_false_positive_abstention(self):
        """Predicted abstain but shouldn't."""
        assert abstention_correct(True, False) is False
    
    def test_false_negative_abstention(self):
        """Didn't abstain but should."""
        assert abstention_correct(False, True) is False


class TestIsSoftAbstention:
    """Tests for is_soft_abstention."""
    
    def test_clear_refusal(self):
        """Clear refusal pattern returns True."""
        assert is_soft_abstention("I don't know based on the documents.") is True
    
    def test_i_do_not_know(self):
        """Full form refusal pattern."""
        assert is_soft_abstention("I do not know the answer.") is True
    
    def test_not_enough_information(self):
        """Insufficient information pattern."""
        assert is_soft_abstention("I don't have enough information to answer that.") is True
    
    def test_does_not_mention(self):
        """Pattern: does not mention."""
        assert is_soft_abstention("The documents does not mention this.") is True
    
    def test_is_unknown(self):
        """Pattern: is unknown."""
        assert is_soft_abstention("The identity is unknown.") is True
    
    def test_cannot_answer(self):
        """Pattern: cannot answer."""
        assert is_soft_abstention("I cannot answer this question.") is True
    
    def test_not_specified_in(self):
        """Pattern: not specified in."""
        assert is_soft_abstention("This detail is not specified in the documents.") is True
    
    def test_case_insensitive(self):
        """Matching is case-insensitive."""
        assert is_soft_abstention("I DON'T KNOW THE ANSWER.") is True
    
    def test_normal_answer_returns_false(self):
        """Normal factual answer returns False."""
        assert is_soft_abstention("Margriet Dekker was born in 1851.") is False
    
    def test_answer_with_uncertainty_but_no_refusal(self):
        """Answer with uncertainty markers but no explicit refusal."""
        assert is_soft_abstention("The painting may have been created around 1880.") is False
    
    def test_empty_string(self):
        """Empty string returns False."""
        assert is_soft_abstention("") is False
    
    def test_do_not_provide_information(self):
        """Pattern: do not provide information."""
        assert is_soft_abstention("The documents do not provide information on that") is True


class TestFaithfulness:
    """Tests for faithfulness."""
    
    def test_all_substrings_present(self):
        """All required substrings found."""
        assert faithfulness(
            "The painting depicts a solitary figure with light",
            ["painting", "light"]
        ) == 1.0
    
    def test_partial_substrings(self):
        """Some substrings found."""
        assert faithfulness(
            "This is a painting",
            ["painting", "sculpture", "drawing"]
        ) == pytest.approx(1.0 / 3)
    
    def test_no_substrings(self):
        """No substrings found."""
        assert faithfulness(
            "This is something else",
            ["painting", "drawing"]
        ) == 0.0
    
    def test_case_insensitive(self):
        """Matching is case-insensitive."""
        assert faithfulness(
            "This is a PAINTING",
            ["painting"]
        ) == 1.0
    
    def test_empty_required_substrings(self):
        """Edge case: empty list returns 1.0 (trivially faithful)."""
        assert faithfulness(
            "Any answer",
            []
        ) == 1.0
    
    def test_empty_answer_with_requirements(self):
        """Empty answer with required substrings."""
        assert faithfulness(
            "",
            ["something"]
        ) == 0.0
    
    def test_substring_matching(self):
        """Substrings should match within larger text."""
        assert faithfulness(
            "The documentation is clear",
            ["document"]
        ) == 1.0


class TestAggregateResults:
    """Tests for aggregate_results."""
    
    def test_empty_results(self):
        """Empty results list."""
        result = aggregate_results([])
        assert result["total_queries"] == 0
        assert result["hit_rate_pct"] == 0.0
        assert result["mean_faithfulness"] == 0.0
    
    def test_single_perfect_result(self):
        """Single correct result."""
        results = [
            {
                "retrieved_ids": ["doc1"],
                "relevant_ids": ["doc1"],
                "predicted_abstain": False,
                "should_abstain": False,
                "answer": "This is a painting",
                "required_substrings": ["painting"],
                "latency_ms": 100.0,
            }
        ]
        agg = aggregate_results(results)
        assert agg["hit_rate_pct"] == 100.0
        assert agg["mean_faithfulness"] == 1.0
        assert agg["total_queries"] == 1
        assert agg["total_abstained"] == 0
        assert agg["mean_latency_ms"] == 100.0
    
    def test_multiple_results_with_abstentions(self):
        """Multiple results with mixed abstentions."""
        results = [
            {
                "retrieved_ids": ["doc1"],
                "relevant_ids": ["doc1"],
                "predicted_abstain": False,
                "should_abstain": False,
                "answer": "The painting is important",
                "required_substrings": ["painting"],
                "latency_ms": 100.0,
            },
            {
                "retrieved_ids": ["doc2"],
                "relevant_ids": [],
                "predicted_abstain": True,
                "should_abstain": True,
                "answer": "I don't know",
                "required_substrings": [],
                "latency_ms": 50.0,
            },
            {
                "retrieved_ids": [],
                "relevant_ids": ["doc3"],
                "predicted_abstain": False,
                "should_abstain": True,
                "answer": "Wrong answer",
                "required_substrings": [],
                "latency_ms": 80.0,
            },
        ]
        agg = aggregate_results(results)
        
        assert agg["total_queries"] == 3
        assert agg["total_abstained"] == 1
        assert agg["hit_rate_pct"] == pytest.approx(50.0, abs=0.1)  # 1 hit out of 2 retrieval questions (abstention question excluded from denominator)
        assert agg["abstention_precision"] == 1.0  # 1 correct abstention out of 1 predicted
        assert agg["abstention_recall"] == 0.5  # 1 correct out of 2 ground-truth abstentions (result 3 missed)
        assert agg["mean_faithfulness"] == 1.0  # only 1 non-abstained answer
        assert agg["mean_latency_ms"] == pytest.approx(76.67, abs=0.1)
    
    def test_abstention_metrics_none_when_no_cases(self):
        """Abstention precision/recall are None when no abstentions."""
        results = [
            {
                "retrieved_ids": ["doc1"],
                "relevant_ids": ["doc1"],
                "predicted_abstain": False,
                "should_abstain": False,
                "answer": "Answer",
                "required_substrings": [],
                "latency_ms": 100.0,
            }
        ]
        agg = aggregate_results(results)
        assert agg["abstention_precision"] is None
        assert agg["abstention_recall"] is None
    
    def test_p95_latency(self):
        """Test p95 latency calculation."""
        # Create 20 results with known latencies
        results = [
            {
                "retrieved_ids": ["doc1"],
                "relevant_ids": ["doc1"],
                "predicted_abstain": False,
                "should_abstain": False,
                "answer": "Answer",
                "required_substrings": [],
                "latency_ms": float(i * 10),  # 0, 10, 20, ..., 190
            }
            for i in range(20)
        ]
        agg = aggregate_results(results)
        # p95 should be around 180-190
        assert 170 <= agg["p95_latency_ms"] <= 190
    
    def test_faithfulness_excludes_abstentions(self):
        """Faithfulness is only computed over non-abstained answers."""
        results = [
            {
                "retrieved_ids": [],
                "relevant_ids": [],
                "predicted_abstain": True,
                "should_abstain": True,
                "answer": "I don't know",
                "required_substrings": ["required"],
                "latency_ms": 50.0,
            },
            {
                "retrieved_ids": ["doc1"],
                "relevant_ids": ["doc1"],
                "predicted_abstain": False,
                "should_abstain": False,
                "answer": "This contains required information",
                "required_substrings": ["required"],
                "latency_ms": 100.0,
            },
        ]
        agg = aggregate_results(results)
        # Faithfulness computed only on the second result
        assert agg["mean_faithfulness"] == 1.0
