"""
Test Debate System
==================
Basic tests for the sophisticated debate orchestration system.
"""

import pytest
import asyncio
from src.debate_orchestrator import DebateOrchestrator
from src.debate_models import (
    ArgumentPoint,
    Presentation,
    Reflection,
    Vote,
    VerdictType
)
from src.models import PackageData
from src.overseer import OverseerAgent


def test_debate_models():
    """Test that debate models can be instantiated."""
    
    # Test ArgumentPoint
    point = ArgumentPoint(
        agent_name="test_agent",
        point_id="point_1",
        category="test",
        claim="Test claim",
        evidence="Test evidence",
        weight=0.8
    )
    assert point.agent_name == "test_agent"
    assert point.weight == 0.8
    
    # Test Vote
    vote = Vote(
        agent_name="test_agent",
        round_number=1,
        verdict=VerdictType.MALICIOUS,
        risk_score=90,
        confidence=85,
        vote_reasoning="Test reasoning",
        deciding_factors=["factor1", "factor2"],
        willing_to_change=False,
        certainty_level="high"
    )
    assert vote.verdict == VerdictType.MALICIOUS
    assert vote.risk_score == 90


def test_orchestrator_initialization():
    """Test debate orchestrator can be initialized."""
    
    orchestrator = DebateOrchestrator(
        max_rounds=3,
        consensus_threshold=0.67,
        enable_overseer=True
    )
    
    assert orchestrator.max_rounds == 3
    assert orchestrator.consensus_threshold == 0.67
    assert orchestrator.enable_overseer == True
    assert not orchestrator.agents_initialized


def test_consensus_detection():
    """Test consensus detection logic."""
    
    orchestrator = DebateOrchestrator()
    
    # Test unanimous consensus
    votes_unanimous = [
        Vote(
            agent_name=f"agent_{i}",
            round_number=1,
            verdict=VerdictType.MALICIOUS,
            risk_score=90,
            confidence=85,
            vote_reasoning="Test",
            deciding_factors=[],
            willing_to_change=False,
            certainty_level="high"
        )
        for i in range(3)
    ]
    
    result = orchestrator._check_consensus(votes_unanimous)
    assert result["reached"] == True
    assert result["verdict"] == VerdictType.MALICIOUS
    assert result["is_tie"] == False
    
    # Test tie
    votes_tie = [
        Vote(
            agent_name="agent_1",
            round_number=1,
            verdict=VerdictType.MALICIOUS,
            risk_score=90,
            confidence=85,
            vote_reasoning="Test",
            deciding_factors=[],
            willing_to_change=False,
            certainty_level="high"
        ),
        Vote(
            agent_name="agent_2",
            round_number=1,
            verdict=VerdictType.BENIGN,
            risk_score=10,
            confidence=85,
            vote_reasoning="Test",
            deciding_factors=[],
            willing_to_change=False,
            certainty_level="high"
        )
    ]
    
    result = orchestrator._check_consensus(votes_tie)
    assert result["is_tie"] == True
    assert result["reached"] == False
    
    # Test no consensus (split but not tie)
    votes_split = [
        Vote(agent_name=f"agent_{i}", round_number=1, verdict=VerdictType.MALICIOUS,
             risk_score=90, confidence=85, vote_reasoning="Test", deciding_factors=[],
             willing_to_change=False, certainty_level="high")
        for i in range(2)
    ] + [
        Vote(agent_name="agent_3", round_number=1, verdict=VerdictType.BENIGN,
             risk_score=10, confidence=85, vote_reasoning="Test", deciding_factors=[],
             willing_to_change=False, certainty_level="high")
    ]
    
    result = orchestrator._check_consensus(votes_split)
    assert result["reached"] == True  # 2/3 = 67% which meets threshold
    assert result["verdict"] == VerdictType.MALICIOUS


def test_overseer_initialization():
    """Test overseer agent can be initialized."""
    
    overseer = OverseerAgent()
    assert overseer.agent_name == "overseer"
    assert overseer.model_name == "gemini-2.0-flash-thinking-exp"


def test_package_data_creation():
    """Test creating package data for debate."""
    
    package = PackageData(
        package_name="test-package",
        version="1.0.0",
        registry="npm",
        description="Test package",
        code_segments=[
            {
                "code": "eval(something)",
                "location": "index.js:10",
                "reason": "dangerous_eval"
            }
        ],
        static_analysis={
            "dangerous_apis": ["eval"],
            "obfuscation_score": 75
        }
    )
    
    assert package.package_name == "test-package"
    assert len(package.code_segments) == 1
    assert package.static_analysis["dangerous_apis"] == ["eval"]


@pytest.mark.asyncio
async def test_debate_orchestrator_basic():
    """Test basic debate orchestration (without API calls)."""
    
    orchestrator = DebateOrchestrator(max_rounds=2, enable_overseer=False)
    
    # This will fail without API keys, but tests the structure
    package = PackageData(
        package_name="test-malicious",
        version="1.0.0",
        description="Test package",
        code_segments=[{"code": "eval('malicious')", "location": "test.js:1", "reason": "eval"}]
    )
    
    # Test that orchestrator is properly set up
    assert orchestrator.max_rounds == 2
    assert not orchestrator.agents_initialized
    
    # Note: Full debate test requires API keys and is in integration tests


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
