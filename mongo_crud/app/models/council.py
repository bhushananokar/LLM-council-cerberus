"""
Database models for LLM Council analysis results.
"""

from datetime import datetime
from typing import Optional, Any, Dict, List
from bson import ObjectId


class CouncilAnalysisModel:
    """
    MongoDB model for storing council analysis results.
    Stores the complete CouncilDecision output from the orchestrator.
    """
    
    def __init__(
        self,
        # Decision ID and package info
        decision_id: str,
        package_name: str,
        package_version: str,
        registry: str,
        
        # Final verdict
        verdict: str,  # malicious, benign, uncertain, error
        final_risk_score: float,
        final_confidence: float,
        
        # Agent responses
        agent_responses: List[Dict[str, Any]],
        
        # Consensus details
        consensus_type: str,  # unanimous, majority, forced, etc.
        agreement_percentage: float,
        
        # Metadata
        total_tokens_used: int = 0,
        estimated_cost_usd: float = 0.0,
        analysis_duration_seconds: float = 0.0,
        
        # Full details
        recommendation: Optional[str] = None,
        explanation: Optional[str] = None,
        risk_factors: Optional[List[str]] = None,
        mitigating_factors: Optional[List[str]] = None,
        key_findings: Optional[Dict[str, Any]] = None,
        
        # Timestamps
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        _id: Optional[ObjectId] = None
    ):
        self._id = _id
        self.decision_id = decision_id
        self.package_name = package_name
        self.package_version = package_version
        self.registry = registry
        self.verdict = verdict
        self.final_risk_score = final_risk_score
        self.final_confidence = final_confidence
        self.agent_responses = agent_responses
        self.consensus_type = consensus_type
        self.agreement_percentage = agreement_percentage
        self.total_tokens_used = total_tokens_used
        self.estimated_cost_usd = estimated_cost_usd
        self.analysis_duration_seconds = analysis_duration_seconds
        self.recommendation = recommendation
        self.explanation = explanation
        self.risk_factors = risk_factors or []
        self.mitigating_factors = mitigating_factors or []
        self.key_findings = key_findings or {}
        self.created_at = created_at or datetime.utcnow()
        self.updated_at = updated_at or datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for MongoDB storage."""
        return {
            "_id": self._id,
            "decision_id": self.decision_id,
            "package_name": self.package_name,
            "package_version": self.package_version,
            "registry": self.registry,
            "verdict": self.verdict,
            "final_risk_score": self.final_risk_score,
            "final_confidence": self.final_confidence,
            "agent_responses": self.agent_responses,
            "consensus_type": self.consensus_type,
            "agreement_percentage": self.agreement_percentage,
            "total_tokens_used": self.total_tokens_used,
            "estimated_cost_usd": self.estimated_cost_usd,
            "analysis_duration_seconds": self.analysis_duration_seconds,
            "recommendation": self.recommendation,
            "explanation": self.explanation,
            "risk_factors": self.risk_factors,
            "mitigating_factors": self.mitigating_factors,
            "key_findings": self.key_findings,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CouncilAnalysisModel":
        """Create model from dictionary."""
        return cls(**data)


class DebateResultModel:
    """
    MongoDB model for storing debate system results.
    Stores the complete DebateResult output from the debate orchestrator.
    """
    
    def __init__(
        self,
        # Debate ID and package info
        debate_id: str,
        package_name: str,
        
        # Final decision
        final_verdict: str,  # malicious, benign, uncertain
        final_risk_score: float,
        final_confidence: float,
        
        # Consensus details
        consensus_type: str,  # unanimous, majority, tiebroken_by_overseer, forced
        rounds_to_consensus: int,
        final_votes: Dict[str, str],  # agent_name -> verdict
        overseer_voted: bool = False,
        
        # Debate history
        debate_history: List[Dict[str, Any]] = None,  # List of DebateRound dicts
        total_interventions: int = 0,
        
        # Analysis
        key_arguments: List[str] = None,
        turning_points: List[str] = None,
        
        # Quality metrics
        debate_quality_score: float = 0.0,
        reasoning_depth: str = "moderate",  # shallow, moderate, deep, exceptional
        
        # Metadata
        total_tokens_used: int = 0,
        total_duration_seconds: float = 0.0,
        completed_at: Optional[datetime] = None,
        
        # Timestamps
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        _id: Optional[ObjectId] = None
    ):
        self._id = _id
        self.debate_id = debate_id
        self.package_name = package_name
        self.final_verdict = final_verdict
        self.final_risk_score = final_risk_score
        self.final_confidence = final_confidence
        self.consensus_type = consensus_type
        self.rounds_to_consensus = rounds_to_consensus
        self.final_votes = final_votes
        self.overseer_voted = overseer_voted
        self.debate_history = debate_history or []
        self.total_interventions = total_interventions
        self.key_arguments = key_arguments or []
        self.turning_points = turning_points or []
        self.debate_quality_score = debate_quality_score
        self.reasoning_depth = reasoning_depth
        self.total_tokens_used = total_tokens_used
        self.total_duration_seconds = total_duration_seconds
        self.completed_at = completed_at or datetime.utcnow()
        self.created_at = created_at or datetime.utcnow()
        self.updated_at = updated_at or datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for MongoDB storage."""
        return {
            "_id": self._id,
            "debate_id": self.debate_id,
            "package_name": self.package_name,
            "final_verdict": self.final_verdict,
            "final_risk_score": self.final_risk_score,
            "final_confidence": self.final_confidence,
            "consensus_type": self.consensus_type,
            "rounds_to_consensus": self.rounds_to_consensus,
            "final_votes": self.final_votes,
            "overseer_voted": self.overseer_voted,
            "debate_history": self.debate_history,
            "total_interventions": self.total_interventions,
            "key_arguments": self.key_arguments,
            "turning_points": self.turning_points,
            "debate_quality_score": self.debate_quality_score,
            "reasoning_depth": self.reasoning_depth,
            "total_tokens_used": self.total_tokens_used,
            "total_duration_seconds": self.total_duration_seconds,
            "completed_at": self.completed_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DebateResultModel":
        """Create model from dictionary."""
        return cls(**data)
