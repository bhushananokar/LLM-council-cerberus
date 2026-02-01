"""
Pydantic schemas for LLM Council analysis results.
Used for API request/response validation.
"""

from datetime import datetime
from typing import Optional, Any, Dict, List
from pydantic import BaseModel, Field


class CouncilAnalysisResponse(BaseModel):
    """Response schema for council analysis results."""
    
    id: str = Field(..., alias="_id", description="MongoDB document ID")
    decision_id: str = Field(..., description="Unique decision identifier")
    package_name: str = Field(..., description="Analyzed package name")
    package_version: str = Field(..., description="Package version")
    registry: str = Field(..., description="Package registry (npm, pypi, etc.)")
    
    verdict: str = Field(..., description="Final verdict: malicious, benign, uncertain, error")
    final_risk_score: float = Field(..., ge=0, le=100, description="Risk score (0-100)")
    final_confidence: float = Field(..., ge=0, le=100, description="Confidence level (0-100)")
    
    agent_responses: List[Dict[str, Any]] = Field(..., description="Individual agent analyses")
    consensus_type: str = Field(..., description="Type of consensus reached")
    agreement_percentage: float = Field(..., ge=0, le=100, description="Agreement percentage")
    
    total_tokens_used: int = Field(default=0, description="Total LLM tokens used")
    estimated_cost_usd: float = Field(default=0.0, description="Estimated cost in USD")
    analysis_duration_seconds: float = Field(default=0.0, description="Analysis duration")
    
    recommendation: Optional[str] = Field(None, description="Actionable recommendation")
    explanation: Optional[str] = Field(None, description="Detailed explanation")
    risk_factors: List[str] = Field(default_factory=list, description="Identified risk factors")
    mitigating_factors: List[str] = Field(default_factory=list, description="Mitigating factors")
    key_findings: Dict[str, Any] = Field(default_factory=dict, description="Key findings")
    
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "_id": "507f1f77bcf86cd799439011",
                "decision_id": "dec_2024_001",
                "package_name": "suspicious-lib",
                "package_version": "1.2.3",
                "registry": "npm",
                "verdict": "malicious",
                "final_risk_score": 92.5,
                "final_confidence": 87.3,
                "consensus_type": "unanimous",
                "agreement_percentage": 100.0,
                "total_tokens_used": 1500,
                "estimated_cost_usd": 0.0045
            }
        }


class DebateResultResponse(BaseModel):
    """Response schema for debate results."""
    
    id: str = Field(..., alias="_id", description="MongoDB document ID")
    debate_id: str = Field(..., description="Unique debate identifier")
    package_name: str = Field(..., description="Analyzed package name")
    
    final_verdict: str = Field(..., description="Final verdict after debate")
    final_risk_score: float = Field(..., ge=0, le=100, description="Final risk score")
    final_confidence: float = Field(..., ge=0, le=100, description="Final confidence")
    
    consensus_type: str = Field(..., description="How consensus was reached")
    rounds_to_consensus: int = Field(..., description="Number of debate rounds")
    final_votes: Dict[str, str] = Field(..., description="Final votes by agent")
    overseer_voted: bool = Field(default=False, description="Whether overseer intervened")
    
    debate_history: List[Dict[str, Any]] = Field(default_factory=list, description="Full debate history")
    total_interventions: int = Field(default=0, description="Overseer interventions")
    
    key_arguments: List[str] = Field(default_factory=list, description="Key arguments made")
    turning_points: List[str] = Field(default_factory=list, description="Critical turning points")
    
    debate_quality_score: float = Field(..., ge=0, le=100, description="Overall quality score")
    reasoning_depth: str = Field(..., description="Depth of reasoning")
    
    total_tokens_used: int = Field(default=0, description="Total tokens used")
    total_duration_seconds: float = Field(..., description="Total debate duration")
    completed_at: datetime = Field(..., description="Completion timestamp")
    
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "_id": "507f1f77bcf86cd799439012",
                "debate_id": "debate_2024_001",
                "package_name": "suspicious-lib",
                "final_verdict": "malicious",
                "final_risk_score": 95.0,
                "final_confidence": 92.0,
                "consensus_type": "unanimous",
                "rounds_to_consensus": 2,
                "overseer_voted": False,
                "debate_quality_score": 88.5,
                "reasoning_depth": "deep"
            }
        }


class CouncilAnalysisListResponse(BaseModel):
    """Response schema for list of council analyses."""
    
    total: int = Field(..., description="Total number of results")
    analyses: List[CouncilAnalysisResponse] = Field(..., description="List of analyses")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total": 10,
                "analyses": []
            }
        }


class DebateResultListResponse(BaseModel):
    """Response schema for list of debate results."""
    
    total: int = Field(..., description="Total number of results")
    debates: List[DebateResultResponse] = Field(..., description="List of debates")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total": 5,
                "debates": []
            }
        }


class CouncilStatsResponse(BaseModel):
    """Response schema for council statistics."""
    
    total_analyses: int = Field(..., description="Total number of analyses")
    verdict_breakdown: Dict[str, int] = Field(..., description="Count by verdict type")
    avg_risk_score: float = Field(..., description="Average risk score")
    avg_confidence: float = Field(..., description="Average confidence")
    total_tokens_used: int = Field(..., description="Total tokens used")
    total_cost_usd: float = Field(..., description="Total cost in USD")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_analyses": 100,
                "verdict_breakdown": {
                    "malicious": 15,
                    "benign": 80,
                    "uncertain": 5
                },
                "avg_risk_score": 35.2,
                "avg_confidence": 82.5,
                "total_tokens_used": 150000,
                "total_cost_usd": 0.45
            }
        }
