"""
Debate System Models
====================
Enhanced models for the sophisticated debate orchestration system.

Models:
- ArgumentPoint: Individual argument point proposed by an agent
- Presentation: Agent's formal presentation of their case
- Reflection: Agent's self-assessment after hearing other arguments
- Vote: Agent's vote with justification
- DebateRound: Complete round of the debate (phases 1-4)
- OverseerIntervention: Overseer's feedback on reasoning quality
- DebateState: Current state of the ongoing debate
- DebateResult: Final outcome with complete history
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator


class VerdictType(str, Enum):
    """Possible verdicts for package analysis."""
    MALICIOUS = "malicious"
    BENIGN = "benign"
    UNCERTAIN = "uncertain"
    ERROR = "error"


class ArgumentPoint(BaseModel):
    """Individual argument point that an agent plans to make."""
    
    agent_name: str = Field(..., description="Agent making this argument")
    point_id: str = Field(..., description="Unique identifier for this point")
    category: str = Field(..., description="Category (e.g., 'code_pattern', 'threat_indicator')")
    claim: str = Field(..., description="The main claim being made")
    evidence: str = Field(..., description="Evidence supporting this claim")
    weight: float = Field(..., ge=0, le=1, description="How important this point is (0-1)")
    targets_agent: Optional[str] = Field(None, description="If countering another agent")
    
    class Config:
        json_schema_extra = {
            "example": {
                "agent_name": "code_intelligence",
                "point_id": "code_intel_01",
                "category": "obfuscation_detection",
                "claim": "Code contains base64-encoded malicious payload",
                "evidence": "Found eval(atob(...)) pattern with 95% entropy",
                "weight": 0.9,
                "targets_agent": None
            }
        }


class Presentation(BaseModel):
    """Agent's formal presentation of their complete argument."""
    
    agent_name: str = Field(..., description="Agent presenting")
    round_number: int = Field(..., description="Which debate round")
    
    # Core position
    position: VerdictType = Field(..., description="Agent's stance")
    risk_score: float = Field(..., ge=0, le=100, description="Assessed risk (0-100)")
    confidence: float = Field(..., ge=0, le=100, description="Confidence in assessment")
    
    # Argument structure
    main_thesis: str = Field(..., description="Primary argument thesis")
    supporting_points: List[ArgumentPoint] = Field(..., description="List of argument points")
    counterarguments: List[str] = Field(default_factory=list, description="Anticipated counterarguments")
    
    # Evidence and reasoning
    key_evidence: List[str] = Field(..., description="Critical pieces of evidence")
    reasoning_chain: str = Field(..., description="Logical reasoning from evidence to conclusion")
    
    # Metadata
    tokens_used: int = Field(default=0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "agent_name": "threat_intelligence",
                "round_number": 1,
                "position": "malicious",
                "risk_score": 92,
                "confidence": 85,
                "main_thesis": "Package exhibits typosquatting and data exfiltration patterns",
                "supporting_points": [],
                "key_evidence": ["Similar to popular package", "Network calls to unknown domain"],
                "reasoning_chain": "Package name is 1 char different from popular lib -> Downloads credentials -> Sends to external server"
            }
        }


class Reflection(BaseModel):
    """Agent's reflection after hearing other agents' presentations."""
    
    agent_name: str = Field(..., description="Agent reflecting")
    round_number: int = Field(..., description="Which debate round")
    
    # Self-assessment
    initial_position: VerdictType = Field(..., description="Agent's original position")
    revised_position: VerdictType = Field(..., description="Position after reflection")
    position_changed: bool = Field(..., description="Whether position changed")
    
    # Analysis of other arguments
    strengths_found: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Strong points from each other agent"
    )
    weaknesses_found: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Weak points from each other agent"
    )
    
    # Updated assessment
    revised_risk_score: float = Field(..., ge=0, le=100)
    revised_confidence: float = Field(..., ge=0, le=100)
    
    # Reasoning
    reflection_reasoning: str = Field(..., description="Why position changed or remained")
    most_convincing_argument: str = Field(..., description="Most persuasive argument heard")
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Vote(BaseModel):
    """Agent's vote with complete justification."""
    
    agent_name: str = Field(..., description="Agent voting")
    round_number: int = Field(..., description="Which debate round")
    
    # Vote
    verdict: VerdictType = Field(..., description="Final vote")
    risk_score: float = Field(..., ge=0, le=100, description="Final risk assessment")
    confidence: float = Field(..., ge=0, le=100, description="Confidence in vote")
    
    # Justification
    vote_reasoning: str = Field(..., description="Complete justification for vote")
    deciding_factors: List[str] = Field(..., description="Key factors that decided the vote")
    
    # Certainty
    willing_to_change: bool = Field(..., description="Open to changing vote in next round")
    certainty_level: str = Field(..., description="'absolute', 'high', 'moderate', 'low'")
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class OverseerIntervention(BaseModel):
    """Overseer's intervention when agent reasoning goes off-track."""
    
    round_number: int = Field(..., description="When intervention occurred")
    target_agent: str = Field(..., description="Agent being corrected")
    
    # Issue identification
    issue_type: str = Field(..., description="Type of reasoning error")
    issue_description: str = Field(..., description="What went wrong")
    
    # Correction
    guidance: str = Field(..., description="How to get back on track")
    severity: str = Field(..., description="'minor', 'moderate', 'severe'")
    
    # Context
    problematic_reasoning: str = Field(..., description="The problematic reasoning snippet")
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class OverseerAssessment(BaseModel):
    """Overseer's assessment of all agents' reasoning quality."""
    
    round_number: int = Field(..., description="Which round assessed")
    
    # Quality scores per agent
    reasoning_quality: Dict[str, float] = Field(
        ...,
        description="Quality score 0-100 for each agent"
    )
    
    # Flags
    intervention_needed: bool = Field(..., description="Whether intervention required")
    agents_off_track: List[str] = Field(default_factory=list, description="Agents needing guidance")
    
    # Analysis
    debate_quality: str = Field(..., description="Overall debate quality: 'excellent', 'good', 'fair', 'poor'")
    convergence_assessment: str = Field(..., description="Are agents converging toward truth?")
    
    # Recommendations
    recommendations: List[str] = Field(default_factory=list, description="Suggestions for agents")
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DebateRound(BaseModel):
    """Complete round of debate including all 4 phases."""
    
    round_number: int = Field(..., description="Round number")
    
    # Phase 1: Decide points
    argument_points: List[ArgumentPoint] = Field(
        default_factory=list,
        description="Points planned by all agents"
    )
    
    # Phase 2: Present
    presentations: List[Presentation] = Field(
        default_factory=list,
        description="Formal presentations by all agents"
    )
    
    # Phase 3: Reflect
    reflections: List[Reflection] = Field(
        default_factory=list,
        description="Agent reflections after hearing others"
    )
    
    # Phase 4: Vote
    votes: List[Vote] = Field(
        default_factory=list,
        description="Agent votes with justifications"
    )
    
    # Overseer involvement
    overseer_assessment: Optional[OverseerAssessment] = Field(
        None,
        description="Overseer's quality assessment"
    )
    interventions: List[OverseerIntervention] = Field(
        default_factory=list,
        description="Any overseer interventions"
    )
    
    # Round outcomes
    consensus_reached: bool = Field(default=False, description="Whether consensus achieved")
    consensus_verdict: Optional[VerdictType] = Field(None, description="Consensus position if reached")
    
    # Metadata
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None


class DebateState(BaseModel):
    """Current state of an ongoing debate."""
    
    debate_id: str = Field(..., description="Unique debate identifier")
    package_name: str = Field(..., description="Package being analyzed")
    
    # Progress
    current_round: int = Field(default=1, description="Current round number")
    current_phase: int = Field(default=1, description="Current phase (1-4)")
    max_rounds: int = Field(default=5, description="Maximum rounds before forcing decision")
    
    # History
    rounds: List[DebateRound] = Field(default_factory=list, description="Completed rounds")
    
    # Status
    is_active: bool = Field(default=True, description="Whether debate ongoing")
    consensus_reached: bool = Field(default=False)
    forced_termination: bool = Field(default=False, description="Stopped due to max rounds")
    
    # Statistics
    total_tokens_used: int = Field(default=0)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "debate_id": "debate_2024_001",
                "package_name": "suspicious-lib",
                "current_round": 2,
                "current_phase": 3,
                "max_rounds": 5,
                "is_active": True,
                "consensus_reached": False
            }
        }


class DebateResult(BaseModel):
    """Final result of completed debate."""
    
    debate_id: str = Field(..., description="Debate identifier")
    package_name: str = Field(..., description="Package analyzed")
    
    # Final decision
    final_verdict: VerdictType = Field(..., description="Final consensus verdict")
    final_risk_score: float = Field(..., ge=0, le=100)
    final_confidence: float = Field(..., ge=0, le=100)
    
    # How consensus was reached
    consensus_type: str = Field(
        ...,
        description="'unanimous', 'majority', 'tiebroken_by_overseer', 'forced'"
    )
    rounds_to_consensus: int = Field(..., description="Number of rounds taken")
    
    # Vote breakdown
    final_votes: Dict[str, VerdictType] = Field(..., description="Each agent's final vote")
    overseer_voted: bool = Field(default=False, description="Whether overseer cast deciding vote")
    
    # Complete history
    debate_history: List[DebateRound] = Field(..., description="All debate rounds")
    total_interventions: int = Field(default=0, description="Number of overseer interventions")
    
    # Summary
    key_arguments: List[str] = Field(..., description="Most important arguments made")
    turning_points: List[str] = Field(
        default_factory=list,
        description="Key moments that changed the debate"
    )
    
    # Quality metrics
    debate_quality_score: float = Field(..., ge=0, le=100, description="Overall debate quality")
    reasoning_depth: str = Field(..., description="'shallow', 'moderate', 'deep', 'exceptional'")
    
    # Metadata
    total_tokens_used: int = Field(default=0)
    total_duration_seconds: float = Field(...)
    completed_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "debate_id": "debate_2024_001",
                "package_name": "suspicious-lib",
                "final_verdict": "malicious",
                "final_risk_score": 95,
                "final_confidence": 92,
                "consensus_type": "unanimous",
                "rounds_to_consensus": 2,
                "final_votes": {
                    "code_intelligence": "malicious",
                    "threat_intelligence": "malicious",
                    "behavioral_intelligence": "malicious"
                },
                "overseer_voted": False,
                "total_interventions": 1,
                "debate_quality_score": 88,
                "reasoning_depth": "deep",
                "total_duration_seconds": 45.2
            }
        }
