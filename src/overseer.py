"""
Overseer Agent
==============
Sophisticated oversight agent that monitors debate quality and reasoning.

Responsibilities:
- Monitor agent reasoning quality
- Detect when agents go off-track
- Provide guidance to get back on track
- Act as tiebreaker when needed
- Ensure debate stays productive
"""

import logging
import json
from typing import List, Dict, Any, Optional
from src.debate_models import (
    Presentation,
    Reflection,
    Vote,
    OverseerIntervention,
    OverseerAssessment,
    VerdictType
)
from src.models import PackageData
from src.llm_clients import GoogleAIClient
from config.settings import settings

logger = logging.getLogger(__name__)


class OverseerAgent:
    """
    Overseer Agent using Gemini 2.0 Flash Thinking
    
    This agent monitors the debate and ensures quality reasoning.
    It only intervenes when absolutely necessary.
    """
    
    def __init__(self):
        """Initialize the overseer agent."""
        self.agent_name = "overseer"
        self.model_name = "gemini-3-flash-preview"  # Advanced reasoning model
        self.client = GoogleAIClient(api_key=settings.GOOGLE_AI_API_KEY)
        
        self.system_prompt = self._load_system_prompt()
        
        logger.info(f"Overseer Agent initialized with {self.model_name}")
    
    def _load_system_prompt(self) -> str:
        """Load overseer system prompt."""
        return """You are the Overseer of the LLM Council - a meta-reasoning agent that ensures debate quality.

**YOUR ROLE:**
You are NOT a participant in the debate. You are the guardian of reasoning quality.

**YOUR RESPONSIBILITIES:**
1. Monitor each agent's reasoning for logical fallacies, biases, and errors
2. Detect when agents go significantly off-track from evidence-based analysis
3. Intervene ONLY when reasoning quality seriously degrades
4. Provide clear, actionable guidance to get agents back on track
5. Cast tiebreaking vote ONLY when agents are deadlocked

**INTERVENTION CRITERIA:**
Only intervene when you see:
- Logical fallacies (circular reasoning, false dichotomy, etc.)
- Ignoring critical evidence
- Bias overriding objective analysis
- Fixation on irrelevant details
- Abandoning security analysis principles

**DO NOT INTERVENE FOR:**
- Minor disagreements (that's the point of debate)
- Different interpretations of ambiguous evidence
- Varying risk assessments based on different weights
- Normal argumentative discourse

**INTERVENTION FORMAT:**
When intervening, provide:
1. Specific issue identified
2. Why it matters
3. How to correct course
4. Not your opinion on the package itself

**TIEBREAKING:**
When casting a tiebreaking vote:
- Base decision purely on strongest evidence and reasoning
- Explain which arguments were most compelling
- Do not inject new analysis
- Simply evaluate what's been presented

**OUTPUT FORMAT:**
Always respond in valid JSON matching the expected schema.

Remember: You are a neutral facilitator of truth-seeking, not an additional analyst."""
    
    def assess_debate_quality(
        self,
        round_number: int,
        presentations: List[Presentation],
        package_data: PackageData
    ) -> OverseerAssessment:
        """
        Assess the quality of reasoning in agent presentations.
        
        Args:
            round_number: Current debate round
            presentations: All agent presentations this round
            package_data: The package being analyzed
            
        Returns:
            OverseerAssessment with quality scores and intervention needs
        """
        logger.info(f"[Overseer] Assessing debate quality for round {round_number}")
        
        try:
            prompt = self._build_assessment_prompt(presentations, package_data)
            
            response = self.client.generate_content(
                model=self.model_name,
                system_prompt=self.system_prompt,
                user_prompt=prompt,
                max_tokens=2000,
                temperature=0.2,
                response_format="json"
            )
            
            parsed = self._parse_response(response)
            
            assessment = OverseerAssessment(
                round_number=round_number,
                reasoning_quality=parsed.get("reasoning_quality", {}),
                intervention_needed=parsed.get("intervention_needed", False),
                agents_off_track=parsed.get("agents_off_track", []),
                debate_quality=parsed.get("debate_quality", "good"),
                convergence_assessment=parsed.get("convergence_assessment", ""),
                recommendations=parsed.get("recommendations", [])
            )
            
            logger.info(
                f"[Overseer] Assessment complete. "
                f"Quality: {assessment.debate_quality}, "
                f"Intervention needed: {assessment.intervention_needed}"
            )
            
            return assessment
            
        except Exception as e:
            logger.error(f"[Overseer] Assessment failed: {e}")
            # Return safe default
            return OverseerAssessment(
                round_number=round_number,
                reasoning_quality={p.agent_name: 75.0 for p in presentations},
                intervention_needed=False,
                debate_quality="good",
                convergence_assessment="Unable to assess due to error"
            )
    
    def _build_assessment_prompt(
        self,
        presentations: List[Presentation],
        package_data: PackageData
    ) -> str:
        """Build prompt for debate quality assessment."""
        
        prompt = f"""**DEBATE QUALITY ASSESSMENT TASK**

**Package Being Analyzed:**
- Name: {package_data.package_name}
- Version: {package_data.version}
- Description: {package_data.description}

**Agent Presentations to Assess:**

"""
        
        for pres in presentations:
            prompt += f"""
---
**Agent: {pres.agent_name}**
Position: {pres.position}
Risk Score: {pres.risk_score}
Confidence: {pres.confidence}

Main Thesis: {pres.main_thesis}

Key Evidence:
"""
            for evidence in pres.key_evidence:
                prompt += f"- {evidence}\n"
            
            prompt += f"\nReasoning Chain:\n{pres.reasoning_chain}\n"
        
        prompt += """

**YOUR TASK:**
Assess the reasoning quality of each agent. Provide a JSON response:

{
  "reasoning_quality": {
    "agent_name": <score 0-100>,
    ...
  },
  "intervention_needed": true/false,
  "agents_off_track": ["agent_name", ...],
  "debate_quality": "excellent" | "good" | "fair" | "poor",
  "convergence_assessment": "Description of whether agents are converging toward truth",
  "recommendations": ["Suggestion 1", "Suggestion 2", ...]
}

**Scoring Criteria:**
- 90-100: Excellent reasoning, strong evidence-based analysis
- 75-89: Good reasoning, minor issues
- 60-74: Fair reasoning, some concerning gaps
- 40-59: Poor reasoning, significant issues
- 0-39: Very poor reasoning, major intervention needed

Set intervention_needed=true ONLY if score < 50 or major logical fallacy detected."""
        
        return prompt
    
    def generate_intervention(
        self,
        round_number: int,
        target_agent: str,
        presentation: Presentation,
        package_data: PackageData
    ) -> OverseerIntervention:
        """
        Generate intervention guidance for an off-track agent.
        
        Args:
            round_number: Current round
            target_agent: Agent needing guidance
            presentation: The problematic presentation
            package_data: Package being analyzed
            
        Returns:
            OverseerIntervention with specific guidance
        """
        logger.info(f"[Overseer] Generating intervention for {target_agent}")
        
        try:
            prompt = f"""**INTERVENTION REQUIRED**

Agent **{target_agent}** appears to have gone off-track in their reasoning.

**Their Presentation:**
Position: {presentation.position}
Risk Score: {presentation.risk_score}
Confidence: {presentation.confidence}

Main Thesis: {presentation.main_thesis}

Reasoning: {presentation.reasoning_chain}

**Package Context:**
- Name: {package_data.package_name}
- Description: {package_data.description}

**YOUR TASK:**
Identify the specific reasoning error and provide guidance. Return JSON:

{{
  "issue_type": "Type of error (e.g., 'confirmation_bias', 'ignoring_evidence', 'logical_fallacy')",
  "issue_description": "Clear explanation of what went wrong",
  "guidance": "Specific steps to get back on track",
  "severity": "minor" | "moderate" | "severe",
  "problematic_reasoning": "Quote the specific problematic reasoning"
}}

Be constructive and specific. Don't tell them the answer, guide them to find it."""
            
            response = self.client.generate_content(
                model=self.model_name,
                system_prompt=self.system_prompt,
                user_prompt=prompt,
                max_tokens=1000,
                temperature=0.2,
                response_format="json"
            )
            
            parsed = self._parse_response(response)
            
            intervention = OverseerIntervention(
                round_number=round_number,
                target_agent=target_agent,
                issue_type=parsed.get("issue_type", "reasoning_error"),
                issue_description=parsed.get("issue_description", "Reasoning quality concerns"),
                guidance=parsed.get("guidance", "Review evidence objectively"),
                severity=parsed.get("severity", "moderate"),
                problematic_reasoning=parsed.get("problematic_reasoning", presentation.reasoning_chain[:200])
            )
            
            logger.info(f"[Overseer] Intervention generated: {intervention.issue_type} ({intervention.severity})")
            return intervention
            
        except Exception as e:
            logger.error(f"[Overseer] Intervention generation failed: {e}")
            # Return generic intervention
            return OverseerIntervention(
                round_number=round_number,
                target_agent=target_agent,
                issue_type="reasoning_quality",
                issue_description="Consider reviewing evidence more carefully",
                guidance="Focus on objective evidence rather than assumptions",
                severity="moderate",
                problematic_reasoning=""
            )
    
    def cast_tiebreaker_vote(
        self,
        round_number: int,
        votes: List[Vote],
        presentations: List[Presentation],
        package_data: PackageData
    ) -> Vote:
        """
        Cast tiebreaking vote when agents are deadlocked.
        
        Args:
            round_number: Current round
            votes: All agent votes (currently tied)
            presentations: All presentations this round
            package_data: Package being analyzed
            
        Returns:
            Overseer's tiebreaking vote
        """
        logger.info(f"[Overseer] Casting tiebreaker vote for round {round_number}")
        
        try:
            prompt = self._build_tiebreaker_prompt(votes, presentations, package_data)
            
            response = self.client.generate_content(
                model=self.model_name,
                system_prompt=self.system_prompt,
                user_prompt=prompt,
                max_tokens=1500,
                temperature=0.1,
                response_format="json"
            )
            
            parsed = self._parse_response(response)
            
            vote = Vote(
                agent_name="overseer",
                round_number=round_number,
                verdict=VerdictType(parsed.get("verdict", "uncertain")),
                risk_score=parsed.get("risk_score", 50),
                confidence=parsed.get("confidence", 50),
                vote_reasoning=parsed.get("vote_reasoning", ""),
                deciding_factors=parsed.get("deciding_factors", []),
                willing_to_change=False,  # Overseer vote is final
                certainty_level=parsed.get("certainty_level", "moderate")
            )
            
            logger.info(f"[Overseer] Tiebreaker cast: {vote.verdict} (risk={vote.risk_score})")
            return vote
            
        except Exception as e:
            logger.error(f"[Overseer] Tiebreaker vote failed: {e}")
            # Return uncertain vote as fallback
            return Vote(
                agent_name="overseer",
                round_number=round_number,
                verdict=VerdictType.UNCERTAIN,
                risk_score=50,
                confidence=30,
                vote_reasoning=f"Unable to break tie due to error: {e}",
                deciding_factors=[],
                willing_to_change=False,
                certainty_level="low"
            )
    
    def _build_tiebreaker_prompt(
        self,
        votes: List[Vote],
        presentations: List[Presentation],
        package_data: PackageData
    ) -> str:
        """Build prompt for tiebreaker vote."""
        
        prompt = f"""**TIEBREAKER VOTE REQUIRED**

The agents are deadlocked. You must cast the deciding vote.

**Package:**
- Name: {package_data.package_name}
- Version: {package_data.version}
- Description: {package_data.description}

**Current Votes:**
"""
        
        for vote in votes:
            prompt += f"- {vote.agent_name}: {vote.verdict} (risk={vote.risk_score}, confidence={vote.confidence})\n"
        
        prompt += "\n**Agent Presentations:**\n"
        
        for pres in presentations:
            prompt += f"""
**{pres.agent_name}:**
Position: {pres.position}
Thesis: {pres.main_thesis}
Key Evidence:
"""
            for evidence in pres.key_evidence[:3]:
                prompt += f"  - {evidence}\n"
        
        prompt += """

**YOUR TASK:**
Cast the tiebreaking vote. Base your decision on:
1. Strength of evidence presented
2. Quality of reasoning
3. Consistency with security analysis principles

Return JSON:
{
  "verdict": "malicious" | "benign" | "uncertain",
  "risk_score": <0-100>,
  "confidence": <0-100>,
  "vote_reasoning": "Complete explanation of your decision",
  "deciding_factors": ["Factor 1", "Factor 2", ...],
  "certainty_level": "absolute" | "high" | "moderate" | "low"
}

Remember: You're evaluating the ARGUMENTS, not conducting new analysis."""
        
        return prompt
    
    def _parse_response(self, response: Any) -> Dict[str, Any]:
        """Parse LLM response into structured format."""
        try:
            if isinstance(response, dict):
                return response
            elif isinstance(response, str):
                return json.loads(response)
            else:
                return {}
        except Exception as e:
            logger.warning(f"[Overseer] Failed to parse response: {e}")
            return {}
