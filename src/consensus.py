"""
Consensus Builder
=================
Combines multiple agent responses into a unified council decision.

Features:
- Variance-based agreement detection
- Weighted voting mechanism
- Multi-round debate with LLM arguments
- Confidence scoring
- Explanation synthesis
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from statistics import mean, stdev

from src.models import AgentResponse, ConsensusResult, CouncilDecision, PackageData
from config.settings import settings

logger = logging.getLogger(__name__)


class VarianceCalculator:
    """Calculate variance and agreement metrics between agent responses."""
    
    @staticmethod
    def calculate_variance(responses: List[AgentResponse]) -> float:
        """
        Calculate variance in risk scores.
        
        Args:
            responses: List of agent responses
            
        Returns:
            Variance value (max - min)
        """
        if not responses:
            return 0.0
        
        risk_scores = [r.risk_score for r in responses]
        variance = max(risk_scores) - min(risk_scores)
        
        logger.debug(f"Risk scores: {risk_scores}, Variance: {variance}")
        return variance
    
    @staticmethod
    def calculate_standard_deviation(responses: List[AgentResponse]) -> float:
        """
        Calculate standard deviation of risk scores.
        
        Args:
            responses: List of agent responses
            
        Returns:
            Standard deviation
        """
        if len(responses) < 2:
            return 0.0
        
        risk_scores = [r.risk_score for r in responses]
        return stdev(risk_scores)
    
    @staticmethod
    def check_agreement_level(variance: float) -> str:
        """
        Determine agreement level based on variance.
        
        Args:
            variance: Variance value
            
        Returns:
            Agreement level: "strong", "moderate", or "weak"
        """
        if variance < settings.VARIANCE_THRESHOLD_STRONG:
            return "strong"
        elif variance < settings.VARIANCE_THRESHOLD_MODERATE:
            return "moderate"
        else:
            return "weak"


class DebateHandler:
    """Handle agent disagreements through multi-round LLM debate."""
    
    def __init__(self):
        """Initialize debate handler."""
        self.min_debate_rounds = 9  # Minimum number of debate rounds
    
    @staticmethod
    def conduct_debate(responses: List[AgentResponse], agents: List, package_data: PackageData) -> Dict[str, Any]:
        """
        Conduct extended multi-round debate where agents argue with each other.
        
        Args:
            responses: List of initial agent responses
            agents: List of agent instances to call for debate
            package_data: Original package data
            
        Returns:
            Debate results with transcript and final resolution
        """
        logger.info("Conducting extended multi-round debate due to agent disagreement")
        
        # Create agent map and response map
        agent_map = {agent.agent_name: agent for agent in agents}
        response_map = {r.agent_name: r for r in responses}
        
        # Track current votes throughout debate
        # Safely handle None/empty verdicts
        current_votes = {
            r.agent_name: r.verdict[0].upper() if r.verdict else "U" 
            for r in responses
        }  # M, B, or U
        current_scores = {r.agent_name: r.risk_score for r in responses}
        
        debate_transcript = []
        round_num = 1
        
        # Build package context for debate
        package_context = f"""Package: {package_data.package_name} v{package_data.version}
Description: {package_data.description}"""
        
        if package_data.code_segments:
            package_context += f"\n\nCode found: {package_data.code_segments[0].get('code', '')}"
        
        # Initial positions
        for response in responses:
            debate_transcript.append({
                "round": round_num,
                "speaker": response.agent_name,
                "vote": response.verdict[0].upper() if response.verdict else "U",
                "reason": f"Initial assessment: {response.explanation}",
                "risk_score": response.risk_score
            })
        round_num += 1
        
        # Conduct debate rounds
        for debate_round in range(1, 10):  # 9 rounds of debate
            logger.info(f"=== DEBATE ROUND {debate_round} ===")
            
            # Rotate through agents for fairness
            speaker_order = [responses[i % len(responses)] for i in range(debate_round - 1, debate_round + len(responses) - 1)]
            
            for speaker_response in speaker_order[:len(agents)]:
                speaker_name = speaker_response.agent_name
                speaker_agent = agent_map.get(speaker_name)
                
                if not speaker_agent:
                    continue
                
                # Get other agents' current positions
                other_positions = []
                for other_name, other_vote in current_votes.items():
                    if other_name != speaker_name:
                        other_score = current_scores[other_name]
                        vote_full = {"M": "MALICIOUS", "B": "BENIGN", "U": "UNCERTAIN"}[other_vote]
                        other_positions.append(f"- {other_name}: {vote_full} (risk: {other_score}/100)")
                
                # Build debate prompt
                debate_prompt = f"""You are debating the security of this package with other experts:

{package_context}

YOUR INITIAL ASSESSMENT:
- Vote: {current_votes[speaker_name]} ({'MALICIOUS' if current_votes[speaker_name]=='M' else 'BENIGN' if current_votes[speaker_name]=='B' else 'UNCERTAIN'})
- Risk Score: {current_scores[speaker_name]}/100
- Reasoning: {response_map[speaker_name].explanation}

OTHER EXPERTS' CURRENT POSITIONS:
{chr(10).join(other_positions)}

RECENT DEBATE POINTS:
{chr(10).join([f"- {t['speaker']}: {t['reason'][:100]}" for t in debate_transcript[-6:] if t['speaker'] != speaker_name])}

This is debate round {debate_round}/9. Based on others' arguments, do you:
1. Maintain your position (provide stronger evidence)
2. Partially concede (adjust risk score)
3. Fully change your verdict

Respond ONLY with:
VOTE: M or B or U
RISK_SCORE: <number 0-100>
REASON: <1-2 sentences explaining your current position>"""
                
                try:
                    # Call LLM for debate response
                    if hasattr(speaker_agent.client, 'chat_completion'):
                        debate_response = speaker_agent.client.chat_completion(
                            model=speaker_agent.model_name,
                            system_prompt="You are a security expert in a professional debate. Be evidence-based and open to changing your view if presented with strong arguments.",
                            user_prompt=debate_prompt,
                            max_tokens=300,
                            temperature=0.4,
                            response_format="text"
                        )
                    else:
                        debate_response = speaker_agent.client.generate(
                            model=speaker_agent.model_name,
                            system_prompt="You are a security expert in a professional debate. Be evidence-based and open to changing your view if presented with strong arguments.",
                            user_prompt=debate_prompt,
                            max_tokens=300,
                            temperature=0.4,
                            response_format="text"
                        )
                    
                    response_text = debate_response if isinstance(debate_response, str) else str(debate_response)
                    
                    # Parse response
                    new_vote = DebateHandler._extract_vote(response_text, current_votes[speaker_name])
                    new_score = DebateHandler._extract_score(response_text, current_scores[speaker_name])
                    reason = DebateHandler._extract_reason(response_text)
                    
                    # Update current positions
                    current_votes[speaker_name] = new_vote
                    current_scores[speaker_name] = new_score
                    
                    debate_transcript.append({
                        "round": round_num,
                        "speaker": speaker_name,
                        "vote": new_vote,
                        "reason": reason,
                        "risk_score": new_score
                    })
                    
                    logger.info(f"Round {round_num} - {speaker_name}: {new_vote} (score: {new_score}) - {reason[:50] if reason else 'N/A'}")
                    round_num += 1
                    
                except Exception as e:
                    logger.error(f"Debate round failed for {speaker_name}: {e}")
                    debate_transcript.append({
                        "round": round_num,
                        "speaker": speaker_name,
                        "vote": current_votes[speaker_name],
                        "reason": f"Unable to respond: {e}",
                        "risk_score": current_scores[speaker_name]
                    })
                    round_num += 1
        
        # Final vote count
        final_malicious = sum(1 for v in current_votes.values() if v == "M")
        final_benign = sum(1 for v in current_votes.values() if v == "B")
        final_uncertain = sum(1 for v in current_votes.values() if v == "U")
        
        # Determine winner
        if final_malicious > final_benign and final_malicious > final_uncertain:
            winner = "malicious"
            confidence_boost = 10
        elif final_benign > final_malicious and final_benign > final_uncertain:
            winner = "benign"
            confidence_boost = -5
        else:
            winner = "uncertain"
            confidence_boost = -15
        
        # Count initial votes
        initial_malicious = sum(1 for r in responses if r.verdict == "malicious")
        initial_benign = sum(1 for r in responses if r.verdict == "benign")
        initial_uncertain = sum(1 for r in responses if r.verdict == "uncertain")
        
        debate_result = {
            "conducted": True,
            "rounds": len(debate_transcript),
            "transcript": debate_transcript,
            "final_positions": {
                name: {
                    "vote": {"M": "malicious", "B": "benign", "U": "uncertain"}[vote],
                    "risk_score": current_scores[name],
                    "initial_score": response_map[name].risk_score,
                    "changed_position": vote != (response_map[name].verdict[0].upper() if response_map[name].verdict else "U")
                }
                for name, vote in current_votes.items()
            },
            "vote_counts": {
                "initial": {"malicious": initial_malicious, "benign": initial_benign, "uncertain": initial_uncertain},
                "final": {"malicious": final_malicious, "benign": final_benign, "uncertain": final_uncertain}
            },
            "winner": winner,
            "confidence_adjustment": confidence_boost,
            "flag_for_human_review": final_uncertain >= 2
        }
        
        logger.info(f"Debate concluded after {len(debate_transcript)} exchanges")
        logger.info(f"Winner: {winner} | Initial votes (M:{initial_malicious},B:{initial_benign},U:{initial_uncertain}) -> Final (M:{final_malicious},B:{final_benign},U:{final_uncertain})")
        
        return debate_result
    
    @staticmethod
    def _extract_vote(text: str, default: str) -> str:
        """Extract vote (M/B/U) from debate response."""
        import re
        match = re.search(r'VOTE:\s*([MBU])', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        # Check for keywords
        text_upper = text.upper()
        if "MALICIOUS" in text_upper:
            return "M"
        if "BENIGN" in text_upper:
            return "B"
        return default
    
    @staticmethod
    def _extract_score(text: str, default: float) -> float:
        """Extract risk score from debate response."""
        import re
        match = re.search(r'RISK_SCORE:\s*(\d+)', text, re.IGNORECASE)
        if match:
            return float(match.group(1))
        match = re.search(r'SCORE:\s*(\d+)', text, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return default
    
    @staticmethod
    def _extract_reason(text: str) -> str:
        """Extract reason from debate response."""
        import re
        match = re.search(r'REASON:\s*(.+?)(?:\n|$)', text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()[:200]  # Limit length
        # Fallback: return first 150 chars or empty string
        return text[:150] if text else "No reason provided"
    
    @staticmethod
    def _extract_final_score(text: str, default: float) -> float:
        """Extract final risk score from debate response."""
        import re
        match = re.search(r'FINAL_RISK_SCORE:\s*(\d+)', text, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return default
    
    @staticmethod
    def _extract_final_verdict(text: str, default: str) -> str:
        """Extract final verdict from debate response."""
        import re
        match = re.search(r'FINAL_VERDICT:\s*(malicious|benign|uncertain)', text, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        return default


class ConsensusBuilder:
    """Build consensus from multiple agent responses."""
    
    def __init__(self):
        """Initialize consensus builder."""
        self.variance_calculator = VarianceCalculator()
        self.debate_handler = DebateHandler()
        
        # Agent weights for weighted average
        self.agent_weights = {
            "code_intelligence": settings.AGENT1_WEIGHT,
            "threat_intelligence": settings.AGENT2_WEIGHT,
            "behavioral_intelligence": settings.AGENT3_WEIGHT
        }
    
    def build_consensus(self, responses: List[AgentResponse], agents: List = None, package_data: PackageData = None) -> ConsensusResult:
        """
        Build consensus from agent responses.
        
        Args:
            responses: List of agent responses
            agents: List of agent instances (needed for debate)
            package_data: Original package data (needed for debate)
            
        Returns:
            ConsensusResult with final decision
        """
        logger.info(f"Building consensus from {len(responses)} agent responses")
        
        if not responses:
            logger.error("No agent responses provided")
            return self._create_error_consensus("No agent responses")
        
        # Calculate variance
        variance = self.variance_calculator.calculate_variance(responses)
        agreement_level = self.variance_calculator.check_agreement_level(variance)
        
        logger.info(f"Variance: {variance}, Agreement level: {agreement_level}")
        
        # Check if debate is needed
        debate_result = None
        confidence_adjustment = 0
        
        if variance >= settings.VARIANCE_THRESHOLD_MODERATE and settings.ENABLE_DEBATE:
            # Conduct debate if we have agents and package_data
            if agents and package_data:
                try:
                    debate_result = self.debate_handler.conduct_debate(responses, agents, package_data)
                    confidence_adjustment = debate_result.get("confidence_adjustment", 0)
                except Exception as e:
                    logger.error(f"Debate failed: {e}, falling back to simple voting")
                    debate_result = {"conducted": False, "error": str(e)}
            else:
                logger.warning("Debate needed but agents/package_data not provided, using simple voting")
        elif agreement_level == "strong":
            # Strong agreement, boost confidence
            confidence_adjustment = 10
        
        # Calculate weighted average risk score
        final_risk_score = self._calculate_weighted_score(responses)
        
        # Calculate average confidence
        avg_confidence = mean([r.confidence for r in responses])
        final_confidence = max(0, min(100, avg_confidence + confidence_adjustment))
        
        # Determine final verdict
        final_verdict = self._determine_verdict(responses, debate_result)
        
        # Determine threat level
        threat_level = self._determine_threat_level(final_risk_score)
        
        # Generate explanation
        explanation = self._generate_explanation(responses, variance, agreement_level, debate_result)
        
        # Calculate total tokens used
        total_tokens = sum(r.tokens_used for r in responses)
        
        # Create consensus result
        consensus = ConsensusResult(
            final_risk_score=round(final_risk_score, 2),
            final_confidence=round(final_confidence, 2),
            final_verdict=final_verdict,
            threat_level=threat_level,
            agreement_level=agreement_level,
            variance=variance,
            agent_scores={r.agent_name: r.risk_score for r in responses},
            agent_verdicts={r.agent_name: r.verdict for r in responses},
            debate_conducted=debate_result is not None,
            debate_result=debate_result,
            flag_for_review=debate_result.get("flag_for_human_review", False) if debate_result else False,
            explanation=explanation,
            total_tokens_used=total_tokens
        )
        
        logger.info(f"Consensus built: Risk={consensus.final_risk_score}, Verdict={consensus.final_verdict}, Confidence={consensus.final_confidence}")
        
        return consensus
    
    def _calculate_weighted_score(self, responses: List[AgentResponse]) -> float:
        """
        Calculate weighted average risk score.
        
        Args:
            responses: List of agent responses
            
        Returns:
            Weighted average score
        """
        weighted_sum = 0
        total_weight = 0
        
        for response in responses:
            weight = self.agent_weights.get(response.agent_name, 0.33)
            weighted_sum += response.risk_score * weight
            total_weight += weight
        
        if total_weight == 0:
            return mean([r.risk_score for r in responses])
        
        return weighted_sum / total_weight
    
    def _determine_verdict(self, responses: List[AgentResponse], debate_result: Optional[Dict]) -> str:
        """
        Determine final verdict.
        
        Args:
            responses: List of agent responses
            debate_result: Debate result if conducted
            
        Returns:
            Final verdict: "malicious", "benign", or "uncertain"
        """
        # If debate was conducted, use debate winner
        if debate_result:
            return debate_result.get("winner", "uncertain")
        
        # Otherwise, use majority voting
        verdicts = [r.verdict for r in responses]
        malicious_count = verdicts.count("malicious")
        benign_count = verdicts.count("benign")
        
        if malicious_count > benign_count:
            return "malicious"
        elif benign_count > malicious_count:
            return "benign"
        else:
            return "uncertain"
    
    def _determine_threat_level(self, risk_score: float) -> str:
        """
        Determine threat level based on risk score.
        
        Args:
            risk_score: Final risk score
            
        Returns:
            Threat level: "critical", "high", "medium", or "low"
        """
        if risk_score >= 90:
            return "critical"
        elif risk_score >= 70:
            return "high"
        elif risk_score >= 40:
            return "medium"
        else:
            return "low"
    
    def _generate_explanation(self, 
                             responses: List[AgentResponse],
                             variance: float,
                             agreement_level: str,
                             debate_result: Optional[Dict]) -> str:
        """
        Generate human-readable explanation of consensus.
        
        Args:
            responses: List of agent responses
            variance: Score variance
            agreement_level: Agreement level
            debate_result: Debate result if conducted
            
        Returns:
            Explanation string
        """
        explanation_parts = []
        
        # Agent scores summary
        scores_summary = ", ".join([
            f"{r.agent_name}: {r.risk_score}/100"
            for r in responses
        ])
        explanation_parts.append(f"Agent risk scores: {scores_summary}.")
        
        # Agreement level
        if agreement_level == "strong":
            explanation_parts.append(f"Strong agreement detected (variance: {variance:.1f}).")
        elif agreement_level == "moderate":
            explanation_parts.append(f"Moderate agreement (variance: {variance:.1f}).")
        else:
            explanation_parts.append(f"Significant disagreement detected (variance: {variance:.1f}).")
        
        # Debate summary if conducted
        if debate_result:
            winner = debate_result.get("winner")
            votes = debate_result.get("vote_counts", {})
            explanation_parts.append(
                f"Debate conducted: {votes.get('malicious', 0)} agents flagged malicious, "
                f"{votes.get('benign', 0)} flagged benign. "
                f"Consensus: {winner}."
            )
        
        # Key findings from each agent
        key_findings = []
        for response in responses:
            if response.explanation:
                # Extract first sentence or up to 100 chars
                summary = response.explanation.split('.')[0][:100]
                key_findings.append(f"{response.agent_name}: {summary}")
        
        if key_findings:
            explanation_parts.append("Key findings: " + "; ".join(key_findings) + ".")
        
        return " ".join(explanation_parts)
    
    def _create_error_consensus(self, error_message: str) -> ConsensusResult:
        """
        Create error consensus result.
        
        Args:
            error_message: Error description
            
        Returns:
            ConsensusResult with error state
        """
        return ConsensusResult(
            final_risk_score=50.0,
            final_confidence=0.0,
            final_verdict="error",
            threat_level="unknown",
            agreement_level="none",
            variance=0.0,
            agent_scores={},
            agent_verdicts={},
            debate_conducted=False,
            debate_result=None,
            flag_for_review=True,
            explanation=f"Error building consensus: {error_message}",
            total_tokens_used=0
        )
    
    def create_council_decision(self,
                               consensus: ConsensusResult,
                               package_name: str,
                               package_version: str,
                               agent_responses: List[AgentResponse]) -> CouncilDecision:
        """
        Create final council decision from consensus.
        
        Args:
            consensus: Consensus result
            package_name: Package name
            package_version: Package version
            agent_responses: Original agent responses
            
        Returns:
            CouncilDecision with complete analysis
        """
        # No recommendations/suggestions per user request
        recommended_actions = []
        
        # Create decision
        decision = CouncilDecision(
            package_name=package_name,
            package_version=package_version,
            final_risk_score=consensus.final_risk_score,
            final_confidence=consensus.final_confidence,
            verdict=consensus.final_verdict,
            threat_level=consensus.threat_level,
            consensus_result=consensus,
            agent_responses=agent_responses,
            recommended_actions=recommended_actions,
            requires_human_review=consensus.flag_for_review
        )
        
        logger.info(f"Council decision created for {package_name}@{package_version}: {decision.verdict} ({decision.threat_level})")
        
        return decision
    
    # Removed per user request - no suggestions/recommendations
    # def _generate_recommended_actions(self,
    #                                  verdict: str,
    #                                  threat_level: str,
    #                                  flag_for_review: bool) -> List[str]:
    #     """Generate recommended actions based on verdict and threat level."""
    #     return []


# Convenience functions
def build_consensus(responses: List[AgentResponse]) -> ConsensusResult:
    """
    Build consensus from agent responses.
    
    Args:
        responses: List of agent responses
        
    Returns:
        ConsensusResult
    """
    builder = ConsensusBuilder()
    return builder.build_consensus(responses)


def create_decision(consensus: ConsensusResult,
                   package_name: str,
                   package_version: str,
                   agent_responses: List[AgentResponse]) -> CouncilDecision:
    """
    Create council decision from consensus.
    
    Args:
        consensus: Consensus result
        package_name: Package name
        package_version: Package version
        agent_responses: Agent responses
        
    Returns:
        CouncilDecision
    """
    builder = ConsensusBuilder()
    return builder.create_council_decision(
        consensus,
        package_name,
        package_version,
        agent_responses
    )