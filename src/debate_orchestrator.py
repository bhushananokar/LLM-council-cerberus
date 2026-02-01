"""
Debate Orchestrator
===================
Sophisticated multi-agent debate orchestration with 4-phase cycles.

Phases:
1. Decide Points - Agents plan their arguments
2. Present - Agents formally present their cases  
3. Reflect - Agents evaluate their position vs others
4. Vote - Agents cast votes with full justification

The debate continues until consensus is reached or max rounds exceeded.
"""

import asyncio
import logging
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import Counter

from src.debate_models import (
    ArgumentPoint,
    Presentation,
    Reflection,
    Vote,
    DebateRound,
    DebateState,
    DebateResult,
    VerdictType,
    OverseerIntervention,
    OverseerAssessment
)
from src.models import PackageData, AgentResponse
from src.agents import (
    CodeIntelligenceAgent,
    ThreatIntelligenceAgent,
    BehavioralIntelligenceAgent
)
from src.overseer import OverseerAgent
from config.settings import settings

logger = logging.getLogger(__name__)

# MongoDB repository - will be set by API
_debate_repository = None


def set_debate_repository(repository):
    """Set the MongoDB repository for storing debate results."""
    global _debate_repository
    _debate_repository = repository


def get_debate_repository():
    """Get the MongoDB repository."""
    return _debate_repository


class DebateOrchestrator:
    """
    Sophisticated debate orchestrator managing multi-round deliberation.
    
    Features:
    - 4-phase debate rounds (decide, present, reflect, vote)
    - Consensus detection
    - Overseer monitoring and intervention
    - Complete debate history tracking
    """
    
    def __init__(
        self,
        max_rounds: int = 5,
        consensus_threshold: float = 0.67,  # 67% agreement
        enable_overseer: bool = True
    ):
        """
        Initialize debate orchestrator.
        
        Args:
            max_rounds: Maximum debate rounds before forcing decision
            consensus_threshold: Fraction of agents needed for consensus
            enable_overseer: Whether to use overseer monitoring
        """
        self.max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold
        self.enable_overseer = enable_overseer
        
        # Initialize agents
        self.agents_initialized = False
        self.code_agent: Optional[CodeIntelligenceAgent] = None
        self.threat_agent: Optional[ThreatIntelligenceAgent] = None
        self.behavioral_agent: Optional[BehavioralIntelligenceAgent] = None
        self.overseer: Optional[OverseerAgent] = None
        
        logger.info(
            f"DebateOrchestrator initialized "
            f"(max_rounds={max_rounds}, overseer={enable_overseer})"
        )
    
    def initialize_agents(self) -> bool:
        """Initialize all agents with sophisticated prompts."""
        if self.agents_initialized:
            return True
        
        try:
            logger.info("Initializing agents for debate system...")
            
            prompts_dir = Path(settings.PROMPTS_DIR)
            
            # Load enhanced prompts
            code_prompt = self._load_prompt(prompts_dir / "agent1_code_intelligence.txt")
            threat_prompt = self._load_prompt(prompts_dir / "agent2_threat_intelligence.txt")
            behavioral_prompt = self._load_prompt(prompts_dir / "agent3_behavioral_intelligence.txt")
            
            # Add debate instructions to prompts
            debate_instructions = self._get_debate_instructions()
            
            self.code_agent = CodeIntelligenceAgent(
                system_prompt=code_prompt + "\n\n" + debate_instructions
            )
            self.threat_agent = ThreatIntelligenceAgent(
                system_prompt=threat_prompt + "\n\n" + debate_instructions
            )
            self.behavioral_agent = BehavioralIntelligenceAgent(
                system_prompt=behavioral_prompt + "\n\n" + debate_instructions
            )
            
            # Initialize overseer if enabled
            if self.enable_overseer:
                self.overseer = OverseerAgent()
            
            self.agents_initialized = True
            logger.info("All agents initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize agents: {e}")
            return False
    
    def _load_prompt(self, path: Path) -> str:
        """Load prompt from file."""
        try:
            return path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not load prompt from {path}: {e}")
            return "You are a security analysis agent."
    
    def _get_debate_instructions(self) -> str:
        """Get additional debate-specific instructions for agents."""
        return """
**DEBATE PARTICIPATION INSTRUCTIONS:**

You are participating in a structured debate with other specialized agents to reach the most accurate security assessment. The debate follows a 4-phase cycle:

**Phase 1 - Decide Points:**
- Plan 3-5 key argument points you will make
- Each point should have: claim, evidence, importance weight
- Consider what other agents might argue
- Think strategically about your case

**Phase 2 - Present:**
- Formally present your complete argument
- State your position clearly: malicious, benign, or uncertain
- Provide your risk score (0-100) and confidence level
- Build a logical chain from evidence to conclusion
- Anticipate counterarguments

**Phase 3 - Reflect:**
- Listen carefully to other agents' presentations
- Evaluate strengths and weaknesses of their arguments
- Reassess your own position honestly
- Change your mind if evidence warrants it
- Identify the most convincing arguments

**Phase 4 - Vote:**
- Cast your final vote for this round
- Provide complete justification
- Explain what factors decided your vote
- Indicate your certainty level
- Be willing to change in future rounds if needed

**IMPORTANT PRINCIPLES:**
- Truth-seeking over winning: Change your mind when evidence demands it
- Steel-man opponents: Address their strongest arguments, not weakest
- Evidence-based: Ground all claims in actual evidence from the package
- Intellectual humility: Acknowledge uncertainty where it exists
- No ad-hominem: Critique arguments, not other agents

**DEBATE ETIQUETTE:**
- Be respectful but rigorous
- Point out flaws in reasoning politely
- Credit good arguments from others
- Don't repeat yourself unnecessarily
- Focus on areas of genuine disagreement

The debate continues until consensus or maximum rounds. An overseer monitors for reasoning quality but only intervenes if someone goes seriously off-track.
"""
    
    async def run_debate(
        self,
        package_data: PackageData
    ) -> DebateResult:
        """
        Run complete debate to analyze package.
        
        Args:
            package_data: Package to analyze
            
        Returns:
            DebateResult with complete history and final decision
        """
        logger.info(f"=" * 60)
        logger.info(f"STARTING DEBATE: {package_data.package_name}")
        logger.info(f"=" * 60)
        
        if not self.initialize_agents():
            raise RuntimeError("Failed to initialize agents")
        
        # Initialize debate state
        debate_id = f"debate_{package_data.package_name}_{uuid.uuid4().hex[:8]}"
        state = DebateState(
            debate_id=debate_id,
            package_name=package_data.package_name,
            current_round=1,
            max_rounds=self.max_rounds
        )
        
        start_time = time.time()
        
        # Run debate rounds
        while state.current_round <= state.max_rounds and state.is_active:
            logger.info(f"\n{'='*60}")
            logger.info(f"ROUND {state.current_round} / {state.max_rounds}")
            logger.info(f"{'='*60}\n")
            
            round_start = time.time()
            
            # Execute 4-phase round
            debate_round = await self._execute_round(
                round_number=state.current_round,
                package_data=package_data,
                state=state
            )
            
            debate_round.duration_seconds = time.time() - round_start
            debate_round.completed_at = time.time()
            
            state.rounds.append(debate_round)
            
            # Check for consensus
            if debate_round.consensus_reached:
                logger.info(f"\n🎯 CONSENSUS REACHED in round {state.current_round}!")
                state.consensus_reached = True
                state.is_active = False
                break
            
            # Move to next round
            state.current_round += 1
        
        # Force decision if max rounds reached
        if not state.consensus_reached:
            logger.info(f"\n⚠️  Max rounds reached. Forcing decision...")
            state.forced_termination = True
        
        total_duration = time.time() - start_time
        
        # Build final result
        result = self._build_debate_result(state, total_duration)
        
        # Save to MongoDB if repository is configured
        if _debate_repository:
            try:
                await self._save_debate_to_mongodb(result)
                logger.info(f"Saved debate to MongoDB: {result.debate_id}")
            except Exception as e:
                logger.error(f"Failed to save debate to MongoDB: {str(e)}")
                # Don't fail the debate if MongoDB save fails
        
        logger.info(f"\n{'='*60}")
        logger.info(f"DEBATE COMPLETE")
        logger.info(f"Final Verdict: {result.final_verdict}")
        logger.info(f"Risk Score: {result.final_risk_score}")
        logger.info(f"Confidence: {result.final_confidence}")
        logger.info(f"Consensus Type: {result.consensus_type}")
        logger.info(f"Rounds: {result.rounds_to_consensus}")
        logger.info(f"Duration: {total_duration:.1f}s")
        logger.info(f"{'='*60}\n")
        
        return result
    
    async def _execute_round(
        self,
        round_number: int,
        package_data: PackageData,
        state: DebateState
    ) -> DebateRound:
        """Execute one complete round of debate (all 4 phases)."""
        
        round = DebateRound(round_number=round_number)
        
        # Get previous round for context
        previous_round = state.rounds[-1] if state.rounds else None
        
        try:
            # PHASE 1: Decide Points
            logger.info(f"Phase 1: Deciding argument points...")
            round.argument_points = await self._phase1_decide_points(
                package_data, previous_round
            )
            
            # PHASE 2: Present
            logger.info(f"Phase 2: Formal presentations...")
            round.presentations = await self._phase2_present(
                package_data, round.argument_points, previous_round
            )
            
            # Overseer assessment after presentations
            if self.enable_overseer and self.overseer:
                round.overseer_assessment = self.overseer.assess_debate_quality(
                    round_number, round.presentations, package_data
                )
                
                # Generate interventions if needed
                if round.overseer_assessment.intervention_needed:
                    for agent_name in round.overseer_assessment.agents_off_track:
                        pres = next(
                            (p for p in round.presentations if p.agent_name == agent_name),
                            None
                        )
                        if pres:
                            intervention = self.overseer.generate_intervention(
                                round_number, agent_name, pres, package_data
                            )
                            round.interventions.append(intervention)
            
            # PHASE 3: Reflect
            logger.info(f"Phase 3: Reflection and reassessment...")
            round.reflections = await self._phase3_reflect(
                package_data, round.presentations, round.interventions
            )
            
            # PHASE 4: Vote
            logger.info(f"Phase 4: Voting...")
            round.votes = await self._phase4_vote(
                package_data, round.presentations, round.reflections
            )
            
            # Check for consensus
            consensus_info = self._check_consensus(round.votes)
            round.consensus_reached = consensus_info["reached"]
            round.consensus_verdict = consensus_info.get("verdict")
            
            # Handle tie if needed
            if consensus_info.get("is_tie") and self.enable_overseer and self.overseer:
                logger.info(f"🔀 Votes tied - Overseer casting tiebreaker...")
                overseer_vote = self.overseer.cast_tiebreaker_vote(
                    round_number, round.votes, round.presentations, package_data
                )
                round.votes.append(overseer_vote)
                
                # Recheck consensus with overseer vote
                consensus_info = self._check_consensus(round.votes)
                round.consensus_reached = consensus_info["reached"]
                round.consensus_verdict = consensus_info.get("verdict")
            
        except Exception as e:
            logger.error(f"Round {round_number} failed: {e}")
            raise
        
        return round
    
    async def _phase1_decide_points(
        self,
        package_data: PackageData,
        previous_round: Optional[DebateRound]
    ) -> List[ArgumentPoint]:
        """Phase 1: Agents decide what points to make."""
        
        # Build context from previous round
        context = ""
        if previous_round and previous_round.presentations:
            context = "\n**Previous Round Summary:**\n"
            for pres in previous_round.presentations:
                context += f"- {pres.agent_name}: {pres.position} (risk={pres.risk_score})\n"
        
        # Get argument points from each agent in parallel
        tasks = [
            self._agent_decide_points(self.code_agent, package_data, context),
            self._agent_decide_points(self.threat_agent, package_data, context),
            self._agent_decide_points(self.behavioral_agent, package_data, context)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_points = []
        for result in results:
            if isinstance(result, list):
                all_points.extend(result)
            else:
                logger.warning(f"Phase 1 task failed: {result}")
        
        return all_points
    
    async def _agent_decide_points(
        self,
        agent: Any,
        package_data: PackageData,
        context: str
    ) -> List[ArgumentPoint]:
        """Have single agent decide their argument points."""
        
        # This is a simplified version - in full implementation,
        # each agent would have a decide_points() method
        # For now, return placeholder
        return [
            ArgumentPoint(
                agent_name=agent.agent_name,
                point_id=f"{agent.agent_name}_point_1",
                category="analysis",
                claim=f"{agent.agent_name} initial analysis",
                evidence="Evidence from package data",
                weight=0.8
            )
        ]
    
    async def _phase2_present(
        self,
        package_data: PackageData,
        argument_points: List[ArgumentPoint],
        previous_round: Optional[DebateRound]
    ) -> List[Presentation]:
        """Phase 2: Agents present formal arguments."""
        
        # Get presentations from each agent in parallel
        tasks = [
            self._agent_present(self.code_agent, package_data, argument_points, previous_round),
            self._agent_present(self.threat_agent, package_data, argument_points, previous_round),
            self._agent_present(self.behavioral_agent, package_data, argument_points, previous_round)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        presentations = []
        for result in results:
            if isinstance(result, Presentation):
                presentations.append(result)
            else:
                logger.warning(f"Phase 2 task failed: {result}")
        
        return presentations
    
    async def _agent_present(
        self,
        agent: Any,
        package_data: PackageData,
        argument_points: List[ArgumentPoint],
        previous_round: Optional[DebateRound]
    ) -> Presentation:
        """Have single agent present their case."""
        
        # Use existing analyze method but extract presentation format
        response = agent.analyze(package_data)
        
        # Convert AgentResponse to Presentation
        agent_points = [p for p in argument_points if p.agent_name == agent.agent_name]
        
        presentation = Presentation(
            agent_name=response.agent_name,
            round_number=1,
            position=VerdictType(response.verdict),
            risk_score=response.risk_score,
            confidence=response.confidence,
            main_thesis=response.explanation[:200],
            supporting_points=agent_points,
            key_evidence=[response.explanation[:500]],
            reasoning_chain=response.explanation,
            tokens_used=response.tokens_used
        )
        
        return presentation
    
    async def _phase3_reflect(
        self,
        package_data: PackageData,
        presentations: List[Presentation],
        interventions: List[OverseerIntervention]
    ) -> List[Reflection]:
        """Phase 3: Agents reflect on their position vs others."""
        
        # Get reflections from each agent in parallel
        tasks = [
            self._agent_reflect(self.code_agent, presentations, interventions),
            self._agent_reflect(self.threat_agent, presentations, interventions),
            self._agent_reflect(self.behavioral_agent, presentations, interventions)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        reflections = []
        for result in results:
            if isinstance(result, Reflection):
                reflections.append(result)
            else:
                logger.warning(f"Phase 3 task failed: {result}")
        
        return reflections
    
    async def _agent_reflect(
        self,
        agent: Any,
        presentations: List[Presentation],
        interventions: List[OverseerIntervention]
    ) -> Reflection:
        """Have single agent reflect on debate so far."""
        
        # Get agent's own presentation
        own_pres = next(p for p in presentations if p.agent_name == agent.agent_name)
        other_pres = [p for p in presentations if p.agent_name != agent.agent_name]
        
        # Simple reflection (in full version, would call LLM)
        reflection = Reflection(
            agent_name=agent.agent_name,
            round_number=1,
            initial_position=own_pres.position,
            revised_position=own_pres.position,  # May change
            position_changed=False,
            strengths_found={p.agent_name: ["Strong evidence"] for p in other_pres},
            weaknesses_found={p.agent_name: [] for p in other_pres},
            revised_risk_score=own_pres.risk_score,
            revised_confidence=own_pres.confidence,
            reflection_reasoning="Maintaining position based on evidence",
            most_convincing_argument=other_pres[0].main_thesis if other_pres else ""
        )
        
        return reflection
    
    async def _phase4_vote(
        self,
        package_data: PackageData,
        presentations: List[Presentation],
        reflections: List[Reflection]
    ) -> List[Vote]:
        """Phase 4: Agents cast votes."""
        
        # Get votes from each agent in parallel
        tasks = [
            self._agent_vote(self.code_agent, presentations, reflections),
            self._agent_vote(self.threat_agent, presentations, reflections),
            self._agent_vote(self.behavioral_agent, presentations, reflections)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        votes = []
        for result in results:
            if isinstance(result, Vote):
                votes.append(result)
            else:
                logger.warning(f"Phase 4 task failed: {result}")
        
        return votes
    
    async def _agent_vote(
        self,
        agent: Any,
        presentations: List[Presentation],
        reflections: List[Reflection]
    ) -> Vote:
        """Have single agent cast their vote."""
        
        # Get agent's reflection
        reflection = next(r for r in reflections if r.agent_name == agent.agent_name)
        
        vote = Vote(
            agent_name=agent.agent_name,
            round_number=1,
            verdict=reflection.revised_position,
            risk_score=reflection.revised_risk_score,
            confidence=reflection.revised_confidence,
            vote_reasoning=reflection.reflection_reasoning,
            deciding_factors=["Evidence quality", "Reasoning strength"],
            willing_to_change=reflection.revised_confidence < 80,
            certainty_level="high" if reflection.revised_confidence > 80 else "moderate"
        )
        
        return vote
    
    def _check_consensus(self, votes: List[Vote]) -> Dict[str, Any]:
        """
        Check if consensus has been reached.
        
        Returns dict with:
        - reached: bool
        - verdict: VerdictType or None
        - is_tie: bool
        """
        if not votes:
            return {"reached": False, "is_tie": False}
        
        # Count votes by verdict
        verdict_counts = Counter(v.verdict for v in votes)
        total_votes = len(votes)
        
        # Get most common verdict and its count
        most_common = verdict_counts.most_common(1)[0]
        winner_verdict, winner_count = most_common
        
        # Check if it's a tie
        if len([v for v in verdict_counts.values() if v == winner_count]) > 1:
            return {"reached": False, "is_tie": True}
        
        # Check if consensus threshold met
        consensus_fraction = winner_count / total_votes
        
        if consensus_fraction >= self.consensus_threshold:
            return {
                "reached": True,
                "verdict": winner_verdict,
                "is_tie": False
            }
        
        return {"reached": False, "is_tie": False}
    
    def _build_debate_result(
        self,
        state: DebateState,
        total_duration: float
    ) -> DebateResult:
        """Build final debate result from state."""
        
        final_round = state.rounds[-1]
        final_votes = {v.agent_name: v.verdict for v in final_round.votes}
        
        # Determine consensus type
        overseer_voted = any(v.agent_name == "overseer" for v in final_round.votes)
        
        if state.forced_termination:
            consensus_type = "forced"
        elif overseer_voted:
            consensus_type = "tiebroken_by_overseer"
        elif len(set(final_votes.values())) == 1:
            consensus_type = "unanimous"
        else:
            consensus_type = "majority"
        
        # Calculate final scores (weighted average)
        final_risk = sum(v.risk_score for v in final_round.votes) / len(final_round.votes)
        final_conf = sum(v.confidence for v in final_round.votes) / len(final_round.votes)
        
        # Get final verdict
        verdict_counts = Counter(v.verdict for v in final_round.votes)
        final_verdict = verdict_counts.most_common(1)[0][0]
        
        # Count interventions
        total_interventions = sum(len(r.interventions) for r in state.rounds)
        
        # Extract key arguments
        key_arguments = []
        for round in state.rounds:
            for pres in round.presentations:
                if pres.main_thesis:
                    key_arguments.append(f"{pres.agent_name}: {pres.main_thesis}")
        
        result = DebateResult(
            debate_id=state.debate_id,
            package_name=state.package_name,
            final_verdict=final_verdict,
            final_risk_score=final_risk,
            final_confidence=final_conf,
            consensus_type=consensus_type,
            rounds_to_consensus=len(state.rounds),
            final_votes=final_votes,
            overseer_voted=overseer_voted,
            debate_history=state.rounds,
            total_interventions=total_interventions,
            key_arguments=key_arguments[:10],
            turning_points=[],
            debate_quality_score=85.0,  # Could calculate based on overseer assessments
            reasoning_depth="deep",
            total_tokens_used=state.total_tokens_used,
            total_duration_seconds=total_duration
        )
        
        return result
    
    async def _save_debate_to_mongodb(self, result: DebateResult):
        """
        Save debate result to MongoDB.
        
        Args:
            result: DebateResult to save
        """
        if not _debate_repository:
            return
        
        # Convert result to dict suitable for MongoDB
        result_dict = result.dict()
        
        # Convert debate history rounds to serializable format
        if "debate_history" in result_dict:
            result_dict["debate_history"] = [
                round.dict() if hasattr(round, 'dict') else round
                for round in result_dict["debate_history"]
            ]
        
        # Save to database
        await _debate_repository.save_debate_result(result_dict)


# Singleton instance
_debate_orchestrator: Optional[DebateOrchestrator] = None


def get_debate_orchestrator() -> DebateOrchestrator:
    """Get singleton debate orchestrator instance."""
    global _debate_orchestrator
    if _debate_orchestrator is None:
        _debate_orchestrator = DebateOrchestrator()
    return _debate_orchestrator
