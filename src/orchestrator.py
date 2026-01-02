"""
Council Orchestrator
====================
Main orchestrator coordinating the LLM Council workflow.

Responsibilities:
- Load system prompts
- Coordinate agent execution (parallel)
- Manage workflow state
- Handle caching
- Build consensus
- Track costs and performance
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.models import (
    PackageData,
    AgentResponse,
    ConsensusResult,
    CouncilDecision,
    AnalysisRequest,
    AnalysisResponse
)
from src.agents import (
    CodeIntelligenceAgent,
    ThreatIntelligenceAgent,
    BehavioralIntelligenceAgent
)
from src.consensus import ConsensusBuilder
from src.cache import get_cache_service
from src.utils import generate_decision_id, calculate_estimated_cost
from config.settings import settings

logger = logging.getLogger(__name__)


class CouncilOrchestrator:
    """Main orchestrator for the LLM Council."""
    
    def __init__(self):
        """Initialize the council orchestrator."""
        self.agents_initialized = False
        self.agent1: Optional[CodeIntelligenceAgent] = None
        self.agent2: Optional[ThreatIntelligenceAgent] = None
        self.agent3: Optional[BehavioralIntelligenceAgent] = None
        self.consensus_builder = ConsensusBuilder()
        self.cache = get_cache_service()
        
        # Statistics
        self.stats = {
            "total_analyses": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_tokens_used": 0,
            "total_cost_usd": 0.0,
            "avg_analysis_time": 0.0
        }
        
        logger.info("Council Orchestrator initialized")
    
    def initialize_agents(self) -> bool:
        """
        Initialize all three agents with system prompts.
        
        Returns:
            True if successful
        """
        if self.agents_initialized:
            logger.info("Agents already initialized")
            return True
        
        try:
            logger.info("Initializing agents with system prompts...")
            
            # Load system prompts
            prompts_dir = Path(settings.PROMPTS_DIR)
            
            agent1_prompt = self._load_prompt(prompts_dir / "agent1_code_intelligence.txt")
            agent2_prompt = self._load_prompt(prompts_dir / "agent2_threat_intelligence.txt")
            agent3_prompt = self._load_prompt(prompts_dir / "agent3_behavioral_intelligence.txt")
            
            # Create agents
            self.agent1 = CodeIntelligenceAgent(system_prompt=agent1_prompt)
            self.agent2 = ThreatIntelligenceAgent(system_prompt=agent2_prompt)
            self.agent3 = BehavioralIntelligenceAgent(system_prompt=agent3_prompt)
            
            self.agents_initialized = True
            logger.info("All agents initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize agents: {str(e)}")
            self.agents_initialized = False
            return False
    
    def _load_prompt(self, prompt_path: Path) -> str:
        """
        Load system prompt from file.
        
        Args:
            prompt_path: Path to prompt file
            
        Returns:
            Prompt text
            
        Raises:
            FileNotFoundError: If prompt file doesn't exist
        """
        if not prompt_path.exists():
            # Try to create default prompt
            logger.warning(f"Prompt file not found: {prompt_path}. Using default prompt.")
            return self._get_default_prompt(prompt_path.stem)
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
        
        logger.debug(f"Loaded prompt from {prompt_path} ({len(prompt)} chars)")
        return prompt
    
    def _get_default_prompt(self, agent_name: str) -> str:
        """
        Get default prompt for agent if file doesn't exist.
        
        Args:
            agent_name: Agent identifier
            
        Returns:
            Default prompt string
        """
        defaults = {
            "agent1_code_intelligence": """You are an expert security analyst specializing in code analysis and malware detection.

Your role is to:
1. Analyze code for malicious intent and behavior
2. Deobfuscate encoded or hidden payloads
3. Identify semantic mismatches between claimed purpose and actual behavior
4. Detect dangerous API usage patterns (eval, exec, network calls, file operations)
5. Assess whether code behavior matches the package's stated purpose

Provide your assessment in JSON format with:
- risk_score (0-100)
- confidence (0-100)
- verdict ("malicious", "benign", or "uncertain")
- explanation (detailed findings)
- details (specific technical findings)

Be precise, thorough, and security-focused.""",
            
            "agent2_threat_intelligence": """You are a cybersecurity threat intelligence expert specializing in supply chain attacks.

Your role is to:
1. Detect social engineering tactics in package descriptions and metadata
2. Identify typosquatting and impersonation attempts
3. Assess overall threat level and attacker sophistication
4. Attempt attribution to known threat actors or campaigns
5. Evaluate psychological manipulation techniques

Provide your assessment in JSON format with:
- risk_score (0-100)
- confidence (0-100)
- verdict ("malicious", "benign", or "uncertain")
- explanation (threat assessment)
- details (social engineering tactics, attribution, sophistication)

Consider the broader threat landscape and attack patterns.""",
            
            "agent3_behavioral_intelligence": """You are a malware behavioral analysis specialist.

Your role is to:
1. Analyze runtime behavior patterns from sandbox execution
2. Match behaviors against known malware families and attack patterns
3. Reconstruct complete attack scenarios step-by-step
4. Identify novel or unusual attack techniques
5. Assess whether runtime behavior matches claimed functionality

Provide your assessment in JSON format with:
- risk_score (0-100)
- confidence (0-100)
- verdict ("malicious", "benign", or "uncertain")
- explanation (behavioral analysis)
- details (attack scenario, pattern matches, novelty assessment)

Focus on what the code actually does when executed."""
        }
        
        return defaults.get(agent_name, "You are a security analysis assistant.")
    
    async def analyze_package(self, 
                             package_data: PackageData,
                             force_reanalysis: bool = False,
                             skip_cache: bool = False) -> CouncilDecision:
        """
        Analyze package using the LLM Council.
        
        Args:
            package_data: Package information to analyze
            force_reanalysis: Force reanalysis even if cached
            skip_cache: Skip cache entirely
            
        Returns:
            CouncilDecision with complete analysis
        """
        start_time = time.time()
        
        logger.info(f"Starting analysis for {package_data.package_name}@{package_data.version}")
        
        # Initialize agents if needed
        if not self.agents_initialized:
            success = self.initialize_agents()
            if not success:
                raise RuntimeError("Failed to initialize agents")
        
        # Check cache (unless skipped or forced)
        if not skip_cache and not force_reanalysis:
            cached_decision = self._check_cache(package_data)
            if cached_decision:
                logger.info(f"Cache HIT for {package_data.package_name}")
                self.stats["cache_hits"] += 1
                return cached_decision
            else:
                logger.info(f"Cache MISS for {package_data.package_name}")
                self.stats["cache_misses"] += 1
        
        # Run agent analysis (parallel)
        agent_responses = await self._run_agents_parallel(package_data)
        
        # Build consensus (pass agents and package_data for debate)
        consensus = self.consensus_builder.build_consensus(
            agent_responses, 
            agents=[self.agent1, self.agent2, self.agent3],
            package_data=package_data
        )
        
        # Create council decision
        decision = self.consensus_builder.create_council_decision(
            consensus=consensus,
            package_name=package_data.package_name,
            package_version=package_data.version,
            agent_responses=agent_responses
        )
        
        # Add metadata
        decision.decision_id = generate_decision_id()
        decision.analysis_duration_seconds = round(time.time() - start_time, 2)
        decision.total_tokens_used = consensus.total_tokens_used
        decision.estimated_cost_usd = calculate_estimated_cost(
            consensus.total_tokens_used,
            [r.model_name for r in agent_responses]
        )
        decision.registry = package_data.registry
        
        # Cache result (unless skipped)
        if not skip_cache:
            self._cache_decision(package_data, decision)
        
        # Update statistics
        self._update_stats(decision)
        
        logger.info(
            f"Analysis complete for {package_data.package_name}: "
            f"Verdict={decision.verdict}, Risk={decision.final_risk_score}, "
            f"Time={decision.analysis_duration_seconds}s, Cost=${decision.estimated_cost_usd:.4f}"
        )
        
        return decision
    
    def analyze_package_sync(self,
                            package_data: PackageData,
                            force_reanalysis: bool = False,
                            skip_cache: bool = False) -> CouncilDecision:
        """
        Synchronous wrapper for analyze_package.
        
        Args:
            package_data: Package data
            force_reanalysis: Force reanalysis
            skip_cache: Skip cache
            
        Returns:
            CouncilDecision
        """
        return asyncio.run(self.analyze_package(package_data, force_reanalysis, skip_cache))
    
    async def _run_agents_parallel(self, package_data: PackageData) -> List[AgentResponse]:
        """
        Run all three agents in parallel.
        
        Args:
            package_data: Package data to analyze
            
        Returns:
            List of agent responses
        """
        logger.info("Running agents in parallel...")
        
        # Create tasks for parallel execution
        tasks = [
            asyncio.to_thread(self.agent1.analyze, package_data),
            asyncio.to_thread(self.agent2.analyze, package_data),
            asyncio.to_thread(self.agent3.analyze, package_data)
        ]
        
        # Execute in parallel
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any errors
        agent_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                agent_name = ["code_intelligence", "threat_intelligence", "behavioral_intelligence"][i]
                logger.error(f"Agent {agent_name} failed: {str(response)}")
                # Create error response
                agent_responses.append(AgentResponse(
                    agent_name=agent_name,
                    model_name="error",
                    risk_score=50,
                    confidence=0,
                    verdict="error",
                    explanation=f"Agent execution failed: {str(response)}",
                    details={"error": str(response)},
                    tokens_used=0
                ))
            else:
                agent_responses.append(response)
        
        logger.info(f"All agents completed: {len(agent_responses)} responses")
        return agent_responses
    
    def _check_cache(self, package_data: PackageData) -> Optional[CouncilDecision]:
        """
        Check cache for existing decision.
        
        Args:
            package_data: Package data
            
        Returns:
            Cached decision or None
        """
        try:
            cached = self.cache.get_council_decision(package_data.dict())
            if cached:
                # Reconstruct CouncilDecision from cached dict
                return CouncilDecision(**cached)
            return None
        except Exception as e:
            logger.warning(f"Cache lookup failed: {str(e)}")
            return None
    
    def _cache_decision(self, package_data: PackageData, decision: CouncilDecision):
        """
        Cache council decision.
        
        Args:
            package_data: Package data
            decision: Decision to cache
        """
        try:
            self.cache.set_council_decision(
                package_data.dict(),
                decision.dict(),
                ttl=settings.CACHE_TTL_SECONDS
            )
            logger.debug(f"Cached decision for {package_data.package_name}")
        except Exception as e:
            logger.warning(f"Cache storage failed: {str(e)}")
    
    def _update_stats(self, decision: CouncilDecision):
        """
        Update orchestrator statistics.
        
        Args:
            decision: Completed decision
        """
        self.stats["total_analyses"] += 1
        self.stats["total_tokens_used"] += decision.total_tokens_used or 0
        self.stats["total_cost_usd"] += decision.estimated_cost_usd or 0.0
        
        # Update average analysis time
        prev_avg = self.stats["avg_analysis_time"]
        n = self.stats["total_analyses"]
        self.stats["avg_analysis_time"] = (
            (prev_avg * (n - 1) + decision.analysis_duration_seconds) / n
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get orchestrator statistics.
        
        Returns:
            Dictionary with statistics
        """
        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        cache_hit_rate = (
            (self.stats["cache_hits"] / total_requests * 100)
            if total_requests > 0 else 0.0
        )
        
        return {
            **self.stats,
            "cache_hit_rate": round(cache_hit_rate, 2),
            "avg_cost_per_analysis": round(
                self.stats["total_cost_usd"] / self.stats["total_analyses"]
                if self.stats["total_analyses"] > 0 else 0.0,
                4
            )
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all components.
        
        Returns:
            Dictionary with health status
        """
        health = {
            "status": "healthy",
            "components": {}
        }
        
        # Check agents initialized
        health["components"]["agents_initialized"] = self.agents_initialized
        
        # Check cache
        health["components"]["cache"] = self.cache.health_check()
        
        # Check LLM APIs (would require test calls - skip for now)
        health["components"]["groq_api"] = True  # Assume healthy
        health["components"]["google_ai_api"] = True  # Assume healthy
        
        # Determine overall status
        if not all(health["components"].values()):
            health["status"] = "degraded"
        
        # Add stats
        health["stats"] = self.get_stats()
        
        return health
    
    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = {
            "total_analyses": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_tokens_used": 0,
            "total_cost_usd": 0.0,
            "avg_analysis_time": 0.0
        }
        logger.info("Statistics reset")


# Singleton instance
_orchestrator_instance: Optional[CouncilOrchestrator] = None


def get_orchestrator() -> CouncilOrchestrator:
    """
    Get singleton orchestrator instance.
    
    Returns:
        CouncilOrchestrator instance
    """
    global _orchestrator_instance
    
    if _orchestrator_instance is None:
        _orchestrator_instance = CouncilOrchestrator()
        _orchestrator_instance.initialize_agents()
    
    return _orchestrator_instance


# Convenience functions
async def analyze_package(package_data: PackageData, **kwargs) -> CouncilDecision:
    """
    Analyze package using the council.
    
    Args:
        package_data: Package data
        **kwargs: Additional options
        
    Returns:
        CouncilDecision
    """
    orchestrator = get_orchestrator()
    return await orchestrator.analyze_package(package_data, **kwargs)


def analyze_package_sync(package_data: PackageData, **kwargs) -> CouncilDecision:
    """
    Synchronous package analysis.
    
    Args:
        package_data: Package data
        **kwargs: Additional options
        
    Returns:
        CouncilDecision
    """
    orchestrator = get_orchestrator()
    return orchestrator.analyze_package_sync(package_data, **kwargs)


def get_council_stats() -> Dict[str, Any]:
    """Get council statistics."""
    orchestrator = get_orchestrator()
    return orchestrator.get_stats()


def health_check() -> Dict[str, Any]:
    """Perform health check."""
    orchestrator = get_orchestrator()
    return orchestrator.health_check()