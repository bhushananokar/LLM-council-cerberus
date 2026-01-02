"""
LLM Council Agents
==================
Contains the three specialized AI agents for supply chain security analysis.

Agents:
- CodeIntelligenceAgent: Llama 3.3 70B via Groq (code analysis & deobfuscation)
- ThreatIntelligenceAgent: Qwen 2.5 72B via Groq (threat assessment & social engineering)
- BehavioralIntelligenceAgent: Gemini 2.0 Flash via Google AI (behavioral analysis & pattern matching)
"""

import json
import logging
from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod

from src.models import AgentResponse, PackageData
from src.llm_clients import GroqClient, GoogleAIClient
from config.settings import settings

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all LLM agents."""
    
    def __init__(self, agent_name: str, model_name: str, system_prompt: str):
        """
        Initialize base agent.
        
        Args:
            agent_name: Name of the agent (e.g., "code_intelligence")
            model_name: LLM model identifier
            system_prompt: System prompt defining agent's role and behavior
        """
        self.agent_name = agent_name
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_tokens = settings.MAX_OUTPUT_TOKENS
        self.temperature = settings.TEMPERATURE
        
        logger.info(f"Initialized {agent_name} with model {model_name}")
    
    @abstractmethod
    def analyze(self, package_data: PackageData) -> AgentResponse:
        """
        Analyze package data and return agent's assessment.
        
        Args:
            package_data: Package information to analyze
            
        Returns:
            AgentResponse with risk score, confidence, and explanation
        """
        pass
    
    def _build_user_prompt(self, package_data: PackageData) -> str:
        """
        Build user prompt from package data.
        
        Args:
            package_data: Package information
            
        Returns:
            Formatted prompt string
        """
        pass
    
    def _parse_response(self, response_text: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parse LLM response into structured format.
        
        Args:
            response_text: Raw response from LLM (string or already parsed dict)
            
        Returns:
            Parsed dictionary with risk_score, confidence, etc.
        """
        try:
            # If already a dict, return it
            if isinstance(response_text, dict):
                return response_text
            
            # Try to parse as JSON if string
            result = json.loads(response_text)
            return result
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"{self.agent_name}: Response not valid JSON, attempting extraction: {e}")
            # Fallback: extract key information from text
            if isinstance(response_text, str):
                return self._extract_from_text(response_text)
            else:
                return {"risk_score": 50, "confidence": 10, "verdict": "error", "explanation": "Invalid response format"}
    
    def _extract_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract structured data from plain text response.
        
        Args:
            text: Plain text response
            
        Returns:
            Dictionary with extracted values
        """
        # Basic extraction logic
        result = {
            "risk_score": 50,
            "confidence": 50,
            "explanation": text
        }
        
        # Try to find risk score
        if "risk" in text.lower() or "score" in text.lower():
            import re
            score_match = re.search(r'(?:risk|score)[:\s]+(\d+)', text, re.IGNORECASE)
            if score_match:
                result["risk_score"] = int(score_match.group(1))
        
        # Try to find confidence
        if "confidence" in text.lower():
            import re
            conf_match = re.search(r'confidence[:\s]+(\d+)', text, re.IGNORECASE)
            if conf_match:
                result["confidence"] = int(conf_match.group(1))
        
        return result


class CodeIntelligenceAgent(BaseAgent):
    """
    Agent 1: Code Intelligence (Llama 3.3 70B via Groq)
    
    Specialization:
    - Code intent analysis
    - Deobfuscation of encoded payloads
    - Semantic understanding
    - Dangerous API detection
    """
    
    def __init__(self, system_prompt: str):
        super().__init__(
            agent_name="code_intelligence",
            model_name=settings.AGENT1_MODEL,
            system_prompt=system_prompt
        )
        self.client = GroqClient(api_key=settings.GROQ_API_KEY)
    
    def analyze(self, package_data: PackageData) -> AgentResponse:
        """
        Analyze code for malicious intent and obfuscation.
        
        Args:
            package_data: Package information including code segments
            
        Returns:
            AgentResponse with code analysis results
        """
        logger.info(f"[{self.agent_name}] Analyzing package: {package_data.package_name}")
        
        try:
            # Build user prompt
            user_prompt = self._build_user_prompt(package_data)
            
            # Call Groq API
            response = self.client.chat_completion(
                model=self.model_name,
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format="json"
            )
            
            # Parse response
            parsed = self._parse_response(response)
            
            # Build AgentResponse
            agent_response = AgentResponse(
                agent_name=self.agent_name,
                model_name=self.model_name,
                risk_score=parsed.get("risk_score", 50),
                confidence=parsed.get("confidence", 50),
                verdict=parsed.get("verdict", "uncertain"),
                explanation=parsed.get("explanation", ""),
                details=parsed.get("details", {}),
                tokens_used=response.get("usage", {}).get("total_tokens", 0) if isinstance(response, dict) else 0
            )
            
            logger.info(f"[{self.agent_name}] Analysis complete: Risk={agent_response.risk_score}, Confidence={agent_response.confidence}")
            return agent_response
            
        except Exception as e:
            logger.error(f"[{self.agent_name}] Analysis failed: {str(e)}")
            # Return error response
            return AgentResponse(
                agent_name=self.agent_name,
                model_name=self.model_name,
                risk_score=50,
                confidence=0,
                verdict="error",
                explanation=f"Analysis failed: {str(e)}",
                details={"error": str(e)},
                tokens_used=0
            )
    
    def _build_user_prompt(self, package_data: PackageData) -> str:
        """Build code intelligence specific prompt."""
        
        prompt = f"""Analyze the following package for malicious code patterns:

**Package Information:**
- Name: {package_data.package_name}
- Version: {package_data.version}
- Claimed Purpose: {package_data.description}

**Code Segments to Analyze:**
"""
        
        # Add suspicious code segments
        if package_data.code_segments:
            for idx, segment in enumerate(package_data.code_segments, 1):
                prompt += f"\n**Segment {idx}:**\n```\n{segment.get('code', '')}\n```\n"
                prompt += f"Location: {segment.get('location', 'unknown')}\n"
                prompt += f"Reason flagged: {segment.get('reason', 'unknown')}\n"
        
        # Add static analysis results
        if package_data.static_analysis:
            prompt += f"\n**Static Analysis Results:**\n"
            prompt += f"- High entropy strings: {package_data.static_analysis.get('high_entropy_count', 0)}\n"
            prompt += f"- Dangerous APIs detected: {', '.join(package_data.static_analysis.get('dangerous_apis', []))}\n"
            prompt += f"- Obfuscation score: {package_data.static_analysis.get('obfuscation_score', 0)}/100\n"
        
        prompt += """

**Your Task:**
Analyze the code and provide your assessment in JSON format:

{
  "risk_score": <0-100>,
  "confidence": <0-100>,
  "verdict": "malicious" | "benign" | "uncertain",
  "explanation": "Detailed explanation of findings",
  "details": {
    "code_behavior": "What the code actually does",
    "intent_match": true/false,
    "deobfuscated_code": "If obfuscated, show deobfuscated version",
    "malicious_indicators": ["list", "of", "specific", "concerns"]
  }
}

Focus on:
1. Does the actual code behavior match the claimed package purpose?
2. Is there obfuscation or encoding hiding malicious intent?
3. Are dangerous APIs used inappropriately?
4. What would happen if this code executes?
"""
        
        return prompt


class ThreatIntelligenceAgent(BaseAgent):
    """
    Agent 2: Threat Intelligence (Qwen 2.5 72B via Groq)
    
    Specialization:
    - Social engineering detection
    - Threat assessment
    - Attack attribution
    - Typosquatting identification
    """
    
    def __init__(self, system_prompt: str):
        super().__init__(
            agent_name="threat_intelligence",
            model_name=settings.AGENT2_MODEL,
            system_prompt=system_prompt
        )
        self.client = GroqClient(api_key=settings.GROQ_API_KEY)
    
    def analyze(self, package_data: PackageData) -> AgentResponse:
        """
        Analyze threat level and social engineering tactics.
        
        Args:
            package_data: Package information including metadata
            
        Returns:
            AgentResponse with threat assessment
        """
        logger.info(f"[{self.agent_name}] Analyzing package: {package_data.package_name}")
        
        try:
            # Build user prompt
            user_prompt = self._build_user_prompt(package_data)
            
            # Call Groq API
            response = self.client.chat_completion(
                model=self.model_name,
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format="json"
            )
            
            # Parse response
            parsed = self._parse_response(response)
            
            # Build AgentResponse
            agent_response = AgentResponse(
                agent_name=self.agent_name,
                model_name=self.model_name,
                risk_score=parsed.get("risk_score", 50),
                confidence=parsed.get("confidence", 50),
                verdict=parsed.get("verdict", "uncertain"),
                explanation=parsed.get("explanation", ""),
                details=parsed.get("details", {}),
                tokens_used=response.get("usage", {}).get("total_tokens", 0) if isinstance(response, dict) else 0
            )
            
            logger.info(f"[{self.agent_name}] Analysis complete: Risk={agent_response.risk_score}, Confidence={agent_response.confidence}")
            return agent_response
            
        except Exception as e:
            logger.error(f"[{self.agent_name}] Analysis failed: {str(e)}")
            return AgentResponse(
                agent_name=self.agent_name,
                model_name=self.model_name,
                risk_score=50,
                confidence=0,
                verdict="error",
                explanation=f"Analysis failed: {str(e)}",
                details={"error": str(e)},
                tokens_used=0
            )
    
    def _build_user_prompt(self, package_data: PackageData) -> str:
        """Build threat intelligence specific prompt."""
        
        prompt = f"""Package: {package_data.package_name} v{package_data.version}
Description: {package_data.description}
Author: {package_data.author or 'unknown'}
"""
        
        # Add dependency analysis if present
        if package_data.dependency_analysis:
            if package_data.dependency_analysis.get('typosquatting'):
                typo = package_data.dependency_analysis['typosquatting']
                prompt += f"TYPOSQUATTING: Similar to '{typo.get('similar_to')}'\n"
        
        prompt += "\nProvide JSON assessment. Keep explanation brief."
        
        return prompt


class BehavioralIntelligenceAgent(BaseAgent):
    """
    Agent 3: Behavioral Intelligence (Gemini 2.0 Flash via Google AI)
    
    Specialization:
    - Runtime behavior analysis
    - Pattern recognition
    - Known malware matching
    - Attack scenario reconstruction
    """
    
    def __init__(self, system_prompt: str):
        super().__init__(
            agent_name="behavioral_intelligence",
            model_name=settings.AGENT3_MODEL,
            system_prompt=system_prompt
        )
        self.client = GoogleAIClient(api_key=settings.GOOGLE_AI_API_KEY)
    
    def analyze(self, package_data: PackageData) -> AgentResponse:
        """
        Analyze runtime behavior and match patterns.
        
        Args:
            package_data: Package information including behavioral data
            
        Returns:
            AgentResponse with behavioral analysis
        """
        logger.info(f"[{self.agent_name}] Analyzing package: {package_data.package_name}")
        
        try:
            # Build user prompt
            user_prompt = self._build_user_prompt(package_data)
            
            # Call Google AI API
            response = self.client.generate_content(
                model=self.model_name,
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format="json"
            )
            
            # Parse response
            parsed = self._parse_response(response)
            
            # Build AgentResponse
            agent_response = AgentResponse(
                agent_name=self.agent_name,
                model_name=self.model_name,
                risk_score=parsed.get("risk_score", 50),
                confidence=parsed.get("confidence", 50),
                verdict=parsed.get("verdict", "uncertain"),
                explanation=parsed.get("explanation", ""),
                details=parsed.get("details", {}),
                tokens_used=parsed.get("tokens_used", 0)  # Google AI might not provide this
            )
            
            logger.info(f"[{self.agent_name}] Analysis complete: Risk={agent_response.risk_score}, Confidence={agent_response.confidence}")
            return agent_response
            
        except Exception as e:
            logger.error(f"[{self.agent_name}] Analysis failed: {str(e)}")
            return AgentResponse(
                agent_name=self.agent_name,
                model_name=self.model_name,
                risk_score=50,
                confidence=0,
                verdict="error",
                explanation=f"Analysis failed: {str(e)}",
                details={"error": str(e)},
                tokens_used=0
            )
    
    def _build_user_prompt(self, package_data: PackageData) -> str:
        """Build behavioral intelligence specific prompt."""
        
        prompt = f"""Analyze the following package:

**Package Information:**
- Name: {package_data.package_name}
- Version: {package_data.version}
- Claimed Purpose: {package_data.description}
"""
        
        # Add code segments if available
        has_data = False
        if package_data.code_segments:
            has_data = True
            prompt += f"\n**Code Segments:**\n"
            for idx, segment in enumerate(package_data.code_segments, 1):
                prompt += f"\n**Segment {idx}:**\n```\n{segment.get('code', '')}\n```\n"
                prompt += f"Location: {segment.get('location', 'unknown')}\n"
        
        # Add behavioral analysis results
        if package_data.behavioral_analysis:
            has_data = True
            prompt += f"\n**Runtime Behavior Observed:**\n"
            
            behavior = package_data.behavioral_analysis
            
            if behavior.get('network_activity'):
                prompt += f"\n**Network Activity:**\n"
                for activity in behavior['network_activity'][:5]:  # Limit to first 5
                    prompt += f"- {activity.get('type')}: {activity.get('domain')} ({activity.get('method', 'GET')})\n"
            
            if behavior.get('file_operations'):
                prompt += f"\n**File Operations:**\n"
                for op in behavior['file_operations'][:5]:
                    prompt += f"- {op.get('operation')}: {op.get('path')}\n"
            
            if behavior.get('processes_spawned'):
                prompt += f"\n**Processes Spawned:**\n"
                for proc in behavior['processes_spawned'][:3]:
                    prompt += f"- Command: {proc.get('command')}\n"
            
            if behavior.get('environment_access'):
                prompt += f"\n**Environment Variables Accessed:**\n"
                prompt += f"- {', '.join(behavior['environment_access'][:5])}\n"
        
        if not has_data:
            prompt += f"\n**WARNING: No behavioral data or code segments provided. Analysis must be based solely on limited metadata.**\n"
        
        prompt += """

**Your Task:**
Analyze the behavioral patterns and provide assessment in JSON format:

{
  "risk_score": <0-100>,
  "confidence": <0-100>,
  "verdict": "malicious" | "benign" | "uncertain",
  "explanation": "Detailed behavioral analysis",
  "details": {
    "behavior_summary": "What the package does at runtime",
    "matches_claimed_purpose": true/false,
    "attack_scenario": ["step 1", "step 2", "step 3"],
    "pattern_matches": ["known_malware_family_1", "etc"],
    "novelty": "known_pattern | variant | novel_attack"
  }
}

Focus on:
1. Does runtime behavior match the claimed package functionality?
2. Are there data exfiltration patterns?
3. Does this match any known malware families?
4. What is the complete attack scenario?
"""
        
        return prompt


# Factory function to create agents
def create_agents(prompt_loader) -> tuple:
    """
    Create all three agents with loaded prompts.
    
    Args:
        prompt_loader: Function to load prompts from files
        
    Returns:
        Tuple of (CodeIntelligenceAgent, ThreatIntelligenceAgent, BehavioralIntelligenceAgent)
    """
    code_prompt = prompt_loader("agent1_code_intelligence.txt")
    threat_prompt = prompt_loader("agent2_threat_intelligence.txt")
    behavioral_prompt = prompt_loader("agent3_behavioral_intelligence.txt")
    
    agent1 = CodeIntelligenceAgent(system_prompt=code_prompt)
    agent2 = ThreatIntelligenceAgent(system_prompt=threat_prompt)
    agent3 = BehavioralIntelligenceAgent(system_prompt=behavioral_prompt)
    
    return agent1, agent2, agent3