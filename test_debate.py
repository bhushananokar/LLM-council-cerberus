import asyncio
from src.debate_orchestrator import get_debate_orchestrator
from src.models import PackageData
async def main():
    # Create a suspicious package
    package = PackageData(
        package_name="crypto-stealer",
        version="1.0.0",
        description="Cryptocurrency utility library",
        code_segments=[
            {
                "code": "eval(Buffer.from('c2VuZFRvU2VydmVy', 'base64'))",
                "location": "index.js:15",
                "reason": "eval_with_base64"
            }
        ],
        static_analysis={
            "dangerous_apis": ["eval", "exec"],
            "obfuscation_score": 85
        }
    )
    
    # Run the debate
    print("\n🎭 Starting debate...\n")
    orchestrator = get_debate_orchestrator()
    result = await orchestrator.run_debate(package)
    
    # Show results
    print(f"\n{'='*60}")
    print(f"FINAL DECISION")
    print(f"{'='*60}")
    print(f"Verdict:    {result.final_verdict}")
    print(f"Risk:       {result.final_risk_score:.1f}/100")
    print(f"Confidence: {result.final_confidence:.1f}%")
    print(f"Consensus:  {result.consensus_type}")
    print(f"Rounds:     {result.rounds_to_consensus}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    asyncio.run(main())