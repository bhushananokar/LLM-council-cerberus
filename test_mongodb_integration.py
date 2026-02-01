#!/usr/bin/env python3
"""
MongoDB Setup and Test Script for LLM Council

This script helps verify and test the MongoDB integration.
"""

import asyncio
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Add mongo_crud to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mongo_crud'))


async def test_mongodb_connection():
    """Test MongoDB connection and basic operations."""
    print("=" * 60)
    print("LLM COUNCIL - MONGODB INTEGRATION TEST")
    print("=" * 60)
    print()
    
    # Check environment variables
    print("1. Checking environment configuration...")
    mongodb_uri = os.getenv("MONGODB_URI")
    database_name = os.getenv("DATABASE_NAME", "llm_council")
    
    if not mongodb_uri:
        print("   ❌ MONGODB_URI not set in environment")
        print("   💡 Set it in your .env file or environment:")
        print("      export MONGODB_URI=mongodb://localhost:27017")
        return False
    
    print(f"   ✅ MONGODB_URI: {mongodb_uri}")
    print(f"   ✅ DATABASE_NAME: {database_name}")
    print()
    
    # Test connection
    print("2. Testing MongoDB connection...")
    try:
        from mongo_crud.app.database.mongodb import MongoDB
        
        mongodb = MongoDB()
        await mongodb.connect()
        
        print("   ✅ Successfully connected to MongoDB")
        print()
        
        # Test database operations
        print("3. Testing database operations...")
        
        # Create test collection
        test_collection = mongodb.db["test_council"]
        
        # Insert test document
        test_doc = {
            "test_id": "test_001",
            "message": "MongoDB integration test",
            "timestamp": datetime.utcnow()
        }
        
        result = await test_collection.insert_one(test_doc)
        print(f"   ✅ Inserted test document: {result.inserted_id}")
        
        # Read test document
        found = await test_collection.find_one({"test_id": "test_001"})
        if found:
            print(f"   ✅ Retrieved test document: {found['test_id']}")
        
        # Clean up
        await test_collection.delete_one({"test_id": "test_001"})
        print("   ✅ Cleaned up test document")
        print()
        
        # Test council repository
        print("4. Testing CouncilRepository...")
        from mongo_crud.app.database.council_repository import CouncilRepository
        
        repo = CouncilRepository(mongodb.db)
        
        # Test saving analysis
        test_analysis = {
            "decision_id": f"test_dec_{datetime.utcnow().timestamp()}",
            "package_name": "test-package",
            "package_version": "1.0.0",
            "registry": "npm",
            "verdict": "benign",
            "final_risk_score": 10.0,
            "final_confidence": 95.0,
            "agent_responses": [],
            "consensus_type": "unanimous",
            "agreement_percentage": 100.0,
            "total_tokens_used": 100,
            "estimated_cost_usd": 0.001,
            "analysis_duration_seconds": 1.5
        }
        
        saved = await repo.save_council_analysis(test_analysis)
        print(f"   ✅ Saved test analysis: {saved['decision_id']}")
        
        # Test retrieving
        retrieved = await repo.get_analysis_by_decision_id(test_analysis["decision_id"])
        if retrieved:
            print(f"   ✅ Retrieved analysis: {retrieved['package_name']}")
        
        # Test stats (will include our test)
        stats = await repo.get_analysis_stats()
        print(f"   ✅ Analysis stats: {stats['total_analyses']} total analyses")
        
        # Clean up test analysis
        await mongodb.db.council_analyses.delete_one({"decision_id": test_analysis["decision_id"]})
        print("   ✅ Cleaned up test analysis")
        print()
        
        # List collections
        print("5. Database collections:")
        collections = await mongodb.db.list_collection_names()
        for coll in collections:
            count = await mongodb.db[coll].count_documents({})
            print(f"   - {coll}: {count} documents")
        print()
        
        # Close connection
        await mongodb.disconnect()
        print("   ✅ Disconnected from MongoDB")
        print()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print()
        print("Your MongoDB integration is working correctly.")
        print("You can now use the LLM Council with database storage.")
        print()
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        print("   💡 Make sure you've installed dependencies:")
        print("      pip install -r requirements.txt")
        return False
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print()
        print("Troubleshooting tips:")
        print("1. Make sure MongoDB is running:")
        print("   - Local: sudo systemctl status mongodb")
        print("   - Docker: docker ps | grep mongo")
        print()
        print("2. Check your MONGODB_URI is correct")
        print("3. Verify network connectivity to MongoDB")
        print("4. Check MongoDB logs for errors")
        print()
        return False


async def show_existing_data():
    """Show existing council data in MongoDB."""
    print()
    print("=" * 60)
    print("EXISTING COUNCIL DATA")
    print("=" * 60)
    print()
    
    try:
        from mongo_crud.app.database.mongodb import MongoDB
        from mongo_crud.app.database.council_repository import CouncilRepository
        
        mongodb = MongoDB()
        await mongodb.connect()
        
        repo = CouncilRepository(mongodb.db)
        
        # Show analysis stats
        print("Council Analyses:")
        stats = await repo.get_analysis_stats()
        print(f"  Total: {stats['total_analyses']}")
        print(f"  Verdict breakdown: {stats['verdict_breakdown']}")
        print(f"  Avg risk score: {stats['avg_risk_score']}")
        print(f"  Total cost: ${stats['total_cost_usd']:.4f}")
        print()
        
        # Show recent analyses
        if stats['total_analyses'] > 0:
            print("Recent analyses:")
            analyses = await repo.get_recent_analyses(limit=5)
            for analysis in analyses:
                print(f"  - {analysis['package_name']}@{analysis['package_version']}: "
                      f"{analysis['verdict']} (risk: {analysis['final_risk_score']:.1f})")
        print()
        
        # Show debate stats
        print("Debate Results:")
        debate_stats = await repo.get_debate_stats()
        print(f"  Total: {debate_stats['total_debates']}")
        if debate_stats['total_debates'] > 0:
            print(f"  Consensus breakdown: {debate_stats['consensus_breakdown']}")
            print(f"  Avg rounds: {debate_stats['avg_rounds_to_consensus']}")
            print(f"  Avg quality: {debate_stats['avg_quality_score']}")
        print()
        
        await mongodb.disconnect()
        
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="MongoDB integration test for LLM Council"
    )
    parser.add_argument(
        "--show-data",
        action="store_true",
        help="Show existing data in MongoDB"
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    if args.show_data:
        asyncio.run(show_existing_data())
    else:
        success = asyncio.run(test_mongodb_connection())
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
