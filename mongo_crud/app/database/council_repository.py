"""
Repository for LLM Council analysis results and debate outcomes.
Provides CRUD operations for storing and retrieving council data.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.errors import DuplicateKeyError
from fastapi import HTTPException, status

from app.database.repository import BaseRepository


class CouncilRepository:
    """
    Repository for managing council analysis results and debate outcomes.
    Provides specialized methods for querying and analyzing council data.
    """
    
    def __init__(self, database: AsyncIOMotorDatabase):
        """
        Initialize repository with database connection.
        
        Args:
            database: MongoDB database instance
        """
        self.analyses = BaseRepository(database, "council_analyses")
        self.debates = BaseRepository(database, "debate_results")
        self.database = database
    
    # ========================================================================
    # COUNCIL ANALYSIS OPERATIONS
    # ========================================================================
    
    async def save_council_analysis(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save a council analysis result to the database.
        
        Args:
            analysis_data: Complete analysis data including decision_id, verdict, etc.
            
        Returns:
            Created document with _id
        """
        try:
            # Ensure timestamps
            now = datetime.utcnow()
            analysis_data["created_at"] = now
            analysis_data["updated_at"] = now
            
            # Create indexes if not exists (decision_id, package_name)
            await self.database.council_analyses.create_index("decision_id", unique=True)
            await self.database.council_analyses.create_index("package_name")
            await self.database.council_analyses.create_index("verdict")
            await self.database.council_analyses.create_index("created_at")
            
            return await self.analyses.create(analysis_data)
            
        except DuplicateKeyError:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Analysis with decision_id '{analysis_data.get('decision_id')}' already exists"
            )
    
    async def get_analysis_by_decision_id(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve analysis by decision ID.
        
        Args:
            decision_id: Unique decision identifier
            
        Returns:
            Analysis document or None
        """
        return await self.database.council_analyses.find_one({"decision_id": decision_id})
    
    async def get_analyses_by_package(
        self,
        package_name: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get all analyses for a specific package.
        
        Args:
            package_name: Package name to search for
            limit: Maximum number of results
            
        Returns:
            List of analysis documents
        """
        cursor = self.database.council_analyses.find(
            {"package_name": package_name}
        ).sort("created_at", -1).limit(limit)
        
        return await cursor.to_list(length=limit)
    
    async def get_analyses_by_verdict(
        self,
        verdict: str,
        limit: int = 50,
        skip: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get analyses filtered by verdict.
        
        Args:
            verdict: Verdict to filter by (malicious, benign, uncertain, error)
            limit: Maximum results
            skip: Number to skip (pagination)
            
        Returns:
            List of analysis documents
        """
        cursor = self.database.council_analyses.find(
            {"verdict": verdict}
        ).sort("created_at", -1).skip(skip).limit(limit)
        
        return await cursor.to_list(length=limit)
    
    async def get_recent_analyses(
        self,
        limit: int = 20,
        skip: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get most recent analyses.
        
        Args:
            limit: Maximum results
            skip: Number to skip (pagination)
            
        Returns:
            List of analysis documents
        """
        cursor = self.database.council_analyses.find().sort(
            "created_at", -1
        ).skip(skip).limit(limit)
        
        return await cursor.to_list(length=limit)
    
    async def get_analysis_stats(self) -> Dict[str, Any]:
        """
        Get statistics about council analyses.
        
        Returns:
            Dictionary with stats (total, verdict breakdown, averages, etc.)
        """
        pipeline = [
            {
                "$facet": {
                    "total_count": [{"$count": "count"}],
                    "verdict_breakdown": [
                        {"$group": {"_id": "$verdict", "count": {"$sum": 1}}}
                    ],
                    "averages": [
                        {
                            "$group": {
                                "_id": None,
                                "avg_risk_score": {"$avg": "$final_risk_score"},
                                "avg_confidence": {"$avg": "$final_confidence"},
                                "total_tokens": {"$sum": "$total_tokens_used"},
                                "total_cost": {"$sum": "$estimated_cost_usd"}
                            }
                        }
                    ]
                }
            }
        ]
        
        result = await self.database.council_analyses.aggregate(pipeline).to_list(length=1)
        
        if not result:
            return {
                "total_analyses": 0,
                "verdict_breakdown": {},
                "avg_risk_score": 0.0,
                "avg_confidence": 0.0,
                "total_tokens_used": 0,
                "total_cost_usd": 0.0
            }
        
        data = result[0]
        total = data["total_count"][0]["count"] if data["total_count"] else 0
        
        verdict_breakdown = {
            item["_id"]: item["count"] 
            for item in data["verdict_breakdown"]
        }
        
        averages = data["averages"][0] if data["averages"] else {}
        
        return {
            "total_analyses": total,
            "verdict_breakdown": verdict_breakdown,
            "avg_risk_score": round(averages.get("avg_risk_score", 0.0), 2),
            "avg_confidence": round(averages.get("avg_confidence", 0.0), 2),
            "total_tokens_used": averages.get("total_tokens", 0),
            "total_cost_usd": round(averages.get("total_cost", 0.0), 4)
        }
    
    # ========================================================================
    # DEBATE RESULT OPERATIONS
    # ========================================================================
    
    async def save_debate_result(self, debate_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save a debate result to the database.
        
        Args:
            debate_data: Complete debate data including debate_id, verdict, history, etc.
            
        Returns:
            Created document with _id
        """
        try:
            # Ensure timestamps
            now = datetime.utcnow()
            debate_data["created_at"] = now
            debate_data["updated_at"] = now
            
            # Create indexes
            await self.database.debate_results.create_index("debate_id", unique=True)
            await self.database.debate_results.create_index("package_name")
            await self.database.debate_results.create_index("final_verdict")
            await self.database.debate_results.create_index("created_at")
            
            return await self.debates.create(debate_data)
            
        except DuplicateKeyError:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Debate with debate_id '{debate_data.get('debate_id')}' already exists"
            )
    
    async def get_debate_by_id(self, debate_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve debate by debate ID.
        
        Args:
            debate_id: Unique debate identifier
            
        Returns:
            Debate document or None
        """
        return await self.database.debate_results.find_one({"debate_id": debate_id})
    
    async def get_debates_by_package(
        self,
        package_name: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get all debates for a specific package.
        
        Args:
            package_name: Package name to search for
            limit: Maximum number of results
            
        Returns:
            List of debate documents
        """
        cursor = self.database.debate_results.find(
            {"package_name": package_name}
        ).sort("created_at", -1).limit(limit)
        
        return await cursor.to_list(length=limit)
    
    async def get_recent_debates(
        self,
        limit: int = 20,
        skip: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get most recent debates.
        
        Args:
            limit: Maximum results
            skip: Number to skip (pagination)
            
        Returns:
            List of debate documents
        """
        cursor = self.database.debate_results.find().sort(
            "created_at", -1
        ).skip(skip).limit(limit)
        
        return await cursor.to_list(length=limit)
    
    async def get_debate_stats(self) -> Dict[str, Any]:
        """
        Get statistics about debates.
        
        Returns:
            Dictionary with debate statistics
        """
        pipeline = [
            {
                "$facet": {
                    "total_count": [{"$count": "count"}],
                    "consensus_types": [
                        {"$group": {"_id": "$consensus_type", "count": {"$sum": 1}}}
                    ],
                    "averages": [
                        {
                            "$group": {
                                "_id": None,
                                "avg_rounds": {"$avg": "$rounds_to_consensus"},
                                "avg_quality": {"$avg": "$debate_quality_score"},
                                "avg_duration": {"$avg": "$total_duration_seconds"}
                            }
                        }
                    ]
                }
            }
        ]
        
        result = await self.database.debate_results.aggregate(pipeline).to_list(length=1)
        
        if not result:
            return {
                "total_debates": 0,
                "consensus_breakdown": {},
                "avg_rounds_to_consensus": 0.0,
                "avg_quality_score": 0.0,
                "avg_duration_seconds": 0.0
            }
        
        data = result[0]
        total = data["total_count"][0]["count"] if data["total_count"] else 0
        
        consensus_breakdown = {
            item["_id"]: item["count"] 
            for item in data["consensus_types"]
        }
        
        averages = data["averages"][0] if data["averages"] else {}
        
        return {
            "total_debates": total,
            "consensus_breakdown": consensus_breakdown,
            "avg_rounds_to_consensus": round(averages.get("avg_rounds", 0.0), 2),
            "avg_quality_score": round(averages.get("avg_quality", 0.0), 2),
            "avg_duration_seconds": round(averages.get("avg_duration", 0.0), 2)
        }
