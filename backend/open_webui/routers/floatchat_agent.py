"""
FloatChat Specialized Agent for Argo Float Data Analysis
Handles only oceanographic data queries, processing, and visualization
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import re

from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from openai import OpenAI

from open_webui.models.users import UserModel
from open_webui.utils.auth import get_verified_user

log = logging.getLogger(__name__)

router = APIRouter()

# Agent Configuration
NVIDIA_API_KEY = "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"
FLOATCHAT_MODEL = "qwen/qwen3-coder-480b-a35b-instruct"

class TaskType(Enum):
    """Types of tasks the FloatChat agent can handle"""
    SQL_QUERY = "sql_query"
    VECTOR_SEARCH = "vector_search"
    VISUALIZATION = "visualization"
    DATA_PROCESSING = "data_processing"
    EXPORT = "export"
    OUT_OF_SCOPE = "out_of_scope"

class IntentClassifier:
    """Classifies user queries to determine if they're within FloatChat scope"""
    
    FLOATCHAT_KEYWORDS = [
        # Oceanographic terms
        "salinity", "temperature", "pressure", "depth", "profile", "float", "argo",
        "ocean", "sea", "marine", "oceanographic", "hydrographic",
        # Measurements
        "sst", "sss", "conductivity", "density", "oxygen", "chlorophyll", "ph",
        # Geographic
        "arabian sea", "indian ocean", "bay of bengal", "mumbai", "chennai", "kochi",
        "latitude", "longitude", "coordinates", "location", "region",
        # Data operations
        "heatmap", "visualization", "plot", "chart", "graph", "map", "profile",
        "query", "search", "filter", "export", "download", "csv", "netcdf",
        # Time-related
        "monthly", "seasonal", "annual", "trend", "time series", "historical"
    ]
    
    FORBIDDEN_PATTERNS = [
        r"tell.*joke", r"what.*weather", r"how.*day", r"hello", r"hi there",
        r"general.*question", r"chat.*me", r"casual.*talk", r"random.*topic"
    ]
    
    @classmethod
    def classify_intent(cls, query: str) -> TaskType:
        """Classify user query intent"""
        query_lower = query.lower()
        
        # Check for forbidden patterns
        for pattern in cls.FORBIDDEN_PATTERNS:
            if re.search(pattern, query_lower):
                return TaskType.OUT_OF_SCOPE
        
        # Check for FloatChat keywords
        keyword_count = sum(1 for keyword in cls.FLOATCHAT_KEYWORDS if keyword in query_lower)
        
        if keyword_count == 0:
            return TaskType.OUT_OF_SCOPE
        
        # Determine specific task type
        if any(word in query_lower for word in ["heatmap", "plot", "chart", "visualization", "map"]):
            return TaskType.VISUALIZATION
        elif any(word in query_lower for word in ["export", "download", "csv", "netcdf"]):
            return TaskType.EXPORT
        elif any(word in query_lower for word in ["process", "qc", "quality", "statistics"]):
            return TaskType.DATA_PROCESSING
        elif any(word in query_lower for word in ["search", "find", "similar", "semantic"]):
            return TaskType.VECTOR_SEARCH
        else:
            return TaskType.SQL_QUERY

class TaskRouter:
    """Routes classified tasks to appropriate execution modules"""
    
    @staticmethod
    def route_task(task_type: TaskType, query: str, user_context: Dict) -> Dict[str, Any]:
        """Route task to appropriate module"""
        if task_type == TaskType.OUT_OF_SCOPE:
            return {
                "module": "rejection",
                "message": "This agent handles only FloatChat oceanographic data queries.",
                "suggestions": [
                    "Try asking about Argo float data",
                    "Request oceanographic measurements",
                    "Ask for data visualizations or exports"
                ]
            }
        
        routing_map = {
            TaskType.SQL_QUERY: SQLModule,
            TaskType.VECTOR_SEARCH: VectorSearchModule,
            TaskType.VISUALIZATION: VisualizationModule,
            TaskType.DATA_PROCESSING: ProcessingModule,
            TaskType.EXPORT: ExportModule
        }
        
        module_class = routing_map.get(task_type)
        if module_class:
            return {
                "module": module_class.__name__,
                "executor": module_class(),
                "task_type": task_type.value
            }
        
        return {"module": "unknown", "error": "Task type not supported"}

class SQLModule:
    """Handles SQL query generation for PostgreSQL/PostGIS"""
    
    def __init__(self):
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=NVIDIA_API_KEY
        )
    
    def generate_sql(self, query: str, context: Dict) -> str:
        """Generate SQL query from natural language"""
        system_prompt = """
        You are a SQL expert for oceanographic Argo float data.
        
        Database Schema:
        - float_profiles: float_id, latitude, longitude, date, cycle_number
        - measurements: profile_id, depth, temperature, salinity, pressure, oxygen
        - metadata: float_id, deployment_date, status, wmo_id
        
        Generate PostgreSQL/PostGIS queries for oceanographic data requests.
        Include spatial queries using ST_Distance, ST_Within for geographic filters.
        Use appropriate date ranges and measurement filters.
        
        Example patterns:
        - "salinity near Mumbai" → spatial query with ST_Distance
        - "temperature profiles last 6 months" → date filter with JOIN
        - "depth vs temperature" → measurement aggregation
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate SQL for: {query}"}
        ]
        
        completion = self.client.chat.completions.create(
            model=FLOATCHAT_MODEL,
            messages=messages,
            temperature=0.1,  # Low temperature for precise SQL
            max_tokens=1000
        )
        
        return completion.choices[0].message.content
    
    def execute_query(self, sql: str) -> Dict[str, Any]:
        """Execute SQL query (mock implementation)"""
        # In real implementation, this would connect to PostgreSQL/PostGIS
        return {
            "sql": sql,
            "status": "generated",
            "note": "SQL query generated. Connect to PostgreSQL/PostGIS to execute.",
            "example_result": "Mock oceanographic data would be returned here"
        }

class VectorSearchModule:
    """Handles semantic search on float profiles using embeddings"""
    
    def __init__(self):
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=NVIDIA_API_KEY
        )
    
    def semantic_search(self, query: str, context: Dict) -> Dict[str, Any]:
        """Perform semantic search on float data"""
        # Generate embedding for query
        embedding_prompt = f"Oceanographic query for semantic search: {query}"
        
        # Mock vector search implementation
        return {
            "query": query,
            "method": "vector_search",
            "status": "processed",
            "note": "Semantic search on float profiles. Integrate with Chroma/Pinecone.",
            "similar_profiles": [
                {"float_id": "12345", "similarity": 0.89, "location": "Arabian Sea"},
                {"float_id": "67890", "similarity": 0.76, "location": "Bay of Bengal"}
            ]
        }

class VisualizationModule:
    """Handles chart, map, and 3D visualization generation"""
    
    def __init__(self):
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=NVIDIA_API_KEY
        )
    
    def generate_visualization(self, query: str, data: Dict, context: Dict) -> Dict[str, Any]:
        """Generate appropriate visualization"""
        viz_prompt = f"""
        Generate Python code for oceanographic data visualization.
        Query: {query}
        
        Use libraries: Plotly, Matplotlib, Folium for maps
        Create appropriate visualizations:
        - Heatmaps for spatial data
        - Line plots for depth profiles
        - 3D plots for temperature/salinity/depth
        - Maps for geographic distribution
        """
        
        messages = [
            {"role": "system", "content": "You are an expert in oceanographic data visualization."},
            {"role": "user", "content": viz_prompt}
        ]
        
        completion = self.client.chat.completions.create(
            model=FLOATCHAT_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=1500
        )
        
        return {
            "visualization_code": completion.choices[0].message.content,
            "type": "plotly_chart",
            "status": "generated",
            "note": "Visualization code generated. Execute with Plotly/Matplotlib."
        }

class ProcessingModule:
    """Handles data processing, QC, and statistical analysis"""
    
    def process_data(self, query: str, context: Dict) -> Dict[str, Any]:
        """Process oceanographic data"""
        processing_steps = []
        
        if "quality" in query.lower() or "qc" in query.lower():
            processing_steps.append("Apply quality control filters")
        
        if "statistics" in query.lower() or "stats" in query.lower():
            processing_steps.append("Calculate statistical summaries")
        
        if "profile" in query.lower():
            processing_steps.append("Generate depth profiles")
        
        return {
            "processing_steps": processing_steps,
            "status": "planned",
            "note": "Data processing pipeline defined. Integrate with xarray/dask.",
            "output_format": "Processed NetCDF/Parquet files"
        }

class ExportModule:
    """Handles data export in various formats"""
    
    def generate_export(self, query: str, data: Dict, context: Dict) -> Dict[str, Any]:
        """Generate data export"""
        export_formats = []
        
        if "csv" in query.lower():
            export_formats.append("CSV")
        if "netcdf" in query.lower():
            export_formats.append("NetCDF")
        if "parquet" in query.lower():
            export_formats.append("Parquet")
        if "pdf" in query.lower():
            export_formats.append("PDF Report")
        
        if not export_formats:
            export_formats = ["CSV"]  # Default
        
        return {
            "export_formats": export_formats,
            "status": "ready",
            "download_links": [f"mock_download_{fmt.lower()}.{fmt.lower()}" for fmt in export_formats],
            "note": "Export files generated. Implement actual file generation."
        }

class FloatChatAgent:
    """Main FloatChat specialized agent"""
    
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.task_router = TaskRouter()
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=NVIDIA_API_KEY
        )
    
    def process_query(self, query: str, user_context: Dict = None) -> Dict[str, Any]:
        """Main agent workflow"""
        if user_context is None:
            user_context = {}
        
        log.info(f"FloatChat Agent processing query: {query}")
        
        # Step 1: Classify intent
        task_type = self.intent_classifier.classify_intent(query)
        log.info(f"Classified task type: {task_type.value}")
        
        # Step 2: Route to module
        routing_result = self.task_router.route_task(task_type, query, user_context)
        
        # Step 3: Handle out-of-scope queries
        if task_type == TaskType.OUT_OF_SCOPE:
            return {
                "agent": "FloatChat",
                "status": "rejected",
                "message": routing_result["message"],
                "suggestions": routing_result["suggestions"],
                "timestamp": datetime.now().isoformat()
            }
        
        # Step 4: Execute task
        try:
            executor = routing_result.get("executor")
            if not executor:
                raise ValueError("No executor found for task")
            
            # Execute based on task type
            if task_type == TaskType.SQL_QUERY:
                sql_result = executor.generate_sql(query, user_context)
                execution_result = executor.execute_query(sql_result)
            elif task_type == TaskType.VECTOR_SEARCH:
                execution_result = executor.semantic_search(query, user_context)
            elif task_type == TaskType.VISUALIZATION:
                execution_result = executor.generate_visualization(query, {}, user_context)
            elif task_type == TaskType.DATA_PROCESSING:
                execution_result = executor.process_data(query, user_context)
            elif task_type == TaskType.EXPORT:
                execution_result = executor.generate_export(query, {}, user_context)
            else:
                execution_result = {"error": "Unknown task type"}
            
            # Step 5: Generate response
            response = self.generate_response(query, task_type, execution_result)
            
            # Step 6: Log activity
            self.log_activity(query, task_type, execution_result, user_context)
            
            return response
            
        except Exception as e:
            log.error(f"Error executing FloatChat query: {e}")
            return {
                "agent": "FloatChat",
                "status": "error",
                "message": f"Error processing oceanographic query: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def generate_response(self, query: str, task_type: TaskType, execution_result: Dict) -> Dict[str, Any]:
        """Generate comprehensive response"""
        
        # Create summary using LLM
        summary_prompt = f"""
        Summarize this FloatChat oceanographic data analysis result:
        
        Original Query: {query}
        Task Type: {task_type.value}
        Execution Result: {json.dumps(execution_result, indent=2)}
        
        Provide a clear, technical summary for oceanographic researchers.
        Include any SQL queries, visualization descriptions, or data processing steps.
        """
        
        messages = [
            {"role": "system", "content": "You are a FloatChat oceanographic data analyst."},
            {"role": "user", "content": summary_prompt}
        ]
        
        completion = self.client.chat.completions.create(
            model=FLOATCHAT_MODEL,
            messages=messages,
            temperature=0.5,
            max_tokens=1000
        )
        
        summary = completion.choices[0].message.content
        
        return {
            "agent": "FloatChat",
            "status": "success",
            "task_type": task_type.value,
            "summary": summary,
            "execution_result": execution_result,
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "capabilities": [
                "SQL query generation for PostgreSQL/PostGIS",
                "Semantic search on float profiles",
                "Oceanographic data visualization",
                "Data processing and quality control",
                "Multi-format data export"
            ]
        }
    
    def log_activity(self, query: str, task_type: TaskType, result: Dict, user_context: Dict):
        """Log agent activity for audit and feedback"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "task_type": task_type.value,
            "user_context": user_context,
            "result_status": result.get("status", "unknown"),
            "agent": "FloatChat"
        }
        
        log.info(f"FloatChat Agent Activity: {json.dumps(log_entry)}")

# API Endpoints
class FloatChatQuery(BaseModel):
    query: str
    context: Optional[Dict] = {}

@router.post("/floatchat/query")
async def process_floatchat_query(
    request: FloatChatQuery,
    user: UserModel = Depends(get_verified_user)
):
    """Process FloatChat oceanographic data query"""
    agent = FloatChatAgent()
    
    user_context = {
        "user_id": user.id,
        "user_name": user.name,
        "timestamp": datetime.now().isoformat(),
        **request.context
    }
    
    result = agent.process_query(request.query, user_context)
    return result

@router.get("/floatchat/capabilities")
async def get_floatchat_capabilities():
    """Get FloatChat agent capabilities"""
    return {
        "agent": "FloatChat",
        "version": "1.0",
        "scope": "Argo float oceanographic data analysis",
        "capabilities": [
            "Natural language to SQL query generation",
            "Semantic search on float profiles",
            "Oceanographic data visualization",
            "Data processing and quality control",
            "Multi-format data export (CSV, NetCDF, Parquet)",
            "Spatial queries with PostGIS",
            "Time series analysis",
            "Statistical summaries"
        ],
        "supported_regions": [
            "Arabian Sea",
            "Bay of Bengal", 
            "Indian Ocean",
            "Global ocean coverage"
        ],
        "data_types": [
            "Temperature profiles",
            "Salinity measurements", 
            "Pressure/depth data",
            "Dissolved oxygen",
            "Chlorophyll concentrations",
            "pH measurements"
        ],
        "visualization_types": [
            "Depth profiles",
            "Heatmaps",
            "3D scatter plots",
            "Geographic maps",
            "Time series charts",
            "Statistical plots"
        ]
    }

@router.get("/floatchat/examples")
async def get_query_examples():
    """Get example queries for FloatChat agent"""
    return {
        "examples": [
            {
                "query": "Show salinity near Mumbai for the last 6 months",
                "type": "spatial_temporal",
                "expected_output": "SQL query with ST_Distance and date filters"
            },
            {
                "query": "Generate depth vs temperature profile for float_id 12345",
                "type": "profile_visualization",
                "expected_output": "Plotly line chart with depth on y-axis"
            },
            {
                "query": "Create heatmap of SST for Arabian Sea in September 2025",
                "type": "spatial_visualization", 
                "expected_output": "Geographic heatmap using Folium/Plotly"
            },
            {
                "query": "Export temperature data as NetCDF for Bay of Bengal",
                "type": "data_export",
                "expected_output": "NetCDF file download link"
            },
            {
                "query": "Find similar temperature profiles using semantic search",
                "type": "vector_search",
                "expected_output": "List of similar float profiles with similarity scores"
            },
            {
                "query": "Apply quality control to oxygen measurements",
                "type": "data_processing",
                "expected_output": "QC processing pipeline and filtered dataset"
            }
        ],
        "invalid_examples": [
            "Tell me a joke",
            "What's the weather today?",
            "How are you doing?",
            "General conversation topics"
        ]
    }
