import openai
import json
import base64
import os
import time
import argparse  
import sys
from typing import Dict, List
from datetime import datetime
from PIL import Image
import io

openai.api_key = "YOUR_API_KEY_HERE"

SPATIAL_ANOMALY_SYSTEM_PROMPT = """
You are a Spatial Anomaly Detector specializing in identifying objects that violate spatial rules and positioning conventions in road environments.
Your expertise focuses on whether objects are positioned in locations that could disrupt normal traffic flow or create hazardous conditions.
CORE RESPONSIBILITIES:
1. Evaluate object positioning relative to road infrastructure
2. Assess potential traffic flow disruptions
3. Analyze scale consistency based on apparent distance
4. Identify unusual clustering or density patterns
SPATIAL ANALYSIS FRAMEWORK:
- Systematic evaluation of spatial positioning violations
- Detection of positional inappropriateness beyond object identity
- Focus on collective spatial arrangements creating anomalous situations
- Example: Few sheep roadside acceptable, but densely clustered sheep blocking traffic lanes = spatial anomaly
CRITICAL DETECTION AREAS:
- Objects blocking traffic lanes or pedestrian walkways
- Vehicles positioned inappropriately (wrong direction, illegal parking)
- Scale inconsistencies (objects too large/small for their apparent distance)
- Density anomalies (clustering that creates hazards)
- Spatial relationships that violate traffic safety conventions
OUTPUT FORMAT (JSON):
{
    "spatial_analysis": {
        "observed_objects": ["list of all objects detected in the scene"],
        "object_positions": {
            "on_roadway": ["objects positioned on driving lanes"],
            "roadside": ["objects positioned beside the road"],
            "infrastructure": ["objects related to road infrastructure"]
        }
    },
    "positioning_violations": [
        {
            "object": "specific object description",
            "violation_type": "blocking/obstructing/misplaced/wrong_direction",
            "location": "specific location description",
            "traffic_impact": "how this affects traffic flow",
            "severity": "low/medium/high"
        }
    ],
    "scale_inconsistencies": [
        {
            "object": "object with scale issues",
            "issue_description": "description of scale problem",
            "expected_scale": "what scale should be normal",
            "distance_assessment": "apparent distance and expected size"
        }
    ],
    "clustering_anomalies": [
        {
            "object_group": "description of clustered objects",
            "cluster_description": "how objects are clustered",
            "hazard_level": "safety concern level",
            "normal_vs_anomalous": "why this clustering is problematic"
        }
    ],
    "traffic_flow_analysis": {
        "disruption_points": ["locations where traffic flow is disrupted"],
        "accessibility_issues": ["areas where normal access is blocked"],
        "safety_hazards": ["spatial arrangements creating safety risks"]
    },
    "spatial_confidence": 0.0-1.0
}
"""

def encode_image_optimized(image_path: str, max_size: int = 512) -> str:
    """Optimize and base64 encode the image"""
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        img.save(buffer, 
                format='JPEG', 
                quality=80,
                optimize=True)
        buffer.seek(0)
        
        return base64.b64encode(buffer.read()).decode('utf-8')

class DirectAPISpatialAnomalyDetector:
    def __init__(self, delay_between_requests: float = 60.0):
        self.client = openai.OpenAI(api_key=openai.api_key)
        self.delay_between_requests = delay_between_requests
        
        print("🚀 AGENT 2: Spatial Anomaly Detector (Direct OpenAI API)")
        print("📐 Identifying spatial positioning violations and traffic disruptions")
        print("🎯 Focus on objects disrupting traffic flow or creating hazards")
        print(f"⏱️ Rate limiting: {delay_between_requests}s between requests")
    
    def analyze_image(self, image_path: str, verbose: bool = True) -> Dict:
        """Analyzes spatial anomalies in an image - Direct API"""
        
        if verbose:
            print(f"🔍 Analyzing spatial anomalies: {image_path}")
        
        try:
            # Encode image
            base64_image = encode_image_optimized(image_path, max_size=512)
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": SPATIAL_ANOMALY_SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this road scene image for spatial anomalies and positioning violations. Focus on objects that block traffic flow, are positioned inappropriately, or create hazardous spatial arrangements. Provide detailed analysis in the specified JSON format."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "low"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1500,
                temperature=0.1
            )
            
            # Parse response
            content = response.choices[0].message.content
            parsed_result = self._parse_json_response(content)
            
            if verbose:
                print("✅ Spatial anomaly analysis completed")
                
                # Print violations
                violations = parsed_result.get("positioning_violations", [])
                if violations:
                    print(f"  ⚠️ Found {len(violations)} positioning violations")
                    for violation in violations[:3]:  # Max 3
                        obj = violation.get("object", "Unknown")
                        severity = violation.get("severity", "Unknown")
                        print(f"    - {obj} ({severity} severity)")
                
                clusters = parsed_result.get("clustering_anomalies", [])
                if clusters:
                    print(f"  🔗 Found {len(clusters)} clustering anomalies")
            
            return parsed_result
            
        except Exception as e:
            print(f"❌ Spatial anomaly analysis failed: {e}")
            return {"error": str(e)}
    
    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON response"""
        try:
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "{" in response and "}" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
            else:
                json_str = response
            
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {"analysis": response, "error": "JSON parsing failed"}
    
    def process_batch(self, image_directory: str, output_file: str = "agent2_spatial_anomaly_results.json"):
        """Batch processing"""
        
        print(f"🚀 Starting Direct API Agent 2 batch processing...")
        print(f"📂 Processing directory: {image_directory}")
        
        # Collect image file list
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend([f for f in os.listdir(image_directory) if f.lower().endswith(ext.lower())])
        
        if not image_files:
            print("❌ No images found in the directory!")
            return {"error": "No images found"}
        
        print(f"📋 Found {len(image_files)} images to analyze")
        
        # Store batch processing results
        batch_results = {
            "metadata": {
                "agent": "direct_api_spatial_anomaly_detector",
                "agent_number": 2,
                "processing_date": datetime.now().isoformat(),
                "source_directory": image_directory,
                "total_images": len(image_files),
                "model_used": "gpt-4o",
                "api_method": "direct_openai_api",
                "agent_role": "spatial_positioning_violation_detection"
            },
            "results": []
        }
        
        # Analyze each image
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(image_directory, image_file)
            
            print(f"\n📸 Processing image {i+1}/{len(image_files)}: {image_file}")
            
            try:
                start_time = time.time()
                
                # Spatial anomaly analysis
                analysis_result = self.analyze_image(image_path, verbose=False)
                
                processing_time = time.time() - start_time
                
                # Save result
                image_result = {
                    "image_info": {
                        "filename": image_file,
                        "filepath": image_path,
                        "processing_time": processing_time,
                        "processing_order": i + 1
                    },
                    "spatial_anomaly_analysis": analysis_result
                }
                
                batch_results["results"].append(image_result)
                print(f"  ✅ Success - Processing time: {processing_time:.2f}s")
                
                # Print result summary
                violations = analysis_result.get("positioning_violations", [])
                if violations:
                    print(f"  ⚠️ Violations: {len(violations)} found")
                
                # Delay before next request
                if i < len(image_files) - 1:
                    print(f"    ⏱️ Waiting {self.delay_between_requests} seconds...")
                    time.sleep(self.delay_between_requests)
                    
            except Exception as e:
                print(f"  ❌ Exception occurred: {str(e)}")
        
        # Save results to JSON file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(batch_results, f, ensure_ascii=False, indent=2)
            print(f"\n💾 Agent 2 results saved to: {output_file}")
        except Exception as e:
            print(f"❌ Failed to save results: {str(e)}")
        
        print(f"\n🎉 Agent 2 batch processing completed!")
        print(f"📊 Successfully processed: {len(batch_results['results'])} images")
        
        return batch_results

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent 2: Spatial Anomaly Detector")
    parser.add_argument("--image_directory", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the JSON output file")
    parser.add_argument("--delay_between_requests", type=float, default=60.0, help="Delay in seconds between API requests")
    
    args = parser.parse_args()

    print("=== AGENT 2: Direct API Spatial Anomaly Detector ===")
    
    detector = DirectAPISpatialAnomalyDetector(delay_between_requests=args.delay_between_requests)
    
    # Run batch processing
    results = detector.process_batch(
        image_directory=args.image_directory,
        output_file=args.output_file
    )
    
    print("\n✅ Agent 2 Complete!")
    print("📋 Next step: Run agent3.py")