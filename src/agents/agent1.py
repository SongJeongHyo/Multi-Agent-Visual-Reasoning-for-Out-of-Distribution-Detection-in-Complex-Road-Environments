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

SCENE_CONTEXT_SYSTEM_PROMPT = """
You are a Scene Context Analyzer that establishes contextual baselines for road environments.
Your primary role is to understand what constitutes "normal" for the given road environment.
CORE RESPONSIBILITIES:
1. Determine scene type (urban/rural/highway/intersection/residential)
2. Assess environmental conditions (weather, lighting, time of day)
3. Identify expected object inventory based on context
4. Establish context-dependent normality criteria
CRITICAL: Focus on establishing what should be "expected" in this specific road environment context.
OUTPUT FORMAT (JSON):
{
    "scene_analysis": {
        "scene_type": "urban/rural/highway/intersection/residential/mixed",
        "road_infrastructure": "description of visible road elements",
        "environmental_conditions": {
            "weather": "clear/rainy/foggy/snowy/unclear",
            "lighting": "daylight/dusk/night/artificial/mixed",
            "time_period": "morning/afternoon/evening/night/unclear"
        }
    },
    "contextual_baseline": {
        "expected_objects": ["list of objects that should be normal in this context"],
        "expected_behaviors": ["normal traffic patterns and behaviors for this context"],
        "infrastructure_elements": ["normal road infrastructure for this scene type"],
        "typical_layout": "description of what a normal layout should look like"
    },
    "environmental_factors": {
        "visibility_conditions": "assessment of how environmental factors affect visibility",
        "seasonal_indicators": "any seasonal elements that affect context",
        "special_circumstances": "any special environmental factors"
    },
    "normality_criteria": {
        "object_appropriateness": "what objects belong in this environment",
        "spatial_expectations": "normal positioning and spacing expectations",
        "behavioral_norms": "expected movement and interaction patterns"
    },
    "context_confidence": 0.0-1.0
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

class DirectAPISceneContextAnalyzer:
    def __init__(self, delay_between_requests: float = 60.0):
        self.client = openai.OpenAI(api_key=openai.api_key)
        self.delay_between_requests = delay_between_requests
        
        print("🚀 AGENT 1: Scene Context Analyzer (Direct OpenAI API)")
        print("📍 Using direct OpenAI API for reliable image processing")
        print("🎯 Determining what constitutes 'normal' for each scene")
        print(f"⏱️ Rate limiting: {delay_between_requests}s between requests")
    
    def analyze_image(self, image_path: str, verbose: bool = True) -> Dict:
        """Analyzes the scene context of an image - Direct API"""
        
        if verbose:
            print(f"🔍 Analyzing scene context: {image_path}")
        
        try:
            # Encode image
            base64_image = encode_image_optimized(image_path, max_size=512)
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": SCENE_CONTEXT_SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this road scene image to establish the contextual baseline. Provide detailed analysis in the specified JSON format."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "low"  # Reduce token usage
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
                print("✅ Scene context analysis completed")
                if "scene_analysis" in parsed_result:
                    scene_type = parsed_result["scene_analysis"].get("scene_type", "Unknown")
                    print(f"  📍 Scene type: {scene_type}")
                
                if "contextual_baseline" in parsed_result:
                    expected_objects = parsed_result["contextual_baseline"].get("expected_objects", [])
                    print(f"  🎯 Expected objects: {', '.join(expected_objects[:5])}")
            
            return parsed_result
            
        except Exception as e:
            print(f"❌ Scene context analysis failed: {e}")
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
    
    def process_batch(self, image_directory: str, output_file: str = "agent1_scene_context_results.json"):
        """Batch processing"""
        
        print(f"🚀 Starting Direct API Agent 1 batch processing...")
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
                "agent": "direct_api_scene_context_analyzer",
                "agent_number": 1,
                "processing_date": datetime.now().isoformat(),
                "source_directory": image_directory,
                "total_images": len(image_files),
                "model_used": "gpt-4o",
                "api_method": "direct_openai_api",
                "agent_role": "contextual_baseline_establishment"
            },
            "results": []
        }
        
        # Analyze each image
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(image_directory, image_file)
            
            print(f"\n📸 Processing image {i+1}/{len(image_files)}: {image_file}")
            
            try:
                start_time = time.time()
                
                # Scene context analysis
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
                    "scene_context_analysis": analysis_result
                }
                
                batch_results["results"].append(image_result)
                print(f"  ✅ Success - Processing time: {processing_time:.2f}s")
                
                # Print result summary
                if "scene_analysis" in analysis_result:
                    scene_type = analysis_result["scene_analysis"].get("scene_type", "Unknown")
                    print(f"  📍 Scene: {scene_type}")
                
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
            print(f"\n💾 Agent 1 results saved to: {output_file}")
        except Exception as e:
            print(f"❌ Failed to save results: {str(e)}")
        
        print(f"\n🎉 Agent 1 batch processing completed!")
        print(f"📊 Successfully processed: {len(batch_results['results'])} images")
        
        return batch_results

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent 1: Scene Context Analyzer")
    parser.add_argument("--image_directory", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the JSON output file")
    parser.add_argument("--delay_between_requests", type=float, default=60.0, help="Delay in seconds between API requests")
    
    args = parser.parse_args()

    print("=== AGENT 1: Direct API Scene Context Analyzer ===")
    
    analyzer = DirectAPISceneContextAnalyzer(delay_between_requests=args.delay_between_requests)
    
    # Run batch processing
    results = analyzer.process_batch(
        image_directory=args.image_directory,
        output_file=args.output_file
    )
    
    print("\n✅ Agent 1 Complete!")
    print("📋 Next step: Run agent2.py")