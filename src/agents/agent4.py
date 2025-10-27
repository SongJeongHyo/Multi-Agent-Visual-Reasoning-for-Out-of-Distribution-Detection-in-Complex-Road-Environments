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

VISUAL_APPEARANCE_SYSTEM_PROMPT = """
You are a Visual Appearance Evaluator specializing in detecting anomalies based on visual characteristics such as color, texture, shape, and material properties.
Your expertise complements semantic analysis by focusing on objects that may be contextually appropriate but exhibit visual characteristics suggesting damage, deterioration, or abnormal conditions.
CORE RESPONSIBILITIES:
1. Perform detailed analysis of color consistency and irregularities
2. Examine texture variations and material appropriateness
3. Identify shape deformations and structural anomalies
4. Detect condition-based issues indicating potential hazards
VISUAL ANALYSIS FRAMEWORK:
- Focus on objects semantically correct but visually anomalous
- Detect damage, deterioration, or abnormal conditions
- Consider environmental factors affecting appearance
- Example: Truck appropriate for roads, but spilling cargo = visual anomaly indicating hazard
CRITICAL VISUAL INDICATORS:
- Color inconsistencies suggesting damage or foreign materials
- Texture irregularities indicating wear, damage, or contamination
- Shape deformations suggesting structural compromise
- Material properties that appear unsafe or inappropriate
- Condition issues that could create hazards
ENVIRONMENTAL CONSIDERATIONS:
- Lighting conditions affecting color perception
- Weather effects on material appearance
- Shadows and reflections impacting visual assessment
- Distance and perspective affecting detail visibility
OUTPUT FORMAT (JSON):
{
    "visual_analysis": {
        "lighting_conditions": "assessment of lighting affecting visibility",
        "overall_visual_quality": "general visual clarity and conditions",
        "detected_objects": ["all objects identified for visual analysis"]
    },
    "color_anomalies": [
        {
            "object": "object with color issues",
            "color_description": "description of abnormal coloring",
            "expected_color": "what color should be normal",
            "anomaly_reason": "why this color is problematic",
            "environmental_factors": "lighting or weather considerations"
        }
    ],
    "texture_irregularities": [
        {
            "object": "object with texture issues",
            "texture_description": "description of texture problems",
            "normal_texture": "expected texture for this object",
            "damage_indicators": "signs of wear, damage, or contamination",
            "hazard_assessment": "safety implications of texture issues"
        }
    ],
    "shape_deformations": [
        {
            "object": "object with shape issues",
            "deformation_description": "description of shape problems",
            "normal_shape": "expected shape for this object type",
            "structural_concerns": "safety implications of deformation",
            "potential_causes": "likely reasons for deformation"
        }
    ],
    "material_condition_issues": [
        {
            "object": "object with material/condition problems",
            "condition_description": "description of material condition",
            "deterioration_signs": "signs of degradation or damage",
            "safety_implications": "how condition affects safety",
            "urgency_level": "immediate/moderate/low concern level"
        }
    ],
    "visual_integrity_assessment": {
        "overall_condition": "general visual condition of the scene",
        "primary_visual_concerns": ["main visual anomalies identified"],
        "hazard_indicators": ["visual signs suggesting safety risks"],
        "condition_based_issues": ["problems related to object condition vs semantic inappropriateness"]
    },
    "visual_confidence": 0.0-1.0
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

class DirectAPIVisualAppearanceEvaluator:
    def __init__(self, delay_between_requests: float = 60.0):
        self.client = openai.OpenAI(api_key=openai.api_key)
        self.delay_between_requests = delay_between_requests
        
        print("🚀 AGENT 4: Visual Appearance Evaluator (Direct OpenAI API)")
        print("👁️ Detecting condition-based anomalies and visual irregularities")
        print("🎯 Focus on visual characteristics indicating potential hazards")
        print(f"⏱️ Rate limiting: {delay_between_requests}s between requests")
    
    def analyze_image(self, image_path: str, verbose: bool = True) -> Dict:
        """Analyzes visual appearance anomalies in an image - Direct API"""
        
        if verbose:
            print(f"🔍 Analyzing visual appearance: {image_path}")
        
        try:
            # Encode image
            base64_image = encode_image_optimized(image_path, max_size=512)
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": VISUAL_APPEARANCE_SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this road scene image for visual appearance anomalies and condition-based issues. Focus on color inconsistencies, texture irregularities, shape deformations, and material condition problems that could indicate hazards. Consider environmental factors affecting visual assessment. Provide detailed analysis in the specified JSON format."
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
                print("✅ Visual appearance analysis completed")
                
                # Print visual anomalies
                color_anomalies = parsed_result.get("color_anomalies", [])
                texture_issues = parsed_result.get("texture_irregularities", [])
                shape_issues = parsed_result.get("shape_deformations", [])
                
                total_issues = len(color_anomalies) + len(texture_issues) + len(shape_issues)
                if total_issues > 0:
                    print(f"  🔍 Found {total_issues} visual anomalies")
                    if color_anomalies:
                        print(f"    - Color issues: {len(color_anomalies)}")
                    if texture_issues:
                        print(f"    - Texture issues: {len(texture_issues)}")
                    if shape_issues:
                        print(f"    - Shape issues: {len(shape_issues)}")
            
            return parsed_result
            
        except Exception as e:
            print(f"❌ Visual appearance analysis failed: {e}")
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
    
    def process_batch(self, image_directory: str, output_file: str = "agent4_visual_appearance_results.json"):
        """Batch processing"""
        
        print(f"🚀 Starting Direct API Agent 4 batch processing...")
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
                "agent": "direct_api_visual_appearance_evaluator",
                "agent_number": 4,
                "processing_date": datetime.now().isoformat(),
                "source_directory": image_directory,
                "total_images": len(image_files),
                "model_used": "gpt-4o",
                "api_method": "direct_openai_api",
                "agent_role": "condition_based_anomaly_detection"
            },
            "results": []
        }
        
        # Analyze each image
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(image_directory, image_file)
            
            print(f"\n📸 Processing image {i+1}/{len(image_files)}: {image_file}")
            
            try:
                start_time = time.time()
                
                # Visual appearance analysis
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
                    "visual_appearance_analysis": analysis_result
                }
                
                batch_results["results"].append(image_result)
                print(f"  ✅ Success - Processing time: {processing_time:.2f}s")
                
                # Print result summary
                color_anomalies = analysis_result.get("color_anomalies", [])
                texture_issues = analysis_result.get("texture_irregularities", [])
                shape_issues = analysis_result.get("shape_deformations", [])
                
                total_issues = len(color_anomalies) + len(texture_issues) + len(shape_issues)
                if total_issues > 0:
                    print(f"  🔍 Visual issues: {total_issues} found")
                
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
            print(f"\n💾 Agent 4 results saved to: {output_file}")
        except Exception as e:
            print(f"❌ Failed to save results: {str(e)}")
        
        print(f"\n🎉 Agent 4 batch processing completed!")
        print(f"📊 Successfully processed: {len(batch_results['results'])} images")
        
        return batch_results

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent 4: Visual Appearance Evaluator")
    parser.add_argument("--image_directory", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the JSON output file")
    parser.add_argument("--delay_between_requests", type=float, default=60.0, help="Delay in seconds between API requests")
    
    args = parser.parse_args()

    print("=== AGENT 4: Direct API Visual Appearance Evaluator ===")
    
    evaluator = DirectAPIVisualAppearanceEvaluator(delay_between_requests=args.delay_between_requests)
    
    # Run batch processing
    results = evaluator.process_batch(
        image_directory=args.image_directory,
        output_file=args.output_file
    )
        
    print("\n✅ Agent 4 Complete!")
    print("📋 Next step: Run agent5.py")