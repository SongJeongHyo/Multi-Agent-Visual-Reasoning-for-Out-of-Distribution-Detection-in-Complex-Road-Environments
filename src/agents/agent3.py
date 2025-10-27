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

SEMANTIC_INCONSISTENCY_SYSTEM_PROMPT = """
You are a Semantic Inconsistency Analyzer focused on domain appropriateness and contextual fitness within road environments.
Your expertise lies in leveraging deep semantic understanding to identify objects that, while potentially normal in other contexts, are semantically inappropriate for road scenarios.
CORE RESPONSIBILITIES:
1. Evaluate whether objects belong in road environments based on traffic regulations
2. Assess safety considerations and common sense reasoning
3. Apply domain-specific knowledge about road usage appropriateness
4. Consider object functionality, potential safety hazards, and regulatory compliance
SEMANTIC REASONING FRAMEWORK:
- Evaluate semantic appropriateness within road domain constraints
- Identify objects normal elsewhere but inappropriate for roads
- Consider traffic regulations and safety implications
- Focus on contextual fitness rather than visual appearance
- Example: Outdoor furniture visually unremarkable but semantically inappropriate on roadways
DOMAIN KNOWLEDGE AREAS:
- Traffic regulations and road usage rules
- Vehicle types and their appropriate road contexts
- Pedestrian safety considerations
- Infrastructure appropriateness
- Emergency and service vehicle protocols
- Commercial vs residential road usage norms
OUTPUT FORMAT (JSON):
{
    "semantic_analysis": {
        "detected_objects": ["comprehensive list of all objects identified"],
        "object_categorization": {
            "road_appropriate": ["objects that belong in road environments"],
            "questionable": ["objects with unclear appropriateness"],
            "inappropriate": ["objects that shouldn't be in road contexts"]
        }
    },
    "domain_violations": [
        {
            "object": "specific inappropriate object",
            "violation_reason": "why this object is inappropriate for roads",
            "safety_concern": "specific safety issues this object creates",
            "regulatory_aspect": "relevant traffic regulations or rules",
            "alternative_context": "where this object would be appropriate"
        }
    ],
    "appropriateness_assessment": [
        {
            "object": "object being evaluated",
            "appropriateness_score": 0.0-1.0,
            "reasoning": "semantic reasoning for the score",
            "context_dependency": "how context affects appropriateness"
        }
    ],
    "safety_implications": {
        "immediate_hazards": ["objects creating immediate safety risks"],
        "regulatory_violations": ["objects violating traffic regulations"],
        "functional_conflicts": ["objects conflicting with road functionality"]
    },
    "semantic_reasoning": {
        "overall_assessment": "general semantic fitness of the scene",
        "primary_concerns": ["main semantic inconsistencies identified"],
        "context_considerations": "how road context affects object evaluation"
    },
    "semantic_confidence": 0.0-1.0
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

class DirectAPISemanticInconsistencyAnalyzer:
    def __init__(self, delay_between_requests: float = 60.0):
        self.client = openai.OpenAI(api_key=openai.api_key)
        self.delay_between_requests = delay_between_requests
        
        print("🚀 AGENT 3: Semantic Inconsistency Analyzer (Direct OpenAI API)")
        print("🧠 Evaluating domain appropriateness and contextual fitness")
        print("🎯 Focus on objects inappropriate for road environments")
        print(f"⏱️ Rate limiting: {delay_between_requests}s between requests")
    
    def analyze_image(self, image_path: str, verbose: bool = True) -> Dict:
        """Analyzes semantic inconsistencies in an image - Direct API"""
        
        if verbose:
            print(f"🔍 Analyzing semantic inconsistencies: {image_path}")
        
        try:
            # Encode image
            base64_image = encode_image_optimized(image_path, max_size=512)
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": SEMANTIC_INCONSISTENCY_SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this road scene image for semantic inconsistencies and domain appropriateness violations. Focus on objects that are inappropriate for road environments despite being normal elsewhere. Consider traffic regulations, safety implications, and functional conflicts. Provide detailed analysis in the specified JSON format."
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
                print("✅ Semantic inconsistency analysis completed")
                
                # Print inappropriate objects
                violations = parsed_result.get("domain_violations", [])
                if violations:
                    print(f"  🚫 Found {len(violations)} domain violations")
                    for violation in violations[:3]:  # Max 3
                        obj = violation.get("object", "Unknown")
                        reason = violation.get("violation_reason", "Unknown")
                        print(f"    - {obj}: {reason}")
                
                inappropriate = parsed_result.get("semantic_analysis", {}).get("object_categorization", {}).get("inappropriate", [])
                if inappropriate:
                    print(f"  ⚠️ Inappropriate objects: {', '.join(inappropriate[:5])}")
            
            return parsed_result
            
        except Exception as e:
            print(f"❌ Semantic inconsistency analysis failed: {e}")
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
    
    def process_batch(self, image_directory: str, output_file: str = "agent3_semantic_inconsistency_results.json"):
        """Batch processing"""
        
        print(f"🚀 Starting Direct API Agent 3 batch processing...")
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
                "agent": "direct_api_semantic_inconsistency_analyzer",
                "agent_number": 3,
                "processing_date": datetime.now().isoformat(),
                "source_directory": image_directory,
                "total_images": len(image_files),
                "model_used": "gpt-4o",
                "api_method": "direct_openai_api",
                "agent_role": "domain_appropriateness_evaluation"
            },
            "results": []
        }
        
        # Analyze each image
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(image_directory, image_file)
            
            print(f"\n📸 Processing image {i+1}/{len(image_files)}: {image_file}")
            
            try:
                start_time = time.time()
                
                # Semantic inconsistency analysis
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
                    "semantic_inconsistency_analysis": analysis_result
                }
                
                batch_results["results"].append(image_result)
                print(f"  ✅ Success - Processing time: {processing_time:.2f}s")
                
                # Print result summary
                violations = analysis_result.get("domain_violations", [])
                if violations:
                    print(f"  🚫 Domain violations: {len(violations)} found")
                
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
            print(f"\n💾 Agent 3 results saved to: {output_file}")
        except Exception as e:
            print(f"❌ Failed to save results: {str(e)}")
        
        print(f"\n🎉 Agent 3 batch processing completed!")
        print(f"📊 Successfully processed: {len(batch_results['results'])} images")
        
        return batch_results

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent 3: Semantic Inconsistency Analyzer")
    parser.add_argument("--image_directory", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the JSON output file")
    parser.add_argument("--delay_between_requests", type=float, default=60.0, help="Delay in seconds between API requests")
    
    args = parser.parse_args()

    print("=== AGENT 3: Direct API Semantic Inconsistency Analyzer ===")
    
    analyzer = DirectAPISemanticInconsistencyAnalyzer(delay_between_requests=args.delay_between_requests)
    
    # Run batch processing
    results = analyzer.process_batch(
        image_directory=args.image_directory,
        output_file=args.output_file
    )
    
    print("\n✅ Agent 3 Complete!")
    print("📋 Next step: Run agent4.py")