import openai
import json
import os
import time
import argparse  
import sys
from typing import Dict, List
from datetime import datetime

openai.api_key = "YOUR_API_KEY_HERE"

REASONING_SYNTHESIZER_SYSTEM_PROMPT = """
You are a Reasoning Synthesizer responsible for integrating multi-agent findings into coherent final judgments and generating optimized prompts for GroundedSAM.
Your critical role is to coordinate findings from all analysis agents while preventing individual agent errors from propagating to final decisions.
CORE RESPONSIBILITIES:
1. Cross-validate findings from Scene Context, Spatial Anomaly, Semantic Inconsistency, and Visual Appearance agents
2. Resolve conflicts when agents provide contradictory assessments
3. Assign confidence scores based on agreement levels and agent reliability
4. Generate optimized prompts for GroundedSAM compatibility
INTEGRATION LOGIC:
- Multiple agents identifying same object → Higher confidence
- Agent disagreement → Apply reasoning to determine ground truth
- Consider agent reliability and expertise domains
- Focus on the most visually obvious anomaly
CRITICAL PROMPT REQUIREMENTS:
- V1: EXACTLY "adjective + noun" format (e.g., "scattered rocks", "blocking vehicle")
- V2: EXACTLY single noun (e.g., "rocks", "vehicle")
- Choose ONLY the TOP 1 most anomalous object
- Keep it simple and specific for GroundedSAM compatibility
PRIORITY ORDER FOR ANOMALIES:
1. Animals in road environments (sheep, cows, horses, etc.)
2. Vehicles in wrong positions
3. Obstacles blocking traffic
4. Other visual anomalies
AGENT EXPERTISE DOMAINS:
- Agent 1 (Scene Context): Environmental baselines and contextual normality
- Agent 2 (Spatial Anomaly): Positioning violations and traffic disruptions
- Agent 3 (Semantic Inconsistency): Domain appropriateness and regulatory compliance
- Agent 4 (Visual Appearance): Condition-based anomalies and visual irregularities
OUTPUT FORMAT (JSON):
{
    "agent_summary": {
        "scene_context_findings": "key findings from Agent 1",
        "spatial_anomaly_findings": "key findings from Agent 2", 
        "semantic_inconsistency_findings": "key findings from Agent 3",
        "visual_appearance_findings": "key findings from Agent 4"
    },
    "cross_validation": [
        {
            "object": "object identified by multiple agents",
            "supporting_agents": ["agents that identified this object"],
            "confidence_level": 0.0-1.0,
            "reasoning": "why this object has high confidence"
        }
    ],
    "conflict_resolution": [
        {
            "conflicting_object": "object with disagreement",
            "conflicting_assessments": "what different agents said",
            "resolution": "final judgment with reasoning",
            "resolution_confidence": 0.0-1.0
        }
    ],
    "anomaly_ranking": [
        {
            "rank": 1,
            "object": "most anomalous object",
            "confidence_score": 0.8-1.0,
            "supporting_evidence": "evidence from multiple agents",
            "anomaly_type": "spatial/semantic/visual/contextual",
            "priority_reasoning": "why this is the top anomaly"
        }
    ],
    "final_decision": {
        "most_obvious_anomaly": "the clearest anomaly based on all agent observations",
        "primary_anomaly_type": "spatial/semantic/visual/contextual",
        "detection_confidence": 0.0-1.0,
        "justification": "comprehensive reasoning for this choice"
    },
    "grounded_sam_prompts": {
        "prompt_v1": "single adjective + single noun only",
        "prompt_v2": "single noun only",
        "prompt_confidence": 0.0-1.0,
        "prompt_reasoning": "why these prompts will work best for GroundedSAM"
    },
    "synthesis_reasoning": {
        "integration_process": "how agent findings were combined",
        "confidence_factors": "what contributed to final confidence",
        "limitations": "any limitations or uncertainties in the analysis"
    },
    "overall_confidence": 0.0-1.0
}
"""

class DirectAPIReasoningSynthesizer:
    def __init__(self, delay_between_requests: float = 30.0):
        self.client = openai.OpenAI(api_key=openai.api_key)
        self.delay_between_requests = delay_between_requests
        
        print("🚀 AGENT 5: Reasoning Synthesizer (Direct OpenAI API)")
        print("🔗 Integrating multi-agent findings into coherent judgments")
        print("🎯 Generating optimized prompts for GroundedSAM")
        print(f"⏱️ Rate limiting: {delay_between_requests}s between requests")
    
    def synthesize_analysis(self, agent_results: Dict, verbose: bool = True) -> Dict:
        """Synthesize findings from 4 agents to create a final judgment and prompts - Direct API"""
        
        if verbose:
            print(f"🔗 Synthesizing findings from all agents...")
        
        try:
            # Extract results for each agent
            scene_context = agent_results.get('agent1_scene_context', {})
            spatial_anomaly = agent_results.get('agent2_spatial_anomaly', {})
            semantic_inconsistency = agent_results.get('agent3_semantic_inconsistency', {})
            visual_appearance = agent_results.get('agent4_visual_appearance', {})
            
            # Construct the synthesis prompt
            synthesis_prompt = f"""
            Synthesize the following multi-agent analysis results to determine the most likely OOD objects and generate optimal prompts for GroundedSAM.

            AGENT 1 - SCENE CONTEXT ANALYSIS:
            {json.dumps(scene_context, ensure_ascii=False, indent=2)}

            AGENT 2 - SPATIAL ANOMALY DETECTION:
            {json.dumps(spatial_anomaly, ensure_ascii=False, indent=2)}

            AGENT 3 - SEMANTIC INCONSISTENCY ANALYSIS:
            {json.dumps(semantic_inconsistency, ensure_ascii=False, indent=2)}

            AGENT 4 - VISUAL APPEARANCE EVALUATION:
            {json.dumps(visual_appearance, ensure_ascii=False, indent=2)}

            SYNTHESIS REQUIREMENTS:
            1. Cross-validate findings where multiple agents identify the same object
            2. Resolve conflicts between agent assessments using reasoning
            3. Identify the TOP 1 most obvious anomaly based on all evidence
            4. Generate V1 (adjective + noun) and V2 (noun only) prompts for GroundedSAM
            5. Prioritize: Animals > Misplaced vehicles > Traffic obstacles > Other anomalies

            CRITICAL PROMPT GUIDELINES:
            - V1 must be EXACTLY "adjective + noun" (e.g., "scattered sheep", "blocking truck")
            - V2 must be EXACTLY single noun (e.g., "sheep", "truck")
            - Keep prompts simple and specific for GroundedSAM compatibility
            - Choose only the most obvious anomaly

            Based on all agent findings, provide comprehensive synthesis in the specified JSON format.
            """
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": REASONING_SYNTHESIZER_SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": synthesis_prompt
                    }
                ],
                max_tokens=2000,
                temperature=0.1
            )
            
            # Parse response
            content = response.choices[0].message.content
            final_result = self._parse_json_response(content)
            
            if verbose:
                print("✅ Multi-agent synthesis completed successfully!")
                self._print_synthesis_summary(final_result)
            
            return final_result
            
        except Exception as e:
            print(f"❌ Reasoning synthesis failed: {e}")
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
    
    def _print_synthesis_summary(self, results: Dict):
        """Print a summary of the synthesis results"""
        print("\n" + "="*60)
        print("🎯 REASONING SYNTHESIS SUMMARY")
        print("="*60)
        
        # Final decision
        final_decision = results.get('final_decision', {})
        if 'most_obvious_anomaly' in final_decision:
            print(f"🔍 Most Obvious Anomaly: {final_decision['most_obvious_anomaly']}")
            print(f"📊 Detection Confidence: {final_decision.get('detection_confidence', 0.0):.2f}")
        
        # Generated prompts
        prompts = results.get('grounded_sam_prompts', {})
        if 'prompt_v1' in prompts and 'prompt_v2' in prompts:
            print(f"📝 Generated Prompts:")
            print(f"  V1 (adjective + noun): '{prompts['prompt_v1']}'")
            print(f"  V2 (noun only): '{prompts['prompt_v2']}'")
            print(f"  Prompt Confidence: {prompts.get('prompt_confidence', 0.0):.2f}")
        
        # Cross-validation
        cross_validation = results.get('cross_validation', [])
        if cross_validation:
            print(f"✅ Cross-Validated Objects: {len(cross_validation)} found")
            for validation in cross_validation[:3]:  # Max 3
                obj = validation.get('object', 'Unknown')
                agents = validation.get('supporting_agents', [])
                conf = validation.get('confidence_level', 0.0)
                print(f"  - {obj}: {len(agents)} agents, confidence {conf:.2f}")
        
        # Overall confidence
        overall_confidence = results.get('overall_confidence', 0.0)
        print(f"🎯 Overall Confidence: {overall_confidence:.2f}")
        print("="*60)
    
    def load_agent_results(self, agent1_file: str, agent2_file: str, agent3_file: str, agent4_file: str) -> Dict:
        """Load the result files from the 4 analysis agents"""
        
        agent_results = {}
        
        # Load Agent 1 results
        try:
            with open(agent1_file, 'r', encoding='utf-8') as f:
                agent1_data = json.load(f)
                agent_results['agent1_scene_context'] = agent1_data
                print(f"✅ Loaded Agent 1 results: {len(agent1_data.get('results', []))} images")
        except Exception as e:
            print(f"❌ Failed to load Agent 1 results: {e}")
            agent_results['agent1_scene_context'] = {"error": str(e)}
        
        # Load Agent 2 results
        try:
            with open(agent2_file, 'r', encoding='utf-8') as f:
                agent2_data = json.load(f)
                agent_results['agent2_spatial_anomaly'] = agent2_data
                print(f"✅ Loaded Agent 2 results: {len(agent2_data.get('results', []))} images")
        except Exception as e:
            print(f"❌ Failed to load Agent 2 results: {e}")
            agent_results['agent2_spatial_anomaly'] = {"error": str(e)}
        
        # Load Agent 3 results
        try:
            with open(agent3_file, 'r', encoding='utf-8') as f:
                agent3_data = json.load(f)
                agent_results['agent3_semantic_inconsistency'] = agent3_data
                print(f"✅ Loaded Agent 3 results: {len(agent3_data.get('results', []))} images")
        except Exception as e:
            print(f"❌ Failed to load Agent 3 results: {e}")
            agent_results['agent3_semantic_inconsistency'] = {"error": str(e)}
        
        # Load Agent 4 results
        try:
            with open(agent4_file, 'r', encoding='utf-8') as f:
                agent4_data = json.load(f)
                agent_results['agent4_visual_appearance'] = agent4_data
                print(f"✅ Loaded Agent 4 results: {len(agent4_data.get('results', []))} images")
        except Exception as e:
            print(f"❌ Failed to load Agent 4 results: {e}")
            agent_results['agent4_visual_appearance'] = {"error": str(e)}
        
        return agent_results
    
    def process_batch_synthesis(self, 
                              agent1_file: str, 
                              agent2_file: str, 
                              agent3_file: str, 
                              agent4_file: str,
                              output_file: str = "agent5_final_synthesis_results.json"):
        """Read results from 4 agents and perform batch synthesis - Direct API"""
        
        print(f"🚀 Starting Direct API Agent 5 batch synthesis...")
        print(f"📂 Loading results from 4 agents...")
        
        # Load all agent data
        all_agent_data = self.load_agent_results(agent1_file, agent2_file, agent3_file, agent4_file)
        
        # Match results by image
        agent1_results = all_agent_data.get('agent1_scene_context', {}).get('results', [])
        agent2_results = all_agent_data.get('agent2_spatial_anomaly', {}).get('results', [])
        agent3_results = all_agent_data.get('agent3_semantic_inconsistency', {}).get('results', [])
        agent4_results = all_agent_data.get('agent4_visual_appearance', {}).get('results', [])
        
        # Create dictionaries for matching (by filename)
        agent1_dict = {result['image_info']['filename']: result for result in agent1_results}
        agent2_dict = {result['image_info']['filename']: result for result in agent2_results}
        agent3_dict = {result['image_info']['filename']: result for result in agent3_results}
        agent4_dict = {result['image_info']['filename']: result for result in agent4_results}
        
        # Extract common image filenames
        all_filenames = set(agent1_dict.keys()) & set(agent2_dict.keys()) & set(agent3_dict.keys()) & set(agent4_dict.keys())
        
        if not all_filenames:
            print("❌ No common images found across all agents!")
            return {"error": "No common images found"}
        
        print(f"📋 Found {len(all_filenames)} common images to synthesize")
        
        # Store batch processing results
        batch_results = {
            "metadata": {
                "agent": "direct_api_reasoning_synthesizer",
                "agent_number": 5,
                "processing_date": datetime.now().isoformat(),
                "total_images": len(all_filenames),
                "model_used": "gpt-4o",
                "api_method": "direct_openai_api",
                "agent_role": "multi_agent_synthesis_and_prompt_generation",
                "source_files": {
                    "agent1_file": agent1_file,
                    "agent2_file": agent2_file,
                    "agent3_file": agent3_file,
                    "agent4_file": agent4_file
                }
            },
            "results": []
        }
        
        # Synthesize analysis for each image
        for i, filename in enumerate(sorted(all_filenames)):
            print(f"\n📸 Synthesizing image {i+1}/{len(all_filenames)}: {filename}")
            
            try:
                start_time = time.time()
                
                # Collect results for this image from all agents
                image_agent_results = {
                    'agent1_scene_context': agent1_dict[filename].get('scene_context_analysis', {}),
                    'agent2_spatial_anomaly': agent2_dict[filename].get('spatial_anomaly_analysis', {}),
                    'agent3_semantic_inconsistency': agent3_dict[filename].get('semantic_inconsistency_analysis', {}),
                    'agent4_visual_appearance': agent4_dict[filename].get('visual_appearance_analysis', {})
                }
                
                # Run synthesis
                synthesis_result = self.synthesize_analysis(image_agent_results, verbose=False)
                
                processing_time = time.time() - start_time
                
                # Delay to accommodate GPT-4o token limits
                time.sleep(self.delay_between_requests)
                
                if "error" not in synthesis_result:
                    # Construct final result
                    final_result = {
                        "image_info": agent1_dict[filename]['image_info'],  # Get image info from agent 1
                        "individual_agent_results": {
                            "scene_context": agent1_dict[filename].get('scene_context_analysis', {}),
                            "spatial_anomaly": agent2_dict[filename].get('spatial_anomaly_analysis', {}),
                            "semantic_inconsistency": agent3_dict[filename].get('semantic_inconsistency_analysis', {}),
                            "visual_appearance": agent4_dict[filename].get('visual_appearance_analysis', {})
                        },
                        "synthesis_result": synthesis_result,
                        "final_prompts": {
                            "prompt_v1": synthesis_result.get('grounded_sam_prompts', {}).get('prompt_v1', ''),
                            "prompt_v2": synthesis_result.get('grounded_sam_prompts', {}).get('prompt_v2', ''),
                            "overall_confidence": synthesis_result.get('overall_confidence', 0.0),
                            "detection_confidence": synthesis_result.get('final_decision', {}).get('detection_confidence', 0.0)
                        },
                        "processing_time_synthesis": processing_time
                    }
                    
                    batch_results["results"].append(final_result)
                    print(f"  ✅ Success - Processing time: {processing_time:.2f}s")
                    
                    # Print generated prompts
                    prompts = synthesis_result.get('grounded_sam_prompts', {})
                    print(f"  📝 V1: '{prompts.get('prompt_v1', 'N/A')}'")
                    print(f"  🎯 Confidence: {synthesis_result.get('overall_confidence', 0.0):.2f}")
                
                else:
                    print(f"  ❌ Failed: {synthesis_result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"  ❌ Exception occurred: {str(e)}")
        
        # Save results to JSON file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(batch_results, f, ensure_ascii=False, indent=2)
            print(f"\n💾 Agent 5 synthesis results saved to: {output_file}")
        except Exception as e:
            print(f"❌ Failed to save results: {str(e)}")
        
        print(f"\n🎉 Agent 5 batch synthesis completed!")
        print(f"📊 Successfully processed: {len(batch_results['results'])} images")
        print(f"🔗 All multi-agent analysis and prompt generation completed!")
        
        return batch_results

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent 5: Reasoning Synthesizer")
    parser.add_argument("--agent1_file", type=str, required=True, help="Path to agent 1 JSON output")
    parser.add_argument("--agent2_file", type=str, required=True, help="Path to agent 2 JSON output")
    parser.add_argument("--agent3_file", type=str, required=True, help="Path to agent 3 JSON output")
    parser.add_argument("--agent4_file", type=str, required=True, help="Path to agent 4 JSON output")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the final JSON output")
    parser.add_argument("--delay_between_requests", type=float, default=30.0, help="Delay in seconds between API requests")
    
    args = parser.parse_args()

    print("=== AGENT 5: Direct API Reasoning Synthesizer ===")
    
    synthesizer = DirectAPIReasoningSynthesizer(delay_between_requests=args.delay_between_requests)
    
    # Run batch synthesis
    results = synthesizer.process_batch_synthesis(
        agent1_file=args.agent1_file,
        agent2_file=args.agent2_file,
        agent3_file=args.agent3_file,
        agent4_file=args.agent4_file,
        output_file=args.output_file
    )
    
    print("\n✅ All Multi-Agent analysis complete!")
    print(f"📋 Final results: {args.output_file}")
    print("🎯 V1/V2 prompts for GroundedSAM have been generated!")