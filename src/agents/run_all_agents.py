#!/usr/bin/env python3
"""
Master Script for Multi-Agent OOD Detection Framework (Direct API Version)

This script runs all 5 agents sequentially using Direct OpenAI API:
1. Agent 1: Scene Context Analyzer
2. Agent 2: Spatial Anomaly Detector
3. Agent 3: Semantic Inconsistency Analyzer
4. Agent 4: Visual Appearance Evaluator
5. Agent 5: Reasoning Synthesizer

Usage:
    python run_all_agents.py --image_dir /path/to/images --output_dir ./output --delay 60
"""

import os
import sys
import argparse
import time
from datetime import datetime
import sys

def run_agent(agent_script: str, description: str, args_str: str) -> bool:
    """Run an individual agent script with arguments"""
    print(f"\n{'='*80}")
    print(f"🚀 STARTING: {description}")
    print(f"📄 Script: {agent_script}")
    print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    # Execute the agent script with command-line arguments
    command = f"python {agent_script} {args_str}"
    print(f"Executing command: {command}")
    exit_code = os.system(command)
    
    end_time = time.time()
    duration = end_time - start_time
    
    if exit_code == 0:
        print(f"\n✅ COMPLETED: {description}")
        print(f"⏱️ Duration: {duration:.2f} seconds ({duration/60:.1f} minutes)")
    else:
        print(f"\n❌ FAILED: {description}")
        print(f"💥 Exit code: {exit_code}")
        return False
    
    return True

def check_openai_library():
    """Check if the OpenAI library is installed"""
    try:
        import openai
        print("✅ OpenAI library is available")
        return True
    except ImportError:
        print("❌ OpenAI library not found!")
        print("💡 Please install it: pip install openai")
        return False

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Run Multi-Agent OOD Detection Framework (Direct API)")
    parser.add_argument("--image_dir", 
                       required=True,
                       help="Directory containing input images")
    parser.add_argument("--output_dir", 
                       required=True,
                       help="Directory for output files")
    parser.add_argument("--delay", 
                       type=float, 
                       default=60.0,
                       help="Delay between API requests (seconds)")
    
    args = parser.parse_args()
    
    # Check for OpenAI library
    if not check_openai_library():
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🌟 MULTI-AGENT OOD DETECTION FRAMEWORK (DIRECT API)")
    print("=" * 80)
    print(f"📂 Input Directory: {args.image_dir}")
    print(f"📁 Output Directory: {args.output_dir}")
    print(f"⏱️ Request Delay: {args.delay} seconds")
    print(f"🔗 API Method: Direct OpenAI API")
    print(f"🕐 Total Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Define agent output filenames
    agent_outputs = {
        "agent1": "agent1_scene_context_results.json",
        "agent2": "agent2_spatial_anomaly_results.json",
        "agent3": "agent3_semantic_inconsistency_results.json",
        "agent4": "agent4_visual_appearance_results.json",
        "agent5": "agent5_final_synthesis_results.json"
    }
    
    # Define agent execution flow
    agents = [
        {"script": "agent1.py", "description": "Agent 1: Scene Context Analyzer"},
        {"script": "agent2.py", "description": "Agent 2: Spatial Anomaly Detector"},
        {"script": "agent3.py", "description": "Agent 3: Semantic Inconsistency Analyzer"},
        {"script": "agent4.py", "description": "Agent 4: Visual Appearance Evaluator"},
        {"script": "agent5.py", "description": "Agent 5: Reasoning Synthesizer"}
    ]
    
    total_start_time = time.time()
    successful_agents = []
    failed_agents = []
    
    # Run agents 1-4
    for i, agent in enumerate(agents[:4]): # Run first 4 agents
        agent_start_time = time.time()
        
        print(f"\n🔄 STEP {i+1}/5: Running {agent['description']}")
        
        # Prepare arguments for agent 1-4
        output_file = os.path.join(args.output_dir, agent_outputs[f"agent{i+1}"])
        agent_args = (
            f"--image_directory {args.image_dir} "
            f"--output_file {output_file} "
            f"--delay_between_requests {args.delay}"
        )
        
        success = run_agent(agent["script"], agent["description"], agent_args)
        
        agent_duration = time.time() - agent_start_time
        
        if success:
            successful_agents.append(agent["description"])
            # (File check logic removed for brevity, was good though)
        else:
            failed_agents.append(agent["description"])
            print(f"\n❌ {agent['description']} failed!")
            user_input = input("Continue with remaining agents? (y/n): ").lower()
            if user_input != 'y':
                print("🛑 Stopping execution due to agent failure.")
                sys.exit(1)
        
        # Wait before next agent
        if i < len(agents) - 1:
            print(f"\n⏳ Waiting 10 seconds before next agent...")
            time.sleep(10)
    
    # Run agent 5 (Synthesizer)
    if len(failed_agents) == 0:
        print(f"\n🔄 STEP 5/5: Running {agents[4]['description']}")
        agent_start_time = time.time()
        
        # Prepare arguments for agent 5
        agent1_file = os.path.join(args.output_dir, agent_outputs['agent1'])
        agent2_file = os.path.join(args.output_dir, agent_outputs['agent2'])
        agent3_file = os.path.join(args.output_dir, agent_outputs['agent3'])
        agent4_file = os.path.join(args.output_dir, agent_outputs['agent4'])
        output_file = os.path.join(args.output_dir, agent_outputs['agent5'])
        
        agent5_args = (
            f"--agent1_file {agent1_file} "
            f"--agent2_file {agent2_file} "
            f"--agent3_file {agent3_file} "
            f"--agent4_file {agent4_file} "
            f"--output_file {output_file} "
            f"--delay_between_requests {args.delay}"
        )
        
        success = run_agent(agents[4]["script"], agents[4]["description"], agent5_args)
        agent_duration = time.time() - agent_start_time
        
        if success:
            successful_agents.append(agents[4]["description"])
        else:
            failed_agents.append(agents[4]["description"])

    # Final summary
    total_duration = time.time() - total_start_time
    
    print(f"\n{'='*80}")
    print("🎊 MULTI-AGENT FRAMEWORK EXECUTION SUMMARY")
    print(f"{'='*80}")
    print(f"🕐 Total Execution Time: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)")
    print(f"✅ Successful Agents: {len(successful_agents)}/5")
    print(f"❌ Failed Agents: {len(failed_agents)}/5")
    
    if failed_agents:
        print(f"\n❌ FAILED AGENTS:")
        for agent_name in failed_agents:
            print(f"  - {agent_name}")
    
    final_output = os.path.join(args.output_dir, agent_outputs['agent5'])
    if os.path.exists(final_output):
        print(f"\n🎯 FINAL RESULTS AVAILABLE:")
        print(f"📄 File: {final_output}")
        print(f"🚀 Ready for GroundedSAM segmentation!")
    else:
        print(f"\n⚠️ Final synthesis results not found at: {final_output}")
    
    print(f"\n🏁 Framework execution completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n🛑 Execution interrupted by user")
    except Exception as e:
        print(f"\n\n💥 Unexpected error occurred: {str(e)}")
        sys.exit(1)