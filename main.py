import asyncio
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from typing import List, Literal, Optional

import google.generativeai as genai
import pytz
from dotenv import load_dotenv
from magentic import Agent, Runner, function_tool
from magentic.chat_models import GeminiChatCompletionsModel
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

def parse_event(event):
    """Parses and prints stream events from the Runner."""
    GREEN = "\033[92m"
    RESET = "\033[0m"

    if event.type == "run_item_stream_event":
        if event.name == "llm_response_chunk":
            # This is streamed final answer tokens from Gemini
            print(f"{GREEN}{event.item.raw_item}{RESET}", end="", flush=True)
        elif event.name == "tool_called":
            print()
            print(f"{GREEN}> Tool Called: {event.item.raw_item.name}{RESET}")
            print(f"{GREEN}> Tool Args: {event.item.raw_item.arguments}{RESET}")
        elif event.name == "tool_output":
            print(f"{GREEN}> Tool Output: {event.item.raw_item['output']}{RESET}")
    
    elif event.type == "agent_updated_stream_event":
        print(f"{GREEN}> Current Agent: {event.new_agent.name}{RESET}")


@function_tool
def get_current_os():
    """
    Returns the name of the current operating system.
    Uses the platform module's system() function to get a standardized name.
    Returns:
        str: The name of the operating system (e.g., 'Linux', 'Windows', 'Darwin').
    """
    return platform.system()


@function_tool
def execute_terminal_command(command: str):
    """
    Executes a terminal command string using the system's shell.

    Args:
        command: The command string to execute (e.g., "ls -l", "grep 'pattern' file.txt").
                 NOTE: Using shell=True can be a security risk with untrusted input.

    Returns:
        A JSON string containing {"stdout": str, "stderr": str, "return_code": int}.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True
        )
        return json.dumps({"stdout": result.stdout, "stderr": result.stderr, "return_code": result.returncode})
    except FileNotFoundError:
        error_msg = f"Error: The shell or command '{command.split()[0] if command else ''}' was not found."
        print(error_msg, file=sys.stderr)
        return json.dumps({"stdout": "", "stderr": error_msg, "return_code": 127})
    except Exception as e:
        error_msg = f"An unexpected error occurred while executing '{command}': {e}"
        print(error_msg, file=sys.stderr)
        return json.dumps({"stdout": "", "stderr": error_msg, "return_code": -1})


class RequirementAnalysis(BaseModel):
    requirement_number: int
    requirement_specification: str
    relevant_available_files: List[str]
    requirement_satisfied_already: bool
    possible_to_satisfy_requirement: Optional[bool]
    plan_of_action: Optional[str]
    reasoning: str

class AllRequirementsAnalysis(BaseModel):
    all_requirements_analysis: List[RequirementAnalysis]
    can_satisfy_all_requirements: bool
    any_user_input_required: bool
    question_to_user: Optional[str] = Field(description="Ask additional inputs from the user if needed or if all requirements cannot be satisfied.")

class PlannerStep(BaseModel):
    step_number: int
    description: str
    terminal_command: str
    reasoning: str

class AllPlannerSteps(BaseModel):
    all_planner_steps: List[PlannerStep]
    can_satisfy_all_requirements: bool
    any_user_input_required: bool
    question_to_user: Optional[str] = Field(description="Ask additional inputs from the user if needed or if all requirements cannot be satisfied.")

class ExecutorStep(BaseModel):
    step_number: int
    terminal_command: str
    execution_status: Literal["Success", "Failure"]
    command_changed: bool
    updated_command: Optional[str]
    challenges_faced: Optional[str]

class AllExecutorSteps(BaseModel):
    all_executor_steps: List[ExecutorStep]
    can_satisfy_all_requirements: bool
    any_user_input_required: bool
    question_to_user: Optional[str] = Field(description="Ask additional inputs from the user if needed or if all requirements cannot be satisfied.")

class QCStep(BaseModel):
    requirement_number: int
    requirement_specification: str
    relevant_available_files: List[str]
    requirement_satisfied_successfully: bool
    reasoning: str

class AllQCSteps(BaseModel):
    all_qc_steps: List[QCStep]
    all_requirements_satisfied: bool
    any_user_input_required: bool
    question_to_user: Optional[str] = Field(description="Ask additional inputs from the user if needed or if all requirements cannot be satisfied.")


# Configure the Google Gemini API client
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
except Exception as e:
    print(f"Error configuring Gemini API: {e}\nPlease ensure your GEMINI_API_KEY is set in the .env file.")
    sys.exit(1)


ANALYZER_INSTRUCTIONS = """
Your goal is to thoroughly understand the client requirements and decide if they can be achieved or not with the available input files.
You are known for your exceptional ability to clearly understand the client's requirements and decide if those targets are achieveable using the available files and tools.
You will raise a flag if you find that we do not have the necessary files or tools to deliver the client's expectations.
"""

PLANNER_INSTRUCTIONS = """
You are an expert task planner.
You are known for your exceptional ability to understand the client requirements, the available inputs, the desired output and the available tools in hand to devise a detailed step-by-step plan.
Clearly understand the analysis client requirements, the analysis performed by the Client Requirements Analyst and the available input files and set out a detailed step by step plan to achieve the desired outcome.
"""

EXECUTOR_INSTRUCTIONS = """
You are an expert task executor. You are known for your exceptional ability to understand the given sequence of steps and execute them successfully using the available tools.
Execute the sequence of steps provided to you.
"""

QC_INSTRUCTIONS = """
You are an expert quality checker. You are known for your exceptional ability to understand the given client requirements and check if they have been satisfied or not.
Thoroughly understand the client requirements, compare it available files and check if all the requirements have been met or not.
You will raise a flag if you find that the client's expectations are not successfully met.
"""

# Define the Gemini model to be used by the agents
gemini_model = GeminiChatCompletionsModel("gemini-1.5-flash-latest")

analyzer_agent = Agent(
    name="Client Requirements Analyst",
    instructions=ANALYZER_INSTRUCTIONS,
    model=gemini_model,
    tools=[get_current_os, execute_terminal_command],
    output_type=AllRequirementsAnalysis
)

planner_agent = Agent(
    name="Senior Media Post-Production Task Planner",
    instructions=PLANNER_INSTRUCTIONS,
    model=gemini_model,
    tools=[get_current_os, execute_terminal_command],
    output_type=AllPlannerSteps
)

executor_agent = Agent(
    name="Senior Media Post-Production Client Delivery Expert",
    instructions=EXECUTOR_INSTRUCTIONS,
    model=gemini_model,
    tools=[get_current_os, execute_terminal_command],
    output_type=AllExecutorSteps
)

qc_agent = Agent(
    name="Senior Media Post-Production Quality Checker",
    instructions=QC_INSTRUCTIONS,
    model=gemini_model,
    tools=[get_current_os, execute_terminal_command],
    output_type=AllQCSteps
)


async def process_step(agent, messages):
    """Runs an agent, streams the output, and handles user interaction."""
    while True:
        result = Runner.run_streamed(agent, messages)

        async for event in result.stream_events():
            parse_event(event)

        if result.final_output.any_user_input_required:
            print(f"\n{result.last_agent.name} >> {result.final_output.question_to_user}")
            user_input = input("User Response >> ")
            messages = result.to_input_list()
            messages.append({'role': 'user', 'content': user_input})
        else:
            break
    
    # Add a newline for cleaner separation between steps
    print("\n")
    return result


async def main():
    ANALYZER_PROMPT = """
OVERALL WORKFLOW: Analysis (We are here now) --> Task Planning --> Execution --> Quality Check
INSTRUCTIONS:
We need to deliver the media package as per the specifications mentioned in the client's requirements document.
Analyse the client requirements thoroughly and compare it to the available input files.
Decide if it is possible to achieve the expectations using the available input files and tools.

AVAILABLE TOOLS: You can use FFMPEG, FFPROBE, ImageMagick and other terminal commands as required.
WORKING FOLDER: ./assets (All input files and requirements are in this folder)
IMPORTANT: You are only an analyst. Do not perform any actions or modify any files.
ASK QUESTIONS: If you need user input, please ask.
"""

    PLANNER_PROMPT = """
OVERALL WORKFLOW: Analysis --> Task Planning (We are here now) --> Execution --> Quality Check
INSTRUCTIONS:
Create a detailed step-by-step plan to achieve the desired output based on the previous analysis.
All tasks like transformation, extraction, renaming etc should be done in a separate sub-directory called 'temp'.
The 'temp' folder should be created in the './assets' folder. Creating this sub-directory should be the first step if it doesn't exist.
Each terminal command should be complete and self-contained.

CONTEXT: The output of the previous 'Analysis' step is attached above.
AVAILABLE TOOLS: You can use FFMPEG, FFPROBE, ImageMagick and other terminal commands.
WORKING FOLDER: ./assets
IMPORTANT: You are only a task planner. The actual execution will be performed later.
"""

    EXECUTOR_PROMPT = """
OVERALL WORKFLOW: Analysis --> Task Planning --> Execution (We are here now) --> Quality Check
INSTRUCTIONS:
You have been provided with a step-by-step plan. Execute the plan and record your observations.
If any step fails, analyse the reason, make necessary modifications to the command if needed, and try again until it succeeds.

CONTEXT: The output of the previous 'Analysis' and 'Task Planning' steps are attached.
AVAILABLE TOOLS: You can use FFMPEG, FFPROBE, ImageMagick and other terminal commands.
WORKING FOLDER: ./assets
IMPORTANT: Keep task executions safe and localized to the working folder.
"""

    QC_PROMPT = """
OVERALL WORKFLOW: Analysis --> Task Planning --> Execution --> Quality Check (We are here now)
INSTRUCTIONS:
Check if the available files (input files and generated files in the temp folder) satisfy the client's requirements.
You are allowed to make minor modifications if needed (like renaming files or cleaning up unnecessary files in the temp directory).

CONTEXT: The outputs of 'Analysis', 'Task Planning' and 'Execution' are attached.
AVAILABLE TOOLS: You can use FFMPEG, FFPROBE, ImageMagick and other terminal commands.
WORKING FOLDER: ./assets
IMPORTANT: Keep task executions safe and localized to the working folder.
"""

    messages = [{'role': 'user', 'content': ANALYZER_PROMPT}]

    # STEP 1: Analysis
    print("----- STARTING ANALYSIS -----")
    result_1 = await process_step(analyzer_agent, messages)
    output_1 = result_1.final_output.model_dump_json(indent=2)
    print(output_1)

    messages = result_1.to_input_list()
    messages.append({'role': 'user', 'content': PLANNER_PROMPT})

    # STEP 2: Planning
    print("----- STARTING PLANNING -----")
    result_2 = await process_step(planner_agent, messages)
    output_2 = result_2.final_output.model_dump_json(indent=2)
    print(output_2)

    messages = result_2.to_input_list()
    messages.append({'role': 'user', 'content': EXECUTOR_PROMPT})

    # STEP 3: Execution
    print("----- STARTING EXECUTION -----")
    result_3 = await process_step(executor_agent, messages)
    output_3 = result_3.final_output.model_dump_json(indent=2)
    print(output_3)

    messages = result_3.to_input_list()
    messages.append({'role': 'user', 'content': QC_PROMPT})

    # STEP 4: QC
    print("----- STARTING QUALITY CHECK -----")
    result_4 = await process_step(qc_agent, messages)
    output_4 = result_4.final_output.model_dump_json(indent=2)
    print(output_4)

    print("----- WORKFLOW COMPLETE -----")


if __name__ == "__main__":
    asyncio.run(main())
