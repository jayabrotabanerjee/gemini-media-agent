# Gemini-Powered Media Processing Agent Workflow

This project demonstrates a multi-agent AI workflow for automating media post-production tasks using Google's Gemini Pro model. A sequence of four specialized AI agents collaborate to understand requirements, plan terminal commands, execute them, and perform quality control, mimicking a real-world production pipeline.

The entire workflow is orchestrated using Python, the `magentic` library for agent creation, and `pydantic` for reliable, structured data handling.

***

### Problem Statement

In media production, many tasks are repetitive yet require careful execution (e.g., transcoding videos, creating clips, generating thumbnails, renaming files). This project automates this process. Given a set of source files and a text document with client requirements, the AI agents will:
1.  Analyze the feasibility of the requirements.
2.  Create a step-by-step plan using command-line tools like **FFmpeg**.
3.  Execute the plan, handling errors and retries.
4.  Verify that the final output meets all client specifications.

***

### Technology Stack

* **AI Model**: Google Gemini 1.5 Flash
* **Python**: 3.8+
* **Core Libraries**:
    * `magentic`: A library for seamlessly integrating LLMs into Python code.
    * `google-generativeai`: The official Python SDK for the Gemini API.
    * `pydantic`: For data validation and structured outputs from the LLM.
* **External Tools**: `ffmpeg` (must be installed separately).

***

### Project Structure

```
gemini-media-agent-workflow/
├── .gitignore
├── main.py
├── requirements.txt
├── .env.example
└── assets/
    ├── client_requirements.txt  (Your instructions go here)
    └── input_video.mp4          (Your media files go here)
```

***

### Setup and Installation

#### 1. Prerequisites
- Python 3.8+
- [FFmpeg](https://ffmpeg.org/download.html): You must install FFmpeg on your system and ensure it's accessible from your command line (added to your system's PATH).

#### 2. Clone the Repository
```bash
git clone [https://github.com/your-username/gemini-media-agent-workflow.git](https://github.com/your-username/gemini-media-agent-workflow.git)
cd gemini-media-agent-workflow
```

#### 3. Set up a Virtual Environment (optional)

```bash
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### 4. Install Dependencies
Install all the required Python packages from `requirements.txt`.
```bash
pip install -r requirements.txt
```

#### 5. Configure API Key
- Rename the `.env.example` file to `.env`.
- Open the `.env` file and add your Google AI API key. You can get a key from [Google AI Studio](https://aistudio.google.com/app/apikey).
```env
# .env
GEMINI_API_KEY="your_google_ai_studio_api_key"
```

***

### How to Run

1.  Add your source media files (e.g., `input_video.mp4`) to the `assets/` folder.
2.  Edit the `assets/client_requirements.txt` file to specify the tasks you want the agents to perform.
3.  Run the script from your terminal:
    ```bash
    python main.py
    ```
