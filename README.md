# SQA (Search Quality Assurance)

## Overview
This is an end-to-end service that:
1. Generates trends
2. Fetches SRP data for each trend
3. Evaluates the data
4. Generates a combined report

### Workflow Summary
1. **Trend Aggregation**: Collect trending terms from multiple sources (NUWA, Google Trends, Yahoo Trends) and enrich them with contextual summaries using LLM analysis
2. **Search Data Fetching**: Query search APIs (WIZQA) to retrieve structured search results data for each trending term
3. **Multi-Modal Evaluation**: Run parallel quality assessments including content filtering, relevance checking, knowledge graph validation, and image analysis
4. **Report Generation**: Aggregate all identified issues into comprehensive CSV reports for review and action

## Project Structure

```
SQA/
├── .gitignore 
├── README.md            
├── .flake8                 # Linting configuration file
├── .env.example            # Example file for .env
├── env_settings.yaml       # Project environment and configuration settings
├── main.py                 # Main entry point for running the SQA pipeline              
├── requirements.txt        # Python dependencies
├── data/                   # Generated data during run
├── logs/                   # Log files for each run
├── reports/                # Output reports
├── resource/               # Static resources(prompts, blocklists, etc.)
├── service/                
│   ├── llm.py              # LLM wrapper
│   ├── evaluations/        # Evaluation modules (blocklist, relevance, etc.)
│   ├── fetchers/           # Data fetching modules (SRP, trends, etc.)
│   ├── processors/         # Data processing modules (filter, dedup, etc.)
│   └── utils/              # Utility modules (logging, read files, etc.)
```

### Component Breakdown 
Please refer to this [wiki page](https://git.ouryahoo.com/CAKE-AI/SQA.wiki.git) for details.

### Data Flow
<img width="1325" alt="Screenshot 2025-07-28 at 16 59 34" src="https://git.ouryahoo.com/CAKE-AI/SQA/assets/37836/5dc7306f-67fb-4dcf-acc9-49841f254a91">

### Key Features
- **Concurrent Processing:** Configurable parallel execution for improved performance
- **Multi-Source Validation:** Cross-references multiple trend sources for comprehensive coverage
- **LLM-Powered Analysis:** Uses GPT-4 for contextual understanding and image analysis
- **Modular Evaluation:** Independent quality checkers that can be enabled/disabled as needed
- **Comprehensive Logging:** Detailed execution logs for debugging and audit purposes

  

## Setup Instructions

1. **Clone the repository**

2. **Set up Python environment**

   [option 1: with pyenv]
   - Install [pyenv](https://github.com/pyenv/pyenv) if not already installed
   - Create a new Python environment (e.g. Python 3.10):
     ```sh
     pyenv install 3.10.13
     pyenv virtualenv 3.10.13 sqa-env
     pyenv activate sqa-env
     ```

   [option 2: with conda]
   - Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) if not already installed.
   - Create a new conda environment (e.g. Python 3.10):
     ```sh
     conda create -n sqa-env python=3.10
     conda activate sqa-env
     ```
3. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```

4. **API Keys & Configuration**
   - create a `.env` file
   - copy content from `.env.example` into the `.env`
   - put your keys there. The `.env` file will not get tracked by git.
   - external API endpoints or urls are configured in `env_settings.yaml`.

5. **Run the project**
   ```sh
   python main.py
   ```

## Usage

### Command-Line Options

The SQA pipeline supports the following command-line options:

#### Normal Execution (Default)
```sh
python main.py
```
Runs the complete pipeline:
1. Generates trending queries using the trend pipeline
2. Fetches WIZQA responses for each query
3. Runs all evaluations (blocklist, death check, relevance, KG mismatch, trending searches)
4. Generates combined reports

#### Override Mode
```sh
python main.py --override_trend
```
Skips the trend pipeline and reads queries from a CSV file instead:
1. **Reads queries from `resource/override.csv`**
2. Continues with WIZQA fetching and evaluations using the CSV queries
3. Generates reports as usual

**CSV File Requirements:**
- **Location**: `resource/override.csv`
- **Required column**: `queries` (containing search query strings)
- **Format**: Standard CSV with header row

**Example `resource/override.csv`:**
```csv
queries
"Taylor Swift concert"
"iPhone 16 release date"
"World Cup 2024"
"Stock market news"
```

**Error Handling:**
- If `resource/override.csv` doesn't exist, the script will show an error and exit
- If the CSV exists but lacks a `queries` column, an error will be displayed
- Malformed CSV files will result in clear error messages

## Logging

All actions, results, and errors are logged automatically during workflow execution. Log files are stored in the `logs/` directory. You can review these logs for troubleshooting, auditing, or monitoring purposes.

## Linting and Code Style

Project uses [flake8](https://flake8.pycqa.org/) for Python linting and code style checks. To check your code:

```sh
pip install flake8
flake8 . # for whole project
flake8 <file_name> # for specific file
```

## Collaboration & Git Workflow

To ensure smooth collaboration, please follow this simplified git workflow:

1. **Clone the project**
   ```sh
   git clone git@git.ouryahoo.com:CAKE-AI/SQA.git
   cd <project-directory>
   ```
2. **Create a branch**
   - For JIRA-related work, name your branch after the ticket (e.g., `JK/CAKE-6218`).
   - Each branch should relate to only one ticket.
   ```sh
   git branch JK/CAKE-6218
   git checkout JK/CAKE-6218
   ```
3. **Make changes on your branch**
   - Stage files individually for clarity:
     ```sh
     git add <file1> <file2>
     ```
   - Only use `git add .` if you are certain all untracked files should be included.
   - Commit with a clear message:
     ```sh
     git commit -m "Describe your changes"
     ```
4. **Keep your branch up to date**
   - Regularly rebase from `main`:
     ```sh
     git checkout main
     git pull
     git checkout CAKE-6218
     git rebase main
     ```
5. **Push your branch**
   - For the first push:
     ```sh
     git push -u origin CAKE-6218
     ```
   - For subsequent pushes:
     ```sh
     git push
     ```
6. **Open a Pull Request (PR) on GitHub**
   - Include a description, link to the JIRA ticket, and example output.
   - Request 2 reviewers.

