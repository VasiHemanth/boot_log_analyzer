# UEFI Log Analyzer

A Python-based tool for analyzing UEFI boot logs using LangChain and Ollama. This project helps in processing and analyzing boot logs to extract meaningful insights about the boot process.

## Features

- Analyzes UEFI boot logs using AI-powered language models
- Supports multiple file formats (Text, PDF, CSV, Word documents, JSON)
- Generates detailed analysis reports
- Outputs results in multiple formats (JSON, CSV, and text reports)

## Project Structure

```
├── boot_log_analyzer.py    # Main analysis script
├── boot.log               # Sample boot log file
├── log_analyzer.py        # Core log analysis functionality
├── requirements.txt       # Project dependencies
└── analysis_output/       # Output directory for analysis results
    ├── boot_analysis.json
    ├── boot_entries.csv
    └── boot_report.txt
```

## Prerequisites

- Python 3.x
- Ollama (Local LLM server)
- Required Python packages (listed in requirements.txt)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/VasiHemanth/boot_log_analyzer.git
cd boot_log_analyzer
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Place your boot log file in the project directory
2. Run the analyzer:

```bash
python boot_log_analyzer.py
```

The analysis results will be generated in the `analysis_output` directory.

## Dependencies

Key dependencies include:

- langchain
- langchain-community
- langchain-core
- aiohttp
- httpx
- httpx-sse
- dataclasses-json

## Output Formats

The analyzer generates three types of output files:

- `boot_analysis.json`: Detailed JSON format analysis
- `boot_entries.csv`: Structured CSV format for boot entries
- `boot_report.txt`: Human-readable analysis report

## Contributing

Feel free to open issues or submit pull requests with improvements.

## License

[MIT License](LICENSE)

## Author

VasiHemanth

## Acknowledgments

- LangChain community
- Ollama project
