# AI Dashboard Generator

Upload any CSV or Excel file and automatically generate a dashboard.

## Features
- Upload CSV or Excel files
- Automatic data profiling
- Automatic dashboard generation
- Prompt-based chart creation
- KPI cards, trends, comparisons, distributions, and correlations
- Works without an LLM
- Optional place to connect an LLM later

## Project structure

```text
ai_dashboard_generator/
├── app/
│   └── streamlit_app.py
├── core/
│   ├── config.py
│   ├── schemas.py
│   └── state.py
├── services/
│   ├── chart_builder.py
│   ├── dashboard_generator.py
│   ├── data_loader.py
│   ├── data_profiler.py
│   └── prompt_parser.py
├── utils/
│   ├── formatting.py
│   └── helpers.py
├── tests/
│   └── test_prompt_parser.py
├── requirements.txt
└── README.md
```

## Setup

```bash
cd ai_dashboard_generator
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Supported files
- `.csv`
- `.xlsx`
- `.xls`

## How it works
1. Reads the uploaded file into pandas.
2. Detects numeric, categorical, datetime, and ID-like columns.
3. Builds a profile summary.
4. Creates dashboard sections using rule-based logic.
5. Supports simple prompts like:
   - `show monthly sales`
   - `compare category by revenue`
   - `show distribution of age`
   - `show relation between profit and sales`

## Notes
- This is a clean starter MVP.
- You can later add FastAPI, authentication, saved dashboards, or an LLM layer.
- The prompt parser is intentionally safe and rule-based.
