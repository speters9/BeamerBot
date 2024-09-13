# BeamerBot Slide Automation

This project automates the creation of LaTeX Beamer presentations. The script generates slide content for a specified lesson by leveraging the lesson readings, course objectives from the syllabus, and a previous lesson's Beamer presentation as a template.

It contains functionality to generate slides for an instructional presentation, or for a quiz based on course readings.

## Project Overview

The project contains a simple script with associated helper functions:
1. **Source Code**: This folder, named `src_code`, contains the scripts and utilities that automate the slide creation process.

## Workflow

The script follows a systematic process to generate the slides:

1. **Git Update Check**:
    - Ensures the previous lesson's slides are up-to-date (assumes lessons are version controlled with git in a separate repo linked to Overleaf or another Tex editor).
    - Once validated, the script collects the lesson number to be taught.
    - **Note**: All readings directories are assumed to be organized by lesson, and Beamer slides are assumed to be named by lesson. The assumed naming convention in all cases is `f"L{lesson_number}"`.

2. **Lesson Objectives Extraction**:
    - The course syllabus is parsed to extract learning objectives for the specified lesson, as well as for the lessons immediately preceding and following it. This ensures alignment with the course's educational goals.
    - The extraction process assumes the syllabus is in `.docx` format and is structured such that each lesson begins with `f"Lesson {lesson number}:"`.
    - The extraction function will also pull the lesson prior and the next lesson, to construct some "where we came from/where we're going" slides for intro and conclusion.

3. **Readings Aggregation**:
    - All PDF files related to the current lesson are read and their contents are combined into a single text string, which serves as the primary instructional content for the new lesson.
    - **Note**: The script does not perform OCR on PDFs, so it assumes that PDFs have been converted to a readable format.

4. **Previous Lesson Integration**:
    - The LaTeX source code from the previous lesson’s Beamer presentation is loaded to provide an example structure for the new slides.
    - **Note**: The input is LaTeX code, not a compiled PDF.

5. **Prompt Construction**:
    - A prompt is created that includes the lesson objectives, combined readings, and the previous lesson’s presentation. This prompt is passed through a Language Model (LLM) to generate new LaTeX slide content.
    - Prompt can of course vary, but current structure requests inclusion of a discussion question and class exercise, as well as basic intro/conclusion.
    - The current model used is GPT-4o-mini; this can be adjusted as desired, but amount of included context may be limited.

6. **LaTeX Generation**:
    - The generated LaTeX content is cleaned to remove any unnecessary formatting or extraneous characters, and then concatenated with a predefined LaTeX preamble to form a complete Beamer presentation.

7. **Saving the Output**:
    - The final LaTeX content is saved as a `.tex` file in the same directory as the example `.tex` file, ready for compilation into a PDF presentation.


## Dependencies

- **Python 3.x**
- **Langchain**
- **OpenAI API**
- **LaTeX (for Beamer presentation)**

### Python Packages:
- `langchain-openai`
- `langchain-core`
- `dotenv`
- `python-docx`
- `PyPDF2`

## Environment Setup

The script relies on environment variables to define paths and API keys. The `.env` file should include:

```bash
readingsDir=/path/to/readings
slideDir=/path/to/slides

syllabus_path=/path/to/syllabus.docx

openai_key=your_openai_api_key
openai_org=your_openai_organization_id
```

## Usage
- Run the Script:

  - Execute the script to generate the slides for the specified lesson.
  - The script will prompt you to confirm that you have pulled the latest changes for your slides (assuming you're working in Overleaf or some other LaTeX editor) and ask for the lesson number.
- Review and Edit:
  - The generated .tex file can be reviewed and edited in Overleaf or any LaTeX editor.

## License
- This project is licensed under the MIT License. See the LICENSE file for details.
