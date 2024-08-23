"""
This script automates the generation of a LaTeX Beamer presentation for a specified lesson
in a course, leveraging lesson readings, course objectives from the syllabus, and a previous
lesson's Beamer presentation as a template.

Workflow:
1. **Git Update Check**:

2. **Lesson Objectives Extraction**:

3. **Readings Aggregation**:

4. **Previous Lesson Integration**:

5. **Prompt Construction**:

6. **LaTeX Generation**:

7. **Saving the Output**:


The script is designed to be run in an environment where all dependencies are available and assumes
that the necessary files (readings, syllabus, previous lesson) are properly organized in directories
specified by environment variables. See Readme for assumed directory structure.


"""


# base libraries
import os
from pathlib import Path

# rag chain setup
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# self-defined utils
from BeamerBot.src_code.slide_pipeline_utils import (
    check_git_pull,
    extract_lesson_objectives, load_readings, load_beamer_presentation,
    clean_latex_content
)
from BeamerBot.src_code.slide_preamble import preamble

# env setup
from dotenv import load_dotenv
load_dotenv()

OPENAI_KEY = os.getenv('openai_key')
OPENAI_ORG = os.getenv('openai_org')

# Path definitions
readingDir = Path(os.getenv('readingsDir'))
slideDir = Path(os.getenv('slideDir'))
syllabus_path = Path(os.getenv('syllabus_path'))

# %%
# No more than three lessons at a time -- otherwise too much context for model
quiz_range = range(1,4)

all_readings = []
objectives = ['']
for lsn in quiz_range:
    inputDir = readingDir / f'L{lsn}/'
    # load readings from the lesson folder
    if os.path.exists(inputDir):
        for pdf_file in inputDir.iterdir():
            if pdf_file.suffix == '.pdf':
                readings_text = load_readings(pdf_file)
                all_readings.append(readings_text)

    objectives_text = extract_lesson_objectives(syllabus_path, lsn, only_current=True)
    objectives.extend(objectives_text)


combined_readings_text = "\n\n".join(all_readings)
objectives = "\n".join(objectives)



beamer_example = slideDir / f'L{max(quiz_range)}.tex'
beamer_output = slideDir / f'quiz_L{max(quiz_range)}.tex'

# load presentation from last lesson
prior_lesson = load_beamer_presentation(beamer_example)

# %%

prompt = f"""
 You are a LaTeX Beamer specialist and a political scientist with expertise in Amerian politics.
 You will be creating an in-class quiz for an undergraduate-level course on American politics.
 The quiz should align with the below lesson objectives.
 ---
 {{objectives}}
 ---
 Here are the texts the quiz will cover:
 ---
 {{information}}.
 ---
 Please provide 7 multiple choice questions from the readings.
 Questions should be drawn from the readings and align with the provided course objectives.

 Structure the quiz in a way that it can be put in a Beamer presentation.
 Each slide should have a title and a quiz question.
 The slide after the question should gray out the incorrect answers, with the correct answer in black.
 To ensure consistent slide formatting, an example presentation from the preceding lesson is included below.
 Here is the example presentation:
 ---
 {{last_presentation}}
 ---
 Your answer should be returned in valid LaTeX format.
 Begin your slides at point in the preamble where we call '\title'
 """

parser = StrOutputParser()

prompt_template = ChatPromptTemplate.from_template(prompt)


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=OPENAI_KEY,
    organization=OPENAI_ORG,
)

chain = prompt_template | llm | parser


# Generate Beamer slides
response = chain.invoke({"objectives": objectives_text,
                         "information": combined_readings_text,
                         "last_presentation": prior_lesson})


# Assuming `generated_latex` is your LaTeX content
cleaned_latex = clean_latex_content(response)

# Now concatenate the preamble with the cleaned LaTeX content
full_latex = preamble + "\n\n" + cleaned_latex

# %%
# Optionally, save the generated slides to a .tex file
with open(beamer_output, 'w', encoding='utf-8') as f:
    f.write(full_latex)
