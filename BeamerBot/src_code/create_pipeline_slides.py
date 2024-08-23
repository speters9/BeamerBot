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
# Call the check_git_pull function at the start of the script
lesson_no = check_git_pull()

inputDir = readingDir / f'L{lesson_no}/'
beamer_example = slideDir / f'L{lesson_no-1}.tex'
beamer_output = slideDir / f'L{lesson_no}.tex'


# load syllabus objectives for the current lesson
objectives_text = extract_lesson_objectives(syllabus_path, lesson_no)

# load readings from the lesson folder
all_readings = []
for pdf_file in inputDir.iterdir():
    if pdf_file.suffix == '.pdf':
        readings_text = load_readings(pdf_file)
        all_readings.append(readings_text)

# Join all readings into one string
combined_readings_text = "\n\n".join(all_readings)

# load presentation from last lesson
prior_lesson = load_beamer_presentation(beamer_example)

# %%

prompt = f"""
 You are a LaTeX Beamer specialist and a political scientist with expertise in Amerian politics.
 You will be creating the content for a college-level lesson teaching about a series of texts.
 The lesson information should align with the below lesson objectives.
 To understand what has been taught and what will be taught, the lesson objectives
 for the preceding lesson and the next lesson are here. We are on lesson {lesson_no}.
 ---
 {{objectives}}
 ---
 Here are the texts to use for this lesson:
 ---
 {{information}}.
 ---
 Structure the information in a way that it can be put in a Beamer
 presentation. Each slide should have a title and content, with the content
 being points that work toward the lesson objectives. To help guide you,
 an example presentation from the preceding lesson is included below. The lessons should always include
 a slide placeholder for a student current event presentation after the title page,
 then move on to where we are in the course, what we did last, and what we'll be discussing that day.
 After that we should include a slide with an open-ended and thought-provoking discussion question relevant to the subject matter.
 One slide in the presentation should also include an exercise the students might engage in to help with their learning.
 The lesson shuld conclude with what we accomplished today and where we're going next (ie lesson {lesson_no+1})
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
    temperature=0.2,
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
