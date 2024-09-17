"""
Automates the generation of a LaTeX Beamer presentation for quiz questions based on lesson readings,
course objectives, and an example slide structure from a previous lesson. The script integrates various
components such as lesson objectives, readings, and previous lesson slides to produce a consistent and
formatted output.

Workflow:
1. **Lesson Objectives Extraction**: Retrieves the relevant course objectives from the syllabus for the specified lessons.
2. **Readings Aggregation**: Compiles the texts from the specified lesson readings into a single document.
3. **Previous Lesson Integration**: Loads the Beamer presentation from the previous lesson to maintain slide formatting consistency.
4. **Prompt Construction**: Prepares a detailed prompt for the language model, including objectives, readings, and the prior presentation structure.
5. **LaTeX Generation**: Generates quiz questions and their corresponding LaTeX Beamer slides, following the provided structure and formatting guidelines.
6. **Saving the Output**: Saves the generated LaTeX code to a specified file, ready for compilation into a Beamer presentation.

This script should be run in an environment with all necessary dependencies installed and assumes that
the necessary files (readings, syllabus, previous lesson) are organized in directories specified by
environment variables. Refer to the Readme for details on the expected directory structure.

"""


import json
# base libraries
import os
from collections import namedtuple
from pathlib import Path

import pandas as pd
# embedding check for similarity against true questions
import torch
from docx import Document
# env setup
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
# llm chain setup
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# self-defined utils
from BeamerBot.src_code.slide_pipeline_utils import (check_git_pull,
                                                     clean_latex_content,
                                                     extract_lesson_objectives,
                                                     load_beamer_presentation,
                                                     load_readings)
from BeamerBot.src_code.slide_preamble import preamble

load_dotenv()

OPENAI_KEY = os.getenv('openai_key')
OPENAI_ORG = os.getenv('openai_org')

# Path definitions
readingDir = Path(os.getenv('readingsDir'))
slideDir = Path(os.getenv('slideDir'))
syllabus_path = Path(os.getenv('syllabus_path'))

outputDir = syllabus_path.parent / "midterm"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%

prompt = """
     You are a political scientist with expertise in Amerian politics.
     You will be creating an in-class quiz for an undergraduate-level course on American politics.
     The quiz should align with the lesson objectives listed below.
     ---
     {objectives}
     ---
     Here are the texts the quiz will cover:
     ---
     {information}
     ---
     Please provide 10 questions from the readings.
     Questions should align with the provided course objectives.

     Create a variety of question types, including:
     - Multiple choice
     - True/False
     - Fill in the blank (short answer)

     Return a quiz in JSON format with a variety of question types. Structure your response with keys for each question type as follows:

    {{
      "multiple_choice": [
        {{
          "question": "Question text here",
          "A)": "Choice 1",
          "B)": "Choice 2",
          "C)": "Choice 3",
          "D)": "Choice 4",
          "correct_answer": "A, B, C, or D corresponding to the correct choice"
        }},
        ...
      ],

      "true_false": [
        {{
          "question": "True/False question text here",
          "A)": "True",
          "B)": "False",
          "C)": "",
          "D)": "",
          "correct_answer": "A or B"
        }},
        ...
      ],

      "fill_in_the_blank": [
        {{
          "question": "Question text here",
          "A)": "Choice 1",
          "B)": "Choice 2",
          "C)": "Choice 3",
          "D)": "Choice 4",
          "correct_answer": "Correct answer that completes the missing words"
        }},
        ...
      ]
    }}
    ---

    Your answer should be returned in the format specified above, with a mix of multiple choice, true/false, and fill-in-the-blank questions.
    Every question should always contain "A)", "B)", "C)", and "D)" options, even if they are left blank.

    ---

    Generated questions should differ from the current list of questions, located here:

    {midterm}

    ---

    Return in json-serializable format
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


# %%
# Define the step size (e.g., 3 lessons at a time)
step_size = 3
max_value = 15

quizrangelist = []
Quiz = namedtuple("Quiz", ["min", "max"])

# Create a list of ranges in steps of 3, covering from 1 to 15
quiz_ranges = [range(i, min(i + step_size, max_value + 1)) for i in range(1, max_value + 1, step_size)]

# Print out the ranges to see the result
for quiz_range in quiz_ranges:
    quizrangelist.append(Quiz(min(quiz_range), max(quiz_range)+1))  # +1 to account for range functionality


with open(str(outputDir/'midterm_mc.txt'), 'r') as ground_truth:
    midterm = json.load(ground_truth)

# %%
# No more than three lessons at a time -- otherwise too much context for model

responselist = []

for quiz in tqdm(quizrangelist):
    quiz_range = range(quiz.min, quiz.max)

    all_readings = []
    objectives = ['']
    for lsn in quiz_range:
        inputDir = readingDir / f'L{lsn}/'
        # load readings from the lesson folder
        if os.path.exists(inputDir):
            for pdf_file in inputDir.iterdir():
                if pdf_file.suffix in ['.pdf', '.txt']:
                    readings_text = load_readings(pdf_file)
                    all_readings.append(readings_text)

        objectives_text = extract_lesson_objectives(syllabus_path, lsn, only_current=True)
        objectives.extend(objectives_text)

    combined_readings_text = "\n\n".join(all_readings)
    objectives = "\n".join(objectives)

    # Generate Beamer slides
    response = chain.invoke({"objectives": objectives_text,
                             "information": combined_readings_text,
                             'midterm': midterm})

    # Step 1: Remove the code block indicators (```json and ``` at the end)
    response_cleaned = response.replace('```json\n', '').replace('\n```', '')
    quiz_questions = json.loads(response_cleaned)

    if isinstance(quiz_questions, dict):
        # Flatten the questions and add a 'type' key
        for question_type, questions in quiz_questions.items():
            for question in questions:
                # Add the question type as a key to each question
                question['type'] = question_type
                # Add the updated question to responselist
                responselist.append(question)
    else:
        # Step 2: Parse the JSON string
        responselist.extend(quiz_questions)


# %%


# %%


# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2').to(device)  # A lightweight model, adjust based on performance needs


# Function to extract questions into a single list
def extract_quiz_questions(main_quiz):
    main_quiz_questions = []
    for question in main_quiz:
        main_quiz_questions.append(question['question'])
    return main_quiz_questions


# Function to check similarity between generated questions and main quiz questions
def check_similarity(generated_questions, main_quiz_questions, device, threshold=0.7):
    flagged_questions = []

    # Get embeddings for the newly generated questions
    generated_embeddings = model.encode(generated_questions, convert_to_tensor=True, device=device)
    midterm_embeddings = model.encode(main_quiz_questions, convert_to_tensor=True, device=device)

    # Compare each generated question to the main quiz questions
    for i, gen_question in enumerate(generated_questions):
        cosine_scores = util.pytorch_cos_sim(generated_embeddings[i], midterm_embeddings)
        max_score = float(cosine_scores.max())

        if max_score > threshold:
            flagged_questions.append((gen_question, max_score))

    return flagged_questions


# Get embeddings for the main quiz questions
midterm_questions = extract_quiz_questions(midterm)
generated_questions = extract_quiz_questions(responselist)


bad_questions = check_similarity(generated_questions, midterm_questions, device)

# %%


# %%
def flag_similar_questions(responselist, bad_questions):
    for question in responselist:
        question['similar_to_midterm'] = False  # Default value
        question['similarity'] = None

        for bad_question, similarity in bad_questions:
            if question['question'] == bad_question:
                question['similar_to_midterm'] = True
                question['similarity'] = similarity
                break


# Flag the similar questions
flag_similar_questions(responselist, bad_questions)

# %%

# # save to csv
final_quiz = pd.DataFrame(responselist)

col_order = ['type', 'question', 'A)', 'B)', 'C)', 'D)', 'correct_answer', 'similar_to_midterm', 'similarity']

final_quiz = final_quiz[col_order]
final_quiz = final_quiz[~final_quiz['similar_to_midterm']].reset_index(drop=True)

final_quiz = final_quiz.drop(['similar_to_midterm', 'similarity'], axis=1)

final_quiz.to_excel(outputDir / "review_quiz.xlsx", index=False)
