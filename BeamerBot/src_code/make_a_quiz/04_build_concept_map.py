"""
This script automates the generation of quiz questions for an undergraduate-level political science course.
The quiz is based on lesson objectives, lesson readings, and an existing midterm review, and it checks for
similarity to previous questions. The questions are generated using a language model and returned in multiple formats,
including multiple-choice, true/false, and fill-in-the-blank types.

Workflow:
1. **Data Loading**: Loads lesson readings, objectives, and the existing midterm questions.
2. **Quiz Generation**: Uses a pre-defined prompt to generate a variety of question types (multiple choice, true/false,
   fill-in-the-blank) based on the readings and objectives.
3. **Similarity Check**: Compares the generated questions to the existing midterm questions using sentence embeddings
   to ensure minimal overlap.
4. **Question Flagging**: Flags questions that are too similar to the midterm questions based on a similarity threshold.
5. **Saving the Output**: Outputs the final set of quiz questions in Excel format, excluding flagged questions.

Dependencies:
- This script requires access to an OpenAI API key for generating questions via a language model.
- Ensure the necessary environment variables (`openai_key`, `openai_org`, `syllabus_path`, `readingsDir`, etc.) are set.
- Torch and SentenceTransformer are used for similarity checking, while pandas is used for saving the output.

The expected input files (e.g., lesson readings and syllabus) are organized in directories specified by the environment
variables. The output is saved in an Excel file ready for review and further editing.
"""


# base libraries
import json
import os
from collections import namedtuple
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
# env setup
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
# llm chain setup
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# self-defined utils
from BeamerBot.src_code.slide_pipeline_utils import (extract_lesson_objectives,
                                                     load_readings)

load_dotenv()

OPENAI_KEY = os.getenv('openai_key')
OPENAI_ORG = os.getenv('openai_org')

# Path definitions
readingDir = Path(os.getenv('readingsDir'))
slideDir = Path(os.getenv('slideDir'))
syllabus_path = Path(os.getenv('syllabus_path'))

projectDir = Path(os.getenv('projectDir'))

# %%
parser = StrOutputParser()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=OPENAI_KEY,
    organization=OPENAI_ORG,
)


def summarize_text(text, prompt, objectives, parser=StrOutputParser()):
    summary_template = ChatPromptTemplate.from_template(prompt)
    chain = summary_template | llm | parser
    summary = chain.invoke({'text': text,
                            'objectives': objectives})

    return summary


def extract_concepts(summaries, prompt, parser=StrOutputParser()):
    concept_template = ChatPromptTemplate.from_template(prompt)
    chain = concept_template | llm | parser

    concepts = []

    for summary in summaries:
        try:
            concepts_extracted = chain.invoke({'summary': summary})
            concepts_cleaned = concepts_extracted.replace("```json", "").replace("```", "")
            concepts_dict = json.loads(concepts_cleaned)  # Wrap this part in try-except
            for k, v in concepts_dict.items():
                # concepts.append({k:v})
                concepts.append(k.lower())
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            continue

    return concepts


def map_relationships(concept_list, prompt, objectives, parser=StrOutputParser()):
    relationship_template = ChatPromptTemplate.from_template(prompt)
    chain = relationship_template | llm | parser
    relationships = []

    relationship_raw = chain.invoke({'concepts': concept_list,
                                     'objectives': objectives})

    # Clean the response and parse it into a Python dictionary
    try:
        relationships = relationship_raw.replace("```json", "").replace("```", "")
        relationships_list = json.loads(relationships)
        relationship_tuples = [tuple(relationship) for relationship in relationships_list]
    except json.JSONDecodeError as e:
        print(f"Error parsing the relationships JSON: {e}")
        relationship_tuples = []

    return relationship_tuples


# %%
summary_prompt = """You are a political science professor specializing in American government.
                    You will be given a text and asked to summarize this text in light of your lesson objectives.
                    Your lesson objectives are: \n {objectives} \n
                    Summarize the following text: \n {text}"""


# concept_prompt = """You are a political science professor specializing in American government.
#                     Extract key concepts and themes from the following summary: \n {summary}. \n
#                     Key concepts and themes should be judged according to their relevance to an undergraduate American Politics course.
#                     Return results in a valid json of type: "concept or theme": "description" """

concept_prompt = """You are a political science professor specializing in American government.
                    Extract the most important and generally applicable key concepts and themes from the following summary: \n {summary}. \n
                    Focus on high-level concepts or overarching themes relevant to an undergraduate American Politics course.
                    Examples of such concepts might include things like "Separation of Powers", "Federalism", "Standing Armies", or "Representation".

                    Avoid overly specific or narrow topics.

                    Return results in a valid json of type: "concept or theme": "description" """

relationship_prompt = """You are a political science professor specializing in American government.
                        You are instructing an introductory undergraduate American government class.
                        You will be mapping relationships between the concepts this class addresses.
                        The objectives for the lesson are: \n {objectives} \n

                        I will provide a list of concepts below from this lesson. Please analyze the relationships between these concepts. \n
                        {concepts} \n
                        For each concept, identify how it relates to each of the other concepts in this set.

                        Provide the relationships between these concepts in the format:
                            ["Concept 1", "relationship_type", "Concept 2"]

                        If there is no meaningful relationship from the standpoint of lesson objectives and your expertise as a professor of American Government, \
                            just return "None" in the "relationship_type" section".

                        Because you are comparing every concept to every other concept, the json may be long. That's fine.

                        Ensure results are returned in a valid json.
                        """

relationship_list = []
conceptlist = []

for lsn in range(1, 15):
    print(f"Extracting Lesson {lsn}")
    lsn_summaries = []
    readings = []
    objectives = ['']
    inputDir = readingDir / f'L{lsn}/'
    # load readings from the lesson folder
    if os.path.exists(inputDir):
        for file in inputDir.iterdir():
            if file.suffix in ['.pdf', '.txt']:
                readings_text = load_readings(file)
                readings.append(readings_text)

    if not readings:
        continue

    lsn_objectives = extract_lesson_objectives(syllabus_path, lsn, only_current=True)

    for reading in readings:
        summary = summarize_text(reading, prompt=summary_prompt, objectives=lsn_objectives)
        lsn_summaries.append(summary)

    concepts = extract_concepts(lsn_summaries, prompt=concept_prompt)
    conceptlist.extend(concepts)

    relationships = map_relationships(concepts, relationship_prompt, lsn_objectives)
    relationship_list.extend(relationships)


# %%

dataDir = projectDir / "BeamerBot/data"

with open(dataDir / 'conceptlist.json', 'w') as f:
    json.dump(conceptlist, f)

with open(dataDir / 'relationship_list.json', 'w') as f:
    json.dump(relationship_list, f)


# %%

def normalize_concept(concept):
    # Convert to lowercase, strip spaces, and standardize spacing
    cleaned_concept = '_'.join(concept.strip().lower().split())

    return cleaned_concept


def build_graph(relationships):
    # Initialize a directed graph
    G = nx.DiGraph()

    # Add nodes and edges from relationships, normalizing concepts
    for concept1, relationship, concept2 in relationships:
        concept1 = normalize_concept(concept1)
        concept2 = normalize_concept(concept2)

        if relationship != "None":
            # Add an edge, incrementing the weight if the edge already exists
            if G.has_edge(concept1, concept2):
                G[concept1][concept2]['weight'] += 1  # Increase the weight for multiple relationships
            else:
                G.add_edge(concept1, concept2, weight=1)  # Initialize weight as 1

    return G


def visualize_graph(G):
    # Calculate centrality (degree centrality is simple, but you can use betweenness too)
    centrality = nx.degree_centrality(G)

    # Create node sizes based on centrality
    node_size = [5000 * centrality[node] for node in G.nodes()]  # Scale node sizes

    # Get edge weights to adjust thickness
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

    # Spring layout (force-directed), positions most central nodes towards the center
    pos = nx.spring_layout(G, k=0.15, iterations=50)  # Tweak 'k' for spacing

    # Draw the graph with node sizes and edge thicknesses
    nx.draw(G, pos, with_labels=False, node_color='lightblue', node_size=node_size, font_size=10, font_weight='bold', edge_color='gray')

    # Draw node labels with individual font sizes
    for node in G.nodes():
        nx.draw_networkx_labels(G, pos, labels={node: node}, font_size=10 + 50 * centrality[node])  # Adjust size per node

    # Draw edges with thickness based on weight
    nx.draw_networkx_edges(G, pos, width=edge_weights)

    plt.show()


# Build the graph
G = build_graph(relationship_list)

# Visualize the graph
visualize_graph(G)

# %%
nx.write_gexf(G, dataDir/"concept_graph.gexf")
