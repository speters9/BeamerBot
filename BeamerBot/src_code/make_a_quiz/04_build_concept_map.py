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

import inflect
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
from wordcloud import WordCloud

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
    temperature=0,
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


def extract_relationships(text, objectives, prompt, parser=StrOutputParser()):
    combined_template = ChatPromptTemplate.from_template(prompt)
    chain = combined_template | llm | parser

    response = chain.invoke({'objectives': objectives,
                            'text': text})

    try:
        # Clean and parse the JSON output
        response_cleaned = response.replace("```json", "").replace("```", "")
        data = json.loads(response_cleaned)

        # Extract concepts and relationships
        relationships = [tuple(relationship) for relationship in data["relationships"]]

    except json.JSONDecodeError as e:
        print(f"Error parsing the response: {e}")
        return []

    return relationships


def extract_concepts_from_relationships(relationships):
    concepts = set()  # Use a set to avoid duplicates
    for concept1, _, concept2 in relationships:
        concepts.add(concept1)
        concepts.add(concept2)
    return list(concepts)


# %%
summary_prompt = """You are a political science professor specializing in American government.
                    You will be given a text and asked to summarize this text in light of your lesson objectives.
                    Your lesson objectives are: \n {objectives} \n
                    Summarize the following text: \n {text}"""


relationship_prompt = """You are a political science professor specializing in American government.
                        You are instructing an introductory undergraduate American government class.
                        You will be mapping relationships between the concepts this class addresses.
                        The objectives for this lesson are: \n {objectives} \n

                        From the following text for this lesson, extract the key concepts and the relationships between them.
                        Identify the key concepts and then explain how each relates to the others.
                        \n
                        {text}
                        \n

                        Extract the most important and generally applicable key concepts and themes from the following summary.
                        Focus on high-level concepts or overarching themes relevant to an undergraduate American Politics course and the lesson objectives.
                        Examples of such concepts might include things like "Separation of Powers", "Federalism", "Standing Armies", or "Representation".

                        Avoid overly specific or narrow topics.

                        Provide the relationships between each concept with the other discovered concepts in the format:
                            "relationships": [
                              ["Concept 1", "relationship_type", "Concept 2"],
                              ["Concept 1", "relationship_type", "Concept 3"],
                              ...
                            ]

                        If there is no meaningful relationship from the standpoint of lesson objectives and your expertise as a professor of American Government, \
                        return "None" in the "relationship_type" field.

                        Because you are comparing a lot of concepts, the json may be long. That's fine.

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
        relationships = extract_relationships(summary, lsn_objectives, relationship_prompt, parser=StrOutputParser())
        relationship_list.extend(relationships)

        concepts = extract_concepts_from_relationships(relationships)
        conceptlist.extend(concepts)


# %%

dataDir = projectDir / "BeamerBot/data"

with open(dataDir / 'conceptlist.json', 'w') as f:
    json.dump(conceptlist, f)

with open(dataDir / 'relationship_list.json', 'w') as f:
    json.dump(relationship_list, f)


# %%
dataDir = projectDir / "BeamerBot/data"

with open(dataDir / 'conceptlist.json', 'r') as f:
    conceptlist = json.load(f)

with open(dataDir / 'relationship_list.json', 'r') as f:
    relationship_list = json.load(f)


# %%

def normalize_concept(concept):
    p = inflect.engine()

    concept = concept.lower().strip().replace(" ", "_")  # Normalize case and spacing
    words = concept.split("_")
    normalized_words = [p.singular_noun(word) if p.singular_noun(word) else word for word in words]
    return "_".join(normalized_words)


def jaccard_similarity(concept1, concept2, threshold=0.9):
    set1 = set(concept1)
    set2 = set(concept2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    similarity = intersection / union
    return similarity >= threshold


def replace_similar_concepts(existing_concepts, new_concept):
    """
    Check if the new concept matches any existing concept by similarity.
    If a match is found, return the existing concept.
    If no match is found, return the new concept as is.
    """
    for existing_concept in existing_concepts:
        if jaccard_similarity(existing_concept, new_concept):
            return existing_concept
    return new_concept


def process_relationships(relationships):
    # Initialize a set to keep track of all unique concepts
    unique_concepts = set()
    processed_relationships = []

    for concept1, relationship, concept2 in relationships:
        # Normalize concepts
        concept1 = normalize_concept(concept1)
        concept2 = normalize_concept(concept2)

        # Replace similar concepts with existing ones
        concept1 = replace_similar_concepts(unique_concepts, concept1)
        concept2 = replace_similar_concepts(unique_concepts, concept2)

        # Add concepts to the unique set
        unique_concepts.add(concept1)
        unique_concepts.add(concept2)

        # Add the relationship to the processed list
        processed_relationships.append((concept1, relationship, concept2))

    return processed_relationships


def build_graph(relationships):
    # Initialize an undirected graph
    G = nx.Graph()

    processed_relationships = process_relationships(relationships)

    # Add nodes and edges from relationships
    for concept1, relationship, concept2 in processed_relationships:
        if relationship != "None":
            if G.has_edge(concept1, concept2):
                G[concept1][concept2]['weight'] += 1
            else:
                G.add_edge(concept1, concept2, weight=1)

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
nx.write_gexf(G, dataDir/"concept_graph_streamlined.gexf")


# %%
# Create a string with each concept repeated according to its frequency
concept_string = " ".join(conceptlist)

# Generate the word cloud
wordcloud = WordCloud(width=1500, height=1000, background_color='white', max_font_size=150, max_words=250).generate(concept_string)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
