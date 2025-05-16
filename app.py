import streamlit as st
import os
import json
import itertools
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# --- Modular logic ---
def get_openai_client():
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        st.error('OPENAI_API_KEY not set. Please set it in your environment or .env file.')
        st.stop()
    return OpenAI(api_key=api_key)

MODEL = "gpt-4.1"
MODEL_MINI = "gpt-4.1-mini"
TOOLS = [{"type": "web_search"}]
INCLUDE_RESULTS = ["web_search_call.results"]
DEVELOPER_MESSAGE = "You are a deep researcher"

# --- Step 1: Get topic ---
def get_topic():
    return st.text_input('Research topic:', key='topic').strip()

# --- Step 2: Clarifying questions ---
def get_clarifying_questions(client, topic):
    prompt = f"""
Ask 5 numbered clarifying questions to about: {topic}.
The goal of the questions is to understand the intended purpose of the research.
Reply only with the questions."""
    clarify = client.responses.create(
        model=MODEL_MINI,
        input=prompt,
        instructions=DEVELOPER_MESSAGE,
    )
    questions = clarify.output[0].content[0].text.split("\n")
    return questions

# --- Step 3: Collect answers ---
def get_answers(questions):
    answers = []
    for i, question in enumerate(questions):
        answer = st.text_input(question, key=f'answer_{i}')
        answers.append(answer)
    return answers

# --- Step 4: Generate goal and queries ---
def get_goal_and_queries(client, topic, answers, clarify_id):
    prompt_goals = f"""
Using the user answers {answers}, write a short goal sentence for the research about {topic}.
Output a JSON list of 5 web search queries that will reach it.
Format: {{\"goal\": \"...\", \"queries\": [\"q1\", ...]}}
"""
    goal_and_queries = client.responses.create(
        model=MODEL,
        previous_response_id=clarify_id,
        input=prompt_goals,
        instructions=DEVELOPER_MESSAGE,
    )
    plan = json.loads(goal_and_queries.output[0].content[0].text)
    return plan, goal_and_queries.id

# --- Step 5: Run web search ---
def run_search(client, q):
    resp = client.responses.create(
        model=MODEL,
        input=f"Search: {q}",
        tools=TOOLS,
        include=INCLUDE_RESULTS,
    )
    return {"query": q,
            "resp_id": resp.output[0].id,
            "research_output": resp.output[1].content[0].text}

# --- Step 6: Evaluate if goal is met ---
def evaluate(client, goal, collected):
    review = client.responses.create(
        model=MODEL,
        input=[
            {"role": "developer", "content": f"Research goal: {goal}"},
            {"role": "assistant", "content": json.dumps(collected)},
            {"role": "user", "content": "Does this information fully satisfy the goal? Answer Yes or No only."}
        ],
        instructions=DEVELOPER_MESSAGE,
    )
    return "yes" in review.output[0].content[0].text.lower()

# --- Step 7: Final synthesis ---
def synthesize(client, goal, collected):
    final = client.responses.create(
        model=MODEL,
        input=[
            {"role": "developer", "content": (f"Write a complete answer that meets the goal: {goal}. "
                         "Cite sources inline using [n] and append a reference "
                         "list mapping [n] to url.")},
            {"role": "assistant", "content": json.dumps(collected)},
        ],
        instructions=DEVELOPER_MESSAGE,
    )
    return final.output[0].content[0].text

# --- Streamlit UI ---
def main():
    st.title("Deep Research Clone")
    st.write("AI-powered research assistant with web search.")
    client = get_openai_client()

    # Step 1: Topic
    topic = get_topic()
    if not topic:
        st.info("Enter a research topic to begin.")
        return

    # Step 2: Clarifying questions
    if 'clarify' not in st.session_state or st.session_state.get('last_topic') != topic:
        questions = get_clarifying_questions(client, topic)
        st.session_state['clarify'] = questions
        st.session_state['last_topic'] = topic
    else:
        questions = st.session_state['clarify']

    # Step 3: Answers
    answers = get_answers(questions)
    if not all(answers):
        st.info("Please answer all clarifying questions.")
        return

    # Step 4: Goal and queries
    if 'goal_plan' not in st.session_state or st.session_state.get('last_answers') != answers:
        plan, clarify_id = get_goal_and_queries(client, topic, answers, None)
        st.session_state['goal_plan'] = plan
        st.session_state['clarify_id'] = clarify_id
        st.session_state['last_answers'] = answers
    else:
        plan = st.session_state['goal_plan']
        clarify_id = st.session_state['clarify_id']

    goal = plan["goal"]
    queries = plan["queries"]
    st.subheader("Research Goal")
    st.write(goal)
    st.subheader("Initial Search Queries")
    st.write(queries)

    # Step 5: Research loop
    if st.button("Run Research", key="run_research") or st.session_state.get('research_started'):
        st.session_state['research_started'] = True
        collected = st.session_state.get('collected', [])
        queries_to_run = queries if not collected else st.session_state.get('queries', queries)
        for q in queries_to_run:
            if not any(item['query'] == q for item in collected):
                with st.spinner(f"Searching: {q}"):
                    result = run_search(client, q)
                    collected.append(result)
                    st.session_state['collected'] = collected
                    st.session_state['queries'] = queries_to_run
        # Step 6: Evaluate
        if evaluate(client, goal, collected):
            st.success("Goal satisfied! Generating final report...")
            final_report = synthesize(client, goal, collected)
            st.markdown(final_report, unsafe_allow_html=True)
            st.session_state['research_started'] = False
            st.session_state['collected'] = []
        else:
            # Ask for more queries
            more = client.responses.create(
                model=MODEL,
                input=[
                    {"role": "assistant", "content": f"Current data: {json.dumps(collected)}"},
                    {"role": "user", "content": ("We still haven't met the goal. Give 5 more, new, high-value web search queries only as a JSON list.")}
                ],
                previous_response_id=clarify_id
            )
            queries = json.loads(more.output[0].content[0].text)
            st.session_state['queries'] = queries
            st.warning("More research needed. New queries generated. Click 'Run Research' again.")

if __name__ == "__main__":
    main() 