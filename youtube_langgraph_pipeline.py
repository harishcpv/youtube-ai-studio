import os
import warnings
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

# Suppress harmless warnings for clean console output
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure API Key is loaded
api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY is not set in the environment.")

# Helper to normalize LLM output
def parse_llm_response(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parsed = []
        for part in content:
            if isinstance(part, dict):
                parsed.append(part.get("text", ""))
            else:
                parsed.append(str(part))
        return "".join(parsed)
    return str(content)

# ----------------------------------------------------
# 1. State Definition
# ----------------------------------------------------
class AgentState(TypedDict):
    concept: str
    niche: str
    retrieved_policies: str
    policy_report: str
    policy_status: str  # "PASS" or "FLAGGED"
    metadata_output: str

# ----------------------------------------------------
# 2. Models Setup
# ----------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=api_key
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    google_api_key=api_key
)

# ----------------------------------------------------
# 3. Vector Store Setup (Local ChromaDB)
# ----------------------------------------------------
persist_dir = "./chroma_db_langchain"

if not os.path.exists(persist_dir):
    seed_policies = [
        Document(page_content="Vulgarity or heavy profanity within the first 7 to 15 seconds will trigger limited or no ad serving."),
        Document(page_content="Dangerous stunts, animal abuse or cruelty, unverified health claims, or consuming hazardous items violate YouTube Community Guidelines and risk removal."),
        Document(page_content="Content featuring reused video clips without original commentary or substantial educational transformation will be rejected from monetization.")
    ]
    vector_store = Chroma.from_documents(seed_policies, embeddings, persist_directory=persist_dir)
else:
    vector_store = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# ----------------------------------------------------
# 4. Graph Nodes
# ----------------------------------------------------
def retrieve_policies_node(state: AgentState):
    """Node 1: Retrieves matching policy excerpts from vector DB."""
    query = state["concept"]
    docs = retriever.invoke(query)
    combined_docs = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_policies": combined_docs}

def policy_auditor_node(state: AgentState):
    """Node 2 (Agent 1): Evaluates concept against YouTube policies."""
    prompt = ChatPromptTemplate.from_template("""
You are a YouTube Monetization and Policy Compliance Auditor.
Ground your evaluation strictly on the retrieved policies below:

--- POLICIES ---
{policies}
----------------

Evaluate this video concept:
"{concept}"

Provide:
1. Status: State either [PASS] or [FLAGGED]
2. Identified Risk Factors (if any)
3. Actionable Fixes (if flagged)
""")
    chain = prompt | llm
    response = chain.invoke({
        "policies": state["retrieved_policies"],
        "concept": state["concept"]
    })

    report_text = parse_llm_response(response.content)
    status = "PASS" if "[PASS]" in report_text else "FLAGGED"

    return {
        "policy_report": report_text,
        "policy_status": status
    }

def metadata_generator_node(state: AgentState):
    """Node 3 (Agent 2): Generates CTR metadata + Thumbnail concepts."""
    prompt = ChatPromptTemplate.from_template("""
You are a top-tier YouTube SEO Growth Strategist and Visual Creative Director.
Target Niche: {niche}

Generate high-performance metadata and visual thumbnail ideas for this approved concept:
"{concept}"

Please format your response into the following sections:

### 1. Viral Titles (< 60 chars)
- Provide 3 distinct options (curiosity-driven, clear value, high CTR).

### 2. High-CTR Thumbnail Concepts
- Concept A (Emotional/High Stakes): Subject framing, expression, background, and 2-3 bold overlay text words.
- Concept B (Action/Transformation): Before/After or step-in-action layout, focal point, and text hook.
- Ready-to-use AI Image Generator Prompt: A detailed text prompt (cinematic lighting, 16:9 ratio, photorealistic) suitable for Midjourney or Imagen.

### 3. SEO Description
- Hook in first 2 lines
- Keyword-rich summary paragraph
- Logical chapter timestamp placeholders

### 4. Targeted Hashtags
- 5-7 targeted hashtags (#)

### 5. Search Tags
- 15 comma-separated search tags
""")
    chain = prompt | llm
    response = chain.invoke({
        "niche": state["niche"],
        "concept": state["concept"]
    })

    metadata_text = parse_llm_response(response.content)
    return {"metadata_output": metadata_text}

# ----------------------------------------------------
# 5. Routing Logic
# ----------------------------------------------------
def policy_router(state: AgentState):
    """Routes execution forward if passed, or terminates if flagged."""
    if state["policy_status"] == "PASS":
        return "generate_metadata"
    return "flagged_exit"

# ----------------------------------------------------
# 6. Graph Compilation
# ----------------------------------------------------
workflow = StateGraph(AgentState)

workflow.add_node("retrieve_policies", retrieve_policies_node)
workflow.add_node("audit_policy", policy_auditor_node)
workflow.add_node("generate_metadata", metadata_generator_node)

workflow.add_edge(START, "retrieve_policies")
workflow.add_edge("retrieve_policies", "audit_policy")

workflow.add_conditional_edges(
    "audit_policy",
    policy_router,
    {
        "generate_metadata": "generate_metadata",
        "flagged_exit": END
    }
)
workflow.add_edge("generate_metadata", END)

app = workflow.compile()

# ----------------------------------------------------
# 7. Execution (Interactive)
# ----------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🎬 YOUTUBE MULTI-AGENT COMPLIANCE & SEO PIPELINE")
    print("=" * 60)

    user_concept = input("\nEnter your video concept or script:\n> ")
    user_niche = input("\nEnter your channel niche (e.g., Pets & Animals, Tech, Cooking):\n> ")

    test_input = {
        "concept": user_concept,
        "niche": user_niche
    }

    print("\n--- EXECUTING LANGGRAPH PIPELINE ---")
    output = app.invoke(test_input)

    print("\n" + "=" * 60)
    print("[POLICY AUDIT REPORT]:")
    print("=" * 60)
    print(output["policy_report"])

    if output["policy_status"] == "PASS":
        print("\n" + "=" * 60)
        print("[METADATA & THUMBNAIL OUTPUT]:")
        print("=" * 60)
        print(output["metadata_output"])
    else:
        print("\n[PIPELINE STOPPED]: Policy violations detected. Resolve fixes above before metadata generation.")