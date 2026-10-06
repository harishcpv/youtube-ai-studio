import os
import random
import re
import urllib.parse
import warnings
import pandas as pd
import requests
import streamlit as st
from typing import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from youtube_transcript_api import YouTubeTranscriptApi

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------
# Page Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="YouTube AI Studio (100% Free)",
    page_icon="🎬",
    layout="wide"
)

# ---------------------------------------------------------
# Check Gemini API Key
# ---------------------------------------------------------
gemini_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not gemini_key:
    st.error("🔑 Google Gemini API Key missing! Set GOOGLE_API_KEY in your environment.")
    st.stop()

# ---------------------------------------------------------
# YouTube Helper Functions
# ---------------------------------------------------------
def extract_video_id(url: str) -> str:
    """Extracts YouTube 11-character video ID from URLs."""
    pattern = r"(?:v=|\/|youtu\.be\/|embed\/)([0-9A-Za-z_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else url.strip()

def get_transcript(url: str) -> str:
    """Extracts transcript accommodating both older and newer library releases."""
    vid_id = extract_video_id(url)
    api_instance = YouTubeTranscriptApi() if callable(YouTubeTranscriptApi) else YouTubeTranscriptApi

    if hasattr(api_instance, "fetch"):
        try:
            data = api_instance.fetch(vid_id, languages=['en', 'en-US', 'te', 'hi'])
        except Exception:
            data = api_instance.fetch(vid_id)
        parts = [item.text if hasattr(item, "text") else item.get("text", "") for item in data]
        return " ".join(parts)
    elif hasattr(YouTubeTranscriptApi, "get_transcript"):
        try:
            data = YouTubeTranscriptApi.get_transcript(vid_id, languages=['en', 'en-US', 'te', 'hi'])
        except Exception:
            data = YouTubeTranscriptApi.get_transcript(vid_id)
        return " ".join([item['text'] for item in data])
    else:
        raise AttributeError("Could not find a valid method on YouTubeTranscriptApi.")

# ---------------------------------------------------------
# Robust Free Image Generator (No 402 Paywall)
# ---------------------------------------------------------
def generate_free_thumbnail_bytes(prompt: str, width: int = 1280, height: int = 720):
    """Generates an image via free public endpoints with automatic fallback."""
    clean = re.sub(r'[^a-zA-Z0-9 ,.-]', '', prompt)[:220].strip()
    enhanced_prompt = clean + ", cinematic studio lighting, vivid colors, 8k youtube thumbnail"
    encoded_prompt = urllib.parse.quote(enhanced_prompt)
    seed = random.randint(1000, 99999)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    }

    candidate_urls = [
        f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&model=turbo&nologo=true&seed={seed}",
        f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&nologo=true&seed={seed}",
    ]

    last_error = None
    for img_url in candidate_urls:
        try:
            response = requests.get(img_url, headers=headers, timeout=35)
            if response.status_code == 200 and len(response.content) > 1000:
                return response.content, img_url
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"All free generation routes were busy or restricted. ({last_error})")

# ---------------------------------------------------------
# LangGraph State & Agents
# ---------------------------------------------------------
class AgentState(TypedDict):
    concept: str
    niche: str
    retrieved_policies: str
    policy_report: str
    policy_status: str
    metadata_output: str
    image_prompt: str

@st.cache_resource
def build_agent_graph():
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=gemini_key
    )
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=gemini_key
    )

    persist_dir = "./chroma_db_langchain"
    if not os.path.exists(persist_dir):
        seed_policies = [
            Document(page_content="Vulgarity, hate speech, or heavy profanity within the first 7 to 15 seconds will trigger limited or no ad serving."),
            Document(page_content="Dangerous stunts, animal abuse or cruelty, unverified medical claims, or consuming hazardous items violate YouTube Community Guidelines and risk removal."),
            Document(page_content="Content featuring reused video clips without original commentary or substantial educational transformation will be rejected from monetization.")
        ]
        vector_store = Chroma.from_documents(seed_policies, embeddings, persist_directory=persist_dir)
    else:
        vector_store = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    def parse_llm_response(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parsed = [p.get("text", "") if isinstance(p, dict) else str(p) for p in content]
            return "".join(parsed)
        return str(content)

    def retrieve_policies_node(state: AgentState):
        query = state["concept"][:1000]
        docs = retriever.invoke(query)
        combined = "\n\n".join([d.page_content for d in docs])
        return {"retrieved_policies": combined}

    def policy_auditor_node(state: AgentState):
        prompt = ChatPromptTemplate.from_template(
            "You are a YouTube Monetization and Policy Compliance Auditor.\n"
            "Ground your evaluation strictly on the retrieved policies below:\n\n"
            "--- POLICIES ---\n"
            "{policies}\n"
            "----------------\n\n"
            "Evaluate this video transcript or concept:\n"
            "\"{concept}\"\n\n"
            "Provide:\n"
            "1. Status: State either [PASS] or [FLAGGED]\n"
            "2. Identified Risk Factors (if any)\n"
            "3. Actionable Fixes (if flagged)"
        )
        chain = prompt | llm
        response = chain.invoke({
            "policies": state["retrieved_policies"],
            "concept": state["concept"]
        })
        text = parse_llm_response(response.content)
        status = "PASS" if "[PASS]" in text else "FLAGGED"
        return {"policy_report": text, "policy_status": status}

    def metadata_generator_node(state: AgentState):
        prompt = ChatPromptTemplate.from_template(
            "You are a top-tier YouTube SEO Growth Strategist and Visual Creative Director.\n"
            "Target Niche: {niche}\n\n"
            "Generate high-performance metadata and a high-CTR visual thumbnail prompt based on this approved content:\n"
            "\"{concept}\"\n\n"
            "Provide your output formatted with these clear sections:\n\n"
            "### 1. Viral Titles (under 60 chars)\n"
            "- 3 distinct options (curiosity-driven, clear value, high CTR)\n\n"
            "### 2. High-CTR Telugu Hook Texts\n"
            "- 3 catchy Telugu text hooks (2-3 words each, e.g. సీక్రెట్ ట్రిక్! 🔥, ఇలా చేస్తేనే! 😱, షాకింగ్ రిజల్ట్! ⚡)\n\n"
            "### 3. Visual Thumbnail Prompt\n"
            "Write a photorealistic, high-energy scene description for the image generator (no text in image, expressive human or animal subjects, 16:9 cinematic framing, high contrast):\n"
            "THUMBNAIL_PROMPT: [Write your descriptive 16:9 image generation prompt here]\n\n"
            "### 4. SEO Description\n"
            "- Hook in first 2 lines\n"
            "- Keyword-rich summary paragraph\n"
            "- Chapter timestamp placeholders\n\n"
            "### 5. Targeted Hashtags & Tags\n"
            "- 5-7 targeted hashtags (#)\n"
            "- 15 comma-separated search tags"
        )
        chain = prompt | llm
        response = chain.invoke({
            "niche": state["niche"],
            "concept": state["concept"]
        })
        content_str = parse_llm_response(response.content)

        img_prompt_match = re.search(r"THUMBNAIL_PROMPT:\s*(.*)", content_str)
        clean_img_prompt = img_prompt_match.group(1).strip() if img_prompt_match else "Expressive close-up with vivid studio rim lighting, 16:9 youtube thumbnail"

        return {
            "metadata_output": content_str,
            "image_prompt": clean_img_prompt
        }

    def policy_router(state: AgentState):
        return "generate_metadata" if state["policy_status"] == "PASS" else "flagged_exit"

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve_policies", retrieve_policies_node)
    workflow.add_node("audit_policy", policy_auditor_node)
    workflow.add_node("generate_metadata", metadata_generator_node)

    workflow.add_edge(START, "retrieve_policies")
    workflow.add_edge("retrieve_policies", "audit_policy")
    workflow.add_conditional_edges(
        "audit_policy",
        policy_router,
        {"generate_metadata": "generate_metadata", "flagged_exit": END}
    )
    workflow.add_edge("generate_metadata", END)

    return workflow.compile()

app = build_agent_graph()

# ---------------------------------------------------------
# UI Layout
# ---------------------------------------------------------
st.title("🎬 YouTube AI Studio: Policy Audit + Free Thumbnail Maker")
st.caption("100% Free: Powered by Google Gemini & Open Turbo Engine")

with st.sidebar:
    st.header("⚙️ Configuration")
    input_choice = st.radio("Select Input Source:", ["YouTube Video Link (Unlisted/Public)", "Text Script / Concept Outline"])
    selected_niche = st.selectbox(
        "Target Niche:",
        ["Pets & Animals", "Gardening & DIY", "Tech & Gaming", "Cooking & Food", "Education & Training", "Vlogs & Lifestyle", "Other"]
    )
    if selected_niche == "Other":
        selected_niche = st.text_input("Enter Custom Niche:", "General")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📥 Input Video or Script")
    if input_choice == "YouTube Video Link (Unlisted/Public)":
        content_input = st.text_input("Paste Video Link:", placeholder="https://www.youtube.com/watch?v=... or https://youtu.be/...")
    else:
        content_input = st.text_area("Paste Script / Outline:", placeholder="Type or paste your video script...", height=240)

    submit_button = st.button("🚀 Run Policy Audit & Metadata Agents", type="primary", use_container_width=True)

with col_right:
    st.subheader("📊 Output & Generation")

    if submit_button:
        if not content_input.strip():
            st.warning("⚠️ Please provide a video URL or script text first.")
        else:
            final_text = ""
            with st.spinner("Extracting content..."):
                if input_choice == "YouTube Video Link (Unlisted/Public)":
                    try:
                        final_text = get_transcript(content_input)
                        st.info(f"✅ Extracted transcript ({len(final_text.split())} words).")
                    except Exception as err:
                        st.error(f"Could not load transcript: {err}")
                        st.stop()
                else:
                    final_text = content_input

            with st.spinner("Running LangGraph Agents (Audit & Strategy)..."):
                result = app.invoke({"concept": final_text, "niche": selected_niche})
                st.session_state["result"] = result

    # Display Results
    if "result" in st.session_state:
        res = st.session_state["result"]
        status = res.get("policy_status", "FLAGGED")

        if status == "PASS":
            st.success("🎉 **Policy Status: PASSED** — Safe for Monetization!")
        else:
            st.error("⚠️ **Policy Status: FLAGGED** — Violations Detected!")

        with st.expander("🔍 Policy Audit Details", expanded=(status == "FLAGGED")):
            st.markdown(res.get("policy_report", "No report available."))

        if status == "PASS":
            st.markdown("---")
            st.markdown(res.get("metadata_output", ""))

            # Free Thumbnail Generation
            st.markdown("---")
            st.subheader("🖼️ Free AI Thumbnail Generator (16:9)")
            img_prompt = res.get("image_prompt", "")
            
            prompt_input = st.text_area("Image Prompt (Edit if desired):", value=img_prompt, height=80)

            if st.button("🎨 Generate 16:9 Thumbnail (Free)", type="secondary"):
                with st.spinner("Rendering HD thumbnail... (Takes ~5-8 seconds)"):
                    try:
                        img_bytes, final_url = generate_free_thumbnail_bytes(prompt_input)
                        st.session_state["thumb_bytes"] = img_bytes
                        st.session_state["thumb_url"] = final_url
                    except Exception as e:
                        st.error(f"Image generation failed: {e}")

            if "thumb_bytes" in st.session_state:
                st.image(
                    st.session_state["thumb_bytes"], 
                    caption="Generated 16:9 YouTube Thumbnail (1280x720)", 
                    use_container_width=True
                )
                st.download_button(
                    label="💾 Download HD Thumbnail (.jpg)",
                    data=st.session_state["thumb_bytes"],
                    file_name="youtube_free_thumbnail.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )

            # Export Assets
            st.markdown("---")
            st.subheader("💾 Export Text Metadata")
            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    label="📄 Download All Text (.txt)",
                    data=res.get("metadata_output", ""),
                    file_name="youtube_seo_package.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with c2:
                df = pd.DataFrame({
                    "Field": ["Niche", "Policy Status", "Image Prompt"],
                    "Value": [selected_niche, status, img_prompt]
                })
                st.download_button(
                    label="📊 Download Summary (.csv)",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="youtube_seo_summary.csv",
                    mime="text/csv",
                    use_container_width=True
                )