import os
import io
import re
import base64
import warnings
import pandas as pd
import requests
import streamlit as st
import numpy as np
import cv2
import yt_dlp
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
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
    page_title="YouTube AI Studio & Canva Thumbnail Engine",
    page_icon="🎬",
    layout="wide"
)

gemini_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
canva_token = os.environ.get("CANVA_API_TOKEN", "")

if not gemini_key:
    st.error("🔑 Google Gemini API Key missing! Set GOOGLE_API_KEY in your environment.")
    st.stop()

# ---------------------------------------------------------
# YouTube Helpers & Frame Extractor
# ---------------------------------------------------------
def extract_video_id(url: str) -> str:
    pattern = r"(?:v=|\/|youtu\.be\/|embed\/)([0-9A-Za-z_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else url.strip()

def get_transcript(url: str) -> str:
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

def extract_action_frames(youtube_url: str, num_frames: int = 4):
    """Streams video stream directly and extracts evenly spaced crisp frames."""
    ydl_opts = {
        'format': 'best[height<=1080]/best',
        'quiet': True,
        'no_warnings': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        stream_url = info['url']

    cap = cv2.VideoCapture(stream_url)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    extracted = []
    # Sample between 12% and 85% into footage to skip intro/outro screens
    sample_points = np.linspace(total_frames * 0.12, total_frames * 0.85, num_frames, dtype=int)

    for idx in sample_points:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            extracted.append(img)

    cap.release()
    return extracted

# ---------------------------------------------------------
# Canva Connect API Integration
# ---------------------------------------------------------
def create_canva_design_with_asset(image: Image.Image, token: str, design_title: str = "YouTube Thumbnail"):
    """
    1. Uploads the image (video frame or photo) directly to the user's Canva Asset Library.
    2. Creates a 1280x720 YouTube Thumbnail design in Canva.
    Returns the Canva direct edit URL and asset details.
    """
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    # Step 1: Upload Asset to Canva
    upload_url = "https://api.canva.com/rest/v1/asset-uploads"
    name_encoded = base64.b64encode(design_title.encode("utf-8")).decode("utf-8")
    upload_headers = {
        "Authorization": f"Bearer {token}",
        "Asset-Upload-Metadata": f'{{"name_base64": "{name_encoded}"}}'
    }
    files = {"file": ("thumbnail_asset.png", buf, "image/png")}

    up_res = requests.post(upload_url, headers=upload_headers, files=files)
    if up_res.status_code not in [200, 201]:
        raise RuntimeError(f"Canva Asset Upload Failed ({up_res.status_code}): {up_res.text}")
    
    asset_data = up_res.json()
    asset_id = asset_data.get("asset", {}).get("id")

    # Step 2: Create Canva 1280x720 Thumbnail Design
    design_url = "https://api.canva.com/rest/v1/designs"
    design_headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "design_type": {
            "type": "preset",
            "name": "youtube_thumbnail"
        },
        "title": design_title
    }

    des_res = requests.post(design_url, headers=design_headers, json=payload)
    if des_res.status_code not in [200, 201]:
        # Fallback to direct Canva edit link if scope differs
        edit_url = f"https://www.canva.com/create/youtube-thumbnails/"
    else:
        edit_url = des_res.json().get("design", {}).get("urls", {}).get("edit_url", "https://www.canva.com/create/youtube-thumbnails/")

    return edit_url, asset_id

# ---------------------------------------------------------
# High-CTR Local Rendering Engine (Mobile Pop & Telugu Font)
# ---------------------------------------------------------
def apply_ctr_color_grade(pil_img: Image.Image) -> Image.Image:
    w, h = pil_img.size
    target_ratio = 16 / 9
    current_ratio = w / h

    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        offset = (w - new_w) // 2
        pil_img = pil_img.crop((offset, 0, offset + new_w, h))
    elif current_ratio < target_ratio:
        new_h = int(w / target_ratio)
        offset = (h - new_h) // 2
        pil_img = pil_img.crop((0, offset, w, offset + new_h))

    img = pil_img.resize((1280, 720), Image.Resampling.LANCZOS)
    img = ImageEnhance.Color(img).enhance(1.28)
    img = ImageEnhance.Contrast(img).enhance(1.22)
    img = ImageEnhance.Sharpness(img).enhance(1.35)
    return img

def render_telugu_thumbnail(
    base_img: Image.Image,
    telugu_text: str,
    text_color: str = "#FFEE00",
    stroke_color: str = "#000000",
    stroke_width: int = 8,
    font_size: int = 80,
    position: str = "Top Left"
) -> Image.Image:
    img = apply_ctr_color_grade(base_img.copy())
    draw = ImageDraw.Draw(img)

    font_paths = [
        "C:/Windows/Fonts/NirmalaB.ttf",
        "C:/Windows/Fonts/Nirmala.ttf",
        "C:/Windows/Fonts/gautamib.ttf",
        "C:/Windows/Fonts/gautami.ttf"
    ]
    font = None
    for path in font_paths:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, font_size)
                break
            except Exception:
                continue

    if font is None:
        font = ImageFont.load_default()

    img_w, img_h = img.size
    if position == "Top Left":
        x, y = int(img_w * 0.05), int(img_h * 0.08)
    elif position == "Bottom Left":
        x, y = int(img_w * 0.05), int(img_h * 0.72)
    elif position == "Top Right":
        x, y = int(img_w * 0.48), int(img_h * 0.08)
    else:
        x, y = int(img_w * 0.15), int(img_h * 0.40)

    bbox = draw.textbbox((x, y), telugu_text, font=font, stroke_width=stroke_width)
    padding = 16
    draw.rounded_rectangle(
        [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding],
        radius=14,
        fill=(0, 0, 0, 185)
    )

    draw.text(
        (x, y),
        telugu_text,
        font=font,
        fill=text_color,
        stroke_width=stroke_width,
        stroke_fill=stroke_color
    )
    return img

# ---------------------------------------------------------
# LangGraph Workflow (Policy Audit + SEO + Telugu Hooks)
# ---------------------------------------------------------
class AgentState(TypedDict):
    concept: str
    niche: str
    retrieved_policies: str
    policy_report: str
    policy_status: str
    metadata_output: str
    telugu_hooks: list[str]

@st.cache_resource
def build_agent_graph():
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", google_api_key=gemini_key)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2", google_api_key=gemini_key)

    persist_dir = "./chroma_db_langchain"
    if not os.path.exists(persist_dir):
        seed_policies = [
            Document(page_content="Vulgarity or heavy profanity within the first 7 to 15 seconds will trigger limited ad serving."),
            Document(page_content="Dangerous stunts, animal cruelty, or unverified claims violate guidelines and risk removal."),
            Document(page_content="Reused content without commentary or transformation will be rejected from monetization.")
        ]
        vector_store = Chroma.from_documents(seed_policies, embeddings, persist_directory=persist_dir)
    else:
        vector_store = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    def parse_llm_response(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in content])
        return str(content)

    def retrieve_policies_node(state: AgentState):
        query = state["concept"][:1000]
        docs = retriever.invoke(query)
        return {"retrieved_policies": "\n\n".join([d.page_content for d in docs])}

    def policy_auditor_node(state: AgentState):
        prompt = ChatPromptTemplate.from_template(
            "You are a YouTube Policy Auditor. Ground your evaluation on the policies:\n"
            "{policies}\n\nEvaluate:\n\"{concept}\"\n\n"
            "Provide:\n1. Status: [PASS] or [FLAGGED]\n2. Identified Risk Factors\n3. Actionable Fixes"
        )
        chain = prompt | llm
        res = chain.invoke({"policies": state["retrieved_policies"], "concept": state["concept"]})
        text = parse_llm_response(res.content)
        status = "PASS" if "[PASS]" in text else "FLAGGED"
        return {"policy_report": text, "policy_status": status}

    def metadata_generator_node(state: AgentState):
        prompt = ChatPromptTemplate.from_template(
            "You are a YouTube SEO Growth Strategist & Telugu Visual Creative Director.\n"
            "Target Niche: {niche}\n\n"
            "Based on this content:\n\"{concept}\"\n\n"
            "Generate:\n"
            "1. Three high-CTR viral titles (< 60 chars)\n"
            "2. Three high-impact 2-to-3 word Telugu hook texts for the Canva thumbnail. Must create intense curiosity (e.g. ఫుడ్ అగ్రెషన్ ఆపండి!, ఇలా అస్సలు చేయకండి!, సీక్రెట్ ట్రిక్!).\n"
            "Format after 'TELUGU_HOOKS:' with each on a new bullet line.\n"
            "3. SEO Description (Hook, summary, timestamps)\n"
            "4. 15 search tags\n\n"
            "Output format:\n"
            "### 1. Viral Titles\n...\n\n"
            "### 2. High-CTR Telugu Hook Texts\n"
            "TELUGU_HOOKS:\n"
            "- Hook 1\n"
            "- Hook 2\n"
            "- Hook 3\n\n"
            "### 3. SEO Description\n...\n\n"
            "### 4. Targeted Tags\n..."
        )
        chain = prompt | llm
        res = chain.invoke({"niche": state["niche"], "concept": state["concept"]})
        text = parse_llm_response(res.content)

        hooks = []
        hook_section = re.search(r"TELUGU_HOOKS:\s*(.*?)(?=###|\Z)", text, re.DOTALL)
        if hook_section:
            lines = hook_section.group(1).strip().split("\n")
            for line in lines:
                clean = re.sub(r"^[-*0-9.]\s*", "", line).strip()
                if clean:
                    hooks.append(clean)

        if not hooks:
            hooks = ["సీక్రెట్ ట్రిక్! 🔥", "ఇలా చేస్తేనే! 😱", "షాకింగ్ రిజల్ట్! ⚡"]

        return {"metadata_output": text, "telugu_hooks": hooks}

    def policy_router(state: AgentState):
        return "generate_metadata" if state["policy_status"] == "PASS" else "flagged_exit"

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve_policies", retrieve_policies_node)
    workflow.add_node("audit_policy", policy_auditor_node)
    workflow.add_node("generate_metadata", metadata_generator_node)

    workflow.add_edge(START, "retrieve_policies")
    workflow.add_edge("retrieve_policies", "audit_policy")
    workflow.add_conditional_edges("audit_policy", policy_router, {"generate_metadata": "generate_metadata", "flagged_exit": END})
    workflow.add_edge("generate_metadata", END)

    return workflow.compile()

app = build_agent_graph()

# ---------------------------------------------------------
# Streamlit Interface
# ---------------------------------------------------------
st.title("🎬 YouTube AI Studio: Policy Audit & Canva Thumbnail Engine")
st.caption("Pulls video action frames or custom photos, generates Telugu hooks, and connects with Canva.")

with st.sidebar:
    st.header("⚙️ Configuration")
    input_choice = st.radio("Select Input Source:", ["YouTube Video Link (Unlisted/Public)", "Text Script / Concept Outline"])
    selected_niche = st.selectbox(
        "Target Niche:",
        ["Pets & Animals", "Cooking & Food", "Tech & Cloud Tutorials", "Fitness & Gym", "Vlogs & Lifestyle", "Other"]
    )
    if selected_niche == "Other":
        selected_niche = st.text_input("Enter Custom Niche:", "General")

    st.markdown("---")
    st.subheader("🎨 Canva Connect Setup")
    user_canva_token = st.text_input(
        "Canva API Access Token (Optional):",
        value=canva_token,
        type="password",
        help="Optional Personal Access Token from Canva Developer Portal to auto-sync frames into your Canva Uploads library."
    )

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📥 Input Content")
    if input_choice == "YouTube Video Link (Unlisted/Public)":
        content_input = st.text_input("Paste Video Link:", placeholder="https://www.youtube.com/watch?v=... or https://youtu.be/...")
    else:
        content_input = st.text_area("Paste Script / Outline:", placeholder="Type or paste your video script...", height=240)

    submit_button = st.button("🚀 Run Policy Audit & Metadata Agents", type="primary", use_container_width=True)

with col_right:
    st.subheader("📊 Output & Thumbnail Studio")

    if submit_button:
        if not content_input.strip():
            st.warning("⚠️ Please provide a video URL or script text first.")
        else:
            final_text = ""
            with st.spinner("Processing video content..."):
                if input_choice == "YouTube Video Link (Unlisted/Public)":
                    try:
                        final_text = get_transcript(content_input)
                        st.info(f"✅ Extracted transcript ({len(final_text.split())} words).")
                    except Exception as err:
                        st.error(f"Could not load transcript: {err}")
                        st.stop()
                else:
                    final_text = content_input

            with st.spinner("Running LangGraph Agents (Audit & Telugu Hook Strategy)..."):
                result = app.invoke({"concept": final_text, "niche": selected_niche})
                st.session_state["result"] = result
                st.session_state["video_url"] = content_input if input_choice == "YouTube Video Link (Unlisted/Public)" else None

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

            # ---------------------------------------------------------
            # Canva & Thumbnail Creation Engine
            # ---------------------------------------------------------
            st.markdown("---")
            st.subheader("🖼️ Thumbnail Creator (Canva Integrated)")
            
            image_source = st.radio(
                "Choose Image Source for Thumbnail:",
                ["Extract Frames Automatically from Video", "Upload High-Quality Camera Photo"],
                horizontal=True
            )

            if image_source == "Extract Frames Automatically from Video":
                if st.session_state.get("video_url"):
                    if st.button("🎞 Extract 4 Action Frames from Video"):
                        with st.spinner("Extracting clear frames from video footage..."):
                            try:
                                frames = extract_action_frames(st.session_state["video_url"], num_frames=4)
                                st.session_state["video_frames"] = frames
                            except Exception as e:
                                st.error(f"Frame extraction failed: {e}")

                    if "video_frames" in st.session_state:
                        st.write("Click to select the best action/expression frame:")
                        cols = st.columns(4)
                        for i, col in enumerate(cols):
                            with col:
                                st.image(st.session_state["video_frames"][i], use_container_width=True)
                                if st.button(f"Select Frame #{i+1}", key=f"f_pick_{i}"):
                                    st.session_state["selected_base_img"] = st.session_state["video_frames"][i]
                else:
                    st.info("No video link entered. Switch to 'Upload High-Quality Camera Photo' or enter a YouTube URL.")

            else:
                uploaded_file = st.file_uploader("Upload crisp camera photo (JPG/PNG):", type=["jpg", "jpeg", "png"])
                if uploaded_file:
                    st.session_state["selected_base_img"] = Image.open(uploaded_file)

            # When an image is ready
            if "selected_base_img" in st.session_state:
                st.markdown("#### Telugu Hook & Visual Styling")
                agent_hooks = res.get("telugu_hooks", ["సీక్రెట్ ట్రిక్! 🔥"])

                col_h1, col_h2, col_h3 = st.columns([2, 1, 1])
                with col_h1:
                    chosen_hook = st.selectbox("Suggested Telugu Hook:", agent_hooks)
                    custom_text = st.text_input("Or Enter Custom Telugu Text:", value=chosen_hook)
                with col_h2:
                    text_color = st.color_picker("Text Color (Yellow gives max CTR):", "#FFEE00")
                with col_h3:
                    text_pos = st.selectbox("Text Position:", ["Top Left", "Bottom Left", "Top Right", "Center"])

                # Render with color grading & Telugu badge
                final_thumb = render_telugu_thumbnail(
                    base_img=st.session_state["selected_base_img"],
                    telugu_text=custom_text,
                    text_color=text_color,
                    position=text_pos
                )

                st.markdown("#### 16:9 Thumbnail Preview:")
                st.image(final_thumb, use_container_width=True)

                buf = io.BytesIO()
                final_thumb.save(buf, format="JPEG", quality=95)
                thumb_bytes = buf.getvalue()

                # Action Buttons
                c_save, c_canva_direct, c_canva_api = st.columns([1, 1, 1.2])

                with c_save:
                    st.download_button(
                        label="💾 Download Thumbnail (.jpg)",
                        data=thumb_bytes,
                        file_name="youtube_telugu_thumbnail.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )

                with c_canva_direct:
                    st.link_button(
                        label="🎨 Open Canva (1280x720 Canvas)",
                        url="https://www.canva.com/create/youtube-thumbnails/",
                        use_container_width=True
                    )

                with c_canva_api:
                    if user_canva_token:
                        if st.button("🚀 Push to Canva Account via API", use_container_width=True):
                            with st.spinner("Pushing image to your Canva library..."):
                                try:
                                    edit_url, asset_id = create_canva_design_with_asset(
                                        st.session_state["selected_base_img"],
                                        user_canva_token,
                                        design_title=f"Thumb_{custom_text[:15]}"
                                    )
                                    st.success(f"Uploaded to Canva! Asset ID: {asset_id}")
                                    st.link_button("👉 Open Uploaded Asset in Canva", edit_url, use_container_width=True)
                                except Exception as err:
                                    st.error(f"Canva API error: {err}")
                    else:
                        st.caption("💡 Enter your Canva Access Token in the sidebar to enable direct 1-click cloud sync.")

            # Export Metadata
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
                    "Field": ["Niche", "Policy Status", "Video URL"],
                    "Value": [selected_niche, status, st.session_state.get("video_url", "N/A")]
                })
                st.download_button(
                    label="📊 Download Summary (.csv)",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="youtube_seo_summary.csv",
                    mime="text/csv",
                    use_container_width=True
                )