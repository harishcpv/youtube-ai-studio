\# 🎬 YouTube Pre-Upload Studio \& High-CTR Canva Thumbnail Engine



\[!\[Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)

\[!\[Framework](https://img.shields.io/badge/framework-Streamlit-red.svg)](https://streamlit.io/)

\[!\[Orchestration](https://img.shields.io/badge/orchestrator-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)

\[!\[LLM](https://img.shields.io/badge/LLM-Gemini%20Flash%203.5-brightgreen.svg)](https://deepmind.google/technologies/gemini/)

\[!\[VectorDB](https://img.shields.io/badge/VectorDB-Chroma-purple.svg)](https://www.trychroma.com/)

\[!\[Design](https://img.shields.io/badge/integration-Canva%20Connect%20API-00C4CC.svg)](https://www.canva.dev/)



An enterprise-ready AI copilot for YouTube creators that audits pre-upload scripts and videos against monetization compliance policies, generates high-ranking SEO metadata, extracts action keyframes, and builds high-CTR 16:9 clickable thumbnails with Telugu hook typography and Canva Connect cloud synchronization.



\---



\## 🏗️ System Architecture \& Workflow



The pipeline utilizes \*\*LangGraph\*\* to construct a deterministic state machine connected with \*\*RAG (Retrieval-Augmented Generation)\*\* via ChromaDB, \*\*yt-dlp/OpenCV\*\* for stream frame extraction, and the \*\*Canva Connect REST API\*\*.



```mermaid

flowchart TD

&#x20;   %% Styling

&#x20;   classDef inputStyle fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#fff;

&#x20;   classDef agentStyle fill:#0f172a,stroke:#a855f7,stroke-width:2px,color:#fff;

&#x20;   classDef visionStyle fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#fff;

&#x20;   classDef canvaStyle fill:#083344,stroke:#06b6d4,stroke-width:2px,color:#fff;

&#x20;   classDef passStyle fill:#065f46,stroke:#34d399,stroke-width:2px,color:#fff;

&#x20;   classDef failStyle fill:#7f1d1d,stroke:#f87171,stroke-width:2px,color:#fff;



&#x20;   subgraph Inputs \["📥 Ingestion Layer"]

&#x20;       A1\[YouTube Video Link]:::inputStyle

&#x20;       A2\[Raw Script / Concept]:::inputStyle

&#x20;   end



&#x20;   subgraph LangGraphAgent \["🧠 LangGraph Compliance \& Strategy Agents"]

&#x20;       B1\[(ChromaDB Policy Store)]

&#x20;       B2\[Policy Retriever Node]:::agentStyle

&#x20;       B3\[Compliance Auditor Node]:::agentStyle

&#x20;       Decision{Policy Status?}

&#x20;       B4\[SEO \& Telugu Hook Generator]:::agentStyle

&#x20;       B5\[Monetization Flagged Exit]:::failStyle

&#x20;   end



&#x20;   subgraph VisualEngine \["🖼️ Frame Extraction \& Local Image Engine"]

&#x20;       C1\[yt-dlp Stream Interceptor]:::visionStyle

&#x20;       C2\[OpenCV Action Frame Extractor]:::visionStyle

&#x20;       C3\[Custom Photo Upload Camera]:::visionStyle

&#x20;       C4\[16:9 Mobile Color-Grading Pop]:::visionStyle

&#x20;       C5\[Telugu Unicode Typography Renderer]:::visionStyle

&#x20;   end



&#x20;   subgraph CanvaSync \["🎨 Canva Integration Layer"]

&#x20;       D1\[Direct 1280x720 Canvas Launch]:::canvaStyle

&#x20;       D2\[Canva Connect REST API]:::canvaStyle

&#x20;       D3\[User Canva Cloud Asset Library]:::canvaStyle

&#x20;   end



&#x20;   %% Flow Connections

&#x20;   A1 -->|Fetch Transcript| B2

&#x20;   A2 -->|Direct Feed| B2

&#x20;   B1 -->|Cosine Similarity Search| B2

&#x20;   B2 --> B3

&#x20;   B3 --> Decision

&#x20;   Decision -->|FLAGGED| B5

&#x20;   Decision -->|PASS| B4



&#x20;   A1 --> C1

&#x20;   C1 --> C2

&#x20;   C2 --> C4

&#x20;   C3 --> C4

&#x20;   B4 -->|Telugu Hooks| C5

&#x20;   C4 --> C5



&#x20;   C5 --> D1

&#x20;   C5 -->|Base64 Asset Stream| D2

&#x20;   D2 --> D3



sequenceDiagram

&#x20;   autonumber

&#x20;   actor Creator as Content Creator

&#x20;   participant UI as Streamlit Studio UI

&#x20;   participant Agent as LangGraph Orchestrator

&#x20;   participant RAG as ChromaDB Vector Store

&#x20;   participant Media as yt-dlp \& OpenCV Engine

&#x20;   participant Canva as Canva Connect API



&#x20;   Creator->>UI: Provide YouTube URL or Script

&#x20;   UI->>Agent: Initialize workflow state (niche, script)

&#x20;   Agent->>RAG: Retrieve YouTube monetization guidelines

&#x20;   RAG-->>Agent: Matched guideline chunks

&#x20;   Agent->>Agent: Audit for profanity, copyright, policy safety

&#x20;   

&#x20;   alt Policy FLAGGED

&#x20;       Agent-->>UI: Return violation diagnostic report

&#x20;   else Policy PASSED

&#x20;       Agent->>Agent: Generate Viral Titles, SEO Description, 15 Tags \& Telugu Hooks

&#x20;       Agent-->>UI: Display SEO suite

&#x20;       

&#x20;       opt Automatic Frame Capture

&#x20;           Creator->>UI: Request Frame Extraction

&#x20;           UI->>Media: Probe video stream and sample action frames (15% - 80%)

&#x20;           Media-->>UI: Display 4 High-Resolution Candidate Frames

&#x20;       end

&#x20;       

&#x20;       Creator->>UI: Select Frame \& Pick Telugu Hook

&#x20;       UI->>UI: Apply 16:9 dynamic crop, contrast pop, \& Telugu font overlay

&#x20;       

&#x20;       opt Canva Sync

&#x20;           Creator->>UI: Trigger Canva Push

&#x20;           UI->>Canva: Post media payload to /v1/asset-uploads

&#x20;           Canva-->>UI: Return asset ID \& direct Canva design edit URL

&#x20;       end

&#x20;   end

🚀 Key Features🛡️ Pre-Upload Monetization Audit: Employs ChromaDB vector retrieval to cross-reference video transcripts against YouTube community guidelines and advertiser-friendly content rules before publishing.🎯 High-CTR Metadata Generator:3 Viral title variations (< 60 characters).15 High-ranking search tags ready for YouTube Studio.SEO-optimized description with automated chapter timestamp placeholders.🎞️ Automatic Action Frame Extractor: Direct headless streaming with yt-dlp and OpenCV sampling high-energy frames between 15% and 80% to eliminate intro/outro dead time.✍️ Telugu Typography Engine: Native Unicode font rendering (Nirmala UI, Gautami, Mandali) with dark translucent bounding badges and black strokes for small-screen readability on mobile devices.🎨 Canva Connect Cloud Synchronization: Built-in REST client for uploading generated frames straight to the Canva Asset Library and launching 1280x720 canvas templates.📂 Project StructurePlaintextyoutube-ai-studio/

├── .gitignore                      # Excludes venv, binaries, vector DB, cache

├── app.py                          # Core Streamlit app \& LangGraph orchestration

├── requirements.txt                # Pinned production dependencies

├── README.md                       # System architecture \& documentation

└── chroma\_db\_langchain/            # Local vector DB store (git-ignored)

🛠️ Installation \& Setup1. Clone RepositoryPowerShellgit clone \[https://github.com/harishcpv/youtube-ai-studio.git](https://github.com/harishcpv/youtube-ai-studio.git)

cd youtube-ai-studio

2\. Set Up Virtual EnvironmentPowerShellpython -m venv venv

.\\venv\\Scripts\\Activate.ps1

3\. Install DependenciesPowerShellpip install -r requirements.txt

4\. Configure Environment VariablesSet your Google Gemini API key:PowerShell$env:GOOGLE\_API\_KEY="your\_gemini\_api\_key\_here"

(Optional) Configure your Canva Developer Token:PowerShell$env:CANVA\_API\_TOKEN="your\_canva\_bearer\_token\_here"

5\. Launch the ApplicationPowerShellstreamlit run app.py

🛡️ Tech StackComponentTechnologyPurposeUser InterfaceStreamlitResponsive dashboard \& thumbnail customizationWorkflow EngineLangGraphState machine orchestration \& conditional routingLarge Language ModelGoogle Gemini 3.5 FlashPolicy audit, metadata generation, and Telugu hooksEmbeddingsGemini Embedding 2Vector embeddings for guideline retrievalVector DatabaseChromaDBLocal vector store for compliance policiesVideo Processingyt-dlp \& OpenCVVideo stream capture and action frame extractionImage ProcessingPillow (PIL) \& NumPy16:9 crop, color pop grading, and Telugu font renderingCloud DesignCanva Connect APICloud asset upload and thumbnail workspace creation

