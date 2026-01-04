"""
Streamlit Web Application
"""
import streamlit as st
import pandas as pd
import requests
import json
from pathlib import Path
from datetime import datetime
import time
import os
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rag.rag_engine import RAGEngine
from src.core.logger import log
from config.settings import settings

# Page configuration
st.set_page_config(
    page_title="GIS-RAG Geoinformation Retrieval System",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS to improve UI stability
st.markdown("""
<style>
    /* Disable minimal height animation for dataframe container */
    .stDataFrame {
        min-height: 100px;
    }
    /* Simplify scrollbar appearance */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    /* Stabilize sidebar layout */
    [data-testid="stSidebar"] {
        min-width: 300px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'think_mode' not in st.session_state:
    st.session_state.think_mode = False
if 'current_conversation_id' not in st.session_state:
    st.session_state.current_conversation_id = None
if 'conversations' not in st.session_state:
    st.session_state.conversations = {}
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0
if 'pending_question' not in st.session_state:
    st.session_state.pending_question = None

# Conversation history directory
CONVERSATIONS_DIR = settings.PROJECT_ROOT / "data" / "conversations"
CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)

def get_conversation_list():
    """Get all conversation history list"""
    conversations = []
    if CONVERSATIONS_DIR.exists():
        for f in sorted(CONVERSATIONS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                import json
                with open(f, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    conversations.append({
                        "id": f.stem,
                        "title": data.get("title", "Untitled"),
                        "created_at": data.get("created_at", ""),
                        "message_count": len(data.get("messages", []))
                    })
            except:
                pass
    return conversations

def save_conversation(conv_id: str, title: str, messages: list):
    """Save conversation"""
    import json
    from datetime import datetime
    
    data = {
        "id": conv_id,
        "title": title,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "messages": messages
    }
    
    filepath = CONVERSATIONS_DIR / f"{conv_id}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_conversation(conv_id: str):
    """Load conversation"""
    import json
    filepath = CONVERSATIONS_DIR / f"{conv_id}.json"
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("messages", []), data.get("title", "Untitled")
    return [], "Untitled"

def delete_conversation(conv_id: str):
    """Delete conversation"""
    filepath = CONVERSATIONS_DIR / f"{conv_id}.json"
    if filepath.exists():
        filepath.unlink()

def generate_conversation_id():
    """Generate conversation ID"""
    import uuid
    return str(uuid.uuid4())[:8]

def get_conversation_title(messages):
    """Generate conversation title from first message"""
    if messages and len(messages) > 0:
        first_q = messages[0][0] if isinstance(messages[0], tuple) else messages[0].get("question", "")
        return first_q[:20] + "..." if len(first_q) > 20 else first_q
    return "New Chat"


@st.cache_data(show_spinner=False)
def _get_uploaded_files_info(pdf_dir: str, vector_dir: str, raster_dir: str):
    files_info = []

    pdf_path = Path(pdf_dir)
    if pdf_path.exists():
        for p in sorted(pdf_path.glob("*.pdf"), key=lambda x: x.name.lower()):
            files_info.append({
                "Filename": p.name,
                "Type": "PDF",
                "Size(MB)": f"{p.stat().st_size / 1024 / 1024:.2f}",
                "Uploaded": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            })

    vector_path = Path(vector_dir)
    if vector_path.exists():
        for ext in [".shp", ".geojson"]:
            for p in sorted(vector_path.glob(f"*{ext}"), key=lambda x: x.name.lower()):
                files_info.append({
                    "Filename": p.name,
                    "Type": "Vector",
                    "Size(MB)": f"{p.stat().st_size / 1024 / 1024:.2f}",
                    "Uploaded": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                })

    raster_path = Path(raster_dir)
    if raster_path.exists():
        for ext in [".tif", ".tiff", ".jp2"]:
            for p in sorted(raster_path.glob(f"*{ext}"), key=lambda x: x.name.lower()):
                files_info.append({
                    "Filename": p.name,
                    "Type": "Raster",
                    "Size(MB)": f"{p.stat().st_size / 1024 / 1024:.2f}",
                    "Uploaded": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                })

    return files_info

@st.cache_resource
def load_rag_engine():
    """Load RAG engine with caching"""
    try:
        engine = RAGEngine()
        return engine
    except Exception as e:
        return None

@st.cache_data(show_spinner=False)
def count_uploaded_files():
    """Count uploaded files"""
    pdf_count = 0
    vector_count = 0
    raster_count = 0

    # Count PDF files
    pdf_dir = settings.PDF_DATA_DIR
    if pdf_dir.exists():
        pdf_count = len(list(pdf_dir.glob("*.pdf")))

    # Count vector files
    vector_dir = settings.VECTOR_DATA_DIR
    if vector_dir.exists():
        for ext in ['.shp', '.geojson', '.gpkg', '.kml']:
            vector_count += len(list(vector_dir.glob(f"*{ext}")))

    # Count raster files
    raster_dir = settings.RASTER_DATA_DIR
    if raster_dir.exists():
        for ext in ['.tif', '.tiff', '.jp2', '.img', '.nc']:
            raster_count += len(list(raster_dir.glob(f"*{ext}")))

    return pdf_count, vector_count, raster_count

def main():
    """Main application function"""

    # Title and description
    st.title("🗺️ GIS-RAG: Geoinformation Retrieval System")
    st.markdown("**Retrieval-Augmented Generation System for GIS with LLM**")

    # Sidebar
    with st.sidebar:
        st.header("📋 Navigation")

        # Initialize current page
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "💬 Q&A"
        
        # Navigation buttons
        pages = ["💬 Q&A", "📁 Files", "🔍 Search", "⚙️ Settings"]
        for p in pages:
            if st.button(p, key=f"nav_{p}", width="stretch",
                        type="primary" if st.session_state.current_page == p else "secondary"):
                st.session_state.current_page = p
                st.rerun()
        
        page = st.session_state.current_page

        st.divider()
        
        # Chat history (Q&A page only)
        if page == "💬 Q&A":
            st.header("💬 Chat History")
            
            # New chat button
            if st.button("➕ New Chat", width="stretch", type="primary"):
                st.session_state.current_conversation_id = None
                st.session_state.chat_history = []
                st.rerun()
            
            # Display conversation list
            conversations = get_conversation_list()
            for conv in conversations:
                col_title, col_del = st.columns([4, 1])
                with col_title:
                    # Highlight current conversation
                    is_current = st.session_state.current_conversation_id == conv["id"]
                    btn_type = "primary" if is_current else "secondary"
                    if st.button(f"📝 {conv['title']}", key=f"conv_{conv['id']}", 
                                width="stretch", type=btn_type):
                        # Load selected conversation
                        messages, title = load_conversation(conv["id"])
                        st.session_state.chat_history = messages
                        st.session_state.current_conversation_id = conv["id"]
                        st.rerun()
                with col_del:
                    if st.button("🗑️", key=f"del_{conv['id']}", help="Delete this chat"):
                        delete_conversation(conv["id"])
                        if st.session_state.current_conversation_id == conv["id"]:
                            st.session_state.current_conversation_id = None
                            st.session_state.chat_history = []
                        st.rerun()
            
            st.divider()

        # System status
        st.header("📊 System Status")
        
        if st.session_state.rag_engine is not None:

            # Show file statistics
            st.markdown("##### 📈 Status Details")
            
            if st.button("🔄 Refresh", key="refresh_sidebar_status"):
                count_uploaded_files.clear()
            
            try:
                pdf_count, vector_count, raster_count = count_uploaded_files()

                # File statistics display
                total_count = pdf_count + vector_count + raster_count
                st.markdown(f"**📊 Total {total_count} files**")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📄 PDF", pdf_count)
                with col2:
                    st.metric("🗺️ Vector", vector_count)
                with col3:
                    st.metric("🛰️ Raster", raster_count)

                # Show system status
                system_info = st.session_state.rag_engine.get_system_info()
                status = system_info.get('status', 'unknown')
                status_color = "🟢" if status == "ready" else "🟡"
                st.write(f"{status_color} Status: **{status}**")

            except Exception as e:
                st.error(f"Failed to get system info: {e}")
        else:
            st.warning("⏳ System starting...")

    # Main content area - initialize RAG engine
    if st.session_state.rag_engine is None:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### 🚀 Starting System")
            st.markdown("Please wait, loading RAG engine and models...")
            with st.spinner("Initializing..."):
                engine = load_rag_engine()
                if engine:
                    st.session_state.rag_engine = engine
                    st.toast("✅ System started successfully!", icon="🚀")
                    st.rerun()
                else:
                    st.error("❌ System startup failed, please check configuration")
        return

    # Render content based on selected page
    if page == "💬 Q&A":
        show_chat_page()
    elif page == "📁 Files":
        show_file_management_page()
    elif page == "🔍 Search":
        show_search_page()
    elif page == "⚙️ Settings":
        show_settings_page()

def show_chat_page():
    """Q&A Page"""
    st.header("💬 Q&A")

    if "is_answering" not in st.session_state:
        st.session_state.is_answering = False
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    if "active_question" not in st.session_state:
        st.session_state.active_question = None
    if "queued_question" not in st.session_state:
        st.session_state.queued_question = None
    if "clear_question_input" not in st.session_state:
        st.session_state.clear_question_input = False
    if "question_input_field" not in st.session_state:
        st.session_state.question_input_field = ""

    if st.session_state.clear_question_input:
        st.session_state.question_input_field = ""
        st.session_state.clear_question_input = False

    if st.session_state.queued_question and not st.session_state.is_answering:
        st.session_state.question_input_field = st.session_state.queued_question
        st.session_state.queued_question = None

    def _clear_chat():
        st.session_state.chat_history = []
        st.session_state.current_conversation_id = None
        st.session_state.selected_file = None
        st.session_state.last_visualized_file = None
        st.session_state.selected_file_idx = 0
        st.session_state.pending_question = None
        st.session_state.active_question = None
        st.session_state.queued_question = None
        st.session_state.clear_question_input = True

    # Think mode toggle
    status_col, toggle_col = st.columns([7, 3], vertical_alignment="center")

    with status_col:
        if st.session_state.think_mode:
            st.markdown("🧠 Thinking mode enabled · Deep reasoning")
        else:
            st.markdown("⚡ Standard mode · Fast response")

    with toggle_col:
        left_pad, switch_col, label_col = st.columns([2, 1, 3], vertical_alignment="center")
        with switch_col:
            new_think_mode = st.toggle(
                "Deep Think",
                value=st.session_state.think_mode,
                key="deep_think_toggle",
                help="Enable thinking model for deeper analysis, but slower response",
                label_visibility="collapsed",
                disabled=st.session_state.is_answering,
            )
        with label_col:
            st.markdown("🧠 Deep Think")

        if new_think_mode != st.session_state.think_mode:
            st.session_state.think_mode = new_think_mode
            if st.session_state.rag_engine:
                with st.spinner(
                    f"{'🧠 Loading thinking model...' if new_think_mode else '⚡ Loading standard model...'}"
                ):
                    success = st.session_state.rag_engine.switch_model(new_think_mode)
                    if success:
                        st.success(f"Switched to {'thinking mode' if new_think_mode else 'standard mode'}")
                    else:
                        st.error("Model switch failed")
                st.rerun()

    # File selector for GIS queries
    st.markdown("---")
    
    # Load uploaded file list
    uploaded_files = []
    
    # Collect vector files
    if settings.VECTOR_DATA_DIR.exists():
        for ext in ['.shp', '.geojson', '.gpkg']:
            for f in settings.VECTOR_DATA_DIR.glob(f"*{ext}"):
                uploaded_files.append({"name": f.name, "path": str(f), "type": "Vector"})
    
    # Collect raster files
    if settings.RASTER_DATA_DIR.exists():
        for ext in ['.tif', '.tiff', '.jp2']:
            for f in settings.RASTER_DATA_DIR.glob(f"*{ext}"):
                uploaded_files.append({"name": f.name, "path": str(f), "type": "Raster"})
    
    # Collect PDF files
    if settings.PDF_DATA_DIR.exists():
        for f in settings.PDF_DATA_DIR.glob("*.pdf"):
            uploaded_files.append({"name": f.name, "path": str(f), "type": "PDF"})
    
    # File selection
    if 'selected_file' not in st.session_state:
        st.session_state.selected_file = None
    if "selected_file_idx" not in st.session_state:
        st.session_state.selected_file_idx = 0
    
    st.markdown("##### 🎯 Select file to query")
    file_options = ["No specific file (search all)"] + [f"📄 {f['name']} ({f['type']})" for f in uploaded_files]
    selected_idx = st.selectbox(
        "Select file to query",
        range(len(file_options)),
        format_func=lambda x: file_options[x],
        help="Select a file to query, AI will answer based on its content",
        label_visibility="collapsed",
        disabled=st.session_state.is_answering,
        key="selected_file_idx",
    )
    
    if selected_idx > 0:
        st.session_state.selected_file = uploaded_files[selected_idx - 1]
        st.info(f"Selected: **{st.session_state.selected_file['name']}**")
    else:
        st.session_state.selected_file = None

    def _prefill_question(q: str):
        if st.session_state.is_answering:
            st.session_state.queued_question = q
            return
        st.session_state.question_input_field = q

    # Q&A examples (click to fill input without sending)
    with st.expander("💡 Quick Question Examples", expanded=False):
        st.caption("📄 **Document Questions:**")
        doc_questions = [
            "What is the definition of Spatial Data Science?",
            "What is vector data in Python using Shapely?",
            "What are the main spatial indexing algorithms?"
        ]
        
        for i, q in enumerate(doc_questions):
            st.button(
                f"💬 {q}",
                key=f"doc_q_{i}",
                on_click=_prefill_question,
                args=(q,),
                disabled=st.session_state.is_answering,
            )
        
        st.caption("🗺️ **GIS File Questions:**")
        gis_questions = [
            "What is the coordinate system of this file?",
            "What type of file is this?",
            "What is the spatial extent of this data?"
        ]
        
        for i, q in enumerate(gis_questions):
            st.button(
                f"💬 {q}",
                key=f"gis_q_{i}",
                on_click=_prefill_question,
                args=(q,),
                disabled=st.session_state.is_answering,
            )

    # Chat history container
    chat_container = st.container()

    # Render chat history
    with chat_container:
        for i, item in enumerate(st.session_state.chat_history):
            # Support both old (question, answer) and new (question, answer, sources) format
            if len(item) == 3:
                question, answer, sources = item
            else:
                question, answer = item
                sources = []
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                st.write(answer)
                if sources:
                    with st.expander("📚 Related Documents", expanded=False):
                        for j, doc in enumerate(sources):
                            st.write(f"**Document {j+1}:**")
                            st.write(doc.get('content', ''))
                            st.divider()

    # Question input area
    st.markdown("---")
    st.markdown("##### 💬 Enter Your Question")

    with st.form("question_form", clear_on_submit=False):
        col_input, col_btn = st.columns([6, 1], vertical_alignment="center")
        with col_input:
            question_input = st.text_input(
                "Question input",
                placeholder="Enter your question here, then click Send...",
                label_visibility="collapsed",
                key="question_input_field",
                disabled=st.session_state.is_answering,
            )
        with col_btn:
            send_clicked = st.form_submit_button(
                "🚀 Send",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.is_answering,
            )

    if send_clicked and (not st.session_state.is_answering) and question_input.strip():
        st.session_state.pending_question = question_input.strip()
        st.session_state.active_question = None
        st.session_state.is_answering = True
        st.session_state.clear_question_input = True
        st.rerun()

    if st.session_state.is_answering and st.session_state.active_question is None:
        if st.session_state.pending_question:
            st.session_state.active_question = st.session_state.pending_question
            st.session_state.pending_question = None
        else:
            st.session_state.is_answering = False

    question = st.session_state.active_question if st.session_state.is_answering else None

    if question:
        # Append user question to history
        with chat_container:
            with st.chat_message("user"):
                st.write(question)

        # Stream-answer the question
        with chat_container:
            with st.chat_message("assistant"):
                thinking_placeholder = st.empty()
                answer_placeholder = st.empty()
                source_expander = None
                
                thinking_content = ""
                answer_content = ""
                source_docs = []
                is_thinking = False
                
                st.session_state.is_answering = True

                try:
                    # Build file filter if a file is selected
                    file_filter = None
                    if st.session_state.selected_file:
                        file_filter = st.session_state.selected_file.get("path")
                    
                    for chunk in st.session_state.rag_engine.query_stream(question, file_filter=file_filter):
                        chunk_type = chunk.get("type", "")
                        content = chunk.get("content", "")
                        
                        if chunk_type == "source":
                            source_docs = content
                        elif chunk_type == "thinking_start":
                            is_thinking = True
                            thinking_placeholder.info("🧠 Thinking...")
                        elif chunk_type == "thinking":
                            thinking_content += content
                            thinking_placeholder.markdown(f"🧠 **Thinking...**\n\n{thinking_content}")
                        elif chunk_type == "thinking_end":
                            is_thinking = False
                            # Collapse thinking content
                            with thinking_placeholder.container():
                                with st.expander("🧠 View Thinking Process", expanded=False):
                                    st.markdown(thinking_content)
                        elif chunk_type == "answer":
                            answer_content += content
                            answer_placeholder.markdown(answer_content)
                        elif chunk_type == "error":
                            st.error(f"Error: {content}")
                    
                    # Show related documents during streaming
                    if source_docs:
                        with st.expander("📚 Related Documents", expanded=False):
                            for i, doc in enumerate(source_docs):
                                st.write(f"**Document {i+1}:**")
                                st.write(doc.get('content', ''))
                                st.divider()
                    
                    # Add final answer to chat history (now including source_docs)
                    if answer_content:
                        full_answer = answer_content
                        if thinking_content:
                            full_answer = f"[Thinking process collapsed]\n\n{answer_content}"
                        # Store as (question, answer, sources) tuple
                        st.session_state.chat_history.append((question, full_answer, source_docs))
                        
                        # Auto-save conversation
                        if st.session_state.current_conversation_id is None:
                            st.session_state.current_conversation_id = generate_conversation_id()
                        
                        title = get_conversation_title(st.session_state.chat_history)
                        save_conversation(
                            st.session_state.current_conversation_id,
                            title,
                            st.session_state.chat_history
                        )
                        
                except Exception as e:
                    st.error(f"Error processing question: {e}")
                finally:
                    st.session_state.is_answering = False
                    st.session_state.active_question = None
                    st.rerun()

    # Clear chat history button
    st.button(
        "🗑️ Clear Chat History",
        disabled=st.session_state.is_answering,
        on_click=_clear_chat,
    )
    
    # GIS file visualization - auto open when a file is selected
    if (
        (not st.session_state.is_answering)
        and st.session_state.selected_file
        and st.session_state.selected_file.get("type") in ["Vector", "Raster"]
    ):
        # Detect newly selected file
        current_file = st.session_state.selected_file.get("path")
        last_visualized = st.session_state.get("last_visualized_file")
        
        if current_file != last_visualized:
            st.session_state.last_visualized_file = current_file
            show_gis_visualization_dialog()
        
        # Also keep manual button for re-opening visualization
        if st.button("🗺️ View Visualization Again", type="secondary", disabled=st.session_state.is_answering):
            show_gis_visualization_dialog()


@st.dialog("🗺️ GIS File Visualization", width="large")
def show_gis_visualization_dialog():
    """GIS file visualization dialog"""
    if not st.session_state.selected_file:
        st.warning("Please select a GIS file first")
        return
    
    file_path = st.session_state.selected_file["path"]
    file_type = st.session_state.selected_file["type"]
    file_name = st.session_state.selected_file["name"]
    
    st.subheader(f"📄 {file_name}")
    
    try:
        if file_type == "Vector":
            import geopandas as gpd
            import folium
            import re
            
            with st.spinner("🗺️ Loading vector layer..."):
                gdf = gpd.read_file(file_path)
            
                if gdf.crs and gdf.crs != "EPSG:4326":
                    gdf = gdf.to_crs("EPSG:4326")
            
                bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
                center_lat = (bounds[1] + bounds[3]) / 2
                center_lon = (bounds[0] + bounds[2]) / 2
            
                m = folium.Map(location=[center_lat, center_lon], zoom_start=6)
            
                gdf_copy = gdf.copy()
                for col in gdf_copy.columns:
                    if col != 'geometry':
                        if gdf_copy[col].dtype == 'datetime64[ns]' or 'datetime' in str(gdf_copy[col].dtype):
                            gdf_copy[col] = gdf_copy[col].astype(str)
            
                geojson_data = gdf_copy.to_json()
                folium.GeoJson(
                    geojson_data,
                    name="data"
                ).add_to(m)
            
                m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
            
            from streamlit.components.v1 import html
            map_html = m._repr_html_()
            map_var = None
            tile_var = None
            m_map = re.search(r"var\s+(map_[A-Za-z0-9_]+)\s*=\s*L\.map", map_html)
            if m_map:
                map_var = m_map.group(1)
            m_tile = re.search(r"var\s+(tile_layer_[A-Za-z0-9_]+)\s*=\s*L\.tileLayer", map_html)
            if m_tile:
                tile_var = m_tile.group(1)

            if map_var:
                overlay_html = f"""
<style>
.gis-rag-map-overlay {{
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(255, 255, 255, 0.88);
  z-index: 9999;
  font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
}}
.gis-rag-map-overlay .box {{
  padding: 10px 14px;
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 10px;
  background: white;
  box-shadow: 0 8px 24px rgba(0,0,0,0.08);
}}
</style>
<script>
(function() {{
  const map = window.{map_var};
  if (!map || !map.getContainer) return;
  const container = map.getContainer();
  container.style.position = 'relative';
  const overlay = document.createElement('div');
  overlay.className = 'gis-rag-map-overlay';
  overlay.innerHTML = '<div class="box">🗺️ Loading map layers...</div>';
  container.appendChild(overlay);
  function hide() {{
    if (overlay && overlay.parentNode) overlay.parentNode.removeChild(overlay);
  }}
  const tile = {(f"window.{tile_var}" if tile_var else "null")};
  if (tile && tile.on) {{
    tile.on('load', hide);
    tile.on('tileerror', hide);
  }}
  setTimeout(hide, 10000);
}})();
</script>
"""
                map_html = map_html + overlay_html

            html(map_html, height=450)
            
            # Show attribute table preview
            st.caption(f"📊 Attribute Table Preview ({len(gdf)} features)")
            display_df = gdf.drop(columns=['geometry']).head(10)
            for col in display_df.columns:
                if display_df[col].dtype == 'datetime64[ns]':
                    display_df[col] = display_df[col].astype(str)
            st.dataframe(display_df, width="stretch")
            
        elif file_type == "Raster":
            import rasterio
            import numpy as np
            import matplotlib.pyplot as plt
            
            with st.spinner("🛰️ Loading raster preview..."):
                src = rasterio.open(file_path)
            with src:
                # Display metadata
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Size:** {src.width} × {src.height} pixels")
                    st.write(f"**Bands:** {src.count}")
                with col2:
                    st.write(f"**CRS:** {src.crs}")
                    st.write(f"**Data Type:** {src.dtypes[0]}")
                
                if src.bounds:
                    st.write(f"**Spatial Extent:** ({src.bounds.left:.4f}, {src.bounds.bottom:.4f}) to ({src.bounds.right:.4f}, {src.bounds.top:.4f})")
                
                # Read and display image preview
                st.caption("🖼️ Image Preview")
                
                # Downsample for performance
                scale_factor = max(1, max(src.width, src.height) // 500)
                out_shape = (src.count, src.height // scale_factor, src.width // scale_factor)
                
                if src.count >= 3:
                    # RGB display
                    data = src.read([1, 2, 3], out_shape=(3, out_shape[1], out_shape[2]))
                    data = np.moveaxis(data, 0, -1)
                    if data.max() > 1:
                        data = (data - data.min()) / (data.max() - data.min() + 1e-8)
                    st.image(data, caption="RGB Preview", width="stretch")
                else:
                    # Single band grayscale display
                    data = src.read(1, out_shape=(out_shape[1], out_shape[2]))
                    fig, ax = plt.subplots(figsize=(10, 8))
                    im = ax.imshow(data, cmap='viridis')
                    plt.colorbar(im, ax=ax, label='Value')
                    ax.set_title('Band 1')
                    st.pyplot(fig)
                    plt.close()
                    
    except ImportError as e:
        st.warning(f"Additional dependencies required: {e}")
        st.code("pip install folium geopandas rasterio matplotlib", language="bash")
    except Exception as e:
        st.error(f"Visualization failed: {e}")




def show_file_management_page():
    """File management page"""
    st.header("📁 File Management")

    # File upload section
    st.subheader("📄 Upload PDF Documents")
    pdf_file = st.file_uploader(
        "Select PDF file",
        type=['pdf'],
        help="Supports GIS-related courseware, textbooks and other PDF documents",
        key=f"upload_pdf_file_{st.session_state.uploader_key}"
    )

    if pdf_file and st.button("Upload PDF", type="primary", key="upload_pdf_btn"):
        with st.spinner("📝 Processing PDF document..."):
            try:
                # Save PDF file
                pdf_path = settings.PDF_DATA_DIR / pdf_file.name
                pdf_path.parent.mkdir(parents=True, exist_ok=True)

                with open(pdf_path, "wb") as f:
                    f.write(pdf_file.getbuffer())

                # Add to RAG system
                success = st.session_state.rag_engine.add_pdf_document(pdf_path)

                if success:
                    st.success(f"✅ PDF document '{pdf_file.name}' uploaded successfully!")
                    _get_uploaded_files_info.clear()
                    st.session_state.uploader_key += 1  # Reset uploader file list
                    st.rerun()  # Refresh page to update file counters
                else:
                    st.error("❌ PDF document processing failed")

            except Exception as e:
                st.error(f"Upload failed: {e}")
    st.divider()

    st.subheader("🗺️ Upload GIS Data")
    
    gis_files = st.file_uploader(
        "Select GIS data files",
        type=['shp', 'shx', 'dbf', 'prj', 'cpg', 'geojson', 'tif', 'tiff', 'jp2'],
        accept_multiple_files=True,
        help="For Shapefile, upload all related files (.shp/.shx/.dbf/.prj/.cpg); also supports GeoJSON and raster data",
        key=f"upload_gis_file_{st.session_state.uploader_key}"
    )

    if gis_files and st.button("Upload GIS Data", type="primary", key="upload_gis_btn"):
        with st.spinner("🗺️ Processing GIS data..."):
            try:
                # Analyze uploaded file types
                file_names = [f.name for f in gis_files]
                shp_files = [f for f in gis_files if f.name.lower().endswith('.shp')]
                geojson_files = [f for f in gis_files if f.name.lower().endswith('.geojson')]
                raster_files = [f for f in gis_files if f.name.lower().endswith(('.tif', '.tiff', '.jp2'))]
                
                saved_main_files = []
                
                # Handle Shapefile and companion files
                if shp_files:
                    for shp_file in shp_files:
                        base_name = Path(shp_file.name).stem
                        save_dir = settings.VECTOR_DATA_DIR
                        save_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Save all companion files with same basename
                        related_files = [f for f in gis_files if Path(f.name).stem == base_name]
                        for rf in related_files:
                            file_path = save_dir / rf.name
                            with open(file_path, "wb") as f:
                                f.write(rf.getbuffer())
                        
                        # Record main file path
                        shp_path = save_dir / shp_file.name
                        saved_main_files.append(shp_path)
                        st.info(f"📁 Saved {len(related_files)} Shapefile components")
                
                # Handle GeoJSON
                for gf in geojson_files:
                    save_dir = settings.VECTOR_DATA_DIR
                    save_dir.mkdir(parents=True, exist_ok=True)
                    file_path = save_dir / gf.name
                    with open(file_path, "wb") as f:
                        f.write(gf.getbuffer())
                    saved_main_files.append(file_path)
                
                # Handle raster data
                for rf in raster_files:
                    save_dir = settings.RASTER_DATA_DIR
                    save_dir.mkdir(parents=True, exist_ok=True)
                    file_path = save_dir / rf.name
                    with open(file_path, "wb") as f:
                        f.write(rf.getbuffer())
                    saved_main_files.append(file_path)
                
                # Add GIS data to RAG system
                success_count = 0
                for main_file in saved_main_files:
                    if st.session_state.rag_engine.add_gis_data(main_file):
                        success_count += 1
                
                if success_count > 0:
                    st.success(f"✅ Successfully processed {success_count} GIS data file(s)!")
                    _get_uploaded_files_info.clear()
                    st.session_state.uploader_key += 1  # Reset uploader file list
                    st.rerun()
                else:
                    st.error("❌ GIS data processing failed")

            except Exception as e:
                st.error(f"Upload failed: {e}")


    st.divider()

    # Uploaded files list
    st.subheader("📋 Uploaded Files")

    if st.button("🔄 Refresh File List", key="refresh_uploaded_files"):
        _get_uploaded_files_info.clear()

    files_info = _get_uploaded_files_info(
        str(settings.PDF_DATA_DIR),
        str(settings.VECTOR_DATA_DIR),
        str(settings.RASTER_DATA_DIR),
    )

    if files_info:
        df = pd.DataFrame(files_info)
        st.dataframe(df, width="stretch", hide_index=True)

        # File statistics with metrics
        total_files = len(files_info)
        pdf_files = len([f for f in files_info if f['Type'] == 'PDF'])
        vector_files = len([f for f in files_info if f['Type'] == 'Vector'])
        raster_files = len([f for f in files_info if f['Type'] == 'Raster'])

        st.markdown(f"**📊 Total {total_files} files**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 PDF", pdf_files)
        with col2:
            st.metric("🗺️ Vector", vector_files)
        with col3:
            st.metric("🛰️ Raster", raster_files)
    else:
        st.info("No files uploaded yet")

def show_search_page():
    """Document search page"""
    st.header("🔍 Document Search")

    # Search input
    col1, col2 = st.columns([3, 1])

    with col1:
        search_query = st.text_input("🔍 Search Query", placeholder="Enter keywords to search documents...")

    with col2:
        doc_type = st.selectbox("Document Type", ["All", "PDF", "GIS Metadata"])

    if search_query and st.button("Search", type="primary"):
        with st.spinner("🔍 Searching..."):
            try:
                filter_type = None
                if doc_type == "PDF":
                    filter_type = "pdf"
                elif doc_type == "GIS Metadata":
                    filter_type = "gis_metadata"

                # Execute search
                results = st.session_state.rag_engine.search_documents(
                    query=search_query,
                    doc_type=filter_type
                )

                if results:
                    st.success(f"Found {len(results)} related document chunks")

                    # Display search results
                    for i, result in enumerate(results):
                        with st.expander(f"📄 Document Chunk {i+1} (Similarity: {result.get('similarity_score', 0):.3f})"):
                            st.write("**Content:**")
                            st.write(result.get('content', ''))

                            st.write("**Metadata:**")
                            metadata = result.get('metadata', {})

                            # Format metadata for display
                            col1, col2 = st.columns(2)
                            with col1:
                                if metadata.get('source'):
                                    st.write(f"**Source:** {Path(metadata['source']).name}")
                                if metadata.get('doc_type'):
                                    st.write(f"**Type:** {metadata['doc_type']}")

                            with col2:
                                if metadata.get('file_size'):
                                    st.write(f"**Size:** {metadata['file_size'] / 1024 / 1024:.2f} MB")
                                if metadata.get('chunk_id') is not None:
                                    st.write(f"**Chunk ID:** {metadata['chunk_id']}")
                else:
                    st.info("No related documents found")

            except Exception as e:
                st.error(f"Search failed: {e}")

def show_settings_page():
    """System settings page"""
    st.header("⚙️ System Settings")

    st.subheader("📊 System Information")

    try:
        system_info = st.session_state.rag_engine.get_system_info()

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Vector Store Info:**")
            vector_info = system_info.get('vector_store', {})
            st.json(vector_info)

        with col2:
            st.write("**Model Info:**")
            llm_info = system_info.get('llm_model', {})
            if llm_info:
                st.json(llm_info)
            else:
                st.warning("LLM model not configured")

        st.write("**Processor Status:**")
        processors = system_info.get('processors', {})
        for processor, status in processors.items():
            status_icon = "✅" if status else "❌"
            st.write(f"{status_icon} {processor}: {'Available' if status else 'Not Available'}")

        # Detailed file statistics
        st.write("**File Statistics:**")
        pdf_count, vector_count, raster_count = count_uploaded_files()

        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.metric("📄 PDF Documents", pdf_count)
        with metrics_col2:
            st.metric("🗺️ Vector Data", vector_count)
        with metrics_col3:
            st.metric("🛰️ Raster Data", raster_count)

        # Vector database statistics
        if vector_info:
            st.write("**Vector Database Statistics:**")
            doc_chunks = vector_info.get('document_count', 0)
            st.write(f"- Total document chunks: {doc_chunks}")
            st.write(f"- Average chunks per PDF: {doc_chunks / max(pdf_count, 1):.1f}")

    except Exception as e:
        st.error(f"Failed to get system info: {e}")

    st.divider()

    # Database management
    st.subheader("🗄️ Database Management")

    st.warning("⚠️ Dangerous operation: Clearing database will delete all processed document data")

    if st.button("🗑️ Clear Database", type="secondary"):
        if st.button("⚠️ Confirm Clear", type="primary"):
            with st.spinner("Clearing database..."):
                try:
                    success = st.session_state.rag_engine.clear_database()
                    if success:
                        st.success("✅ Database cleared")
                        st.rerun()
                    else:
                        st.error("❌ Failed to clear database")
                except Exception as e:
                    st.error(f"Error clearing database: {e}")

if __name__ == "__main__":
    main()





