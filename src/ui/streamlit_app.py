
# src/ui/streamlit_app.py - SIMPLE VERSION LIKE ORIGINAL REPO
"""
Simple Streamlit App with Dynamic PACS.008 Intelligence
Clean UI like original repo but with enhanced backend processing
"""

import streamlit as st
import os
import json
import tempfile
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any
import logging
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Import modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.document_processor import DocumentProcessor
from exporters.excel_exporter import TestCaseExporter

# Try to import dynamic system
try:
    from ai_engine.dynamic_pacs008_test_generator import DynamicPACS008TestGenerator
    DYNAMIC_SYSTEM_AVAILABLE = True
except ImportError:
    from ai_engine.test_generator import TestCaseGenerator
    DYNAMIC_SYSTEM_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration - same as original
st.set_page_config(
    page_title="ITASSIST - Test Case Generator",
    page_icon="🏦" if DYNAMIC_SYSTEM_AVAILABLE else "🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main Streamlit application - simple like original"""
    
    # Title - enhanced if dynamic system available
    if DYNAMIC_SYSTEM_AVAILABLE:
        st.title("🏦 ITASSIST - Intelligent Test Case Generator")
        st.markdown("**AI-powered test case generation with PACS.008 intelligence**")
    else:
        st.title("🤖 ITASSIST - Intelligent Test Case Generator")
        st.markdown("**AI-powered test case generation from BFSI documents**")
    
    # Sidebar - same as original structure
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input - same as original
        default_api_key = os.getenv("OPENAI_API_KEY", "")
        api_key = st.text_input(
            "OpenAI API Key", 
            value=default_api_key,
            type="password",
            help="API key loaded from environment" if default_api_key else "Enter your OpenAI API key"
        )
        
        # Model selection - same as original
        model_option = st.selectbox(
            "AI Model",
            ["gpt-4.1-mini-2025-04-14", "gpt-4o-mini", "gpt-3.5-turbo"],
            index=0
        )
        
        # Generation options - same as original
        st.subheader("Generation Options")
        num_test_cases = st.slider("Test Cases per Story", 5, 15, 8)
        include_edge_cases = st.checkbox("Include Edge Cases", value=True)
        include_negative_cases = st.checkbox("Include Negative Cases", value=True)
        
        # PACS.008 status indicator
        if DYNAMIC_SYSTEM_AVAILABLE:
            st.subheader("🏦 PACS.008 Intelligence")
            st.success("✅ Enhanced Processing Available")
            st.info("System will automatically:\n• Detect PACS.008 fields\n• Apply banking intelligence\n• Generate domain-specific tests")
        
        # Export format - same as original
        export_format = st.multiselect(
            "Export Formats",
            ["Excel", "CSV", "JSON"],
            default=["Excel"]
        )
    
    # Initialize session state - same as original plus documentation
    if 'generated_test_cases' not in st.session_state:
        st.session_state.generated_test_cases = []
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'workflow_results' not in st.session_state:
        st.session_state.workflow_results = {}
    
    # Main content tabs - same as original structure but with documentation
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload & Process", "🧪 Generated Test Cases", "📋 Processing Report", "💬 Chat Assistant"])
    
    with tab1:
        upload_and_process_tab(api_key, num_test_cases, include_edge_cases, include_negative_cases)
    
    with tab2:
        display_test_cases_tab(export_format)
    
    with tab3:
        processing_report_tab()
    
    with tab4:
        chat_assistant_tab(api_key)

def upload_and_process_tab(api_key: str, num_test_cases: int, include_edge_cases: bool, include_negative_cases: bool):
    """File upload and processing tab - same as original but with enhanced backend"""
    
    st.header("📁 Document Upload & Processing")
    
    # File upload section - same as original
    uploaded_files = st.file_uploader(
        "Upload your documents",
        type=['docx', 'pdf', 'xlsx', 'png', 'jpg', 'jpeg', 'tiff', 'bmp', 'txt', 'eml', 'json', 'xml', 'csv', 'zip'],
        accept_multiple_files=True,
        help="Supported formats: DOCX, PDF, XLSX, Images (PNG/JPG/TIFF/BMP), TXT, EML, JSON, XML, CSV, ZIP"
    )
    
    # Display file validation info - same as original
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} file(s) uploaded successfully")
        
        # Show file details
        with st.expander("📋 File Details"):
            for file in uploaded_files:
                file_size = len(file.getvalue()) / (1024*1024)  # MB
                st.write(f"• **{file.name}** ({file_size:.1f} MB)")
                
                # Validate file size
                if file_size > 50:
                    st.warning(f"⚠️ {file.name} is large ({file_size:.1f} MB). Processing may take longer.")
    
    # Enhanced processing options - same as original
    st.subheader("🔧 Processing Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        process_embedded_content = st.checkbox("📷 Process Embedded Images/Screenshots", value=True)
    with col2:
        extract_tables = st.checkbox("📊 Extract Table Content", value=True)
    with col3:
        enhance_ocr = st.checkbox("🔍 Enhanced OCR Processing", value=True)
    
    # Custom instructions - same as original but with PACS.008 templates
    st.subheader("📝 Custom Instructions")
    
    if DYNAMIC_SYSTEM_AVAILABLE:
        instruction_templates = {
            "Standard": "",
            "Focus on PACS.008 Banking": "Focus on PACS.008 payment processing, banking agents, and cross-border scenarios",
            "Maker-Checker Workflows": "Emphasize maker-checker workflows, approval processes, and banking operations",
            "Focus on Negative Cases": "Generate more negative test cases and error scenarios. Include boundary testing and invalid input validation.",
            "Basic Scenarios Only": "Focus on basic happy path scenarios. Minimize edge cases and complex integration tests.",
            "Comprehensive Coverage": "Generate comprehensive test coverage including positive, negative, edge cases, and integration scenarios.",
            "Security Focus": "Emphasize security testing scenarios including authentication, authorization, and data validation.",
            "Performance Testing": "Include performance-related test scenarios for high-volume and stress testing."
        }
    else:
        instruction_templates = {
            "Standard": "",
            "Focus on Negative Cases": "Generate more negative test cases and error scenarios. Include boundary testing and invalid input validation.",
            "Basic Scenarios Only": "Focus on basic happy path scenarios. Minimize edge cases and complex integration tests.",
            "Comprehensive Coverage": "Generate comprehensive test coverage including positive, negative, edge cases, and integration scenarios.",
            "Security Focus": "Emphasize security testing scenarios including authentication, authorization, and data validation.",
            "Performance Testing": "Include performance-related test scenarios for high-volume and stress testing."
        }
    
    selected_template = st.selectbox("Choose Instruction Template:", list(instruction_templates.keys()))
    
    custom_instructions = st.text_area(
        "Custom Instructions",
        value=instruction_templates[selected_template],
        placeholder="e.g., 'Focus on payment validation scenarios' or 'Create 4 test cases per acceptance criteria'",
        help="Provide specific instructions to customize test case generation"
    )
    
    # Process button - same as original
    if st.button("🚀 Generate Test Cases", type="primary", disabled=not api_key or not uploaded_files):
        if not api_key:
            st.error("Please provide OpenAI API key in the sidebar")
            return
        
        if not uploaded_files:
            st.error("Please upload at least one document")
            return
        
        process_files(uploaded_files, api_key, custom_instructions, num_test_cases, 
                     include_edge_cases, include_negative_cases, process_embedded_content)

def process_files(uploaded_files, api_key: str, custom_instructions: str, 
                 num_test_cases: int, include_edge_cases: bool, include_negative_cases: bool,
                 process_embedded_content: bool):
    """Process uploaded files - enhanced backend but simple UI"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Process each file - same as original
        doc_processor = DocumentProcessor()
        
        all_content = []
        total_files = len(uploaded_files)
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            progress_bar.progress((i + 1) / (total_files + 2))
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Process the file
            result = doc_processor.process_file(tmp_file_path)
            all_content.append(result.get('content', ''))
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
        
        # Combine all extracted content
        status_text.text("Combining extracted content...")
        progress_bar.progress(0.9)
        
        combined_content = '\n\n--- Document Separator ---\n\n'.join(all_content)
        
        # Generate custom instructions - enhanced
        generation_instructions = build_generation_instructions(
            custom_instructions, num_test_cases, include_edge_cases, include_negative_cases
        )
        
        # Generate test cases - enhanced backend but simple UI
        if DYNAMIC_SYSTEM_AVAILABLE:
            status_text.text("Generating test cases with PACS.008 intelligence...")
            
            # Prepare files info for documentation
            files_info = []
            for uploaded_file in uploaded_files:
                file_size = len(uploaded_file.getvalue()) / (1024*1024)
                files_info.append({
                    "name": uploaded_file.name,
                    "size_mb": file_size,
                    "type": uploaded_file.type or "unknown",
                    "status": "processed"
                })
            
            # Use dynamic system for enhanced processing
            generator = DynamicPACS008TestGenerator(api_key)
            workflow_results = generator.process_complete_workflow(combined_content, num_test_cases, files_info)
            
            # Store complete workflow results
            st.session_state.workflow_results = workflow_results
            
            # Extract test cases from workflow
            test_cases = workflow_results.get("step5_test_cases", [])
            
            # Show brief intelligence summary
            if workflow_results.get("step1_analysis", {}).get("is_pacs008_relevant", False):
                st.success("🏦 **PACS.008 content detected** - Applied banking intelligence!")
            
        else:
            status_text.text("Generating test cases with AI...")
            
            # Use standard system
            test_generator = TestCaseGenerator(api_key)
            test_cases = test_generator.generate_test_cases(combined_content, generation_instructions)
            
            # Store basic results
            st.session_state.workflow_results = {"step5_test_cases": test_cases}
        
        progress_bar.progress(0.95)
        
        if test_cases:
            st.session_state.generated_test_cases = test_cases
            st.session_state.processing_complete = True
            
            progress_bar.progress(1.0)
            status_text.text("✅ Processing complete!")
            
            # Display summary - same as original
            st.success(f"Successfully generated {len(test_cases)} test cases!")
            
            # Show content preview
            with st.expander("📄 Extracted Content Preview"):
                preview_content = combined_content[:1000] + "..." if len(combined_content) > 1000 else combined_content
                st.text(preview_content)
                
        else:
            st.error("No test cases could be generated. Please check your documents and try again.")
            
    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        logger.error(f"Processing error: {str(e)}")

def build_generation_instructions(custom_instructions: str, num_test_cases: int, 
                                include_edge_cases: bool, include_negative_cases: bool) -> str:
    """Build generation instructions - same as original but enhanced"""
    instructions = []
    
    if custom_instructions:
        instructions.append(custom_instructions)
    
    instructions.append(f"Generate exactly {num_test_cases} test cases per user story/requirement")
    
    if include_edge_cases:
        instructions.append("Include edge cases and boundary conditions")
    
    if include_negative_cases:
        instructions.append("Include negative test scenarios and error conditions")
    
    # Enhanced instruction for PACS.008 if available
    if DYNAMIC_SYSTEM_AVAILABLE:
        instructions.append("Focus on BFSI domain scenarios with realistic banking data and PACS.008 intelligence")
    else:
        instructions.append("Focus on BFSI domain scenarios with realistic banking data")
    
    return ". ".join(instructions)

def display_test_cases_tab(export_formats: List[str]):
    """Display generated test cases - same as original structure"""
    
    st.header("🧪 Generated Test Cases")
    
    if not st.session_state.generated_test_cases:
        st.info("No test cases generated yet. Please upload documents and process them first.")
        return
    
    test_cases = st.session_state.generated_test_cases
    
    # Display summary metrics - same as original
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Test Cases", len(test_cases))
    with col2:
        high_priority = len([tc for tc in test_cases if tc.get("Priority") == "High"])
        st.metric("High Priority", high_priority)
    with col3:
        regression_tests = len([tc for tc in test_cases if tc.get("Part of Regression") == "Yes"])
        st.metric("Regression Tests", regression_tests)
    with col4:
        unique_stories = len(set(tc.get("User Story ID", "") for tc in test_cases))
        st.metric("User Stories", unique_stories)
    
    # Show PACS.008 enhancement indicator if available
    if DYNAMIC_SYSTEM_AVAILABLE:
        pacs008_enhanced = len([tc for tc in test_cases if tc.get("PACS008_Enhanced") == "Yes"])
        if pacs008_enhanced > 0:
            st.info(f"🏦 {pacs008_enhanced} test cases enhanced with PACS.008 intelligence")
    
    # Filter options - same as original
    with st.expander("🔍 Filter Test Cases"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            priority_filter = st.multiselect(
                "Priority", 
                ["High", "Medium", "Low"],
                default=["High", "Medium", "Low"]
            )
        
        with col2:
            regression_filter = st.multiselect(
                "Regression", 
                ["Yes", "No"],
                default=["Yes", "No"]
            )
        
        with col3:
            story_ids = list(set(tc.get("User Story ID", "") for tc in test_cases))
            story_filter = st.multiselect(
                "User Story ID",
                story_ids,
                default=story_ids
            )
    
    # Apply filters - same as original
    filtered_test_cases = [
        tc for tc in test_cases
        if (tc.get("Priority") in priority_filter and
            tc.get("Part of Regression") in regression_filter and
            tc.get("User Story ID") in story_filter)
    ]
    
    # Display test cases table - same as original
    if filtered_test_cases:
        st.subheader(f"Test Cases ({len(filtered_test_cases)} of {len(test_cases)})")
        
        # Convert to DataFrame for display
        df = pd.DataFrame(filtered_test_cases)
        
        # Configure column display - same as original
        column_config = {
            "Steps": st.column_config.TextColumn(width="large"),
            "Test Case Description": st.column_config.TextColumn(width="medium"),
            "Expected Result": st.column_config.TextColumn(width="medium"),
        }
        
        st.dataframe(
            df,
            use_container_width=True,
            column_config=column_config,
            hide_index=True
        )
        
        # Export section - same as original
        st.subheader("📥 Export Test Cases")
        
        col1, col2, col3 = st.columns(3)
        
        if "Excel" in export_formats:
            with col1:
                if st.button("📊 Download Excel", type="primary"):
                    export_excel(filtered_test_cases)
        
        if "CSV" in export_formats:
            with col2:
                if st.button("📄 Download CSV"):
                    export_csv(filtered_test_cases)
        
        if "JSON" in export_formats:
            with col3:
                if st.button("🔧 Download JSON"):
                    export_json(filtered_test_cases)
    
    else:
        st.warning("No test cases match the selected filters.")

def export_excel(test_cases):
    """Export Excel - same as original"""
    try:
        import io
        
        # Create Excel in memory
        df = pd.DataFrame(test_cases)
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Test Cases', index=False)
        
        output.seek(0)
        
        st.download_button(
            label="📊 Download Excel File",
            data=output.getvalue(),
            file_name=f"test_cases_{len(test_cases)}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    except Exception as e:
        st.error(f"Export error: {str(e)}")

def export_csv(test_cases):
    """Export CSV - same as original"""
    try:
        csv_data = pd.DataFrame(test_cases).to_csv(index=False)
        st.download_button(
            label="📄 Download CSV File",
            data=csv_data,
            file_name=f"test_cases_{len(test_cases)}.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Export error: {str(e)}")

def export_json(test_cases):
    """Export JSON - same as original"""
    try:
        json_data = json.dumps(test_cases, indent=2, ensure_ascii=False)
        st.download_button(
            label="🔧 Download JSON File",
            data=json_data,
            file_name=f"test_cases_{len(test_cases)}.json",
            mime="application/json"
        )
    except Exception as e:
        st.error(f"Export error: {str(e)}")

def processing_report_tab():
    """Processing report and documentation tab"""
    
    st.header("📋 Processing Report & Documentation")
    
    if not st.session_state.workflow_results:
        st.info("📄 No processing report available. Process documents first to see detailed analysis.")
        return
    
    workflow_results = st.session_state.workflow_results
    
    # Check if documentation is available
    documentation = workflow_results.get("documentation", {})
    
    if documentation and documentation.get("report_text"):
        st.success("✅ **Complete Processing Documentation Available**")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            user_stories = len(workflow_results.get("step2_user_stories", []))
            st.metric("📖 User Stories", user_stories)
        
        with col2:
            pacs008_fields = workflow_results.get("step3_pacs008_fields", {}).get("total_unique_fields", 0)
            st.metric("🏦 PACS.008 Fields", pacs008_fields)
        
        with col3:
            test_cases = len(workflow_results.get("step5_test_cases", []))
            st.metric("🧪 Test Cases", test_cases)
        
        with col4:
            maker_checker_items = len(workflow_results.get("step4_maker_checker", {}).get("validation_items", []))
            st.metric("👥 Validation Items", maker_checker_items)
        
        # Processing intelligence indicator
        analysis = workflow_results.get("step1_analysis", {})
        if analysis.get("is_pacs008_relevant", False):
            st.success(f"🎯 **PACS.008 Intelligence Applied** - Confidence: {analysis.get('confidence_score', 0)}%")
        else:
            st.info("📋 **Standard Processing Applied** - No PACS.008 content detected")
        
        # Documentation preview and download
        st.subheader("📄 Complete Processing Documentation")
        
        with st.expander("📋 Processing Report Preview (First 1000 characters)", expanded=False):
            report_text = documentation.get("report_text", "")
            preview = report_text[:1000] + "..." if len(report_text) > 1000 else report_text
            st.text(preview)
        
        # Download options
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download Complete Processing Report", type="primary"):
                try:
                    report_text = documentation.get("report_text", "")
                    
                    # Create filename with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"PACS008_Processing_Report_{timestamp}.md"
                    
                    st.download_button(
                        label="📄 Download Report (Markdown)",
                        data=report_text.encode('utf-8'),
                        file_name=filename,
                        mime="text/markdown",
                        help="Complete processing report with all analysis, decisions, and reasoning"
                    )
                    
                except Exception as e:
                    st.error(f"Error preparing download: {str(e)}")
        
        with col2:
            if st.button("🔧 Download JSON Data"):
                try:
                    json_data = documentation.get("json_data", {})
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"PACS008_Processing_Data_{timestamp}.json"
                    
                    st.download_button(
                        label="📊 Download JSON Data",
                        data=json.dumps(json_data, indent=2, ensure_ascii=False).encode('utf-8'),
                        file_name=filename,
                        mime="application/json",
                        help="Raw processing data in JSON format for programmatic access"
                    )
                    
                except Exception as e:
                    st.error(f"Error preparing JSON download: {str(e)}")
        
        # Processing summary
        st.subheader("🔍 Key Processing Insights")
        
        # Show key insights
        if analysis.get("banking_concepts"):
            st.write("**🏦 Banking Concepts Detected:**")
            for concept in analysis.get("banking_concepts", [])[:5]:
                st.write(f"• {concept}")
        
        if workflow_results.get("processing_errors"):
            st.subheader("⚠️ Processing Warnings")
            for error in workflow_results.get("processing_errors", []):
                st.warning(f"• {error}")
        
        # Processing quality indicators
        quality = workflow_results.get("workflow_summary", {}).get("quality_indicators", {})
        if quality:
            st.subheader("📊 Processing Quality")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                field_quality = quality.get("field_detection_accuracy", "medium")
                color = "🟢" if field_quality == "high" else "🟡"
                st.write(f"{color} **Field Detection**: {field_quality.title()}")
            
            with col2:
                test_quality = quality.get("test_case_relevance", "medium")
                color = "🟢" if test_quality == "high" else "🟡"
                st.write(f"{color} **Test Relevance**: {test_quality.title()}")
            
            with col3:
                business_quality = quality.get("business_alignment", "medium")
                color = "🟢" if business_quality == "high" else "🟡"
                st.write(f"{color} **Business Alignment**: {business_quality.title()}")
        
    else:
        st.warning("⚠️ **Limited Documentation Available**")
        st.info("Complete documentation is only available when using the PACS.008 enhanced system.")
        
        # Show basic info if available
        test_cases = workflow_results.get("step5_test_cases", [])
        if test_cases:
            st.write(f"**Generated:** {len(test_cases)} test cases")
            
            # Basic test case summary
            priorities = {}
            for tc in test_cases:
                priority = tc.get("Priority", "Medium")
                priorities[priority] = priorities.get(priority, 0) + 1
            
            st.write("**Priority Distribution:**")
            for priority, count in priorities.items():
                st.write(f"• {priority}: {count} test cases")

def chat_assistant_tab(api_key: str):
    """Chat assistant - same as original"""
    
    st.header("💬 Chat Assistant")
    st.markdown("Ask questions about your test cases or request modifications")
    
    if not api_key:
        st.warning("Please provide OpenAI API key to enable chat functionality")
        return
    
    if not st.session_state.generated_test_cases:
        st.info("Generate test cases first to enable chat assistance")
        return
    
    # Chat interface - same as original
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    # Display chat history
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your test cases..."):
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_chat_response(prompt, st.session_state.generated_test_cases, api_key)
                st.markdown(response)
        
        # Add assistant message
        st.session_state.chat_messages.append({"role": "assistant", "content": response})

def generate_chat_response(prompt: str, test_cases: List[Dict], api_key: str) -> str:
    """Generate chat response - same as original"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Prepare context
        test_cases_summary = f"Total test cases: {len(test_cases)}\n"
        test_cases_summary += "Sample test cases:\n"
        for i, tc in enumerate(test_cases[:3], 1):
            test_cases_summary += f"{i}. {tc.get('Test Case Description', '')}\n"
        
        chat_prompt = f"""
        You are an expert BFSI test engineer assistant. Answer questions about the generated test cases.
        
        Test Cases Context:
        {test_cases_summary}
        
        User Question: {prompt}
        
        Provide helpful, specific answers about the test cases. If asked to modify test cases, 
        provide specific suggestions or instructions.
        """
        
        response = client.chat.completions.create(
            model="gpt-4.1-mini-2025-04-14",
            messages=[
                {"role": "system", "content": "You are a helpful BFSI testing expert."},
                {"role": "user", "content": chat_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

if __name__ == "__main__":
    main()
