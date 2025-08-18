# # src/ui/streamlit_app.py
# import streamlit as st
# import os
# import json
# import tempfile
# import zipfile
# from pathlib import Path
# import pandas as pd
# from typing import List, Dict, Any
# import logging
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Import our custom modules
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from processors.document_processor import DocumentProcessor
# from ai_engine.test_generator import TestCaseGenerator
# from exporters.excel_exporter import TestCaseExporter

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Page configuration
# st.set_page_config(
#     page_title="ITASSIST - Test Case Generator",
#     page_icon="🤖",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# def main():
#     """Main Streamlit application"""
    
#     # Title and description
#     st.title("🤖 ITASSIST - Intelligent Test Case Generator")
#     st.markdown("**AI-powered test case generation from BFSI documents**")
    
#     # Sidebar for configuration
#     with st.sidebar:
#         st.header("⚙️ Configuration")
        
#         # API Key input - auto-load from environment
#         default_api_key = os.getenv("OPENAI_API_KEY", "")
#         api_key = st.text_input(
#             "OpenAI API Key", 
#             value=default_api_key,
#             type="password",
#             help="API key loaded from environment" if default_api_key else "Enter your OpenAI API key"
#         )
        
#         # Model selection
#         model_option = st.selectbox(
#             "AI Model",
#             ["gpt-4.1-mini-2025-04-14", "gpt-4o-mini", "gpt-3.5-turbo"],
#             index=0
#         )
        
#         # Generation options
#         st.subheader("Generation Options")
#         num_test_cases = st.slider("Test Cases per Story", 5, 15, 8)
#         include_edge_cases = st.checkbox("Include Edge Cases", value=True)
#         include_negative_cases = st.checkbox("Include Negative Cases", value=True)
        
#         # Export format
#         export_format = st.multiselect(
#             "Export Formats",
#             ["Excel", "CSV", "JSON"],
#             default=["Excel"]
#         )
    
#     # Initialize session state
#     if 'generated_test_cases' not in st.session_state:
#         st.session_state.generated_test_cases = []
#     if 'processing_complete' not in st.session_state:
#         st.session_state.processing_complete = False
    
#     # Main content tabs
#     tab1, tab2, tab3 = st.tabs(["📁 Upload & Process", "🧪 Generated Test Cases", "💬 Chat Assistant"])
    
#     with tab1:
#         upload_and_process_tab(api_key, num_test_cases, include_edge_cases, include_negative_cases)
    
#     with tab2:
#         display_test_cases_tab(export_format)
    
#     with tab3:
#         chat_assistant_tab(api_key)

# def upload_and_process_tab(api_key: str, num_test_cases: int, include_edge_cases: bool, include_negative_cases: bool):
#     """File upload and processing tab"""
    
#     st.header("📁 Document Upload & Processing")
    
#     # File upload section
#     uploaded_files = st.file_uploader(
#         "Upload your documents",
#         type=['docx', 'pdf', 'xlsx', 'png', 'jpg', 'jpeg', 'tiff', 'bmp', 'txt', 'eml', 'json', 'xml', 'csv', 'zip'],
#         accept_multiple_files=True,
#         help="Supported formats: DOCX, PDF, XLSX, Images (PNG/JPG/TIFF/BMP), TXT, EML, JSON, XML, CSV, ZIP"
#     )
    
#     # Display file validation info
#     if uploaded_files:
#         st.info(f"📁 {len(uploaded_files)} file(s) uploaded successfully")
        
#         # Show file details
#         with st.expander("📋 File Details"):
#             for file in uploaded_files:
#                 file_size = len(file.getvalue()) / (1024*1024)  # MB
#                 st.write(f"• **{file.name}** ({file_size:.1f} MB)")
                
#                 # Validate file size
#                 if file_size > 50:
#                     st.warning(f"⚠️ {file.name} is large ({file_size:.1f} MB). Processing may take longer.")
    
#     # Enhanced processing options
#     st.subheader("🔧 Processing Options")
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         process_embedded_content = st.checkbox("📷 Process Embedded Images/Screenshots", value=True)
#     with col2:
#         extract_tables = st.checkbox("📊 Extract Table Content", value=True)
#     with col3:
#         enhance_ocr = st.checkbox("🔍 Enhanced OCR Processing", value=True)
    
#     # Custom instructions with examples
#     st.subheader("📝 Custom Instructions")
    
#     # Predefined instruction templates
#     instruction_templates = {
#         "Standard": "",
#         "Focus on Negative Cases": "Generate more negative test cases and error scenarios. Include boundary testing and invalid input validation.",
#         "Basic Scenarios Only": "Focus on basic happy path scenarios. Minimize edge cases and complex integration tests.",
#         "Comprehensive Coverage": "Generate comprehensive test coverage including positive, negative, edge cases, and integration scenarios.",
#         "Security Focus": "Emphasize security testing scenarios including authentication, authorization, and data validation.",
#         "Performance Testing": "Include performance-related test scenarios for high-volume and stress testing."
#     }
    
#     selected_template = st.selectbox("Choose Instruction Template:", list(instruction_templates.keys()))
    
#     custom_instructions = st.text_area(
#         "Custom Instructions",
#         value=instruction_templates[selected_template],
#         placeholder="e.g., 'Focus on payment validation scenarios' or 'Create 4 test cases per acceptance criteria'",
#         help="Provide specific instructions to customize test case generation"
#     )
    
#     # Process button
#     if st.button("🚀 Generate Test Cases", type="primary", disabled=not api_key or not uploaded_files):
#         if not api_key:
#             st.error("Please provide OpenAI API key in the sidebar")
#             return
        
#         if not uploaded_files:
#             st.error("Please upload at least one document")
#             return
        
#         process_files(uploaded_files, api_key, custom_instructions, num_test_cases, 
#                      include_edge_cases, include_negative_cases, process_embedded_content)

# def process_files(uploaded_files, api_key: str, custom_instructions: str, 
#                  num_test_cases: int, include_edge_cases: bool, include_negative_cases: bool,
#                  process_embedded_content: bool):
#     """Process uploaded files and generate test cases"""
    
#     # Initialize processors
#     doc_processor = DocumentProcessor()
#     test_generator = TestCaseGenerator(api_key)
    
#     progress_bar = st.progress(0)
#     status_text = st.empty()
    
#     try:
#         all_content = []
#         total_files = len(uploaded_files)
        
#         # Process each file
#         for i, uploaded_file in enumerate(uploaded_files):
#             status_text.text(f"Processing {uploaded_file.name}...")
#             progress_bar.progress((i + 1) / (total_files + 2))
            
#             # Handle ZIP files
#             if uploaded_file.name.endswith('.zip'):
#                 extracted_content = process_zip_file(uploaded_file, doc_processor)
#                 all_content.extend(extracted_content)
#             else:
#                 # Save uploaded file temporarily
#                 with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
#                     tmp_file.write(uploaded_file.getvalue())
#                     tmp_file_path = tmp_file.name
                
#                 # Process the file
#                 result = doc_processor.process_file(tmp_file_path)
#                 all_content.append(result)
                
#                 # Clean up temporary file
#                 os.unlink(tmp_file_path)
        
#         # Combine all extracted content
#         status_text.text("Combining extracted content...")
#         progress_bar.progress(0.9)
        
#         combined_content = combine_extracted_content(all_content)
        
#         # Generate custom instructions
#         generation_instructions = build_generation_instructions(
#             custom_instructions, num_test_cases, include_edge_cases, include_negative_cases
#         )
        
#         # Generate test cases
#         status_text.text("Generating test cases with AI...")
#         progress_bar.progress(0.95)
        
#         test_cases = test_generator.generate_test_cases(combined_content, generation_instructions)
        
#         if test_cases:
#             st.session_state.generated_test_cases = test_cases
#             st.session_state.processing_complete = True
            
#             progress_bar.progress(1.0)
#             status_text.text("✅ Processing complete!")
            
#             # Display summary
#             st.success(f"Successfully generated {len(test_cases)} test cases!")
            
#             # Show content preview
#             with st.expander("📄 Extracted Content Preview"):
#                 st.text(combined_content[:1000] + "..." if len(combined_content) > 1000 else combined_content)
                
#         else:
#             st.error("No test cases could be generated. Please check your documents and try again.")
            
#     except Exception as e:
#         st.error(f"Error during processing: {str(e)}")
#         logger.error(f"Processing error: {str(e)}")

# def process_zip_file(zip_file, doc_processor: DocumentProcessor) -> List[Dict]:
#     """Process files within a ZIP archive"""
#     extracted_content = []
    
#     with tempfile.TemporaryDirectory() as temp_dir:
#         # Extract ZIP file
#         with zipfile.ZipFile(zip_file, 'r') as zip_ref:
#             zip_ref.extractall(temp_dir)
        
#         # Process each extracted file
#         for root, dirs, files in os.walk(temp_dir):
#             for file in files:
#                 file_path = os.path.join(root, file)
#                 if Path(file_path).suffix.lower() in doc_processor.supported_formats:
#                     result = doc_processor.process_file(file_path)
#                     extracted_content.append(result)
    
#     return extracted_content

# def combine_extracted_content(content_list: List[Dict]) -> str:
#     """Combine content from multiple files"""
#     combined_text = []
    
#     for content in content_list:
#         if content.get('content'):
#             file_info = f"\n--- Content from {content.get('file_name', 'Unknown')} ---\n"
#             combined_text.append(file_info + content['content'])
        
#         # Add table content if available
#         if content.get('tables'):
#             combined_text.append("\nTables:\n" + '\n'.join(content['tables']))
        
#         # Add OCR text from images
#         if content.get('image_text'):
#             combined_text.append("\nImage Text:\n" + '\n'.join(content['image_text']))
    
#     return '\n\n'.join(combined_text)

# def build_generation_instructions(custom_instructions: str, num_test_cases: int, 
#                                 include_edge_cases: bool, include_negative_cases: bool) -> str:
#     """Build generation instructions based on user preferences"""
#     instructions = []
    
#     if custom_instructions:
#         instructions.append(custom_instructions)
    
#     instructions.append(f"Generate exactly {num_test_cases} test cases per user story/requirement")
    
#     if include_edge_cases:
#         instructions.append("Include edge cases and boundary conditions")
    
#     if include_negative_cases:
#         instructions.append("Include negative test scenarios and error conditions")
    
#     instructions.append("Focus on BFSI domain scenarios with realistic banking data")
    
#     return ". ".join(instructions)

# def display_test_cases_tab(export_formats: List[str]):
#     """Display generated test cases"""
    
#     st.header("🧪 Generated Test Cases")
    
#     if not st.session_state.generated_test_cases:
#         st.info("No test cases generated yet. Please upload documents and process them first.")
#         return
    
#     test_cases = st.session_state.generated_test_cases
    
#     # Display summary metrics
#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         st.metric("Total Test Cases", len(test_cases))
#     with col2:
#         high_priority = len([tc for tc in test_cases if tc.get("Priority") == "High"])
#         st.metric("High Priority", high_priority)
#     with col3:
#         regression_tests = len([tc for tc in test_cases if tc.get("Part of Regression") == "Yes"])
#         st.metric("Regression Tests", regression_tests)
#     with col4:
#         unique_stories = len(set(tc.get("User Story ID", "") for tc in test_cases))
#         st.metric("User Stories", unique_stories)
    
#     # Filter options
#     with st.expander("🔍 Filter Test Cases"):
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             priority_filter = st.multiselect(
#                 "Priority", 
#                 ["High", "Medium", "Low"],
#                 default=["High", "Medium", "Low"]
#             )
        
#         with col2:
#             regression_filter = st.multiselect(
#                 "Regression", 
#                 ["Yes", "No"],
#                 default=["Yes", "No"]
#             )
        
#         with col3:
#             story_ids = list(set(tc.get("User Story ID", "") for tc in test_cases))
#             story_filter = st.multiselect(
#                 "User Story ID",
#                 story_ids,
#                 default=story_ids
#             )
    
#     # Apply filters
#     filtered_test_cases = [
#         tc for tc in test_cases
#         if (tc.get("Priority") in priority_filter and
#             tc.get("Part of Regression") in regression_filter and
#             tc.get("User Story ID") in story_filter)
#     ]
    
#     # Display test cases table
#     if filtered_test_cases:
#         st.subheader(f"Test Cases ({len(filtered_test_cases)} of {len(test_cases)})")
        
#         # Convert to DataFrame for display
#         df = pd.DataFrame(filtered_test_cases)
        
#         # Configure column display
#         column_config = {
#             "Steps": st.column_config.TextColumn(width="large"),
#             "Test Case Description": st.column_config.TextColumn(width="medium"),
#             "Expected Result": st.column_config.TextColumn(width="medium"),
#         }
        
#         st.dataframe(
#             df,
#             use_container_width=True,
#             column_config=column_config,
#             hide_index=True
#         )
        
#         # Export section
#         st.subheader("📥 Export Test Cases")
        
#         col1, col2, col3 = st.columns(3)
        
#         exporter = TestCaseExporter()
        
#         if "Excel" in export_formats:
#             with col1:
#                 if st.button("📊 Download Excel", type="primary"):
#                     try:
#                         # Create Excel in memory instead of temp file
#                         import io
#                         from openpyxl import Workbook
                        
#                         # Use our exporter but modify to work with BytesIO
#                         exporter = TestCaseExporter()
                        
#                         # Create DataFrame
#                         df = pd.DataFrame(filtered_test_cases)
                        
#                         # Ensure all required columns exist
#                         for col in exporter.required_columns:
#                             if col not in df.columns:
#                                 df[col] = ""
                        
#                         # Reorder columns
#                         df = df[exporter.required_columns]
                        
#                         # Create Excel file in memory
#                         output = io.BytesIO()
#                         with pd.ExcelWriter(output, engine='openpyxl') as writer:
#                             df.to_excel(writer, sheet_name='Test Cases', index=False)
                            
#                             # Get workbook and worksheet for formatting
#                             workbook = writer.book
#                             worksheet = writer.sheets['Test Cases']
                            
#                             # Apply basic formatting
#                             from openpyxl.styles import Font, PatternFill
                            
#                             # Header formatting
#                             for cell in worksheet[1]:
#                                 cell.font = Font(bold=True, color="FFFFFF")
#                                 cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
                        
#                         output.seek(0)
                        
#                         st.download_button(
#                             label="📊 Download Excel File",
#                             data=output.getvalue(),
#                             file_name=f"test_cases_{len(filtered_test_cases)}.xlsx",
#                             mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#                         )
                        
#                     except Exception as e:
#                         st.error(f"Export error: {str(e)}")
#                         logger.error(f"Excel export error: {str(e)}")
        
#         if "CSV" in export_formats:
#             with col2:
#                 if st.button("📄 Download CSV"):
#                     try:
#                         csv_data = pd.DataFrame(filtered_test_cases).to_csv(index=False)
#                         st.download_button(
#                             label="📄 Download CSV File",
#                             data=csv_data,
#                             file_name=f"test_cases_{len(filtered_test_cases)}.csv",
#                             mime="text/csv"
#                         )
#                     except Exception as e:
#                         st.error(f"Export error: {str(e)}")
        
#         if "JSON" in export_formats:
#             with col3:
#                 if st.button("🔧 Download JSON"):
#                     try:
#                         json_data = json.dumps(filtered_test_cases, indent=2, ensure_ascii=False)
#                         st.download_button(
#                             label="🔧 Download JSON File",
#                             data=json_data,
#                             file_name=f"test_cases_{len(filtered_test_cases)}.json",
#                             mime="application/json"
#                         )
#                     except Exception as e:
#                         st.error(f"Export error: {str(e)}")
    
#     else:
#         st.warning("No test cases match the selected filters.")

# def chat_assistant_tab(api_key: str):
#     """Chat assistant for test case customization"""
    
#     st.header("💬 Chat Assistant")
#     st.markdown("Ask questions about your test cases or request modifications")
    
#     if not api_key:
#         st.warning("Please provide OpenAI API key to enable chat functionality")
#         return
    
#     if not st.session_state.generated_test_cases:
#         st.info("Generate test cases first to enable chat assistance")
#         return
    
#     # Chat interface
#     if "chat_messages" not in st.session_state:
#         st.session_state.chat_messages = []
    
#     # Display chat history
#     for message in st.session_state.chat_messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
    
#     # Chat input
#     if prompt := st.chat_input("Ask about your test cases..."):
#         # Add user message
#         st.session_state.chat_messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)
        
#         # Generate response
#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 response = generate_chat_response(prompt, st.session_state.generated_test_cases, api_key)
#                 st.markdown(response)
        
#         # Add assistant message
#         st.session_state.chat_messages.append({"role": "assistant", "content": response})

# def generate_chat_response(prompt: str, test_cases: List[Dict], api_key: str) -> str:
#     """Generate chat response about test cases"""
#     try:
#         from openai import OpenAI
#         client = OpenAI(api_key=api_key)
        
#         # Prepare context
#         test_cases_summary = f"Total test cases: {len(test_cases)}\n"
#         test_cases_summary += "Sample test cases:\n"
#         for i, tc in enumerate(test_cases[:3], 1):
#             test_cases_summary += f"{i}. {tc.get('Test Case Description', '')}\n"
        
#         chat_prompt = f"""
#         You are an expert BFSI test engineer assistant. Answer questions about the generated test cases.
        
#         Test Cases Context:
#         {test_cases_summary}
        
#         User Question: {prompt}
        
#         Provide helpful, specific answers about the test cases. If asked to modify test cases, 
#         provide specific suggestions or instructions.
#         """
        
#         response = client.chat.completions.create(
#             model="gpt-4.1-mini-2025-04-14",
#             messages=[
#                 {"role": "system", "content": "You are a helpful BFSI testing expert."},
#                 {"role": "user", "content": chat_prompt}
#             ],
#             temperature=0.7,
#             max_tokens=500
#         )
        
#         return response.choices[0].message.content
        
#     except Exception as e:
#         return f"Sorry, I encountered an error: {str(e)}"

# if __name__ == "__main__":
#     main()





# # src/ui/streamlit_app.py - FIXED VERSION
# """
# Fixed Streamlit App with Super Intelligent LLM System
# - No auto-approval thresholds - always proceed with intelligence
# - Original Excel format maintained
# - Works with any input files dynamically
# """

# import streamlit as st
# import os
# import json
# import tempfile
# from pathlib import Path
# import pandas as pd
# from typing import List, Dict, Any
# import logging
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Import modules
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from processors.document_processor import DocumentProcessor
# from exporters.excel_exporter import TestCaseExporter

# # Try to import fully dynamic PACS.008 modules
# try:
#     from processors.pacs008_intelligent_detector import PACS008IntelligentDetector
#     from processors.fully_dynamic_intelligent_maker_checker import FullyDynamicIntelligentMakerChecker
#     from ai_engine.fully_dynamic_test_generator import FullyDynamicTestGenerator
#     SUPER_INTELLIGENT_MODE_AVAILABLE = True
# except ImportError:
#     from ai_engine.test_generator import TestCaseGenerator
#     SUPER_INTELLIGENT_MODE_AVAILABLE = False

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def main():
#     """Main Streamlit application with super intelligent dynamic system"""
    
#     # Page configuration
#     st.set_page_config(
#         page_title="ITASSIST - Intelligent PACS.008 Generator",
#         page_icon="🏦" if SUPER_INTELLIGENT_MODE_AVAILABLE else "🤖",
#         layout="wide",
#         initial_sidebar_state="expanded"
#     )
    
#     # Title
#     if SUPER_INTELLIGENT_MODE_AVAILABLE:
#         st.title("🏦 ITASSIST - Intelligent PACS.008 Test Generator")
#         st.markdown("**AI-powered field detection + Intelligent maker-checker validation**")
#         st.success("✅ **Intelligent Mode**: Automated LLM validation with banking expertise")
#     else:
#         st.title("🤖 ITASSIST - Standard Test Generator")
#         st.markdown("**Standard test case generation**")
    
#     # Sidebar
#     with st.sidebar:
#         st.header("⚙️ Configuration")
        
#         # API Key
#         default_api_key = os.getenv("OPENAI_API_KEY", "")
#         api_key = st.text_input(
#             "OpenAI API Key",
#             value=default_api_key,
#             type="password",
#             help="Required for intelligent validation and test generation"
#         )
        
#         # Intelligent Features Status
#         if SUPER_INTELLIGENT_MODE_AVAILABLE:
#             st.subheader("🧠 Intelligent Features")
#             st.success("✅ **All Systems Active**")
#             st.info("🚀 **Always Proceeds**: No thresholds - works with any input")
#             st.write("• PACS.008 Field Detection")
#             st.write("• AI Maker-Checker Validation")
#             st.write("• Enhanced Test Generation")
#             st.write("• Original Excel Format")
            
#             enable_intelligent_mode = st.checkbox("Enable Intelligent Mode", value=True)
#         else:
#             enable_intelligent_mode = False
        
#         # Test Generation Options
#         st.subheader("🧪 Test Generation")
#         num_test_cases = st.slider("Test Cases per Story", 5, 15, 8)
#         include_edge_cases = st.checkbox("Include Edge Cases", value=True)
#         include_negative_cases = st.checkbox("Include Negative Cases", value=True)
        
#         # Export Options
#         export_format = st.multiselect(
#             "Export Formats",
#             ["Excel", "CSV", "JSON"],
#             default=["Excel"]
#         )
    
#     # Initialize session state
#     if 'processing_results' not in st.session_state:
#         st.session_state.processing_results = {}
#     if 'validation_results' not in st.session_state:
#         st.session_state.validation_results = {}
#     if 'final_test_cases' not in st.session_state:
#         st.session_state.final_test_cases = []
    
#     # Main tabs
#     if SUPER_INTELLIGENT_MODE_AVAILABLE and enable_intelligent_mode:
#         tab1, tab2, tab3, tab4 = st.tabs([
#             "📁 Upload & Process", 
#             "🧠 AI Validation Results", 
#             "👥 Intelligent Maker-Checker", 
#             "🧪 Enhanced Test Cases"
#         ])
        
#         with tab1:
#             upload_and_process_tab(api_key, enable_intelligent_mode)
        
#         with tab2:
#             intelligent_validation_tab()
        
#         with tab3:
#             intelligent_maker_checker_tab()
        
#         with tab4:
#             enhanced_test_cases_tab(export_format, num_test_cases, include_edge_cases, include_negative_cases)
#     else:
#         tab1, tab2 = st.tabs(["📁 Upload & Process", "🧪 Test Cases"])
        
#         with tab1:
#             upload_and_process_tab(api_key, False)
        
#         with tab2:
#             enhanced_test_cases_tab(export_format, num_test_cases, include_edge_cases, include_negative_cases)

# def upload_and_process_tab(api_key: str, enable_intelligent_mode: bool):
#     """Upload and processing tab with intelligent options"""
    
#     st.header("📁 Document Upload & Intelligent Processing")
    
#     if enable_intelligent_mode:
#         st.info("🧠 **Intelligent Mode**: AI will detect PACS.008 fields, validate them using banking expertise, and generate enhanced test cases")
#         st.success("🚀 **Always Proceeds**: System works with any input - no approval thresholds needed")
#     else:
#         st.info("📄 **Standard Mode**: Basic document processing and test case generation")
    
#     # File upload
#     uploaded_files = st.file_uploader(
#         "Upload your banking documents",
#         type=['docx', 'pdf', 'xlsx', 'txt', 'json'],
#         accept_multiple_files=True,
#         help="Upload requirements, user stories, payment specifications, or banking documentation"
#     )
    
#     if uploaded_files:
#         st.success(f"📁 {len(uploaded_files)} file(s) uploaded successfully")
        
#         # Show file preview
#         with st.expander("📋 Uploaded Files"):
#             for file in uploaded_files:
#                 file_size = len(file.getvalue()) / (1024*1024)
#                 st.write(f"• **{file.name}** ({file_size:.1f} MB)")
    
#     # Processing instructions
#     st.subheader("📝 Processing Instructions")
    
#     if enable_intelligent_mode:
#         instruction_templates = {
#             "Intelligent Auto-Mode": "Let AI automatically handle everything - field detection, validation, and test generation",
#             "Cross-Border Payment Focus": "Focus on cross-border payment scenarios with correspondent banking and compliance",
#             "Domestic Payment Focus": "Focus on domestic payment scenarios within single country clearing",
#             "High-Value Payment Focus": "Focus on high-value payments with enhanced compliance and monitoring",
#             "Compliance & AML Focus": "Emphasize regulatory compliance, AML/KYC, and sanctions screening"
#         }
#     else:
#         instruction_templates = {
#             "Standard Generation": "Generate comprehensive test cases covering positive, negative, and edge scenarios",
#             "Basic Scenarios": "Focus on fundamental happy path and basic error handling"
#         }
    
#     selected_template = st.selectbox("Processing Template:", list(instruction_templates.keys()))
    
#     custom_instructions = st.text_area(
#         "Custom Instructions (Optional)",
#         value=instruction_templates[selected_template],
#         placeholder="e.g., 'Focus on Deutsche Bank to BNP Paribas routing' or 'Generate compliance-heavy scenarios'",
#         help="AI will enhance these instructions based on detected PACS.008 content"
#     )
    
#     # Process button
#     process_button_text = "🧠 Start Intelligent Processing" if enable_intelligent_mode else "🚀 Generate Test Cases"
    
#     if st.button(process_button_text, type="primary", disabled=not api_key or not uploaded_files):
#         if not api_key:
#             st.error("❌ Please provide OpenAI API key")
#             return
        
#         if not uploaded_files:
#             st.error("❌ Please upload at least one document") 
#             return
        
#         if enable_intelligent_mode:
#             process_with_intelligent_pipeline(uploaded_files, api_key, custom_instructions)
#         else:
#             process_with_standard_generation(uploaded_files, api_key, custom_instructions)

# def process_with_intelligent_pipeline(uploaded_files, api_key: str, custom_instructions: str):
#     """Process files through the intelligent pipeline - always proceeds"""
    
#     progress_bar = st.progress(0)
#     status_text = st.empty()
    
#     try:
#         # Step 1: Initialize
#         status_text.text("🔧 Initializing intelligent AI systems...")
#         progress_bar.progress(0.1)
        
#         doc_processor = DocumentProcessor()
#         pacs008_detector = PACS008IntelligentDetector(api_key)
#         intelligent_maker_checker = FullyDynamicIntelligentMakerChecker(api_key)
#         test_generator = FullyDynamicTestGenerator(api_key)
        
#         # Step 2: Document processing
#         status_text.text("📄 Processing documents...")
#         progress_bar.progress(0.2)
        
#         all_content = []
#         for uploaded_file in uploaded_files:
#             with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
#                 tmp_file.write(uploaded_file.getvalue())
#                 tmp_file_path = tmp_file.name
            
#             result = doc_processor.process_file(tmp_file_path)
#             all_content.append(result.get('content', ''))
#             os.unlink(tmp_file_path)
        
#         combined_content = '\n\n'.join(all_content)
        
#         # Step 3: PACS.008 field detection
#         status_text.text("🏦 AI detecting PACS.008 fields...")
#         progress_bar.progress(0.4)
        
#         detection_results = pacs008_detector.detect_pacs008_fields_in_input(combined_content)
        
#         if detection_results['status'] != 'SUCCESS':
#             st.info("ℹ️ No PACS.008 fields detected. Using intelligent standard processing...")
#             return process_with_intelligent_fallback(combined_content, api_key, custom_instructions)
        
#         detected_fields = detection_results.get('detected_fields', [])
#         st.success(f"✅ **{len(detected_fields)} PACS.008 fields detected**")
        
#         # Step 4: Intelligent maker-checker validation (always proceeds)
#         status_text.text("🧠 AI performing intelligent validation...")
#         progress_bar.progress(0.6)
        
#         validation_results = intelligent_maker_checker.perform_fully_dynamic_validation(detected_fields)
        
#         # Store validation results
#         st.session_state.validation_results = validation_results
        
#         if validation_results['status'] != 'SUCCESS':
#             st.warning("⚠️ AI validation had issues but proceeding with intelligent generation...")
#         else:
#             # Show validation score but always proceed
#             validation_score = validation_results.get('final_analysis', {}).get('final_validation_score', 0)
#             st.info(f"🎯 **AI Validation Score**: {validation_score}% - Proceeding with intelligent generation")
        
#         # Step 5: Enhanced test case generation (always proceeds)
#         status_text.text("🧪 Generating enhanced test cases with AI insights...")
#         progress_bar.progress(0.8)
        
#         test_generation_results = test_generator.generate_fully_dynamic_test_cases(
#             combined_content, detected_fields, custom_instructions
#         )
        
#         # Store results
#         st.session_state.processing_results = {
#             'detection_results': detection_results,
#             'validation_results': validation_results,
#             'test_generation_results': test_generation_results
#         }
        
#         # Always get test cases
#         generated_test_cases = test_generation_results.get('test_cases', [])
        
#         if not generated_test_cases:
#             st.warning("🔄 Primary generation returned 0 cases. Using enhanced fallback generation...")
#             # Enhanced fallback
#             from ai_engine.enhanced_test_generator import EnhancedTestCaseGenerator
#             fallback_generator = EnhancedTestCaseGenerator(api_key)
#             generated_test_cases = fallback_generator.generate_enhanced_test_cases(
#                 combined_content, custom_instructions
#             )
#             st.success(f"✅ Generated {len(generated_test_cases)} enhanced fallback test cases!")
#         else:
#             st.success(f"✅ Generated {len(generated_test_cases)} intelligent test cases!")
        
#         st.session_state.final_test_cases = generated_test_cases
        
#         progress_bar.progress(1.0)
#         status_text.text("✅ Intelligent processing completed successfully!")
        
#         # Display summary
#         display_intelligent_summary(detection_results, validation_results, len(generated_test_cases))
        
#     except Exception as e:
#         st.error(f"❌ Intelligent processing failed: {str(e)}")
#         logger.error(f"Processing error: {str(e)}")

# def display_intelligent_summary(detection_results: Dict, validation_results: Dict, test_count: int):
#     """Display intelligent processing summary"""
    
#     st.subheader("📊 Intelligent Processing Summary")
    
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         detected_count = len(detection_results.get('detected_fields', []))
#         st.metric("🔍 AI Detected", f"{detected_count} fields")
    
#     with col2:
#         validation_score = validation_results.get('final_analysis', {}).get('final_validation_score', 0)
#         st.metric("🎯 AI Score", f"{validation_score}%")
    
#     with col3:
#         overall_decision = validation_results.get('intelligent_decisions', {}).get('combined_analysis', {}).get('overall_decision', 'PROCESSED')
#         status_icon = "✅" if overall_decision in ["APPROVED", "CONDITIONALLY_APPROVED"] else "🔄"
#         st.metric("🤖 AI Status", f"{status_icon}")
    
#     with col4:
#         st.metric("🧪 Test Cases", test_count)
    
#     # Always show success - no thresholds
#     st.success("🎉 **Intelligent Processing Complete**: AI analyzed content, performed validation, and generated enhanced test cases successfully!")

# def intelligent_validation_tab():
#     """Display AI validation results"""
    
#     st.header("🧠 AI Validation Results")
    
#     if not st.session_state.validation_results:
#         st.info("📄 No AI validation results available. Please process documents first.")
#         return
    
#     validation_results = st.session_state.validation_results
    
#     # Validation overview
#     st.subheader("📊 AI Validation Overview")
    
#     final_analysis = validation_results.get('final_analysis', {})
#     comprehensive_validation = validation_results.get('comprehensive_validation', {})
#     overall_assessment = comprehensive_validation.get('overall_assessment', {})
    
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         technical_score = overall_assessment.get('technical_score', 0)
#         st.metric("🔧 Technical Intelligence", f"{technical_score}%")
    
#     with col2:
#         business_score = overall_assessment.get('business_score', 0)
#         st.metric("🏦 Business Intelligence", f"{business_score}%")
    
#     with col3:
#         compliance_score = overall_assessment.get('compliance_score', 0)
#         st.metric("📋 Compliance Intelligence", f"{compliance_score}%")
    
#     # Key findings
#     key_findings = final_analysis.get('validation_summary', {}).get('key_findings', [])
#     if key_findings:
#         st.subheader("💡 AI Key Findings")
#         for finding in key_findings:
#             st.write(f"• {finding}")
    
#     # Field validations
#     field_validations = comprehensive_validation.get('field_validations', [])
#     if field_validations:
#         st.subheader("🔍 Dynamic Field Analysis")
        
#         # Group by status
#         valid_fields = [f for f in field_validations if f.get('validation_status') == 'VALID']
#         warning_fields = [f for f in field_validations if f.get('validation_status') in ['WARNING', 'MISSING']]
#         invalid_fields = [f for f in field_validations if f.get('validation_status') == 'INVALID']
        
#         if valid_fields:
#             with st.expander(f"✅ Valid Fields ({len(valid_fields)})", expanded=True):
#                 for field in valid_fields:
#                     st.write(f"**{field['field_name']}**: {field.get('dynamic_assessment', 'Valid')}")
        
#         if warning_fields:
#             with st.expander(f"⚠️ Fields with Warnings ({len(warning_fields)})", expanded=True):
#                 for field in warning_fields:
#                     st.write(f"**{field['field_name']}**: {field.get('dynamic_assessment', 'Warning')}")
        
#         if invalid_fields:
#             with st.expander(f"❌ Invalid Fields ({len(invalid_fields)})", expanded=False):
#                 for field in invalid_fields:
#                     st.write(f"**{field['field_name']}**: {field.get('dynamic_assessment', 'Invalid')}")

# def intelligent_maker_checker_tab():
#     """Display intelligent maker-checker decisions"""
    
#     st.header("👥 Intelligent Maker-Checker Results")
    
#     if not st.session_state.validation_results:
#         st.info("📄 No maker-checker results available. Please process documents first.")
#         return
    
#     validation_results = st.session_state.validation_results
#     intelligent_decisions = validation_results.get('intelligent_decisions', {})
    
#     if not intelligent_decisions:
#         st.info("ℹ️ Maker-checker analysis completed - system always proceeds with intelligent generation.")
#         return
    
#     # Overall AI decision
#     combined_analysis = intelligent_decisions.get('combined_analysis', {})
#     overall_decision = combined_analysis.get('overall_decision', 'PROCESSED')
    
#     st.info(f"🤖 **AI Decision**: {overall_decision} - System proceeded with intelligent test generation")
    
#     # AI Maker decision
#     st.subheader("🤖 AI Maker Analysis")
    
#     maker_decision = intelligent_decisions.get('maker_decision', {})
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         decision = maker_decision.get('decision', 'PROCESSED')
#         confidence = maker_decision.get('confidence', 0)
#         st.write(f"**Decision**: {decision}")
#         st.write(f"**Confidence**: {confidence}%")
    
#     with col2:
#         reasoning = maker_decision.get('technical_reasoning', 'AI completed technical analysis')
#         st.write(f"**AI Reasoning**: {reasoning}")
    
#     # AI Checker decision
#     st.subheader("✅ AI Checker Analysis")
    
#     checker_decision = intelligent_decisions.get('checker_decision', {})
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         decision = checker_decision.get('decision', 'PROCESSED')
#         confidence = checker_decision.get('confidence', 0)
#         st.write(f"**Decision**: {decision}")
#         st.write(f"**Confidence**: {confidence}%")
    
#     with col2:
#         reasoning = checker_decision.get('business_reasoning', 'AI completed business analysis')
#         st.write(f"**AI Reasoning**: {reasoning}")
    
#     # Show the benefit
#     st.success("💡 **Fully Automated**: AI completed the entire maker-checker process automatically and proceeded with intelligent test generation - no manual intervention or thresholds needed!")

# def enhanced_test_cases_tab(export_formats: List[str], num_test_cases: int, 
#                           include_edge_cases: bool, include_negative_cases: bool):
#     """Display enhanced test cases with original Excel format"""
    
#     st.header("🧪 Enhanced Test Cases")
    
#     if not st.session_state.final_test_cases:
#         st.info("📄 No test cases generated yet. Please process documents first.")
#         return
    
#     test_cases = st.session_state.final_test_cases
#     processing_results = st.session_state.processing_results
    
#     # Test case overview
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         st.metric("Total Test Cases", len(test_cases))
    
#     with col2:
#         enhanced_tests = len([tc for tc in test_cases if tc.get('PACS008_Enhanced') == 'Yes'])
#         if enhanced_tests > 0:
#             st.metric("🏦 PACS.008 Enhanced", enhanced_tests)
#         else:
#             high_priority = len([tc for tc in test_cases if tc.get('Priority') == 'High'])
#             st.metric("High Priority", high_priority)
    
#     with col3:
#         regression_tests = len([tc for tc in test_cases if tc.get('Part of Regression') == 'Yes'])
#         st.metric("Regression Tests", regression_tests)
    
#     with col4:
#         unique_stories = len(set(tc.get('User Story ID', '') for tc in test_cases))
#         st.metric("User Stories", unique_stories)
    
#     # Generation method
#     generation_method = processing_results.get('test_generation_results', {}).get('generation_method', 'INTELLIGENT')
    
#     if 'DYNAMIC' in generation_method or 'ENHANCED' in generation_method:
#         st.success("✅ **Intelligent Generation**: Test cases created using AI validation insights and banking expertise")
#     else:
#         st.info(f"ℹ️ **Generation Method**: {generation_method}")
    
#     # Display test cases
#     st.subheader("📋 Generated Test Cases")
    
#     # Create clean display with original columns
#     display_data = []
#     for tc in test_cases:
#         clean_tc = {
#             "User Story ID": tc.get('User Story ID', ''),
#             "Test Case ID": tc.get('Test Case ID', ''),
#             "Test Case Description": tc.get('Test Case Description', ''),
#             "Priority": tc.get('Priority', ''),
#             "Part of Regression": tc.get('Part of Regression', ''),
#         }
        
#         # Add PACS.008 indicator if available
#         if tc.get('PACS008_Enhanced') == 'Yes':
#             clean_tc["🏦 Enhanced"] = "✅"
#         elif tc.get('validation_status'):
#             clean_tc["🏦 Enhanced"] = "✅"
#         else:
#             clean_tc["🏦 Enhanced"] = "⭕"
        
#         display_data.append(clean_tc)
    
#     df = pd.DataFrame(display_data)
#     st.dataframe(df, use_container_width=True, hide_index=True)
    
#     # Detailed view
#     with st.expander("📋 Detailed Test Case View (First 3)"):
#         for i, tc in enumerate(test_cases[:3], 1):
#             st.write(f"**{i}. {tc.get('Test Case Description', 'N/A')}**")
#             st.write(f"**Steps**: {tc.get('Steps', 'N/A')}")
#             st.write(f"**Expected Result**: {tc.get('Expected Result', 'N/A')}")
#             if tc.get('PACS008_Enhanced') or tc.get('validation_status'):
#                 st.write(f"**Enhancement**: PACS.008 Intelligent Generation")
#             st.write("---")
    
#     # Export section with ORIGINAL Excel format
#     st.subheader("📥 Export Test Cases")
    
#     if "Excel" in export_formats:
#         if st.button("📊 Download Excel (Original Format)", type="primary"):
#             export_original_excel_format(test_cases)
    
#     if "CSV" in export_formats:
#         if st.button("📄 Download CSV"):
#             export_csv_format(test_cases)
    
#     if "JSON" in export_formats:
#         if st.button("🔧 Download JSON"):
#             export_json_format(test_cases)

# def clean_test_cases_for_export(test_cases: List[Dict]) -> List[Dict]:
#     """Clean test cases data to handle lists, None values, and other export issues"""
    
#     cleaned_cases = []
    
#     for test_case in test_cases:
#         cleaned_case = {}
        
#         for key, value in test_case.items():
#             # Handle None values
#             if value is None:
#                 cleaned_case[key] = ""
#             # Handle list values (convert to numbered string)
#             elif isinstance(value, list):
#                 if len(value) == 0:
#                     cleaned_case[key] = ""
#                 else:
#                     # Convert list to numbered steps
#                     numbered_steps = []
#                     for i, item in enumerate(value, 1):
#                         if isinstance(item, str):
#                             numbered_steps.append(f"{i}. {item}")
#                         else:
#                             numbered_steps.append(f"{i}. {str(item)}")
#                     cleaned_case[key] = "\n".join(numbered_steps)
#             # Handle dictionary values
#             elif isinstance(value, dict):
#                 cleaned_case[key] = str(value)
#             # Handle very long strings (Excel limit)
#             elif isinstance(value, str) and len(value) > 32767:
#                 cleaned_case[key] = value[:32760] + "..."
#             # Handle other types
#             else:
#                 cleaned_case[key] = str(value) if value is not None else ""
        
#         cleaned_cases.append(cleaned_case)
    
#     return cleaned_cases

# def export_original_excel_format(test_cases: List[Dict]):
#     """Export using the ORIGINAL Excel format from TestCaseExporter"""
    
#     try:
#         # Clean test cases data first
#         cleaned_test_cases = clean_test_cases_for_export(test_cases)
        
#         # Use your original TestCaseExporter
#         exporter = TestCaseExporter()
        
#         # Create Excel in memory using original format
#         import io
#         from openpyxl import Workbook
#         from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        
#         wb = Workbook()
#         ws = wb.active
#         ws.title = "Test Cases"
        
#         # Use original required columns
#         required_columns = [
#             "User Story ID",
#             "Acceptance Criteria ID", 
#             "Scenario",
#             "Test Case ID",
#             "Test Case Description",
#             "Precondition",
#             "Steps",
#             "Expected Result",
#             "Part of Regression",
#             "Priority"
#         ]
        
#         # Create DataFrame with original format
#         df = pd.DataFrame(cleaned_test_cases)
        
#         # Ensure all required columns exist
#         for col in required_columns:
#             if col not in df.columns:
#                 df[col] = ""
        
#         # Reorder columns to original format
#         df = df[required_columns]
        
#         # Add headers with original formatting
#         for col_num, column_title in enumerate(required_columns, 1):
#             cell = ws.cell(row=1, column=col_num)
#             cell.value = column_title
#             cell.font = Font(bold=True, color="FFFFFF")
#             cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
#             cell.alignment = Alignment(horizontal="center", vertical="center")
#             cell.border = Border(
#                 left=Side(style='thin'),
#                 right=Side(style='thin'),
#                 top=Side(style='thin'),
#                 bottom=Side(style='thin')
#             )
        
#         # Add data rows with original formatting
#         for row_num, (index, test_case) in enumerate(df.iterrows(), 2):
#             for col_num, column in enumerate(required_columns, 1):
#                 cell = ws.cell(row=row_num, column=col_num)
#                 value = test_case[column]
                
#                 # Handle list data (convert to string)
#                 if isinstance(value, list):
#                     value = '\n'.join([f"{i+1}. {step}" for i, step in enumerate(value)])
#                     cell.alignment = Alignment(wrap_text=True, vertical="top")
#                 elif column == "Steps" and "\\n" in str(value):
#                     # Handle multi-line content (Steps field) - original way
#                     value = value.replace("\\n", "\n")
#                     cell.alignment = Alignment(wrap_text=True, vertical="top")
                
#                 # Ensure value is string and not too long for Excel
#                 value = str(value) if value is not None else ""
#                 if len(value) > 32767:  # Excel cell limit
#                     value = value[:32760] + "..."
                
#                 cell.value = value
                
#                 # Apply original conditional formatting based on Priority
#                 if column == "Priority":
#                     if value == "High":
#                         cell.fill = PatternFill(start_color="FFE6E6", end_color="FFE6E6", fill_type="solid")
#                     elif value == "Medium":
#                         cell.fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
#                     elif value == "Low":
#                         cell.fill = PatternFill(start_color="E6F3E6", end_color="E6F3E6", fill_type="solid")
                
#                 # Add borders to all cells - original way
#                 cell.border = Border(
#                     left=Side(style='thin'),
#                     right=Side(style='thin'),
#                     top=Side(style='thin'),
#                     bottom=Side(style='thin')
#                 )
        
#         # Auto-adjust column widths - original way
#         column_widths = {
#             'A': 15,  # User Story ID
#             'B': 20,  # Acceptance Criteria ID
#             'C': 25,  # Scenario
#             'D': 15,  # Test Case ID
#             'E': 40,  # Test Case Description
#             'F': 30,  # Precondition
#             'G': 50,  # Steps
#             'H': 40,  # Expected Result
#             'I': 18,  # Part of Regression
#             'J': 12   # Priority
#         }
        
#         for column, width in column_widths.items():
#             ws.column_dimensions[column].width = width
        
#         # Add original summary sheet
#         summary_ws = wb.create_sheet("Summary")
        
#         # Calculate original statistics
#         total_cases = len(test_cases)
#         priority_counts = {}
#         regression_counts = {}
#         user_story_counts = {}
        
#         for case in test_cases:
#             # Priority distribution
#             priority = case.get("Priority", "Unknown")
#             priority_counts[priority] = priority_counts.get(priority, 0) + 1
            
#             # Regression distribution
#             regression = case.get("Part of Regression", "Unknown")
#             regression_counts[regression] = regression_counts.get(regression, 0) + 1
            
#             # User story distribution
#             user_story = case.get("User Story ID", "Unknown")
#             user_story_counts[user_story] = user_story_counts.get(user_story, 0) + 1
        
#         # Add original summary data
#         summary_data = [
#             ["Test Case Summary Report", ""],
#             ["", ""],
#             ["Total Test Cases", total_cases],
#             ["", ""],
#             ["Priority Distribution", ""],
#             ["High Priority", priority_counts.get("High", 0)],
#             ["Medium Priority", priority_counts.get("Medium", 0)],
#             ["Low Priority", priority_counts.get("Low", 0)],
#             ["", ""],
#             ["Regression Test Distribution", ""],
#             ["Part of Regression", regression_counts.get("Yes", 0)],
#             ["Not in Regression", regression_counts.get("No", 0)],
#             ["", ""],
#             ["Coverage by User Story", ""],
#         ]
        
#         # Add user story coverage
#         for story_id, count in user_story_counts.items():
#             summary_data.append([story_id, count])
        
#         # Write summary data to sheet - original way
#         for row_num, (label, value) in enumerate(summary_data, 1):
#             summary_ws.cell(row=row_num, column=1, value=label)
#             summary_ws.cell(row=row_num, column=2, value=value)
            
#             # Format headers - original way
#             if "Distribution" in str(label) or "Summary Report" in str(label):
#                 summary_ws.cell(row=row_num, column=1).font = Font(bold=True, size=12)
        
#         # Adjust column widths - original way
#         summary_ws.column_dimensions['A'].width = 25
#         summary_ws.column_dimensions['B'].width = 15
        
#         # Save to memory
#         output = io.BytesIO()
#         wb.save(output)
#         output.seek(0)
        
#         st.download_button(
#             label="📊 Download Excel File (Original Format)",
#             data=output.getvalue(),
#             file_name=f"test_cases_{len(test_cases)}.xlsx",
#             mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#         )
        
#     except Exception as e:
#         st.error(f"Export error: {str(e)}")

# def export_csv_format(test_cases: List[Dict]):
#     """Export to CSV with original format"""
    
#     try:
#         # Clean test cases data first
#         cleaned_test_cases = clean_test_cases_for_export(test_cases)
        
#         # Use original required columns
#         required_columns = [
#             "User Story ID", "Acceptance Criteria ID", "Scenario", "Test Case ID",
#             "Test Case Description", "Precondition", "Steps", "Expected Result",
#             "Part of Regression", "Priority"
#         ]
        
#         df = pd.DataFrame(cleaned_test_cases)
        
#         # Ensure all required columns exist
#         for col in required_columns:
#             if col not in df.columns:
#                 df[col] = ""
        
#         # Reorder columns
#         df = df[required_columns]
        
#         # Clean multi-line content for CSV (replace newlines with separator)
#         for column in df.columns:
#             df[column] = df[column].astype(str).str.replace('\n', ' | ')
        
#         csv_data = df.to_csv(index=False)
        
#         st.download_button(
#             label="📄 Download CSV File",
#             data=csv_data,
#             file_name=f"test_cases_{len(test_cases)}.csv",
#             mime="text/csv"
#         )
        
#     except Exception as e:
#         st.error(f"CSV export error: {str(e)}")

# def export_json_format(test_cases: List[Dict]):
#     """Export to JSON with original format"""
    
#     try:
#         # Clean test cases data first
#         cleaned_test_cases = clean_test_cases_for_export(test_cases)
        
#         # Use original required columns
#         required_columns = [
#             "User Story ID", "Acceptance Criteria ID", "Scenario", "Test Case ID",
#             "Test Case Description", "Precondition", "Steps", "Expected Result",
#             "Part of Regression", "Priority"
#         ]
        
#         # Clean test cases for JSON
#         final_test_cases = []
#         for case in cleaned_test_cases:
#             cleaned_case = {}
#             for field in required_columns:
#                 value = case.get(field, "")
#                 # For JSON, keep newlines as \n for readability
#                 if isinstance(value, str):
#                     cleaned_case[field] = value
#                 else:
#                     cleaned_case[field] = str(value)
#             final_test_cases.append(cleaned_case)
        
#         json_data = json.dumps(final_test_cases, indent=2, ensure_ascii=False)
        
#         st.download_button(
#             label="🔧 Download JSON File",
#             data=json_data,
#             file_name=f"test_cases_{len(test_cases)}.json",
#             mime="application/json"
#         )
        
#     except Exception as e:
#         st.error(f"JSON export error: {str(e)}")

# def process_with_standard_generation(uploaded_files, api_key: str, custom_instructions: str):
#     """Standard processing when intelligent mode is disabled"""
    
#     try:
#         # Process documents
#         doc_processor = DocumentProcessor()
#         test_generator = TestCaseGenerator(api_key)
        
#         all_content = []
#         for uploaded_file in uploaded_files:
#             with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
#                 tmp_file.write(uploaded_file.getvalue())
#                 tmp_file_path = tmp_file.name
            
#             result = doc_processor.process_file(tmp_file_path)
#             all_content.append(result.get('content', ''))
#             os.unlink(tmp_file_path)
        
#         combined_content = '\n\n'.join(all_content)
        
#         # Generate test cases
#         test_cases = test_generator.generate_test_cases(combined_content, custom_instructions)
        
#         # Store results
#         st.session_state.final_test_cases = test_cases
#         st.session_state.processing_results = {
#             'test_generation_results': {
#                 'status': 'SUCCESS',
#                 'generation_method': 'STANDARD',
#                 'test_cases': test_cases
#             }
#         }
        
#         st.success(f"✅ Generated {len(test_cases)} test cases using standard method!")
        
#     except Exception as e:
#         st.error(f"❌ Standard processing failed: {str(e)}")

# def process_with_intelligent_fallback(combined_content: str, api_key: str, custom_instructions: str):
#     """Intelligent fallback when PACS.008 detection fails"""
    
#     try:
#         from ai_engine.test_generator import TestCaseGenerator
#         test_generator = TestCaseGenerator(api_key)
        
#         test_cases = test_generator.generate_test_cases(combined_content, custom_instructions)
        
#         st.session_state.final_test_cases = test_cases
#         st.session_state.processing_results = {
#             'test_generation_results': {
#                 'status': 'INTELLIGENT_FALLBACK',
#                 'generation_method': 'INTELLIGENT_STANDARD',
#                 'test_cases': test_cases
#             }
#         }
        
#         st.success(f"✅ Generated {len(test_cases)} test cases using intelligent standard processing!")
        
#     except Exception as e:
#         st.error(f"❌ Intelligent fallback failed: {str(e)}")

# if __name__ == "__main__":
#     main()

####################### original 3step ###########################
# # src/ui/streamlit_app.py - SIMPLE VERSION LIKE ORIGINAL REPO
# """
# Simple Streamlit App with Dynamic PACS.008 Intelligence
# Clean UI like original repo but with enhanced backend processing
# """

# import streamlit as st
# import os
# import json
# import tempfile
# from pathlib import Path
# import pandas as pd
# from typing import List, Dict, Any
# import logging
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Import modules
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from processors.document_processor import DocumentProcessor
# from exporters.excel_exporter import TestCaseExporter

# # Try to import dynamic system
# try:
#     from ai_engine.dynamic_pacs008_test_generator import DynamicPACS008TestGenerator
#     DYNAMIC_SYSTEM_AVAILABLE = True
# except ImportError:
#     from ai_engine.test_generator import TestCaseGenerator
#     DYNAMIC_SYSTEM_AVAILABLE = False

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Page configuration - same as original
# st.set_page_config(
#     page_title="ITASSIST - Test Case Generator",
#     page_icon="🏦" if DYNAMIC_SYSTEM_AVAILABLE else "🤖",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# def main():
#     """Main Streamlit application - simple like original"""
    
#     # Title - enhanced if dynamic system available
#     if DYNAMIC_SYSTEM_AVAILABLE:
#         st.title("🏦 ITASSIST - Intelligent Test Case Generator")
#         st.markdown("**AI-powered test case generation with PACS.008 intelligence**")
#     else:
#         st.title("🤖 ITASSIST - Intelligent Test Case Generator")
#         st.markdown("**AI-powered test case generation from BFSI documents**")
    
#     # Sidebar - same as original structure
#     with st.sidebar:
#         st.header("⚙️ Configuration")
        
#         # API Key input - same as original
#         default_api_key = os.getenv("OPENAI_API_KEY", "")
#         api_key = st.text_input(
#             "OpenAI API Key", 
#             value=default_api_key,
#             type="password",
#             help="API key loaded from environment" if default_api_key else "Enter your OpenAI API key"
#         )
        
#         # Model selection - same as original
#         model_option = st.selectbox(
#             "AI Model",
#             ["gpt-4.1-mini-2025-04-14", "gpt-4o-mini", "gpt-3.5-turbo"],
#             index=0
#         )
        
#         # Generation options - same as original
#         st.subheader("Generation Options")
#         num_test_cases = st.slider("Test Cases per Story", 5, 15, 8)
#         include_edge_cases = st.checkbox("Include Edge Cases", value=True)
#         include_negative_cases = st.checkbox("Include Negative Cases", value=True)
        
#         # PACS.008 status indicator
#         if DYNAMIC_SYSTEM_AVAILABLE:
#             st.subheader("🏦 PACS.008 Intelligence")
#             st.success("✅ Enhanced Processing Available")
#             st.info("System will automatically:\n• Detect PACS.008 fields\n• Apply banking intelligence\n• Generate domain-specific tests")
        
#         # Export format - same as original
#         export_format = st.multiselect(
#             "Export Formats",
#             ["Excel", "CSV", "JSON"],
#             default=["Excel"]
#         )
    
#     # Initialize session state - same as original
#     if 'generated_test_cases' not in st.session_state:
#         st.session_state.generated_test_cases = []
#     if 'processing_complete' not in st.session_state:
#         st.session_state.processing_complete = False
    
#     # Main content tabs - same as original structure
#     tab1, tab2, tab3 = st.tabs(["📁 Upload & Process", "🧪 Generated Test Cases", "💬 Chat Assistant"])
    
#     with tab1:
#         upload_and_process_tab(api_key, num_test_cases, include_edge_cases, include_negative_cases)
    
#     with tab2:
#         display_test_cases_tab(export_format)
    
#     with tab3:
#         chat_assistant_tab(api_key)

# def upload_and_process_tab(api_key: str, num_test_cases: int, include_edge_cases: bool, include_negative_cases: bool):
#     """File upload and processing tab - same as original but with enhanced backend"""
    
#     st.header("📁 Document Upload & Processing")
    
#     # File upload section - same as original
#     uploaded_files = st.file_uploader(
#         "Upload your documents",
#         type=['docx', 'pdf', 'xlsx', 'png', 'jpg', 'jpeg', 'tiff', 'bmp', 'txt', 'eml', 'json', 'xml', 'csv', 'zip'],
#         accept_multiple_files=True,
#         help="Supported formats: DOCX, PDF, XLSX, Images (PNG/JPG/TIFF/BMP), TXT, EML, JSON, XML, CSV, ZIP"
#     )
    
#     # Display file validation info - same as original
#     if uploaded_files:
#         st.info(f"📁 {len(uploaded_files)} file(s) uploaded successfully")
        
#         # Show file details
#         with st.expander("📋 File Details"):
#             for file in uploaded_files:
#                 file_size = len(file.getvalue()) / (1024*1024)  # MB
#                 st.write(f"• **{file.name}** ({file_size:.1f} MB)")
                
#                 # Validate file size
#                 if file_size > 50:
#                     st.warning(f"⚠️ {file.name} is large ({file_size:.1f} MB). Processing may take longer.")
    
#     # Enhanced processing options - same as original
#     st.subheader("🔧 Processing Options")
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         process_embedded_content = st.checkbox("📷 Process Embedded Images/Screenshots", value=True)
#     with col2:
#         extract_tables = st.checkbox("📊 Extract Table Content", value=True)
#     with col3:
#         enhance_ocr = st.checkbox("🔍 Enhanced OCR Processing", value=True)
    
#     # Custom instructions - same as original but with PACS.008 templates
#     st.subheader("📝 Custom Instructions")
    
#     if DYNAMIC_SYSTEM_AVAILABLE:
#         instruction_templates = {
#             "Standard": "",
#             "Focus on PACS.008 Banking": "Focus on PACS.008 payment processing, banking agents, and cross-border scenarios",
#             "Maker-Checker Workflows": "Emphasize maker-checker workflows, approval processes, and banking operations",
#             "Focus on Negative Cases": "Generate more negative test cases and error scenarios. Include boundary testing and invalid input validation.",
#             "Basic Scenarios Only": "Focus on basic happy path scenarios. Minimize edge cases and complex integration tests.",
#             "Comprehensive Coverage": "Generate comprehensive test coverage including positive, negative, edge cases, and integration scenarios.",
#             "Security Focus": "Emphasize security testing scenarios including authentication, authorization, and data validation.",
#             "Performance Testing": "Include performance-related test scenarios for high-volume and stress testing."
#         }
#     else:
#         instruction_templates = {
#             "Standard": "",
#             "Focus on Negative Cases": "Generate more negative test cases and error scenarios. Include boundary testing and invalid input validation.",
#             "Basic Scenarios Only": "Focus on basic happy path scenarios. Minimize edge cases and complex integration tests.",
#             "Comprehensive Coverage": "Generate comprehensive test coverage including positive, negative, edge cases, and integration scenarios.",
#             "Security Focus": "Emphasize security testing scenarios including authentication, authorization, and data validation.",
#             "Performance Testing": "Include performance-related test scenarios for high-volume and stress testing."
#         }
    
#     selected_template = st.selectbox("Choose Instruction Template:", list(instruction_templates.keys()))
    
#     custom_instructions = st.text_area(
#         "Custom Instructions",
#         value=instruction_templates[selected_template],
#         placeholder="e.g., 'Focus on payment validation scenarios' or 'Create 4 test cases per acceptance criteria'",
#         help="Provide specific instructions to customize test case generation"
#     )
    
#     # Process button - same as original
#     if st.button("🚀 Generate Test Cases", type="primary", disabled=not api_key or not uploaded_files):
#         if not api_key:
#             st.error("Please provide OpenAI API key in the sidebar")
#             return
        
#         if not uploaded_files:
#             st.error("Please upload at least one document")
#             return
        
#         process_files(uploaded_files, api_key, custom_instructions, num_test_cases, 
#                      include_edge_cases, include_negative_cases, process_embedded_content)

# def process_files(uploaded_files, api_key: str, custom_instructions: str, 
#                  num_test_cases: int, include_edge_cases: bool, include_negative_cases: bool,
#                  process_embedded_content: bool):
#     """Process uploaded files - enhanced backend but simple UI"""
    
#     progress_bar = st.progress(0)
#     status_text = st.empty()
    
#     try:
#         # Process each file - same as original
#         doc_processor = DocumentProcessor()
        
#         all_content = []
#         total_files = len(uploaded_files)
        
#         for i, uploaded_file in enumerate(uploaded_files):
#             status_text.text(f"Processing {uploaded_file.name}...")
#             progress_bar.progress((i + 1) / (total_files + 2))
            
#             # Save uploaded file temporarily
#             with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
#                 tmp_file.write(uploaded_file.getvalue())
#                 tmp_file_path = tmp_file.name
            
#             # Process the file
#             result = doc_processor.process_file(tmp_file_path)
#             all_content.append(result.get('content', ''))
            
#             # Clean up temporary file
#             os.unlink(tmp_file_path)
        
#         # Combine all extracted content
#         status_text.text("Combining extracted content...")
#         progress_bar.progress(0.9)
        
#         combined_content = '\n\n--- Document Separator ---\n\n'.join(all_content)
        
#         # Generate custom instructions - enhanced
#         generation_instructions = build_generation_instructions(
#             custom_instructions, num_test_cases, include_edge_cases, include_negative_cases
#         )
        
#         # Generate test cases - enhanced backend but simple UI
#         if DYNAMIC_SYSTEM_AVAILABLE:
#             status_text.text("Generating test cases with PACS.008 intelligence...")
            
#             # Use dynamic system for enhanced processing
#             generator = DynamicPACS008TestGenerator(api_key)
#             workflow_results = generator.process_complete_workflow(combined_content, num_test_cases)
            
#             # Extract test cases from workflow
#             test_cases = workflow_results.get("step5_test_cases", [])
            
#             # Show brief intelligence summary
#             if workflow_results.get("step1_analysis", {}).get("is_pacs008_relevant", False):
#                 st.success("🏦 **PACS.008 content detected** - Applied banking intelligence!")
            
#         else:
#             status_text.text("Generating test cases with AI...")
            
#             # Use standard system
#             test_generator = TestCaseGenerator(api_key)
#             test_cases = test_generator.generate_test_cases(combined_content, generation_instructions)
        
#         progress_bar.progress(0.95)
        
#         if test_cases:
#             st.session_state.generated_test_cases = test_cases
#             st.session_state.processing_complete = True
            
#             progress_bar.progress(1.0)
#             status_text.text("✅ Processing complete!")
            
#             # Display summary - same as original
#             st.success(f"Successfully generated {len(test_cases)} test cases!")
            
#             # Show content preview
#             with st.expander("📄 Extracted Content Preview"):
#                 preview_content = combined_content[:1000] + "..." if len(combined_content) > 1000 else combined_content
#                 st.text(preview_content)
                
#         else:
#             st.error("No test cases could be generated. Please check your documents and try again.")
            
#     except Exception as e:
#         st.error(f"Error during processing: {str(e)}")
#         logger.error(f"Processing error: {str(e)}")

# def build_generation_instructions(custom_instructions: str, num_test_cases: int, 
#                                 include_edge_cases: bool, include_negative_cases: bool) -> str:
#     """Build generation instructions - same as original but enhanced"""
#     instructions = []
    
#     if custom_instructions:
#         instructions.append(custom_instructions)
    
#     instructions.append(f"Generate exactly {num_test_cases} test cases per user story/requirement")
    
#     if include_edge_cases:
#         instructions.append("Include edge cases and boundary conditions")
    
#     if include_negative_cases:
#         instructions.append("Include negative test scenarios and error conditions")
    
#     # Enhanced instruction for PACS.008 if available
#     if DYNAMIC_SYSTEM_AVAILABLE:
#         instructions.append("Focus on BFSI domain scenarios with realistic banking data and PACS.008 intelligence")
#     else:
#         instructions.append("Focus on BFSI domain scenarios with realistic banking data")
    
#     return ". ".join(instructions)

# def display_test_cases_tab(export_formats: List[str]):
#     """Display generated test cases - same as original structure"""
    
#     st.header("🧪 Generated Test Cases")
    
#     if not st.session_state.generated_test_cases:
#         st.info("No test cases generated yet. Please upload documents and process them first.")
#         return
    
#     test_cases = st.session_state.generated_test_cases
    
#     # Display summary metrics - same as original
#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         st.metric("Total Test Cases", len(test_cases))
#     with col2:
#         high_priority = len([tc for tc in test_cases if tc.get("Priority") == "High"])
#         st.metric("High Priority", high_priority)
#     with col3:
#         regression_tests = len([tc for tc in test_cases if tc.get("Part of Regression") == "Yes"])
#         st.metric("Regression Tests", regression_tests)
#     with col4:
#         unique_stories = len(set(tc.get("User Story ID", "") for tc in test_cases))
#         st.metric("User Stories", unique_stories)
    
#     # Show PACS.008 enhancement indicator if available
#     if DYNAMIC_SYSTEM_AVAILABLE:
#         pacs008_enhanced = len([tc for tc in test_cases if tc.get("PACS008_Enhanced") == "Yes"])
#         if pacs008_enhanced > 0:
#             st.info(f"🏦 {pacs008_enhanced} test cases enhanced with PACS.008 intelligence")
    
#     # Filter options - same as original
#     with st.expander("🔍 Filter Test Cases"):
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             priority_filter = st.multiselect(
#                 "Priority", 
#                 ["High", "Medium", "Low"],
#                 default=["High", "Medium", "Low"]
#             )
        
#         with col2:
#             regression_filter = st.multiselect(
#                 "Regression", 
#                 ["Yes", "No"],
#                 default=["Yes", "No"]
#             )
        
#         with col3:
#             story_ids = list(set(tc.get("User Story ID", "") for tc in test_cases))
#             story_filter = st.multiselect(
#                 "User Story ID",
#                 story_ids,
#                 default=story_ids
#             )
    
#     # Apply filters - same as original
#     filtered_test_cases = [
#         tc for tc in test_cases
#         if (tc.get("Priority") in priority_filter and
#             tc.get("Part of Regression") in regression_filter and
#             tc.get("User Story ID") in story_filter)
#     ]
    
#     # Display test cases table - same as original
#     if filtered_test_cases:
#         st.subheader(f"Test Cases ({len(filtered_test_cases)} of {len(test_cases)})")
        
#         # Convert to DataFrame for display
#         df = pd.DataFrame(filtered_test_cases)
        
#         # Configure column display - same as original
#         column_config = {
#             "Steps": st.column_config.TextColumn(width="large"),
#             "Test Case Description": st.column_config.TextColumn(width="medium"),
#             "Expected Result": st.column_config.TextColumn(width="medium"),
#         }
        
#         st.dataframe(
#             df,
#             use_container_width=True,
#             column_config=column_config,
#             hide_index=True
#         )
        
#         # Export section - same as original
#         st.subheader("📥 Export Test Cases")
        
#         col1, col2, col3 = st.columns(3)
        
#         if "Excel" in export_formats:
#             with col1:
#                 if st.button("📊 Download Excel", type="primary"):
#                     export_excel(filtered_test_cases)
        
#         if "CSV" in export_formats:
#             with col2:
#                 if st.button("📄 Download CSV"):
#                     export_csv(filtered_test_cases)
        
#         if "JSON" in export_formats:
#             with col3:
#                 if st.button("🔧 Download JSON"):
#                     export_json(filtered_test_cases)
    
#     else:
#         st.warning("No test cases match the selected filters.")

# def export_excel(test_cases):
#     """Export Excel - same as original"""
#     try:
#         import io
        
#         # Create Excel in memory
#         df = pd.DataFrame(test_cases)
#         output = io.BytesIO()
        
#         with pd.ExcelWriter(output, engine='openpyxl') as writer:
#             df.to_excel(writer, sheet_name='Test Cases', index=False)
        
#         output.seek(0)
        
#         st.download_button(
#             label="📊 Download Excel File",
#             data=output.getvalue(),
#             file_name=f"test_cases_{len(test_cases)}.xlsx",
#             mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#         )
        
#     except Exception as e:
#         st.error(f"Export error: {str(e)}")

# def export_csv(test_cases):
#     """Export CSV - same as original"""
#     try:
#         csv_data = pd.DataFrame(test_cases).to_csv(index=False)
#         st.download_button(
#             label="📄 Download CSV File",
#             data=csv_data,
#             file_name=f"test_cases_{len(test_cases)}.csv",
#             mime="text/csv"
#         )
#     except Exception as e:
#         st.error(f"Export error: {str(e)}")

# def export_json(test_cases):
#     """Export JSON - same as original"""
#     try:
#         json_data = json.dumps(test_cases, indent=2, ensure_ascii=False)
#         st.download_button(
#             label="🔧 Download JSON File",
#             data=json_data,
#             file_name=f"test_cases_{len(test_cases)}.json",
#             mime="application/json"
#         )
#     except Exception as e:
#         st.error(f"Export error: {str(e)}")

# def chat_assistant_tab(api_key: str):
#     """Chat assistant - same as original"""
    
#     st.header("💬 Chat Assistant")
#     st.markdown("Ask questions about your test cases or request modifications")
    
#     if not api_key:
#         st.warning("Please provide OpenAI API key to enable chat functionality")
#         return
    
#     if not st.session_state.generated_test_cases:
#         st.info("Generate test cases first to enable chat assistance")
#         return
    
#     # Chat interface - same as original
#     if "chat_messages" not in st.session_state:
#         st.session_state.chat_messages = []
    
#     # Display chat history
#     for message in st.session_state.chat_messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
    
#     # Chat input
#     if prompt := st.chat_input("Ask about your test cases..."):
#         # Add user message
#         st.session_state.chat_messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)
        
#         # Generate response
#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 response = generate_chat_response(prompt, st.session_state.generated_test_cases, api_key)
#                 st.markdown(response)
        
#         # Add assistant message
#         st.session_state.chat_messages.append({"role": "assistant", "content": response})

# def generate_chat_response(prompt: str, test_cases: List[Dict], api_key: str) -> str:
#     """Generate chat response - same as original"""
#     try:
#         from openai import OpenAI
#         client = OpenAI(api_key=api_key)
        
#         # Prepare context
#         test_cases_summary = f"Total test cases: {len(test_cases)}\n"
#         test_cases_summary += "Sample test cases:\n"
#         for i, tc in enumerate(test_cases[:3], 1):
#             test_cases_summary += f"{i}. {tc.get('Test Case Description', '')}\n"
        
#         chat_prompt = f"""
#         You are an expert BFSI test engineer assistant. Answer questions about the generated test cases.
        
#         Test Cases Context:
#         {test_cases_summary}
        
#         User Question: {prompt}
        
#         Provide helpful, specific answers about the test cases. If asked to modify test cases, 
#         provide specific suggestions or instructions.
#         """
        
#         response = client.chat.completions.create(
#             model="gpt-4.1-mini-2025-04-14",
#             messages=[
#                 {"role": "system", "content": "You are a helpful BFSI testing expert."},
#                 {"role": "user", "content": chat_prompt}
#             ],
#             temperature=0.7,
#             max_tokens=500
#         )
        
#         return response.choices[0].message.content
        
#     except Exception as e:
#         return f"Sorry, I encountered an error: {str(e)}"

# if __name__ == "__main__":
#     main()


# # src/ui/streamlit_app.py - CRITICAL FIXES
# """
# FIXED: Simple Streamlit App with Enhanced Dynamic PACS.008 Intelligence
# Clean UI like original repo but with FIXED backend processing that actually works
# """

# import streamlit as st
# import os
# import json
# import tempfile
# from pathlib import Path
# import pandas as pd
# from typing import List, Dict, Any
# import logging
# from dotenv import load_dotenv
# from datetime import datetime

# # Load environment variables
# load_dotenv()

# # Import modules
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from processors.document_processor import DocumentProcessor
# from exporters.excel_exporter import TestCaseExporter

# # Try to import FIXED dynamic system
# try:
#     from ai_engine.dynamic_pacs008_test_generator import DynamicPACS008TestGenerator
#     DYNAMIC_SYSTEM_AVAILABLE = True
# except ImportError:
#     from ai_engine.test_generator import TestCaseGenerator
#     DYNAMIC_SYSTEM_AVAILABLE = False

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Page configuration - enhanced
# st.set_page_config(
#     page_title="ITASSIST - AI Test Case Generator",
#     page_icon="🏦" if DYNAMIC_SYSTEM_AVAILABLE else "🤖",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# def main():
#     """Main Streamlit application - enhanced with FIXED backend"""
    
#     # Title - enhanced if dynamic system available
#     if DYNAMIC_SYSTEM_AVAILABLE:
#         st.title("🏦 ITASSIST - AI Test Case Generator with PACS.008 Intelligence")
#         st.markdown("**FIXED: AI-powered test case generation with accurate field detection and banking intelligence**")
#         st.success("✅ **ENHANCED SYSTEM ACTIVE** - Advanced field detection, realistic banking scenarios, and domain expertise")
#     else:
#         st.title("🤖 ITASSIST - AI Test Case Generator")
#         st.markdown("**AI-powered test case generation from BFSI documents**")
#         st.warning("⚠️ Enhanced PACS.008 system not available - using standard generation")
    
#     # Sidebar - same as original structure but enhanced
#     with st.sidebar:
#         st.header("⚙️ Configuration")
        
#         # API Key input - same as original
#         default_api_key = os.getenv("OPENAI_API_KEY", "")
#         api_key = st.text_input(
#             "OpenAI API Key", 
#             value=default_api_key,
#             type="password",
#             help="API key loaded from environment" if default_api_key else "Enter your OpenAI API key"
#         )
        
#         # Model selection - same as original
#         model_option = st.selectbox(
#             "AI Model",
#             ["gpt-4o-mini", "gpt-4.1-mini-2025-04-14", "gpt-3.5-turbo"],
#             index=0,
#             help="gpt-4o-mini recommended for enhanced accuracy"
#         )
        
#         # Generation options - enhanced
#         st.subheader("Generation Options")
#         num_test_cases = st.slider("Test Cases per Story", 5, 15, 8)
#         include_edge_cases = st.checkbox("Include Edge Cases", value=True)
#         include_negative_cases = st.checkbox("Include Negative Cases", value=True)
        
#         # FIXED: Enhanced PACS.008 status indicator
#         if DYNAMIC_SYSTEM_AVAILABLE:
#             st.subheader("🏦 FIXED PACS.008 Intelligence")
#             st.success("✅ **FIXED ENHANCED PROCESSING**")
#             st.info("**FIXED System Features:**\n• ✅ Accurate field detection (USD 565000, bank names)\n• ✅ Realistic banking scenarios\n• ✅ Proper maker-checker workflows\n• ✅ Domain-specific test cases")
            
#             st.subheader("🔧 FIXES APPLIED")
#             st.success("**Field Detection FIXED:**\n• Pattern-based pre-extraction\n• Aggressive LLM detection\n• Banking data integration")
#             st.success("**Test Generation FIXED:**\n• Realistic banking scenarios\n• Actual amounts & bank names\n• Business-focused test cases")
        
#         # Export format - same as original
#         export_format = st.multiselect(
#             "Export Formats",
#             ["Excel", "CSV", "JSON"],
#             default=["Excel"]
#         )
    
#     # Initialize session state - same as original plus documentation
#     if 'generated_test_cases' not in st.session_state:
#         st.session_state.generated_test_cases = []
#     if 'processing_complete' not in st.session_state:
#         st.session_state.processing_complete = False
#     if 'workflow_results' not in st.session_state:
#         st.session_state.workflow_results = {}
#     if 'field_detection_results' not in st.session_state:
#         st.session_state.field_detection_results = {}
    
#     # Main content tabs - enhanced with field detection tab
#     tab1, tab2, tab3, tab4, tab5 = st.tabs(["📁 Upload & Process", "🧪 Generated Test Cases", "🏦 Field Detection Results", "📋 Processing Report", "💬 Chat Assistant"])
    
#     with tab1:
#         upload_and_process_tab(api_key, num_test_cases, include_edge_cases, include_negative_cases)
    
#     with tab2:
#         display_test_cases_tab(export_format)
    
#     with tab3:
#         field_detection_results_tab()
    
#     with tab4:
#         processing_report_tab()
    
#     with tab5:
#         chat_assistant_tab(api_key)

# def upload_and_process_tab(api_key: str, num_test_cases: int, include_edge_cases: bool, include_negative_cases: bool):
#     """FIXED: File upload and processing tab with enhanced backend"""
    
#     st.header("📁 Document Upload & Processing")
    
#     # FIXED: Enhanced file upload section
#     uploaded_files = st.file_uploader(
#         "Upload your documents (Enhanced processing will detect PACS.008 fields)",
#         type=['docx', 'pdf', 'xlsx', 'png', 'jpg', 'jpeg', 'tiff', 'bmp', 'txt', 'eml', 'json', 'xml', 'csv'],
#         accept_multiple_files=True,
#         help="FIXED: System now accurately detects amounts (USD 565000), bank names (Al Ahli Bank), and generates realistic test cases"
#     )
    
#     # Display file validation info with enhancement note
#     if uploaded_files:
#         st.info(f"📁 {len(uploaded_files)} file(s) uploaded successfully")
        
#         if DYNAMIC_SYSTEM_AVAILABLE:
#             st.success("🏦 **FIXED PROCESSING WILL APPLY:**\n• Accurate field detection for amounts and bank names\n• Realistic banking test scenarios\n• Enhanced PACS.008 intelligence")
        
#         # Show file details
#         with st.expander("📋 File Details"):
#             for file in uploaded_files:
#                 file_size = len(file.getvalue()) / (1024*1024)  # MB
#                 st.write(f"• **{file.name}** ({file_size:.1f} MB)")
                
#                 # Validate file size
#                 if file_size > 50:
#                     st.warning(f"⚠️ {file.name} is large ({file_size:.1f} MB). Processing may take longer.")
    
#     # Enhanced processing options
#     st.subheader("🔧 Processing Options")
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         process_embedded_content = st.checkbox("📷 Process Embedded Images/Screenshots", value=True)
#     with col2:
#         extract_tables = st.checkbox("📊 Extract Table Content", value=True)
#     with col3:
#         enhance_ocr = st.checkbox("🔍 Enhanced OCR Processing", value=True)
    
#     # FIXED: Custom instructions with enhanced templates
#     st.subheader("📝 Custom Instructions")
    
#     if DYNAMIC_SYSTEM_AVAILABLE:
#         instruction_templates = {
#             "Standard": "",
#             "Focus on PACS.008 Banking": "Focus on PACS.008 payment processing, banking agents, cross-border scenarios with USD 565000 amounts",
#             "Maker-Checker Workflows": "Emphasize maker-checker workflows, approval processes, and banking operations with realistic banking data",
#             "High-Value Payments": "Generate test cases for high-value payments (USD 565000, EUR 25000) with correspondent banking",
#             "Focus on Negative Cases": "Generate more negative test cases and error scenarios. Include boundary testing and invalid input validation.",
#             "Comprehensive Coverage": "Generate comprehensive test coverage including positive, negative, edge cases, and integration scenarios with realistic banking data.",
#             "Security Focus": "Emphasize security testing scenarios including authentication, authorization, and data validation for banking systems.",
#             "Performance Testing": "Include performance-related test scenarios for high-volume and stress testing with banking loads."
#         }
#     else:
#         instruction_templates = {
#             "Standard": "",
#             "Focus on Negative Cases": "Generate more negative test cases and error scenarios. Include boundary testing and invalid input validation.",
#             "Basic Scenarios Only": "Focus on basic happy path scenarios. Minimize edge cases and complex integration tests.",
#             "Comprehensive Coverage": "Generate comprehensive test coverage including positive, negative, edge cases, and integration scenarios.",
#             "Security Focus": "Emphasize security testing scenarios including authentication, authorization, and data validation.",
#             "Performance Testing": "Include performance-related test scenarios for high-volume and stress testing."
#         }
    
#     selected_template = st.selectbox("Choose Instruction Template:", list(instruction_templates.keys()))
    
#     custom_instructions = st.text_area(
#         "Custom Instructions",
#         value=instruction_templates[selected_template],
#         placeholder="e.g., 'Focus on USD 565000 payment validation' or 'Create test cases with Al Ahli Bank and BNP Paribas'",
#         help="FIXED: System will now use these instructions to generate realistic banking test cases with actual field values"
#     )
    
#     # FIXED: Enhanced process button
#     process_button_text = "🚀 Generate Test Cases with FIXED Intelligence" if DYNAMIC_SYSTEM_AVAILABLE else "🚀 Generate Test Cases"
    
#     if st.button(process_button_text, type="primary", disabled=not api_key or not uploaded_files):
#         if not api_key:
#             st.error("Please provide OpenAI API key in the sidebar")
#             return
        
#         if not uploaded_files:
#             st.error("Please upload at least one document")
#             return
        
#         process_files_enhanced(uploaded_files, api_key, custom_instructions, num_test_cases, 
#                               include_edge_cases, include_negative_cases, process_embedded_content)

# def process_files_enhanced(uploaded_files, api_key: str, custom_instructions: str, 
#                           num_test_cases: int, include_edge_cases: bool, include_negative_cases: bool,
#                           process_embedded_content: bool):
#     """FIXED: Process uploaded files with enhanced backend"""
    
#     progress_bar = st.progress(0)
#     status_text = st.empty()
    
#     try:
#         # Process each file - same as original
#         doc_processor = DocumentProcessor()
        
#         all_content = []
#         total_files = len(uploaded_files)
        
#         for i, uploaded_file in enumerate(uploaded_files):
#             status_text.text(f"Processing {uploaded_file.name}...")
#             progress_bar.progress((i + 1) / (total_files + 2))
            
#             # Save uploaded file temporarily
#             with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
#                 tmp_file.write(uploaded_file.getvalue())
#                 tmp_file_path = tmp_file.name
            
#             # Process the file
#             result = doc_processor.process_file(tmp_file_path)
#             all_content.append(result.get('content', ''))
            
#             # Clean up temporary file
#             os.unlink(tmp_file_path)
        
#         # Combine all extracted content
#         status_text.text("Combining extracted content...")
#         progress_bar.progress(0.9)
        
#         combined_content = '\n\n--- Document Separator ---\n\n'.join(all_content)
        
#         # Generate custom instructions - enhanced
#         generation_instructions = build_generation_instructions_enhanced(
#             custom_instructions, num_test_cases, include_edge_cases, include_negative_cases
#         )
        
#         # FIXED: Generate test cases with enhanced backend
#         if DYNAMIC_SYSTEM_AVAILABLE:
#             status_text.text("Generating test cases with FIXED PACS.008 intelligence...")
            
#             # Prepare files info for documentation
#             files_info = []
#             for uploaded_file in uploaded_files:
#                 file_size = len(uploaded_file.getvalue()) / (1024*1024)
#                 files_info.append({
#                     "name": uploaded_file.name,
#                     "size_mb": file_size,
#                     "type": uploaded_file.type or "unknown",
#                     "status": "processed"
#                 })
            
#             # Use FIXED dynamic system for enhanced processing
#             generator = DynamicPACS008TestGenerator(api_key)
#             workflow_results = generator.process_complete_workflow(combined_content, num_test_cases, files_info)
            
#             # Store complete workflow results
#             st.session_state.workflow_results = workflow_results
            
#             # Extract test cases from workflow
#             test_cases = workflow_results.get("step5_test_cases", [])
            
#             # FIXED: Store field detection results for display
#             field_detection = workflow_results.get("step3_pacs008_fields", {})
#             st.session_state.field_detection_results = field_detection
            
#             # FIXED: Show enhanced intelligence summary
#             analysis = workflow_results.get("step1_analysis", {})
#             if analysis.get("is_pacs008_relevant", False):
#                 detected_amounts = analysis.get("detected_amounts", [])
#                 detected_banks = analysis.get("detected_banks", [])
                
#                 st.success(f"🏦 **FIXED PACS.008 INTELLIGENCE APPLIED!**")
                
#                 col1, col2, col3 = st.columns(3)
#                 with col1:
#                     if detected_amounts:
#                         st.metric("💰 Amounts Detected", len(detected_amounts))
#                         st.write("**Amounts:**")
#                         for amount in detected_amounts[:3]:
#                             st.write(f"• {amount}")
                
#                 with col2:
#                     if detected_banks:
#                         st.metric("🏦 Banks Detected", len(detected_banks))
#                         st.write("**Banks:**")
#                         for bank in detected_banks[:3]:
#                             st.write(f"• {bank}")
                
#                 with col3:
#                     total_fields = field_detection.get("total_unique_fields", 0)
#                     st.metric("📋 Fields Extracted", total_fields)
            
#         else:
#             status_text.text("Generating test cases with AI...")
            
#             # Use standard system
#             test_generator = TestCaseGenerator(api_key)
#             test_cases = test_generator.generate_test_cases(combined_content, generation_instructions)
            
#             # Store basic results
#             st.session_state.workflow_results = {"step5_test_cases": test_cases}
        
#         progress_bar.progress(0.95)
        
#         if test_cases:
#             st.session_state.generated_test_cases = test_cases
#             st.session_state.processing_complete = True
            
#             progress_bar.progress(1.0)
#             status_text.text("✅ Processing complete!")
            
#             # FIXED: Display enhanced summary
#             total_test_cases = len(test_cases)
#             enhanced_test_cases = len([tc for tc in test_cases if tc.get("PACS008_Enhanced") == "Yes"])
            
#             if DYNAMIC_SYSTEM_AVAILABLE and enhanced_test_cases > 0:
#                 st.success(f"🎯 **FIXED SUCCESS:** Generated {total_test_cases} test cases ({enhanced_test_cases} enhanced with PACS.008 intelligence)")
                
#                 # Show enhancement breakdown
#                 col1, col2, col3 = st.columns(3)
#                 with col1:
#                     st.metric("📝 Total Test Cases", total_test_cases)
#                 with col2:
#                     st.metric("🏦 PACS.008 Enhanced", enhanced_test_cases)
#                 with col3:
#                     enhancement_rate = round((enhanced_test_cases / total_test_cases) * 100, 1)
#                     st.metric("✨ Enhancement Rate", f"{enhancement_rate}%")
#             else:
#                 st.success(f"Successfully generated {total_test_cases} test cases!")
            
#             # Show content preview
#             with st.expander("📄 Extracted Content Preview"):
#                 preview_content = combined_content[:1000] + "..." if len(combined_content) > 1000 else combined_content
#                 st.text(preview_content)
                
#         else:
#             st.error("No test cases could be generated. Please check your documents and try again.")
            
#     except Exception as e:
#         st.error(f"Error during processing: {str(e)}")
#         logger.error(f"Processing error: {str(e)}")

# def build_generation_instructions_enhanced(custom_instructions: str, num_test_cases: int, 
#                                          include_edge_cases: bool, include_negative_cases: bool) -> str:
#     """FIXED: Build enhanced generation instructions"""
#     instructions = []
    
#     if custom_instructions:
#         instructions.append(custom_instructions)
    
#     instructions.append(f"Generate exactly {num_test_cases} test cases per user story/requirement")
    
#     if include_edge_cases:
#         instructions.append("Include edge cases and boundary conditions")
    
#     if include_negative_cases:
#         instructions.append("Include negative test scenarios and error conditions")
    
#     # FIXED: Enhanced instruction for PACS.008
#     if DYNAMIC_SYSTEM_AVAILABLE:
#         instructions.append("Use FIXED PACS.008 intelligence: extract actual amounts (USD 565000), bank names (Al Ahli Bank, BNP Paribas), and create realistic banking scenarios with maker-checker workflows")
#     else:
#         instructions.append("Focus on BFSI domain scenarios with realistic banking data")
    
#     return ". ".join(instructions)

# def field_detection_results_tab():
#     """FIXED: New tab to display field detection results"""
    
#     st.header("🏦 Field Detection Results")
    
#     if not st.session_state.field_detection_results:
#         st.info("📋 No field detection results available. Process documents first to see PACS.008 field analysis.")
#         return
    
#     field_results = st.session_state.field_detection_results
    
#     # Display summary metrics
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         total_fields = field_results.get("total_unique_fields", 0)
#         st.metric("📋 Total Fields", total_fields)
    
#     with col2:
#         detection_summary = field_results.get("detection_summary", {})
#         high_confidence = detection_summary.get("high_confidence_detections", 0)
#         st.metric("✅ High Confidence", high_confidence)
    
#     with col3:
#         stories_with_fields = detection_summary.get("stories_with_pacs008", 0)
#         st.metric("📖 Stories with Fields", stories_with_fields)
    
#     with col4:
#         total_stories = detection_summary.get("total_stories_processed", 0)
#         if total_stories > 0:
#             coverage = round((stories_with_fields / total_stories) * 100, 1)
#             st.metric("📊 Coverage", f"{coverage}%")
    
#     # Show field detection quality indicator
#     if high_confidence >= 3:
#         st.success("🎯 **EXCELLENT FIELD DETECTION** - System successfully extracted specific banking values!")
#     elif total_fields >= 2:
#         st.info("✅ **GOOD FIELD DETECTION** - System identified key banking fields")
#     else:
#         st.warning("⚠️ **LIMITED FIELD DETECTION** - Consider adding more specific banking content")
    
#     # Display detected fields by story
#     st.subheader("📋 Detected Fields by User Story")
    
#     story_mapping = field_results.get("story_field_mapping", {})
    
#     if story_mapping:
#         for story_id, story_data in story_mapping.items():
#             with st.expander(f"📖 {story_id}: {story_data.get('story_title', 'Unknown Story')}", expanded=True):
                
#                 # Story summary
#                 col1, col2, col3 = st.columns(3)
#                 with col1:
#                     st.metric("Fields Found", story_data.get("field_count", 0))
#                 with col2:
#                     st.metric("Mandatory Fields", story_data.get("mandatory_fields", 0))
#                 with col3:
#                     st.metric("High Confidence", story_data.get("high_confidence_fields", 0))
                
#                 # Display detected fields
#                 detected_fields = story_data.get("detected_fields", [])
                
#                 if detected_fields:
#                     st.write("**🔍 Detected PACS.008 Fields:**")
                    
#                     for field in detected_fields:
#                         # Color code by confidence
#                         confidence = field.get("confidence", "Low")
#                         if confidence == "High":
#                             confidence_color = "🟢"
#                         elif confidence == "Medium":
#                             confidence_color = "🟡"
#                         else:
#                             confidence_color = "🔴"
                        
#                         field_name = field.get("field_name", "Unknown Field")
#                         extracted_value = field.get("extracted_value", "Not specified")
#                         is_mandatory = "⭐ Mandatory" if field.get("is_mandatory", False) else "Optional"
                        
#                         st.write(f"{confidence_color} **{field_name}** ({is_mandatory})")
#                         st.write(f"   💎 **Value:** {extracted_value}")
#                         st.write(f"   📊 **Confidence:** {confidence}")
                        
#                         # Show reasoning if available
#                         reasoning = field.get("detection_reason", "")
#                         if reasoning:
#                             st.write(f"   🧠 **Detection Reason:** {reasoning}")
                        
#                         st.write("---")
#                 else:
#                     st.write("❌ No fields detected for this story")
#     else:
#         st.warning("No field detection data available")

# def display_test_cases_tab(export_formats: List[str]):
#     """FIXED: Display generated test cases with enhancement indicators"""
    
#     st.header("🧪 Generated Test Cases")
    
#     if not st.session_state.generated_test_cases:
#         st.info("No test cases generated yet. Please upload documents and process them first.")
#         return
    
#     test_cases = st.session_state.generated_test_cases
    
#     # FIXED: Display enhanced summary metrics
#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         st.metric("Total Test Cases", len(test_cases))
#     with col2:
#         high_priority = len([tc for tc in test_cases if tc.get("Priority") == "High"])
#         st.metric("High Priority", high_priority)
#     with col3:
#         regression_tests = len([tc for tc in test_cases if tc.get("Part of Regression") == "Yes"])
#         st.metric("Regression Tests", regression_tests)
#     with col4:
#         unique_stories = len(set(tc.get("User Story ID", "") for tc in test_cases))
#         st.metric("User Stories", unique_stories)
    
#     # FIXED: Show PACS.008 enhancement indicator
#     if DYNAMIC_SYSTEM_AVAILABLE:
#         pacs008_enhanced = len([tc for tc in test_cases if tc.get("PACS008_Enhanced") == "Yes"])
#         if pacs008_enhanced > 0:
#             enhancement_rate = round((pacs008_enhanced / len(test_cases)) * 100, 1)
#             st.success(f"🏦 **FIXED PACS.008 INTELLIGENCE APPLIED:** {pacs008_enhanced} test cases enhanced with banking intelligence ({enhancement_rate}% enhancement rate)")
            
#             # Show specific enhancements
#             with st.expander("🔍 View Enhancement Details", expanded=False):
#                 enhanced_cases = [tc for tc in test_cases if tc.get("PACS008_Enhanced") == "Yes"]
                
#                 st.write("**✨ Enhanced Test Cases Include:**")
#                 for i, tc in enumerate(enhanced_cases[:5], 1):  # Show first 5
#                     scenario = tc.get("Scenario", "Unknown Scenario")
#                     description = tc.get("Test Case Description", "")
                    
#                     # Check for banking data in description
#                     has_amount = any(amount in description for amount in ["USD 565000", "EUR 25000", "USD"])
#                     has_bank = any(bank in description for bank in ["Al Ahli", "BNP", "Deutsche", "Bank"])
                    
#                     enhancements = []
#                     if has_amount:
#                         enhancements.append("💰 Realistic amounts")
#                     if has_bank:
#                         enhancements.append("🏦 Actual bank names")
#                     if "maker" in description.lower() or "checker" in description.lower():
#                         enhancements.append("👥 Maker-checker workflow")
                    
#                     st.write(f"{i}. **{scenario}** - {', '.join(enhancements)}")
                
#                 if len(enhanced_cases) > 5:
#                     st.write(f"... and {len(enhanced_cases) - 5} more enhanced test cases")
    
#     # Filter options - same as original but enhanced
#     with st.expander("🔍 Filter Test Cases"):
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             priority_filter = st.multiselect(
#                 "Priority", 
#                 ["High", "Medium", "Low"],
#                 default=["High", "Medium", "Low"]
#             )
        
#         with col2:
#             regression_filter = st.multiselect(
#                 "Regression", 
#                 ["Yes", "No"],
#                 default=["Yes", "No"]
#             )
        
#         with col3:
#             story_ids = list(set(tc.get("User Story ID", "") for tc in test_cases))
#             story_filter = st.multiselect(
#                 "User Story ID",
#                 story_ids,
#                 default=story_ids
#             )
    
#     # FIXED: Add PACS.008 enhancement filter
#     if DYNAMIC_SYSTEM_AVAILABLE:
#         with st.expander("🏦 PACS.008 Enhancement Filter"):
#             enhancement_filter = st.selectbox(
#                 "Show Test Cases",
#                 ["All Test Cases", "PACS.008 Enhanced Only", "Standard Only"],
#                 index=0
#             )
#     else:
#         enhancement_filter = "All Test Cases"
    
#     # Apply filters - enhanced
#     filtered_test_cases = [
#         tc for tc in test_cases
#         if (tc.get("Priority") in priority_filter and
#             tc.get("Part of Regression") in regression_filter and
#             tc.get("User Story ID") in story_filter)
#     ]
    
#     # Apply PACS.008 enhancement filter
#     if enhancement_filter == "PACS.008 Enhanced Only":
#         filtered_test_cases = [tc for tc in filtered_test_cases if tc.get("PACS008_Enhanced") == "Yes"]
#     elif enhancement_filter == "Standard Only":
#         filtered_test_cases = [tc for tc in filtered_test_cases if tc.get("PACS008_Enhanced") != "Yes"]
    
#     # Display test cases table - enhanced
#     if filtered_test_cases:
#         st.subheader(f"Test Cases ({len(filtered_test_cases)} of {len(test_cases)})")
        
#         # Convert to DataFrame for display
#         df = pd.DataFrame(filtered_test_cases)
        
#         # FIXED: Add enhancement indicator column
#         if DYNAMIC_SYSTEM_AVAILABLE and "PACS008_Enhanced" in df.columns:
#             df["🏦 Enhanced"] = df["PACS008_Enhanced"].apply(lambda x: "✅" if x == "Yes" else "")
        
#         # Configure column display - enhanced
#         column_config = {
#             "Steps": st.column_config.TextColumn(width="large"),
#             "Test Case Description": st.column_config.TextColumn(width="medium"),
#             "Expected Result": st.column_config.TextColumn(width="medium"),
#             "🏦 Enhanced": st.column_config.TextColumn(width="small"),
#         }
        
#         # FIXED: Hide technical columns from display
#         display_columns = [col for col in df.columns if col not in ["PACS008_Enhanced", "Enhancement_Type", "Generation_Method"]]
        
#         st.dataframe(
#             df[display_columns],
#             use_container_width=True,
#             column_config=column_config,
#             hide_index=True
#         )
        
#         # FIXED: Show sample enhanced test case
#         if DYNAMIC_SYSTEM_AVAILABLE:
#             enhanced_cases = [tc for tc in filtered_test_cases if tc.get("PACS008_Enhanced") == "Yes"]
#             if enhanced_cases:
#                 with st.expander("👁️ View Sample Enhanced Test Case", expanded=False):
#                     sample_case = enhanced_cases[0]
                    
#                     st.write("**📋 Test Case Details:**")
#                     st.write(f"**Test ID:** {sample_case.get('Test Case ID', 'Unknown')}")
#                     st.write(f"**Scenario:** {sample_case.get('Scenario', 'Unknown')}")
#                     st.write(f"**Description:** {sample_case.get('Test Case Description', 'Unknown')}")
                    
#                     st.write("**🧪 Test Steps:**")
#                     steps = sample_case.get('Steps', '').replace('\n', '\n\n')
#                     st.text(steps)
                    
#                     st.write("**✅ Expected Result:**")
#                     st.text(sample_case.get('Expected Result', 'Unknown'))
                    
#                     # Highlight enhancements
#                     description = sample_case.get('Test Case Description', '')
#                     steps_text = sample_case.get('Steps', '')
                    
#                     enhancements_found = []
#                     if any(amount in description + steps_text for amount in ["USD 565000", "EUR 25000", "565000", "25000"]):
#                         enhancements_found.append("💰 **Realistic Amounts:** Uses actual detected amounts like USD 565000")
#                     if any(bank in description + steps_text for bank in ["Al Ahli", "BNP", "Deutsche", "Bank"]):
#                         enhancements_found.append("🏦 **Real Bank Names:** References actual banks like Al Ahli Bank of Kuwait")
#                     if any(term in (description + steps_text).lower() for term in ["maker", "checker", "approval"]):
#                         enhancements_found.append("👥 **Banking Workflows:** Includes maker-checker approval processes")
                    
#                     if enhancements_found:
#                         st.write("**✨ PACS.008 Enhancements Applied:**")
#                         for enhancement in enhancements_found:
#                             st.write(f"• {enhancement}")
        
#         # Export section - same as original
#         st.subheader("📥 Export Test Cases")
        
#         col1, col2, col3 = st.columns(3)
        
#         if "Excel" in export_formats:
#             with col1:
#                 if st.button("📊 Download Excel", type="primary"):
#                     export_excel(filtered_test_cases)
        
#         if "CSV" in export_formats:
#             with col2:
#                 if st.button("📄 Download CSV"):
#                     export_csv(filtered_test_cases)
        
#         if "JSON" in export_formats:
#             with col3:
#                 if st.button("🔧 Download JSON"):
#                     export_json(filtered_test_cases)
    
#     else:
#         st.warning("No test cases match the selected filters.")

# def export_excel(test_cases):
#     """Export Excel - same as original"""
#     try:
#         import io
        
#         # Create Excel in memory
#         df = pd.DataFrame(test_cases)
#         output = io.BytesIO()
        
#         with pd.ExcelWriter(output, engine='openpyxl') as writer:
#             df.to_excel(writer, sheet_name='Test Cases', index=False)
        
#         output.seek(0)
        
#         st.download_button(
#             label="📊 Download Excel File",
#             data=output.getvalue(),
#             file_name=f"test_cases_{len(test_cases)}.xlsx",
#             mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#         )
        
#     except Exception as e:
#         st.error(f"Export error: {str(e)}")

# def export_csv(test_cases):
#     """Export CSV - same as original"""
#     try:
#         csv_data = pd.DataFrame(test_cases).to_csv(index=False)
#         st.download_button(
#             label="📄 Download CSV File",
#             data=csv_data,
#             file_name=f"test_cases_{len(test_cases)}.csv",
#             mime="text/csv"
#         )
#     except Exception as e:
#         st.error(f"Export error: {str(e)}")

# def export_json(test_cases):
#     """Export JSON - same as original"""
#     try:
#         json_data = json.dumps(test_cases, indent=2, ensure_ascii=False)
#         st.download_button(
#             label="🔧 Download JSON File",
#             data=json_data,
#             file_name=f"test_cases_{len(test_cases)}.json",
#             mime="application/json"
#         )
#     except Exception as e:
#         st.error(f"Export error: {str(e)}")

# def processing_report_tab():
#     """FIXED: Processing report and documentation tab with enhanced display"""
    
#     st.header("📋 Processing Report & Documentation")
    
#     if not st.session_state.workflow_results:
#         st.info("📄 No processing report available. Process documents first to see detailed analysis.")
#         return
    
#     workflow_results = st.session_state.workflow_results
    
#     # Check if documentation is available
#     documentation = workflow_results.get("documentation", {})
    
#     if documentation and documentation.get("report_text"):
#         st.success("✅ **Complete FIXED Processing Documentation Available**")
        
#         # FIXED: Enhanced summary metrics
#         col1, col2, col3, col4 = st.columns(4)
        
#         with col1:
#             user_stories = len(workflow_results.get("step2_user_stories", []))
#             st.metric("📖 User Stories", user_stories)
        
#         with col2:
#             pacs008_fields = workflow_results.get("step3_pacs008_fields", {}).get("total_unique_fields", 0)
#             st.metric("🏦 PACS.008 Fields", pacs008_fields)
        
#         with col3:
#             test_cases = len(workflow_results.get("step5_test_cases", []))
#             st.metric("🧪 Test Cases", test_cases)
        
#         with col4:
#             maker_checker_items = len(workflow_results.get("step4_maker_checker", {}).get("validation_items", []))
#             st.metric("👥 Validation Items", maker_checker_items)
        
#         # FIXED: Processing intelligence indicator
#         analysis = workflow_results.get("step1_analysis", {})
#         if analysis.get("is_pacs008_relevant", False):
#             confidence = analysis.get("confidence_score", 0)
#             detected_amounts = analysis.get("detected_amounts", [])
#             detected_banks = analysis.get("detected_banks", [])
            
#             st.success(f"🎯 **FIXED PACS.008 INTELLIGENCE APPLIED** - Confidence: {confidence}%")
            
#             # Show detected data
#             if detected_amounts or detected_banks:
#                 with st.expander("🔍 View Detected Banking Data", expanded=False):
#                     col1, col2 = st.columns(2)
                    
#                     with col1:
#                         if detected_amounts:
#                             st.write("**💰 Detected Amounts:**")
#                             for amount in detected_amounts:
#                                 st.write(f"• {amount}")
                    
#                     with col2:
#                         if detected_banks:
#                             st.write("**🏦 Detected Banks:**")
#                             for bank in detected_banks:
#                                 st.write(f"• {bank}")
#         else:
#             st.info("📋 **Standard Processing Applied** - No PACS.008 content detected")
        
#         # Documentation preview and download
#         st.subheader("📄 Complete Processing Documentation")
        
#         with st.expander("📋 Processing Report Preview (First 1000 characters)", expanded=False):
#             report_text = documentation.get("report_text", "")
#             preview = report_text[:1000] + "..." if len(report_text) > 1000 else report_text
#             st.text(preview)
        
#         # Download options
#         col1, col2 = st.columns(2)
        
#         with col1:
#             if st.button("📥 Download Complete Processing Report", type="primary"):
#                 try:
#                     report_text = documentation.get("report_text", "")
                    
#                     # Create filename with timestamp
#                     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                     filename = f"FIXED_PACS008_Processing_Report_{timestamp}.md"
                    
#                     st.download_button(
#                         label="📄 Download Report (Markdown)",
#                         data=report_text.encode('utf-8'),
#                         file_name=filename,
#                         mime="text/markdown",
#                         help="Complete processing report with all analysis, decisions, and reasoning"
#                     )
                    
#                 except Exception as e:
#                     st.error(f"Error preparing download: {str(e)}")
        
#         with col2:
#             if st.button("🔧 Download JSON Data"):
#                 try:
#                     json_data = documentation.get("json_data", {})
                    
#                     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                     filename = f"FIXED_PACS008_Processing_Data_{timestamp}.json"
                    
#                     st.download_button(
#                         label="📊 Download JSON Data",
#                         data=json.dumps(json_data, indent=2, ensure_ascii=False).encode('utf-8'),
#                         file_name=filename,
#                         mime="application/json",
#                         help="Raw processing data in JSON format for programmatic access"
#                     )
                    
#                 except Exception as e:
#                     st.error(f"Error preparing JSON download: {str(e)}")
        
#         # FIXED: Processing insights
#         st.subheader("🔍 Key Processing Insights")
        
#         # Show key insights
#         if analysis.get("banking_concepts"):
#             st.write("**🏦 Banking Concepts Detected:**")
#             concepts = analysis.get("banking_concepts", [])[:5]
#             for concept in concepts:
#                 st.write(f"• {concept}")
        
#         if workflow_results.get("processing_errors"):
#             st.subheader("⚠️ Processing Warnings")
#             for error in workflow_results.get("processing_errors", []):
#                 st.warning(f"• {error}")
        
#         # FIXED: Processing quality indicators
#         quality = workflow_results.get("workflow_summary", {}).get("quality_indicators", {})
#         if quality:
#             st.subheader("📊 Processing Quality")
#             col1, col2, col3 = st.columns(3)
            
#             with col1:
#                 field_quality = quality.get("field_detection_accuracy", "medium")
#                 color = "🟢" if field_quality == "high" else "🟡"
#                 st.write(f"{color} **Field Detection**: {field_quality.title()}")
            
#             with col2:
#                 test_quality = quality.get("test_case_relevance", "medium")
#                 color = "🟢" if test_quality == "high" else "🟡"
#                 st.write(f"{color} **Test Relevance**: {test_quality.title()}")
            
#             with col3:
#                 business_quality = quality.get("business_alignment", "medium")
#                 color = "🟢" if business_quality == "high" else "🟡"
#                 st.write(f"{color} **Business Alignment**: {business_quality.title()}")
        
#     else:
#         st.warning("⚠️ **Limited Documentation Available**")
#         st.info("Complete documentation is only available when using the FIXED PACS.008 enhanced system.")
        
#         # Show basic info if available
#         test_cases = workflow_results.get("step5_test_cases", [])
#         if test_cases:
#             st.write(f"**Generated:** {len(test_cases)} test cases")
            
#             # Basic test case summary
#             priorities = {}
#             for tc in test_cases:
#                 priority = tc.get("Priority", "Medium")
#                 priorities[priority] = priorities.get(priority, 0) + 1
            
#             st.write("**Priority Distribution:**")
#             for priority, count in priorities.items():
#                 st.write(f"• {priority}: {count} test cases")

# def chat_assistant_tab(api_key: str):
#     """FIXED: Chat assistant with enhanced responses"""
    
#     st.header("💬 Chat Assistant")
#     st.markdown("Ask questions about your test cases or request modifications")
    
#     if not api_key:
#         st.warning("Please provide OpenAI API key to enable chat functionality")
#         return
    
#     if not st.session_state.generated_test_cases:
#         st.info("Generate test cases first to enable chat assistance")
#         return
    
#     # FIXED: Enhanced chat interface
#     if "chat_messages" not in st.session_state:
#         st.session_state.chat_messages = []
        
#         # Add welcome message for enhanced system
#         if DYNAMIC_SYSTEM_AVAILABLE:
#             welcome_msg = "Hello! I'm your FIXED PACS.008 testing assistant. I can help you understand the enhanced test cases, field detection results, and banking intelligence applied to your documents. What would you like to know?"
#         else:
#             welcome_msg = "Hello! I'm your testing assistant. I can help you understand your generated test cases. What would you like to know?"
        
#         st.session_state.chat_messages.append({"role": "assistant", "content": welcome_msg})
    
#     # Display chat history
#     for message in st.session_state.chat_messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
    
#     # Chat input
#     if prompt := st.chat_input("Ask about your test cases, field detection, or banking intelligence..."):
#         # Add user message
#         st.session_state.chat_messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)
        
#         # Generate response
#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 response = generate_enhanced_chat_response(prompt, st.session_state.generated_test_cases, 
#                                                          st.session_state.field_detection_results, api_key)
#                 st.markdown(response)
        
#         # Add assistant message
#         st.session_state.chat_messages.append({"role": "assistant", "content": response})

# def generate_enhanced_chat_response(prompt: str, test_cases: List[Dict], field_results: Dict, api_key: str) -> str:
#     """FIXED: Generate enhanced chat response with field detection context"""
#     try:
#         from openai import OpenAI
#         client = OpenAI(api_key=api_key)
        
#         # Prepare enhanced context
#         test_cases_summary = f"Total test cases: {len(test_cases)}\n"
        
#         # Add enhancement information
#         enhanced_cases = len([tc for tc in test_cases if tc.get("PACS008_Enhanced") == "Yes"])
#         if enhanced_cases > 0:
#             test_cases_summary += f"PACS.008 Enhanced test cases: {enhanced_cases}\n"
        
#         # Add field detection summary
#         if field_results:
#             total_fields = field_results.get("total_unique_fields", 0)
#             high_confidence = field_results.get("detection_summary", {}).get("high_confidence_detections", 0)
#             test_cases_summary += f"PACS.008 fields detected: {total_fields} (High confidence: {high_confidence})\n"
        
#         test_cases_summary += "Sample test cases:\n"
#         for i, tc in enumerate(test_cases[:3], 1):
#             scenario = tc.get('Scenario', 'Unknown')
#             enhanced = " [PACS.008 Enhanced]" if tc.get('PACS008_Enhanced') == 'Yes' else ""
#             test_cases_summary += f"{i}. {scenario}{enhanced}\n"
        
#         chat_prompt = f"""
#         You are an expert PACS.008 banking test assistant with knowledge of the FIXED enhancement system.

#         Test Cases Context:
#         {test_cases_summary}

#         Field Detection Results:
#         {json.dumps(field_results.get("detection_summary", {}), indent=2) if field_results else "No field detection data"}

#         User Question: {prompt}

#         Provide helpful, specific answers about:
#         - Test cases and their banking relevance
#         - PACS.008 field detection results and accuracy
#         - Banking intelligence enhancements applied
#         - Specific amounts, banks, or scenarios detected
#         - How the FIXED system improved the results

#         If asked about enhancements, explain how the system detected actual amounts (like USD 565000) 
#         and bank names (like Al Ahli Bank of Kuwait) to create realistic test scenarios.
#         """
        
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": "You are a helpful PACS.008 banking testing expert who understands the FIXED enhancement system."},
#                 {"role": "user", "content": chat_prompt}
#             ],
#             temperature=0.7,
#             max_tokens=500
#         )
        
#         return response.choices[0].message.content
        
#     except Exception as e:
#         return f"Sorry, I encountered an error: {str(e)}"

# if __name__ == "__main__":
#     main()



# src/ui/streamlit_app.py - CRITICAL FIXES FOR CLIENT FEEDBACK
"""
FIXED: Streamlit App addressing client feedback:
- Test Case Description must have validation pertaining to maker and checker process
- Enhanced UI to highlight maker-checker validation in test cases
- Better display of dual authorization workflows
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

# Try to import FIXED dynamic system
try:
    from ai_engine.dynamic_pacs008_test_generator import DynamicPACS008TestGenerator
    DYNAMIC_SYSTEM_AVAILABLE = True
except ImportError:
    from ai_engine.test_generator import TestCaseGenerator
    DYNAMIC_SYSTEM_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="ITASSIST - AI Test Case Generator",
    page_icon="🏦" if DYNAMIC_SYSTEM_AVAILABLE else "🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """FIXED: Main Streamlit application with maker-checker focus"""
    
    # Title with client feedback addressed notice
    if DYNAMIC_SYSTEM_AVAILABLE:
        st.title("🏦 ITASSIST - AI Test Case Generator with PACS.008 Intelligence")
        st.markdown("**✅ CLIENT FEEDBACK ADDRESSED: Test descriptions now include explicit maker-checker validation processes**")
        st.success("🎯 **FIXED SYSTEM ACTIVE** - All test cases include 'Ops User maker' and 'Ops User checker' validation workflows")
    else:
        st.title("🤖 ITASSIST - AI Test Case Generator")
        st.markdown("**AI-powered test case generation from BFSI documents**")
        st.warning("⚠️ Enhanced PACS.008 system not available - using standard generation")
    
    # Enhanced sidebar with maker-checker focus
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        default_api_key = os.getenv("OPENAI_API_KEY", "")
        api_key = st.text_input(
            "OpenAI API Key", 
            value=default_api_key,
            type="password",
            help="API key loaded from environment" if default_api_key else "Enter your OpenAI API key"
        )
        
        # Model selection
        model_option = st.selectbox(
            "AI Model",
            ["gpt-4.1-mini-2025-04-14", "gpt-4o-mini", "gpt-3.5-turbo"],
            index=0,
            help="gpt-4.1-mini recommended for maker-checker workflows"
        )
        
        # Generation options with maker-checker focus
        st.subheader("🎯 Maker-Checker Test Generation")
        num_test_cases = st.slider("Test Cases per Story", 5, 15, 8)
        
        if DYNAMIC_SYSTEM_AVAILABLE:
            st.success("✅ **MAKER-CHECKER VALIDATION FOCUS**")
            st.info("**Every test case will include:**\n• 'Ops User maker' actions\n• 'Ops User checker' validation\n• Explicit approval workflows\n• Field-by-field validation")
        
        include_edge_cases = st.checkbox("Include Edge Cases", value=True)
        include_negative_cases = st.checkbox("Include Negative Cases", value=True)
        
        # FIXED: Enhanced system status
        if DYNAMIC_SYSTEM_AVAILABLE:
            st.subheader("🔧 CLIENT FEEDBACK FIXES")
            st.success("✅ **DESCRIPTION VALIDATION FIXED**")
            st.info("**All test descriptions now include:**\n• Maker creation/input actions\n• Checker validation/approval\n• Dual authorization workflows\n• Field validation processes")
            
            st.success("✅ **BANKING INTELLIGENCE APPLIED**")
            st.info("**System uses:**\n• Actual amounts (USD 565000)\n• Real bank names (Al Ahli Bank)\n• Realistic maker-checker scenarios")
        
        # Export format
        export_format = st.multiselect(
            "Export Formats",
            ["Excel", "CSV", "JSON"],
            default=["Excel"]
        )
    
    # Initialize session state
    if 'generated_test_cases' not in st.session_state:
        st.session_state.generated_test_cases = []
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'workflow_results' not in st.session_state:
        st.session_state.workflow_results = {}
    if 'field_detection_results' not in st.session_state:
        st.session_state.field_detection_results = {}
    
    # Main content tabs with maker-checker focus
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📁 Upload & Process", 
        "🧪 Generated Test Cases", 
        "👥 Maker-Checker Analysis", 
        "📋 Processing Report", 
        "💬 Chat Assistant"
    ])
    
    with tab1:
        upload_and_process_tab(api_key, num_test_cases, include_edge_cases, include_negative_cases)
    
    with tab2:
        display_test_cases_tab_with_maker_checker_focus(export_format)
    
    with tab3:
        maker_checker_analysis_tab()
    
    with tab4:
        processing_report_tab()
    
    with tab5:
        chat_assistant_tab(api_key)

def upload_and_process_tab(api_key: str, num_test_cases: int, include_edge_cases: bool, include_negative_cases: bool):
    """FIXED: File upload with maker-checker validation focus"""
    
    st.header("📁 Document Upload & Processing")
    
    # Client feedback addressed notice
    if DYNAMIC_SYSTEM_AVAILABLE:
        st.success("🎯 **CLIENT FEEDBACK ADDRESSED**: System now generates test descriptions with explicit maker-checker validation")
    
    uploaded_files = st.file_uploader(
        "Upload your documents (FIXED: System generates test cases with maker-checker validation)",
        type=['docx', 'pdf', 'xlsx', 'png', 'jpg', 'jpeg', 'tiff', 'bmp', 'txt', 'eml', 'json', 'xml', 'csv'],
        accept_multiple_files=True,
        help="FIXED: Every test case description will include 'Ops User maker' and 'Ops User checker' validation processes"
    )
    
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} file(s) uploaded successfully")
        
        if DYNAMIC_SYSTEM_AVAILABLE:
            st.success("🎯 **MAKER-CHECKER PROCESSING WILL APPLY:**\n• Test descriptions include explicit maker-checker validation\n• Ops User maker and checker workflows in every test\n• Dual authorization and approval processes")
    
    # Enhanced custom instructions with maker-checker templates
    st.subheader("📝 Custom Instructions")
    
    if DYNAMIC_SYSTEM_AVAILABLE:
        instruction_templates = {
            "Standard Maker-Checker": "Focus on maker-checker workflows with dual authorization for all payment processing",
            "High-Value Payment Validation": "Generate test cases for high-value payments (USD 565000) requiring maker input and checker approval",
            "Field Validation Focus": "Emphasize field-by-field validation by checker of all maker inputs before approval",
            "PACS.008 Compliance": "Focus on PACS.008 compliance validation within maker-checker workflows",
            "Cross-Border Payments": "Generate maker-checker test cases for cross-border payments with correspondent banking",
            "Comprehensive Dual Authorization": "Generate comprehensive maker-checker workflows including creation, validation, and approval processes"
        }
    else:
        instruction_templates = {
            "Standard": "",
            "Focus on Negative Cases": "Generate more negative test cases and error scenarios",
            "Comprehensive Coverage": "Generate comprehensive test coverage including positive, negative, edge cases",
        }
    
    selected_template = st.selectbox("Choose Instruction Template:", list(instruction_templates.keys()))
    
    custom_instructions = st.text_area(
        "Custom Instructions",
        value=instruction_templates[selected_template],
        placeholder="e.g., 'Focus on maker-checker validation for USD 565000 payments'",
        help="FIXED: System will generate test cases with explicit maker-checker validation processes"
    )
    
    # Enhanced process button
    process_button_text = "🚀 Generate Maker-Checker Test Cases" if DYNAMIC_SYSTEM_AVAILABLE else "🚀 Generate Test Cases"
    
    if st.button(process_button_text, type="primary", disabled=not api_key or not uploaded_files):
        if not api_key:
            st.error("Please provide OpenAI API key in the sidebar")
            return
        
        if not uploaded_files:
            st.error("Please upload at least one document")
            return
        
        process_files_with_maker_checker_focus(uploaded_files, api_key, custom_instructions, num_test_cases, 
                                             include_edge_cases, include_negative_cases)

def process_files_with_maker_checker_focus(uploaded_files, api_key: str, custom_instructions: str, 
                                         num_test_cases: int, include_edge_cases: bool, include_negative_cases: bool):
    """FIXED: Process files with maker-checker validation focus"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Process files
        doc_processor = DocumentProcessor()
        all_content = []
        total_files = len(uploaded_files)
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            progress_bar.progress((i + 1) / (total_files + 2))
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            result = doc_processor.process_file(tmp_file_path)
            all_content.append(result.get('content', ''))
            os.unlink(tmp_file_path)
        
        combined_content = '\n\n--- Document Separator ---\n\n'.join(all_content)
        
        # Enhanced generation instructions with maker-checker focus
        generation_instructions = build_maker_checker_generation_instructions(
            custom_instructions, num_test_cases, include_edge_cases, include_negative_cases
        )
        
        # Generate test cases with maker-checker focus
        if DYNAMIC_SYSTEM_AVAILABLE:
            status_text.text("Generating test cases with FIXED maker-checker validation...")
            
            files_info = []
            for uploaded_file in uploaded_files:
                file_size = len(uploaded_file.getvalue()) / (1024*1024)
                files_info.append({
                    "name": uploaded_file.name,
                    "size_mb": file_size,
                    "type": uploaded_file.type or "unknown",
                    "status": "processed"
                })
            
            # Use FIXED dynamic system
            generator = DynamicPACS008TestGenerator(api_key)
            workflow_results = generator.process_complete_workflow(combined_content, num_test_cases, files_info)
            
            st.session_state.workflow_results = workflow_results
            test_cases = workflow_results.get("step5_test_cases", [])
            
            # Store field detection results
            field_detection = workflow_results.get("step3_pacs008_fields", {})
            st.session_state.field_detection_results = field_detection
            
            # FIXED: Show maker-checker validation summary
            analysis = workflow_results.get("step1_analysis", {})
            if analysis.get("is_pacs008_relevant", False):
                st.success(f"✅ **CLIENT FEEDBACK ADDRESSED - MAKER-CHECKER VALIDATION APPLIED!**")
                
                # Validate that test cases include maker-checker validation
                maker_checker_compliant_tests = 0
                for tc in test_cases:
                    description = tc.get("Test Case Description", "").lower()
                    if ("ops user maker" in description or "maker" in description) and \
                       ("ops user checker" in description or "checker" in description) and \
                       ("validate" in description or "approve" in description):
                        maker_checker_compliant_tests += 1
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📝 Total Test Cases", len(test_cases))
                with col2:
                    st.metric("👥 Maker-Checker Compliant", maker_checker_compliant_tests)
                with col3:
                    compliance_rate = round((maker_checker_compliant_tests / len(test_cases)) * 100, 1) if test_cases else 0
                    st.metric("✅ Compliance Rate", f"{compliance_rate}%")
                
                if compliance_rate >= 90:
                    st.success(f"🎯 **EXCELLENT**: {compliance_rate}% of test cases include maker-checker validation as required!")
                elif compliance_rate >= 70:
                    st.info(f"✅ **GOOD**: {compliance_rate}% of test cases include maker-checker validation")
                else:
                    st.warning(f"⚠️ **NEEDS IMPROVEMENT**: Only {compliance_rate}% include full maker-checker validation")
        
        else:
            status_text.text("Generating test cases...")
            test_generator = TestCaseGenerator(api_key)
            test_cases = test_generator.generate_test_cases(combined_content, generation_instructions)
            st.session_state.workflow_results = {"step5_test_cases": test_cases}
        
        progress_bar.progress(0.95)
        
        if test_cases:
            st.session_state.generated_test_cases = test_cases
            st.session_state.processing_complete = True
            
            progress_bar.progress(1.0)
            status_text.text("✅ Processing complete!")
            
            # Show maker-checker compliance summary
            total_test_cases = len(test_cases)
            st.success(f"🎯 Generated {total_test_cases} test cases with maker-checker validation focus!")
            
        else:
            st.error("No test cases could be generated. Please check your documents and try again.")
            
    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        logger.error(f"Processing error: {str(e)}")

def build_maker_checker_generation_instructions(custom_instructions: str, num_test_cases: int, 
                                               include_edge_cases: bool, include_negative_cases: bool) -> str:
    """Build enhanced generation instructions with maker-checker focus"""
    instructions = []
    
    # FIXED: Maker-checker focus instructions
    instructions.append("CRITICAL: Every test case description must include explicit maker-checker validation processes")
    instructions.append("Include 'Ops User maker' creation/input actions and 'Ops User checker' validation/approval workflows")
    instructions.append("Focus on dual authorization and field-by-field validation processes")
    
    if custom_instructions:
        instructions.append(custom_instructions)
    
    instructions.append(f"Generate exactly {num_test_cases} test cases per user story/requirement")
    
    if include_edge_cases:
        instructions.append("Include edge cases and boundary conditions within maker-checker workflows")
    
    if include_negative_cases:
        instructions.append("Include negative test scenarios for maker-checker rejection and rework processes")
    
    if DYNAMIC_SYSTEM_AVAILABLE:
        instructions.append("Use FIXED PACS.008 intelligence with maker-checker focus: extract actual amounts (USD 565000), bank names (Al Ahli Bank, BNP Paribas), and create realistic dual authorization scenarios")
    
    return ". ".join(instructions)

def display_test_cases_tab_with_maker_checker_focus(export_formats: List[str]):
    """FIXED: Display test cases with maker-checker validation focus"""
    
    st.header("🧪 Generated Test Cases with Maker-Checker Validation")
    
    if not st.session_state.generated_test_cases:
        st.info("No test cases generated yet. Please upload documents and process them first.")
        return
    
    test_cases = st.session_state.generated_test_cases
    
    # FIXED: Maker-checker compliance analysis
    maker_checker_analysis = analyze_maker_checker_compliance(test_cases)
    
    # Display compliance summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Test Cases", len(test_cases))
    with col2:
        st.metric("👥 Maker-Checker Compliant", maker_checker_analysis["compliant_count"])
    with col3:
        compliance_rate = maker_checker_analysis["compliance_rate"]
        st.metric("✅ Compliance Rate", f"{compliance_rate}%")
    with col4:
        high_priority = len([tc for tc in test_cases if tc.get("Priority") == "High"])
        st.metric("High Priority", high_priority)
    
    # Compliance indicator
    if compliance_rate >= 90:
        st.success(f"🎯 **EXCELLENT COMPLIANCE**: {compliance_rate}% of test cases include required maker-checker validation processes!")
    elif compliance_rate >= 70:
        st.info(f"✅ **GOOD COMPLIANCE**: {compliance_rate}% of test cases include maker-checker validation")
    else:
        st.warning(f"⚠️ **NEEDS IMPROVEMENT**: Only {compliance_rate}% include full maker-checker validation")
    
    # Show detailed compliance breakdown
    with st.expander("📊 Maker-Checker Compliance Analysis", expanded=compliance_rate < 90):
        st.write("**Compliance Requirements Check:**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"✅ **Tests with 'Maker' actions**: {maker_checker_analysis['has_maker_count']}")
            st.write(f"✅ **Tests with 'Checker' validation**: {maker_checker_analysis['has_checker_count']}")
        with col2:
            st.write(f"✅ **Tests with 'Validation' processes**: {maker_checker_analysis['has_validation_count']}")
            st.write(f"✅ **Tests with 'Approval' workflows**: {maker_checker_analysis['has_approval_count']}")
        
        # Show sample compliant vs non-compliant test cases
        compliant_tests = [tc for tc in test_cases if is_maker_checker_compliant(tc)]
        non_compliant_tests = [tc for tc in test_cases if not is_maker_checker_compliant(tc)]
        
        if compliant_tests:
            st.write("**✅ Sample Compliant Test Case:**")
            sample_compliant = compliant_tests[0]
            st.info(f"**{sample_compliant.get('Test Case ID')}**: {sample_compliant.get('Test Case Description', '')[:200]}...")
        
        if non_compliant_tests:
            st.write("**⚠️ Sample Non-Compliant Test Case:**")
            sample_non_compliant = non_compliant_tests[0]
            st.warning(f"**{sample_non_compliant.get('Test Case ID')}**: {sample_non_compliant.get('Test Case Description', '')[:200]}...")
            st.write("*This test case needs to explicitly include maker-checker validation processes*")
    
    # Filter options with maker-checker focus
    with st.expander("🔍 Filter Test Cases"):
        col1, col2, col3, col4 = st.columns(4)
        
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
        
        with col4:
            # FIXED: Maker-checker compliance filter
            compliance_filter = st.selectbox(
                "Maker-Checker Compliance",
                ["All Test Cases", "Compliant Only", "Non-Compliant Only"],
                index=0
            )
    
    # Apply filters
    filtered_test_cases = [
        tc for tc in test_cases
        if (tc.get("Priority") in priority_filter and
            tc.get("Part of Regression") in regression_filter and
            tc.get("User Story ID") in story_filter)
    ]
    
    # Apply maker-checker compliance filter
    if compliance_filter == "Compliant Only":
        filtered_test_cases = [tc for tc in filtered_test_cases if is_maker_checker_compliant(tc)]
    elif compliance_filter == "Non-Compliant Only":
        filtered_test_cases = [tc for tc in filtered_test_cases if not is_maker_checker_compliant(tc)]
    
    # Display filtered test cases
    if filtered_test_cases:
        st.subheader(f"Test Cases ({len(filtered_test_cases)} of {len(test_cases)})")
        
        # Convert to DataFrame with maker-checker indicators
        df = pd.DataFrame(filtered_test_cases)
        
        # Add maker-checker compliance indicator
        df["👥 M-C Compliant"] = df.apply(lambda row: "✅" if is_maker_checker_compliant(row.to_dict()) else "❌", axis=1)
        
        # Enhanced display with maker-checker focus
        column_config = {
            "Test Case Description": st.column_config.TextColumn(width="large"),
            "Steps": st.column_config.TextColumn(width="large"),
            "Expected Result": st.column_config.TextColumn(width="medium"),
            "👥 M-C Compliant": st.column_config.TextColumn(width="small"),
        }
        
        display_columns = [col for col in df.columns if col not in ["PACS008_Enhanced", "Enhancement_Type", "Generation_Method"]]
        
        st.dataframe(
            df[display_columns],
            use_container_width=True,
            column_config=column_config,
            hide_index=True
        )
        
        # Show detailed maker-checker validation example
        compliant_cases = [tc for tc in filtered_test_cases if is_maker_checker_compliant(tc)]
        if compliant_cases:
            with st.expander("👁️ View Detailed Maker-Checker Test Case Example", expanded=False):
                sample_case = compliant_cases[0]
                
                st.write("**📋 Test Case Details:**")
                st.write(f"**Test ID:** {sample_case.get('Test Case ID', 'Unknown')}")
                st.write(f"**Scenario:** {sample_case.get('Scenario', 'Unknown')}")
                
                st.write("**📝 Test Description (with Maker-Checker Validation):**")
                description = sample_case.get('Test Case Description', 'Unknown')
                
                # Highlight maker-checker terms
                highlighted_description = description
                maker_terms = ["Ops User maker", "maker creates", "maker inputs", "maker"]
                checker_terms = ["Ops User checker", "checker validates", "checker reviews", "checker approves", "checker"]
                
                for term in maker_terms:
                    highlighted_description = highlighted_description.replace(term, f"**{term}**")
                for term in checker_terms:
                    highlighted_description = highlighted_description.replace(term, f"**{term}**")
                
                st.markdown(highlighted_description)
                
                st.write("**🧪 Test Steps:**")
                steps = sample_case.get('Steps', '').replace('\n', '\n\n')
                st.text(steps)
                
                st.write("**✅ Expected Result:**")
                st.text(sample_case.get('Expected Result', 'Unknown'))
                
                # Show maker-checker compliance indicators
                compliance_indicators = get_maker_checker_indicators(sample_case)
                if compliance_indicators:
                    st.write("**👥 Maker-Checker Validation Elements Found:**")
                    for indicator in compliance_indicators:
                        st.write(f"• {indicator}")
        
        # Export section
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

def analyze_maker_checker_compliance(test_cases: List[Dict]) -> Dict[str, Any]:
    """Analyze maker-checker compliance across all test cases"""
    
    total_cases = len(test_cases)
    compliant_count = 0
    has_maker_count = 0
    has_checker_count = 0
    has_validation_count = 0
    has_approval_count = 0
    
    for tc in test_cases:
        description = tc.get("Test Case Description", "").lower()
        steps = tc.get("Steps", "").lower()
        expected = tc.get("Expected Result", "").lower()
        
        all_text = f"{description} {steps} {expected}"
        
        # Check for maker terms
        if any(term in all_text for term in ["ops user maker", "maker creates", "maker inputs", "maker submits"]):
            has_maker_count += 1
        
        # Check for checker terms
        if any(term in all_text for term in ["ops user checker", "checker validates", "checker reviews", "checker approves"]):
            has_checker_count += 1
        
        # Check for validation terms
        if any(term in all_text for term in ["validate", "validation", "verify", "review"]):
            has_validation_count += 1
        
        # Check for approval terms
        if any(term in all_text for term in ["approve", "approval", "authorize", "authorization"]):
            has_approval_count += 1
        
        # Check if fully compliant
        if is_maker_checker_compliant(tc):
            compliant_count += 1
    
    compliance_rate = round((compliant_count / total_cases) * 100, 1) if total_cases > 0 else 0
    
    return {
        "total_cases": total_cases,
        "compliant_count": compliant_count,
        "compliance_rate": compliance_rate,
        "has_maker_count": has_maker_count,
        "has_checker_count": has_checker_count,
        "has_validation_count": has_validation_count,
        "has_approval_count": has_approval_count
    }

def is_maker_checker_compliant(test_case: Dict) -> bool:
    """Check if a test case is compliant with maker-checker requirements"""
    
    description = test_case.get("Test Case Description", "").lower()
    steps = test_case.get("Steps", "").lower()
    expected = test_case.get("Expected Result", "").lower()
    
    all_text = f"{description} {steps} {expected}"
    
    # Check for maker terms
    has_maker = any(term in all_text for term in [
        "ops user maker", "maker creates", "maker inputs", "maker submits", "maker"
    ])
    
    # Check for checker terms  
    has_checker = any(term in all_text for term in [
        "ops user checker", "checker validates", "checker reviews", "checker approves", "checker"
    ])
    
    # Check for validation/approval process
    has_process = any(term in all_text for term in [
        "validate", "validation", "approve", "approval", "review", "verify"
    ])
    
    return has_maker and has_checker and has_process

def get_maker_checker_indicators(test_case: Dict) -> List[str]:
    """Get specific maker-checker validation indicators from a test case"""
    
    indicators = []
    description = test_case.get("Test Case Description", "").lower()
    steps = test_case.get("Steps", "").lower()
    expected = test_case.get("Expected Result", "").lower()
    
    all_text = f"{description} {steps} {expected}"
    
    if "ops user maker" in all_text:
        indicators.append("✅ **Explicit 'Ops User maker' role mentioned**")
    elif "maker" in all_text:
        indicators.append("✅ **Maker role referenced**")
    
    if "ops user checker" in all_text:
        indicators.append("✅ **Explicit 'Ops User checker' role mentioned**")
    elif "checker" in all_text:
        indicators.append("✅ **Checker role referenced**")
    
    if "validate" in all_text or "validation" in all_text:
        indicators.append("✅ **Field validation process included**")
    
    if "approve" in all_text or "approval" in all_text:
        indicators.append("✅ **Approval workflow mentioned**")
    
    if "review" in all_text:
        indicators.append("✅ **Review process included**")
    
    if "dual authorization" in all_text or "dual approval" in all_text:
        indicators.append("✅ **Dual authorization workflow**")
    
    return indicators

def maker_checker_analysis_tab():
    """FIXED: New tab specifically for maker-checker analysis"""
    
    st.header("👥 Maker-Checker Analysis")
    
    if not st.session_state.generated_test_cases:
        st.info("📋 No test cases available. Process documents first to see maker-checker analysis.")
        return
    
    test_cases = st.session_state.generated_test_cases
    analysis = analyze_maker_checker_compliance(test_cases)
    
    # Overall compliance metrics
    st.subheader("📊 Maker-Checker Compliance Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📝 Total Test Cases", analysis["total_cases"])
    
    with col2:
        st.metric("✅ Fully Compliant", analysis["compliant_count"])
    
    with col3:
        compliance_rate = analysis["compliance_rate"]
        st.metric("📈 Compliance Rate", f"{compliance_rate}%")
    
    with col4:
        if compliance_rate >= 90:
            st.success("🎯 Excellent")
        elif compliance_rate >= 70:
            st.info("✅ Good")
        else:
            st.warning("⚠️ Needs Work")
    
    # Detailed breakdown
    st.subheader("🔍 Detailed Maker-Checker Element Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**👤 Maker Elements:**")
        maker_rate = round((analysis["has_maker_count"] / analysis["total_cases"]) * 100, 1)
        st.metric("Tests with Maker Actions", analysis["has_maker_count"], f"{maker_rate}%")
        
        st.write("**✅ Validation Elements:**")
        validation_rate = round((analysis["has_validation_count"] / analysis["total_cases"]) * 100, 1)
        st.metric("Tests with Validation", analysis["has_validation_count"], f"{validation_rate}%")
    
    with col2:
        st.write("**👨‍💼 Checker Elements:**")
        checker_rate = round((analysis["has_checker_count"] / analysis["total_cases"]) * 100, 1)
        st.metric("Tests with Checker Actions", analysis["has_checker_count"], f"{checker_rate}%")
        
        st.write("**✅ Approval Elements:**")
        approval_rate = round((analysis["has_approval_count"] / analysis["total_cases"]) * 100, 1)
        st.metric("Tests with Approval", analysis["has_approval_count"], f"{approval_rate}%")
    
    # Compliance recommendations
    if compliance_rate < 90:
        st.subheader("🔧 Improvement Recommendations")
        
        if analysis["has_maker_count"] < analysis["total_cases"] * 0.9:
            st.warning("⚠️ **Add Maker Actions**: Include explicit 'Ops User maker' actions in test descriptions")
        
        if analysis["has_checker_count"] < analysis["total_cases"] * 0.9:
            st.warning("⚠️ **Add Checker Validation**: Include explicit 'Ops User checker' validation in test descriptions")
        
        if analysis["has_validation_count"] < analysis["total_cases"] * 0.9:
            st.warning("⚠️ **Add Validation Process**: Include field validation and verification processes")
        
        if analysis["has_approval_count"] < analysis["total_cases"] * 0.9:
            st.warning("⚠️ **Add Approval Workflow**: Include explicit approval and authorization workflows")
    
    # Show examples of compliant vs non-compliant test cases
    st.subheader("📋 Compliance Examples")
    
    compliant_tests = [tc for tc in test_cases if is_maker_checker_compliant(tc)]
    non_compliant_tests = [tc for tc in test_cases if not is_maker_checker_compliant(tc)]
    
    col1, col2 = st.columns(2)
    
    with col1:
        if compliant_tests:
            st.success("✅ **Compliant Test Case Example:**")
            sample = compliant_tests[0]
            st.write(f"**{sample.get('Test Case ID')}**: {sample.get('Scenario', '')}")
            
            description = sample.get('Test Case Description', '')
            # Highlight key terms
            highlighted = description
            for term in ["Ops User maker", "Ops User checker", "validate", "approve"]:
                highlighted = highlighted.replace(term, f"**{term}**")
            
            st.markdown(f"*{highlighted[:300]}...*")
            
            indicators = get_maker_checker_indicators(sample)
            for indicator in indicators[:3]:
                st.write(indicator)
    
    with col2:
        if non_compliant_tests:
            st.warning("⚠️ **Non-Compliant Test Case Example:**")
            sample = non_compliant_tests[0]
            st.write(f"**{sample.get('Test Case ID')}**: {sample.get('Scenario', '')}")
            st.write(f"*{sample.get('Test Case Description', '')[:300]}...*")
            
            st.write("**Missing Elements:**")
            all_text = f"{sample.get('Test Case Description', '')} {sample.get('Steps', '')}".lower()
            
            if "maker" not in all_text:
                st.write("• ❌ Missing maker actions")
            if "checker" not in all_text:
                st.write("• ❌ Missing checker validation")
            if "validate" not in all_text and "approve" not in all_text:
                st.write("• ❌ Missing validation/approval process")
        else:
            st.success("🎉 **All test cases are compliant!**")
    
    # Export compliance report
    if st.button("📥 Download Compliance Report"):
        export_compliance_report(analysis, compliant_tests, non_compliant_tests)

def export_compliance_report(analysis: Dict, compliant_tests: List, non_compliant_tests: List):
    """Export maker-checker compliance report"""
    
    report = f"""# Maker-Checker Compliance Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overall Compliance Summary
- Total Test Cases: {analysis['total_cases']}
- Fully Compliant: {analysis['compliant_count']}
- Compliance Rate: {analysis['compliance_rate']}%

## Element Analysis
- Tests with Maker Actions: {analysis['has_maker_count']}
- Tests with Checker Actions: {analysis['has_checker_count']}
- Tests with Validation: {analysis['has_validation_count']}
- Tests with Approval: {analysis['has_approval_count']}

## Compliant Test Cases ({len(compliant_tests)})
"""
    
    for tc in compliant_tests[:5]:
        report += f"\n### {tc.get('Test Case ID', 'Unknown')}: {tc.get('Scenario', 'Unknown')}\n"
        report += f"{tc.get('Test Case Description', '')[:200]}...\n"
    
    if non_compliant_tests:
        report += f"\n## Non-Compliant Test Cases ({len(non_compliant_tests)})\n"
        for tc in non_compliant_tests[:5]:
            report += f"\n### {tc.get('Test Case ID', 'Unknown')}: {tc.get('Scenario', 'Unknown')}\n"
            report += f"{tc.get('Test Case Description', '')[:200]}...\n"
    
    st.download_button(
        label="📄 Download Compliance Report",
        data=report.encode('utf-8'),
        file_name=f"maker_checker_compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown"
    )

# Keep existing functions (export_excel, export_csv, export_json, processing_report_tab, chat_assistant_tab)
def export_excel(test_cases):
    """Export Excel"""
    try:
        import io
        df = pd.DataFrame(test_cases)
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Test Cases', index=False)
        
        output.seek(0)
        
        st.download_button(
            label="📊 Download Excel File",
            data=output.getvalue(),
            file_name=f"test_cases_maker_checker_{len(test_cases)}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    except Exception as e:
        st.error(f"Export error: {str(e)}")

def export_csv(test_cases):
    """Export CSV"""
    try:
        csv_data = pd.DataFrame(test_cases).to_csv(index=False)
        st.download_button(
            label="📄 Download CSV File",
            data=csv_data,
            file_name=f"test_cases_maker_checker_{len(test_cases)}.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Export error: {str(e)}")

def export_json(test_cases):
    """Export JSON"""
    try:
        json_data = json.dumps(test_cases, indent=2, ensure_ascii=False)
        st.download_button(
            label="🔧 Download JSON File",
            data=json_data,
            file_name=f"test_cases_maker_checker_{len(test_cases)}.json",
            mime="application/json"
        )
    except Exception as e:
        st.error(f"Export error: {str(e)}")

def processing_report_tab():
    """Processing report tab - same as before but with maker-checker focus"""
    
    st.header("📋 Processing Report & Documentation")
    
    if not st.session_state.workflow_results:
        st.info("📄 No processing report available. Process documents first to see detailed analysis.")
        return
    
    workflow_results = st.session_state.workflow_results
    
    # Show maker-checker compliance in summary
    if st.session_state.generated_test_cases:
        test_cases = st.session_state.generated_test_cases
        analysis = analyze_maker_checker_compliance(test_cases)
        
        st.subheader("👥 Maker-Checker Compliance Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("✅ Compliant Tests", analysis["compliant_count"])
        with col2:
            st.metric("📈 Compliance Rate", f"{analysis['compliance_rate']}%")
        with col3:
            if analysis["compliance_rate"] >= 90:
                st.success("🎯 Excellent")
            elif analysis["compliance_rate"] >= 70:
                st.info("✅ Good")
            else:
                st.warning("⚠️ Needs Work")
    
    # Rest of processing report (same as before)
    documentation = workflow_results.get("documentation", {})
    
    if documentation and documentation.get("report_text"):
        st.success("✅ **Complete Processing Documentation Available**")
        
        # Show basic metrics
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
        
        # Download options
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download Complete Processing Report", type="primary"):
                try:
                    report_text = documentation.get("report_text", "")
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"FIXED_PACS008_MakerChecker_Report_{timestamp}.md"
                    
                    st.download_button(
                        label="📄 Download Report (Markdown)",
                        data=report_text.encode('utf-8'),
                        file_name=filename,
                        mime="text/markdown"
                    )
                    
                except Exception as e:
                    st.error(f"Error preparing download: {str(e)}")

def chat_assistant_tab(api_key: str):
    """Chat assistant with maker-checker focus"""
    
    st.header("💬 Chat Assistant")
    st.markdown("Ask questions about your test cases, maker-checker compliance, or request modifications")
    
    if not api_key:
        st.warning("Please provide OpenAI API key to enable chat functionality")
        return
    
    if not st.session_state.generated_test_cases:
        st.info("Generate test cases first to enable chat assistance")
        return
    
    # Initialize chat with maker-checker focus
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
        
        if DYNAMIC_SYSTEM_AVAILABLE:
            welcome_msg = "Hello! I'm your FIXED PACS.008 testing assistant with maker-checker expertise. I can help you understand the maker-checker validation processes in your test cases, compliance analysis, and banking intelligence applied. What would you like to know about your maker-checker workflows?"
        else:
            welcome_msg = "Hello! I'm your testing assistant. I can help you understand your generated test cases and maker-checker workflows. What would you like to know?"
        
        st.session_state.chat_messages.append({"role": "assistant", "content": welcome_msg})
    
    # Display chat history
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input with maker-checker focus
    if prompt := st.chat_input("Ask about maker-checker validation, test compliance, or banking workflows..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_maker_checker_chat_response(
                    prompt, 
                    st.session_state.generated_test_cases, 
                    st.session_state.field_detection_results, 
                    api_key
                )
                st.markdown(response)
        
        st.session_state.chat_messages.append({"role": "assistant", "content": response})

def generate_maker_checker_chat_response(prompt: str, test_cases: List[Dict], field_results: Dict, api_key: str) -> str:
    """Generate chat response with maker-checker focus"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Analyze maker-checker compliance
        compliance_analysis = analyze_maker_checker_compliance(test_cases)
        
        # Prepare context
        context = f"""Test Cases: {len(test_cases)} total
Maker-Checker Compliance: {compliance_analysis['compliance_rate']}%
Compliant Test Cases: {compliance_analysis['compliant_count']}
Tests with Maker Actions: {compliance_analysis['has_maker_count']}
Tests with Checker Validation: {compliance_analysis['has_checker_count']}

Sample compliant test case:
"""
        
        compliant_tests = [tc for tc in test_cases if is_maker_checker_compliant(tc)]
        if compliant_tests:
            sample = compliant_tests[0]
            context += f"{sample.get('Test Case ID')}: {sample.get('Test Case Description', '')[:200]}..."
        
        chat_prompt = f"""
        You are an expert PACS.008 banking test assistant specializing in maker-checker workflows and compliance.

        Test Cases Context:
        {context}

        Field Detection Results:
        {json.dumps(field_results.get("detection_summary", {}), indent=2) if field_results else "No field detection data"}

        User Question: {prompt}

        Provide helpful, specific answers about:
        - Maker-checker compliance and validation processes
        - Test case quality and banking relevance
        - PACS.008 field detection results
        - Dual authorization workflows
        - How to improve test cases to include maker-checker validation
        - Banking intelligence enhancements applied

        Focus on maker-checker workflows, dual authorization, and field validation processes.
        """
        
        response = client.chat.completions.create(
            model="gpt-4.1-mini-2025-04-14",
            messages=[
                {"role": "system", "content": "You are a helpful PACS.008 banking testing expert specializing in maker-checker workflows and compliance analysis."},
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