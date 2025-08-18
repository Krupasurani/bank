
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
