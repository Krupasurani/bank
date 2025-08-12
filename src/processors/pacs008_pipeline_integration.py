# src/processors/pacs008_pipeline_integration.py
"""
Complete PACS.008 Pipeline Integration
Ties together intelligent user story detection, PACS.008 field detection, and enhanced test generation
"""

import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import os

logger = logging.getLogger(__name__)

def integrate_pacs008_into_streamlit_pipeline(uploaded_files, api_key: str, custom_instructions: str, pacs008_knowledge: str = None) -> Dict[str, Any]:
    """
    Complete pipeline integration for PACS.008-enhanced test case generation
    
    Returns:
        Dict containing:
        - generated_test_cases: List of all generated test cases
        - enhanced_processing_results: PACS.008 analysis results per file
        - processing_summary: Summary statistics
    """
    
    logger.info("Starting complete PACS.008-enhanced pipeline...")
    
    try:
        # Initialize processors
        from processors.document_processor import DocumentProcessor
        doc_processor = DocumentProcessor()
        
        # Try to initialize enhanced components
        try:
            from processors.pacs008_intelligent_detector import PACS008IntelligentDetector
            from ai_engine.enhanced_test_generator import EnhancedTestCaseGenerator
            
            pacs008_detector = PACS008IntelligentDetector(api_key)
            test_generator = EnhancedTestCaseGenerator(api_key)
            pacs008_mode = True
            logger.info("✅ PACS.008 enhanced components loaded successfully")
            
        except ImportError as e:
            logger.warning(f"PACS.008 components not available: {str(e)}")
            from ai_engine.test_generator import TestCaseGenerator
            test_generator = TestCaseGenerator(api_key)
            pacs008_mode = False
        
        # Process all uploaded files
        logger.info(f"Processing {len(uploaded_files)} uploaded files...")
        
        all_content = []
        file_processing_results = []
        pacs008_relevant_count = 0
        
        for i, uploaded_file in enumerate(uploaded_files):
            logger.info(f"Processing file {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
            
            # Save file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Process document content
                doc_result = doc_processor.process_file(tmp_file_path)
                file_content = doc_result.get('content', '')
                
                # Initialize file processing result
                file_result = {
                    'file_name': uploaded_file.name,
                    'file_type': doc_result.get('file_type', 'unknown'),
                    'content_length': len(file_content),
                    'processing_status': 'SUCCESS'
                }
                
                if file_content:
                    all_content.append(file_content)
                    
                    # PACS.008 analysis if available
                    if pacs008_mode:
                        logger.info(f"Running PACS.008 analysis on {uploaded_file.name}")
                        
                        pacs008_analysis = pacs008_detector.detect_pacs008_fields_in_input(file_content)
                        
                        if pacs008_analysis['status'] == 'SUCCESS':
                            pacs008_relevant_count += 1
                            logger.info(f"✅ PACS.008 content detected in {uploaded_file.name}")
                            
                            file_result.update({
                                'pacs008_intelligence': pacs008_analysis,
                                'maker_checker_prep': {
                                    'fields_requiring_validation': pacs008_analysis.get('maker_checker_items', []),
                                    'validation_summary': {
                                        'total_fields': len(pacs008_analysis.get('detected_fields', [])),
                                        'high_confidence': len([f for f in pacs008_analysis.get('detected_fields', []) if f.get('confidence') == 'High']),
                                        'needs_review': len(pacs008_analysis.get('maker_checker_items', []))
                                    }
                                }
                            })
                        else:
                            logger.info(f"ℹ️ No PACS.008 content detected in {uploaded_file.name}")
                            file_result['pacs008_intelligence'] = pacs008_analysis
                    
                    file_processing_results.append(file_result)
                    
                else:
                    logger.warning(f"No content extracted from {uploaded_file.name}")
                    file_result['processing_status'] = 'NO_CONTENT'
                    file_processing_results.append(file_result)
            
            finally:
                # Cleanup temporary file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
        
        # Combine all content
        combined_content = '\n\n--- DOCUMENT SEPARATOR ---\n\n'.join(all_content)
        
        if not combined_content:
            logger.error("No content extracted from any files")
            return {
                'generated_test_cases': [],
                'enhanced_processing_results': file_processing_results,
                'processing_summary': {
                    'total_files': len(uploaded_files),
                    'files_processed': 0,
                    'pacs008_relevant_files': 0,
                    'test_cases_generated': 0,
                    'intelligence_level': 'FAILED'
                }
            }
        
        # Generate enhanced test cases
        logger.info("Generating intelligent test cases...")
        
        if pacs008_mode and pacs008_relevant_count > 0:
            logger.info(f"Using PACS.008-enhanced generation for {pacs008_relevant_count} relevant files")
            
            # Enhanced instructions with PACS.008 context
            enhanced_instructions = custom_instructions
            if custom_instructions and "test cases per story" not in custom_instructions.lower():
                enhanced_instructions += ". Generate comprehensive test cases for each detected user story."
            
            # Generate enhanced test cases
            generated_test_cases = test_generator.generate_enhanced_test_cases(
                combined_content, 
                enhanced_instructions,
                pacs008_knowledge
            )
            
            intelligence_level = 'PACS008_ENHANCED'
            
        else:
            logger.info("Using standard intelligent generation")
            
            # Standard intelligent generation
            generated_test_cases = test_generator.generate_test_cases(
                combined_content,
                custom_instructions
            )
            
            intelligence_level = 'INTELLIGENT_STANDARD'
        
        # Create processing summary
        processing_summary = {
            'total_files': len(uploaded_files),
            'files_processed': len([r for r in file_processing_results if r['processing_status'] == 'SUCCESS']),
            'pacs008_relevant_files': pacs008_relevant_count,
            'test_cases_generated': len(generated_test_cases),
            'intelligence_level': intelligence_level,
            'content_length': len(combined_content),
            'generation_method': 'Enhanced' if pacs008_mode else 'Standard'
        }
        
        # Add test case breakdown
        if generated_test_cases:
            # Count by user story
            story_counts = {}
            pacs008_enhanced_count = 0
            high_priority_count = 0
            regression_count = 0
            
            for tc in generated_test_cases:
                story_id = tc.get('User Story ID', 'Unknown')
                story_counts[story_id] = story_counts.get(story_id, 0) + 1
                
                if tc.get('PACS008_Enhanced') == 'Yes':
                    pacs008_enhanced_count += 1
                if tc.get('Priority') == 'High':
                    high_priority_count += 1
                if tc.get('Part of Regression') == 'Yes':
                    regression_count += 1
            
            processing_summary.update({
                'unique_user_stories': len(story_counts),
                'user_story_breakdown': story_counts,
                'pacs008_enhanced_tests': pacs008_enhanced_count,
                'high_priority_tests': high_priority_count,
                'regression_tests': regression_count,
                'test_distribution': {
                    'enhanced': pacs008_enhanced_count,
                    'standard': len(generated_test_cases) - pacs008_enhanced_count
                }
            })
        
        logger.info(f"Pipeline complete: {len(generated_test_cases)} test cases generated across {processing_summary['unique_user_stories']} user stories")
        
        return {
            'generated_test_cases': generated_test_cases,
            'enhanced_processing_results': file_processing_results,
            'processing_summary': processing_summary
        }
        
    except Exception as e:
        logger.error(f"Pipeline integration failed: {str(e)}")
        return {
            'generated_test_cases': [],
            'enhanced_processing_results': [],
            'processing_summary': {
                'total_files': len(uploaded_files) if uploaded_files else 0,
                'files_processed': 0,
                'pacs008_relevant_files': 0,
                'test_cases_generated': 0,
                'intelligence_level': 'FAILED',
                'error': str(e)
            }
        }

def validate_pipeline_components():
    """Validate that all pipeline components are available"""
    
    component_status = {
        'document_processor': False,
        'test_generator': False,
        'pacs008_detector': False,
        'enhanced_generator': False,
        'excel_exporter': False
    }
    
    try:
        from processors.document_processor import DocumentProcessor
        component_status['document_processor'] = True
    except ImportError:
        pass
    
    try:
        from ai_engine.test_generator import TestCaseGenerator
        component_status['test_generator'] = True
    except ImportError:
        pass
    
    try:
        from processors.pacs008_intelligent_detector import PACS008IntelligentDetector
        component_status['pacs008_detector'] = True
    except ImportError:
        pass
    
    try:
        from ai_engine.enhanced_test_generator import EnhancedTestCaseGenerator
        component_status['enhanced_generator'] = True
    except ImportError:
        pass
    
    try:
        from exporters.excel_exporter import TestCaseExporter
        component_status['excel_exporter'] = True
    except ImportError:
        pass
    
    return component_status

def get_pipeline_capabilities(api_key: str = None) -> Dict[str, Any]:
    """Get current pipeline capabilities"""
    
    components = validate_pipeline_components()
    
    # Determine capabilities
    has_basic_generation = components['document_processor'] and components['test_generator']
    has_pacs008_intelligence = components['pacs008_detector'] and components['enhanced_generator']
    has_export_capabilities = components['excel_exporter']
    
    capability_level = 'NONE'
    if has_basic_generation and has_pacs008_intelligence:
        capability_level = 'FULL_PACS008_ENHANCED'
    elif has_basic_generation:
        capability_level = 'INTELLIGENT_STANDARD'
    
    capabilities = {
        'capability_level': capability_level,
        'components_available': components,
        'features': {
            'intelligent_user_story_detection': components['test_generator'],
            'document_processing': components['document_processor'],
            'pacs008_field_detection': components['pacs008_detector'],
            'enhanced_test_generation': components['enhanced_generator'],
            'maker_checker_workflow': components['pacs008_detector'],
            'excel_export': components['excel_exporter']
        },
        'supported_file_types': [
            '.docx', '.pdf', '.xlsx', '.txt', '.json', '.csv'
        ] if components['document_processor'] else [],
        'api_key_required': api_key is not None,
        'recommended_setup': []
    }
    
    # Add setup recommendations
    if not components['document_processor']:
        capabilities['recommended_setup'].append("Install document processing dependencies: python-docx, PyPDF2, openpyxl")
    
    if not components['pacs008_detector']:
        capabilities['recommended_setup'].append("Ensure PACS.008 intelligent detector is properly configured")
    
    if not api_key:
        capabilities['recommended_setup'].append("Provide OpenAI API key for LLM intelligence")
    
    return capabilities

# Enhanced Streamlit integration function
def create_enhanced_streamlit_interface():
    """Create enhanced Streamlit interface with full pipeline integration"""
    
    import streamlit as st
    
    # Get pipeline capabilities
    capabilities = get_pipeline_capabilities()
    
    # Display capability status
    if capabilities['capability_level'] == 'FULL_PACS008_ENHANCED':
        st.success("🏦 Full PACS.008 Enhanced Pipeline Active")
        st.info("✅ Intelligent user story detection, PACS.008 field analysis, enhanced test generation")
    elif capabilities['capability_level'] == 'INTELLIGENT_STANDARD':
        st.info("🤖 Intelligent Standard Pipeline Active")
        st.warning("⚠️ PACS.008 enhancements not available")
    else:
        st.error("❌ Pipeline components missing")
        if capabilities['recommended_setup']:
            st.write("**Required setup:**")
            for setup_item in capabilities['recommended_setup']:
                st.write(f"• {setup_item}")
    
    return capabilities

# Usage example and testing
if __name__ == "__main__":
    # Test pipeline components
    print("Testing PACS.008 Pipeline Integration...")
    
    components = validate_pipeline_components()
    print("\nComponent Status:")
    for component, status in components.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {component}: {status}")
    
    capabilities = get_pipeline_capabilities()
    print(f"\nPipeline Capability Level: {capabilities['capability_level']}")
    print(f"Available Features: {list(capabilities['features'].keys())}")
    
    if capabilities['recommended_setup']:
        print("\nRecommended Setup:")
        for item in capabilities['recommended_setup']:
            print(f"• {item}")
    
    # Test with sample data (if API key available)
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and capabilities['capability_level'] != 'NONE':
        print(f"\n🧪 Testing with API key available...")
        
        # Create mock uploaded files for testing
        class MockFile:
            def __init__(self, name, content):
                self.name = name
                self.content = content.encode('utf-8')
            
            def getvalue(self):
                return self.content
        
        mock_files = [
            MockFile("test_requirements.txt", """
            User Story 1: As a bank customer, I want to transfer money internationally 
            so that I can pay my overseas suppliers.
            
            User Story 2: As a compliance officer, I want to validate all payment parties 
            so that we remain compliant with AML regulations.
            
            Requirements:
            - Support PACS.008 messaging format
            - Validate BIC codes for all agents
            - Support multiple currencies: EUR, USD, GBP
            - Handle cross-border settlement
            """)
        ]
        
        try:
            result = integrate_pacs008_into_streamlit_pipeline(
                mock_files, 
                api_key, 
                "Generate exactly 4 test cases per user story"
            )
            
            summary = result['processing_summary']
            print(f"\n📊 Test Results:")
            print(f"   Files processed: {summary['files_processed']}/{summary['total_files']}")
            print(f"   PACS.008 relevant: {summary['pacs008_relevant_files']}")
            print(f"   Test cases generated: {summary['test_cases_generated']}")
            print(f"   User stories detected: {summary.get('unique_user_stories', 'Unknown')}")
            print(f"   Intelligence level: {summary['intelligence_level']}")
            
            if summary.get('user_story_breakdown'):
                print(f"\n📋 User Story Breakdown:")
                for story_id, count in summary['user_story_breakdown'].items():
                    print(f"   {story_id}: {count} test cases")
            
            print(f"\n✅ Pipeline test completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Pipeline test failed: {str(e)}")
    
    else:
        print(f"\n⏭️ Skipping API test (no API key or missing components)")
    
    print(f"\n🏁 Pipeline validation complete!")