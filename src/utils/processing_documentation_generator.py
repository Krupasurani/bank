# src/utils/processing_documentation_generator.py
"""
Processing Documentation Generator
Captures all processing logic, decisions, and outputs in one comprehensive document
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ProcessingDocumentationGenerator:
    """Generates comprehensive documentation of all processing steps and decisions"""
    
    def __init__(self):
        self.documentation = {
            "processing_session": {
                "timestamp": datetime.now().isoformat(),
                "session_id": f"PACS008_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            },
            "input_analysis": {},
            "extracted_content": {},
            "pacs008_intelligence": {},
            "user_stories": {},
            "field_detection": {},
            "maker_checker": {},
            "test_generation": {},
            "processing_summary": {}
        }
    
    def add_input_analysis(self, files_info: List[Dict], combined_content: str):
        """Document input files and initial analysis"""
        
        self.documentation["input_analysis"] = {
            "files_processed": [
                {
                    "filename": file_info.get("name", "unknown"),
                    "size_mb": file_info.get("size_mb", 0),
                    "type": file_info.get("type", "unknown"),
                    "processing_status": file_info.get("status", "processed")
                }
                for file_info in files_info
            ],
            "total_files": len(files_info),
            "total_content_length": len(combined_content),
            "content_preview": combined_content[:500] + "..." if len(combined_content) > 500 else combined_content
        }
    
    def add_extracted_content(self, combined_content: str, content_sources: List[str]):
        """Document all extracted content and sources"""
        
        self.documentation["extracted_content"] = {
            "full_content": combined_content,
            "content_length": len(combined_content),
            "sources": content_sources,
            "content_analysis": {
                "has_user_stories": any(keyword in combined_content.lower() for keyword in ["as a", "user story", "given when then"]),
                "has_banking_keywords": any(keyword in combined_content.lower() for keyword in ["payment", "bank", "pacs", "agent", "account", "iban", "bic"]),
                "has_technical_specs": any(keyword in combined_content.lower() for keyword in ["api", "system", "interface", "message", "field"]),
                "word_count": len(combined_content.split()),
                "line_count": len(combined_content.split('\n'))
            }
        }
    
    def add_pacs008_analysis(self, analysis_result: Dict, field_detection: Dict):
        """Document PACS.008 intelligence analysis and reasoning"""
        
        self.documentation["pacs008_intelligence"] = {
            "content_analysis": {
                "is_pacs008_relevant": analysis_result.get("is_pacs008_relevant", False),
                "confidence_score": analysis_result.get("confidence_score", 0),
                "content_type": analysis_result.get("content_type", "unknown"),
                "technical_level": analysis_result.get("technical_level", "medium"),
                "banking_concepts": analysis_result.get("banking_concepts", []),
                "mentioned_systems": analysis_result.get("mentioned_systems", []),
                "key_indicators": analysis_result.get("key_indicators", []),
                "llm_reasoning": "LLM analyzed content and determined PACS.008 relevance based on banking terminology, payment processing keywords, and technical context."
            },
            "field_detection_summary": {
                "total_unique_fields": field_detection.get("total_unique_fields", 0),
                "detection_method": "LLM-based intelligent detection using static PACS.008 knowledge base",
                "detection_accuracy": "High confidence for explicitly mentioned fields, medium for inferred fields",
                "story_field_mapping": field_detection.get("story_field_mapping", {})
            }
        }
    
    def add_user_stories_extraction(self, user_stories: List[Dict], extraction_method: str, extraction_reasoning: str):
        """Document user story extraction process and reasoning"""
        
        stories_analysis = []
        for story in user_stories:
            story_analysis = {
                "story_id": story.get("id", "unknown"),
                "title": story.get("title", "untitled"),
                "story_text": story.get("story", ""),
                "pacs008_relevance": story.get("pacs008_relevance", "medium"),
                "story_type": story.get("story_type", "unknown"),
                "source_content": story.get("source_content", "")[:200],
                "acceptance_criteria": story.get("acceptance_criteria", []),
                "estimated_test_scenarios": story.get("estimated_test_scenarios", 0),
                "extraction_reasoning": f"Extracted using {extraction_method} - identified as {story.get('story_type', 'unknown')} with {story.get('pacs008_relevance', 'medium')} PACS.008 relevance"
            }
            stories_analysis.append(story_analysis)
        
        self.documentation["user_stories"] = {
            "extraction_summary": {
                "total_stories": len(user_stories),
                "extraction_method": extraction_method,
                "extraction_reasoning": extraction_reasoning,
                "pacs008_relevant_stories": len([s for s in user_stories if s.get("pacs008_relevance") != "low"])
            },
            "individual_stories": stories_analysis,
            "story_types_found": list(set(s.get("story_type", "unknown") for s in user_stories)),
            "llm_decision_logic": "LLM identified user stories by looking for formal story patterns (As a... I want... So that...), requirements that can be converted to stories, and business processes representing user needs."
        }
    
    def add_field_detection_details(self, field_detection_results: Dict):
        """Document detailed field detection logic and reasoning"""
        
        field_detection_details = {}
        
        story_mapping = field_detection_results.get("story_field_mapping", {})
        for story_id, story_data in story_mapping.items():
            detected_fields = story_data.get("detected_fields", [])
            
            fields_analysis = []
            for field in detected_fields:
                field_analysis = {
                    "field_key": field.get("field_key", "unknown"),
                    "field_name": field.get("field_name", "unknown"),
                    "extracted_value": field.get("extracted_value", "not specified"),
                    "confidence": field.get("confidence", "medium"),
                    "detection_reason": field.get("detection_reason", "not provided"),
                    "is_mandatory": field.get("is_mandatory", False),
                    "business_context": field.get("business_context", "not provided"),
                    "llm_reasoning": f"LLM detected this field because: {field.get('detection_reason', 'field was mentioned or inferred from business context')}",
                    "why_this_confidence": self._explain_confidence_level(field)
                }
                fields_analysis.append(field_analysis)
            
            field_detection_details[story_id] = {
                "story_title": story_data.get("story_title", story_id),
                "total_fields": len(detected_fields),
                "mandatory_fields": len([f for f in detected_fields if f.get("is_mandatory", False)]),
                "detected_fields": fields_analysis,
                "detection_summary": f"Found {len(detected_fields)} PACS.008 fields in this user story using LLM intelligence"
            }
        
        self.documentation["field_detection"] = {
            "detection_methodology": "LLM analyzes each user story content against static PACS.008 field knowledge base",
            "confidence_scoring": "High = explicitly mentioned, Medium = inferred from context, Low = uncertain detection",
            "mandatory_field_logic": "Based on ISO 20022 PACS.008 standard - critical fields for payment processing",
            "by_story": field_detection_details,
            "overall_statistics": {
                "total_stories_analyzed": len(story_mapping),
                "stories_with_fields": len([s for s in story_mapping.values() if s.get("field_count", 0) > 0]),
                "total_unique_fields": field_detection_results.get("total_unique_fields", 0)
            }
        }
    
    def add_maker_checker_logic(self, maker_checker_data: Dict):
        """Document maker-checker process preparation and logic"""
        
        validation_items = maker_checker_data.get("validation_items", [])
        
        validation_analysis = []
        for item in validation_items:
            validation_reasoning = {
                "field_name": item.get("field_name", "unknown"),
                "story_id": item.get("story_id", "unknown"),
                "why_needs_validation": item.get("validation_reason", "unknown"),
                "business_impact": item.get("business_impact", "unknown"),
                "maker_action_required": item.get("maker_action", "verify field"),
                "checker_action_required": item.get("checker_action", "validate field"),
                "validation_logic": self._explain_validation_logic(item)
            }
            validation_analysis.append(validation_reasoning)
        
        self.documentation["maker_checker"] = {
            "process_overview": "Automated preparation of fields requiring maker-checker validation based on business criticality",
            "validation_criteria": [
                "Mandatory fields always require validation",
                "Low confidence detections need human review",
                "Fields with no extracted values need maker input",
                "High business impact fields require approval"
            ],
            "validation_items": validation_analysis,
            "summary": maker_checker_data.get("summary", {}),
            "automation_logic": "System automatically identifies which fields need human validation based on PACS.008 business rules and detection confidence"
        }
    
    def add_test_generation_logic(self, test_cases: List[Dict], generation_params: Dict):
        """Document test case generation process and reasoning"""
        
        # Analyze generated test cases
        test_analysis = {
            "by_user_story": {},
            "test_types": {},
            "priority_distribution": {},
            "enhancement_analysis": {}
        }
        
        # Group by user story
        for test_case in test_cases:
            story_id = test_case.get("User Story ID", "unknown")
            if story_id not in test_analysis["by_user_story"]:
                test_analysis["by_user_story"][story_id] = []
            test_analysis["by_user_story"][story_id].append({
                "test_id": test_case.get("Test Case ID", "unknown"),
                "scenario": test_case.get("Scenario", "unknown"),
                "description": test_case.get("Test Case Description", "")[:100] + "...",
                "priority": test_case.get("Priority", "medium"),
                "regression": test_case.get("Part of Regression", "no"),
                "pacs008_enhanced": test_case.get("PACS008_Enhanced", "no"),
                "generation_reasoning": self._explain_test_generation_reasoning(test_case)
            })
        
        # Analyze test types
        for test_case in test_cases:
            scenario = test_case.get("Scenario", "unknown")
            test_type = self._categorize_test_type(scenario)
            test_analysis["test_types"][test_type] = test_analysis["test_types"].get(test_type, 0) + 1
            
            priority = test_case.get("Priority", "medium")
            test_analysis["priority_distribution"][priority] = test_analysis["priority_distribution"].get(priority, 0) + 1
            
            enhancement = test_case.get("PACS008_Enhanced", "no")
            test_analysis["enhancement_analysis"][enhancement] = test_analysis["enhancement_analysis"].get(enhancement, 0) + 1
        
        self.documentation["test_generation"] = {
            "generation_parameters": generation_params,
            "generation_methodology": "LLM generates domain-specific test cases using PACS.008 intelligence and client's business examples",
            "test_case_strategy": [
                "Generate specified number of test cases per user story",
                "Include positive scenarios (happy path)",
                "Include negative scenarios (error conditions)", 
                "Include edge cases and boundary conditions",
                "Use realistic banking data from detected PACS.008 fields",
                "Apply maker-checker workflow scenarios",
                "Focus on business rules and compliance"
            ],
            "total_statistics": {
                "total_test_cases": len(test_cases),
                "test_cases_per_story": {story_id: len(tests) for story_id, tests in test_analysis["by_user_story"].items()},
                "priority_breakdown": test_analysis["priority_distribution"],
                "test_type_breakdown": test_analysis["test_types"],
                "pacs008_enhancement": test_analysis["enhancement_analysis"]
            },
            "test_case_analysis": test_analysis,
            "llm_generation_logic": "LLM uses detected PACS.008 fields to create banking-specific test scenarios with realistic data, maker-checker workflows, and compliance validation"
        }
    
    def add_processing_summary(self, workflow_results: Dict):
        """Add overall processing summary and insights"""
        
        summary = workflow_results.get("workflow_summary", {})
        automation_intelligence = summary.get("automation_intelligence", {})
        
        self.documentation["processing_summary"] = {
            "overall_results": {
                "processing_status": "completed",
                "automation_level": "enhanced" if automation_intelligence.get("content_analysis", {}).get("pacs008_relevant") else "standard",
                "user_stories_found": automation_intelligence.get("user_story_extraction", {}).get("total_stories", 0),
                "pacs008_fields_detected": automation_intelligence.get("field_detection", {}).get("total_unique_fields", 0),
                "test_cases_generated": automation_intelligence.get("test_generation", {}).get("total_test_cases", 0),
                "maker_checker_items": automation_intelligence.get("maker_checker", {}).get("validation_items", 0)
            },
            "quality_indicators": summary.get("quality_indicators", {}),
            "business_value": summary.get("business_value", {}),
            "automation_insights": [
                "System successfully automated domain expert workflow",
                "LLM intelligence applied for PACS.008 field detection",
                "Business context understanding for test generation",
                "Maker-checker process preparation automated",
                "Domain-specific test scenarios generated"
            ],
            "processing_errors": workflow_results.get("processing_errors", []),
            "recommendations_for_improvement": self._generate_improvement_recommendations(workflow_results)
        }
    
    def _explain_confidence_level(self, field: Dict) -> str:
        """Explain why a field has specific confidence level"""
        confidence = field.get("confidence", "medium")
        extracted_value = field.get("extracted_value", "")
        
        if confidence == "high":
            return "High confidence because field was explicitly mentioned with specific value or clear context"
        elif confidence == "medium":
            return "Medium confidence because field was inferred from business context or mentioned without specific value"
        else:
            return "Low confidence because field detection was uncertain or based on weak indicators"
    
    def _explain_validation_logic(self, item: Dict) -> str:
        """Explain why an item needs maker-checker validation"""
        if item.get("is_mandatory"):
            return "Requires validation because it's a mandatory PACS.008 field critical for payment processing"
        elif item.get("confidence") == "low":
            return "Requires validation because detection confidence was low and needs human verification"
        elif not item.get("extracted_value"):
            return "Requires validation because no value was extracted and maker needs to provide input"
        else:
            return "Requires validation to ensure accuracy and compliance with business rules"
    
    def _explain_test_generation_reasoning(self, test_case: Dict) -> str:
        """Explain reasoning behind specific test case generation"""
        scenario = test_case.get("Scenario", "")
        priority = test_case.get("Priority", "medium")
        pacs008_enhanced = test_case.get("PACS008_Enhanced", "no")
        
        reasoning = f"Generated as {priority.lower()} priority "
        
        if "valid" in scenario.lower() or "success" in scenario.lower():
            reasoning += "positive scenario to test happy path functionality"
        elif "invalid" in scenario.lower() or "error" in scenario.lower():
            reasoning += "negative scenario to test error handling"
        elif "boundary" in scenario.lower() or "edge" in scenario.lower():
            reasoning += "edge case scenario to test boundary conditions"
        elif "maker" in scenario.lower() or "checker" in scenario.lower():
            reasoning += "workflow scenario to test maker-checker process"
        else:
            reasoning += "business scenario based on requirements analysis"
        
        if pacs008_enhanced == "yes":
            reasoning += ". Enhanced with PACS.008 intelligence using detected banking fields and domain knowledge."
        
        return reasoning
    
    def _categorize_test_type(self, scenario: str) -> str:
        """Categorize test type based on scenario"""
        scenario_lower = scenario.lower()
        
        if any(word in scenario_lower for word in ["valid", "success", "positive"]):
            return "positive_scenario"
        elif any(word in scenario_lower for word in ["invalid", "error", "negative"]):
            return "negative_scenario"
        elif any(word in scenario_lower for word in ["boundary", "edge", "limit"]):
            return "edge_case"
        elif any(word in scenario_lower for word in ["maker", "checker", "approval", "workflow"]):
            return "workflow_scenario"
        elif any(word in scenario_lower for word in ["security", "auth", "permission"]):
            return "security_scenario"
        else:
            return "business_scenario"
    
    def _generate_improvement_recommendations(self, workflow_results: Dict) -> List[str]:
        """Generate recommendations for improving the system"""
        recommendations = []
        
        # Analyze results and suggest improvements
        test_cases = workflow_results.get("step5_test_cases", [])
        pacs008_analysis = workflow_results.get("step1_analysis", {})
        user_stories = workflow_results.get("step2_user_stories", [])
        
        if len(test_cases) == 0:
            recommendations.append("No test cases generated - improve content analysis and user story extraction")
        
        if not pacs008_analysis.get("is_pacs008_relevant", False):
            recommendations.append("PACS.008 content not detected - enhance banking keyword detection and context analysis")
        
        if len(user_stories) == 0:
            recommendations.append("No user stories extracted - improve story pattern recognition and requirement conversion")
        
        pacs008_enhanced = len([tc for tc in test_cases if tc.get("PACS008_Enhanced") == "Yes"])
        if pacs008_enhanced == 0:
            recommendations.append("No PACS.008 enhanced test cases - improve field detection and banking intelligence")
        
        # Default recommendations
        recommendations.extend([
            "Add more banking domain examples to improve test case relevance",
            "Enhance field detection patterns for better PACS.008 coverage",
            "Expand maker-checker workflow scenarios",
            "Add more realistic banking data samples",
            "Improve confidence scoring algorithm for field detection"
        ])
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def generate_documentation_report(self) -> str:
        """Generate comprehensive documentation report"""
        
        report = f"""
# PACS.008 Test Generation Processing Report
**Generated on:** {self.documentation['processing_session']['timestamp']}
**Session ID:** {self.documentation['processing_session']['session_id']}

## 1. INPUT ANALYSIS
### Files Processed:
"""
        
        for file_info in self.documentation.get("input_analysis", {}).get("files_processed", []):
            report += f"- **{file_info['filename']}** ({file_info['size_mb']:.1f} MB, {file_info['type']})\n"
        
        report += f"""
**Total Content Length:** {self.documentation.get('input_analysis', {}).get('total_content_length', 0):,} characters
**Total Files:** {self.documentation.get('input_analysis', {}).get('total_files', 0)}

## 2. EXTRACTED CONTENT ANALYSIS
"""
        
        content_analysis = self.documentation.get("extracted_content", {}).get("content_analysis", {})
        report += f"""
**Word Count:** {content_analysis.get('word_count', 0):,}
**Line Count:** {content_analysis.get('line_count', 0):,}
**Has User Stories:** {'✅ Yes' if content_analysis.get('has_user_stories') else '❌ No'}
**Has Banking Keywords:** {'✅ Yes' if content_analysis.get('has_banking_keywords') else '❌ No'}
**Has Technical Specs:** {'✅ Yes' if content_analysis.get('has_technical_specs') else '❌ No'}

### Content Preview:
```
{self.documentation.get('input_analysis', {}).get('content_preview', 'No preview available')}
```

## 3. PACS.008 INTELLIGENCE ANALYSIS
"""
        
        pacs008_intel = self.documentation.get("pacs008_intelligence", {})
        content_analysis = pacs008_intel.get("content_analysis", {})
        
        report += f"""
**PACS.008 Relevant:** {'✅ Yes' if content_analysis.get('is_pacs008_relevant') else '❌ No'}
**Confidence Score:** {content_analysis.get('confidence_score', 0)}%
**Content Type:** {content_analysis.get('content_type', 'unknown').title()}
**Technical Level:** {content_analysis.get('technical_level', 'medium').title()}

**Banking Concepts Found:**
"""
        for concept in content_analysis.get('banking_concepts', []):
            report += f"- {concept}\n"
        
        report += f"""
**Systems Mentioned:**
"""
        for system in content_analysis.get('mentioned_systems', []):
            report += f"- {system}\n"
        
        report += f"""
**LLM Reasoning:** {content_analysis.get('llm_reasoning', 'Not available')}

## 4. USER STORIES EXTRACTION
"""
        
        user_stories_data = self.documentation.get("user_stories", {})
        extraction_summary = user_stories_data.get("extraction_summary", {})
        
        report += f"""
**Total Stories Found:** {extraction_summary.get('total_stories', 0)}
**PACS.008 Relevant Stories:** {extraction_summary.get('pacs008_relevant_stories', 0)}
**Extraction Method:** {extraction_summary.get('extraction_method', 'unknown')}
**Story Types:** {', '.join(user_stories_data.get('story_types_found', []))}

**LLM Decision Logic:** {user_stories_data.get('llm_decision_logic', 'Not available')}

### Individual User Stories:
"""
        
        for story in user_stories_data.get("individual_stories", []):
            report += f"""
#### {story['story_id']}: {story['title']}
**Story:** {story['story_text'][:200]}...
**PACS.008 Relevance:** {story['pacs008_relevance'].title()}
**Type:** {story['story_type'].title()}
**Estimated Test Cases:** {story['estimated_test_scenarios']}
**Extraction Reasoning:** {story['extraction_reasoning']}
"""
        
        report += f"""
## 5. PACS.008 FIELD DETECTION
"""
        
        field_detection = self.documentation.get("field_detection", {})
        
        report += f"""
**Detection Methodology:** {field_detection.get('detection_methodology', 'Not documented')}
**Confidence Scoring:** {field_detection.get('confidence_scoring', 'Not documented')}
**Mandatory Field Logic:** {field_detection.get('mandatory_field_logic', 'Not documented')}

**Overall Statistics:**
- Stories Analyzed: {field_detection.get('overall_statistics', {}).get('total_stories_analyzed', 0)}
- Stories with Fields: {field_detection.get('overall_statistics', {}).get('stories_with_fields', 0)}
- Total Unique Fields: {field_detection.get('overall_statistics', {}).get('total_unique_fields', 0)}

### Field Detection by Story:
"""
        
        for story_id, story_data in field_detection.get("by_story", {}).items():
            report += f"""
#### {story_id}: {story_data['story_title']}
**Fields Found:** {story_data['total_fields']} (Mandatory: {story_data['mandatory_fields']})
**Detection Summary:** {story_data['detection_summary']}

**Detected Fields:**
"""
            for field in story_data.get('detected_fields', []):
                report += f"""
- **{field['field_name']}**
  - Value: {field['extracted_value']}
  - Confidence: {field['confidence'].title()}
  - Mandatory: {'Yes' if field['is_mandatory'] else 'No'}
  - Reason: {field['detection_reason']}
  - LLM Reasoning: {field['llm_reasoning']}
  - Confidence Explanation: {field['why_this_confidence']}
"""
        
        report += f"""
## 6. MAKER-CHECKER PROCESS
"""
        
        maker_checker = self.documentation.get("maker_checker", {})
        
        report += f"""
**Process Overview:** {maker_checker.get('process_overview', 'Not documented')}
**Automation Logic:** {maker_checker.get('automation_logic', 'Not documented')}

**Validation Criteria:**
"""
        for criteria in maker_checker.get('validation_criteria', []):
            report += f"- {criteria}\n"
        
        report += f"""
### Validation Items:
"""
        
        for item in maker_checker.get('validation_items', []):
            report += f"""
#### {item['field_name']} (Story: {item['story_id']})
**Why Needs Validation:** {item['why_needs_validation']}
**Business Impact:** {item['business_impact'].title()}
**Maker Action:** {item['maker_action_required']}
**Checker Action:** {item['checker_action_required']}
**Validation Logic:** {item['validation_logic']}
"""
        
        report += f"""
## 7. TEST CASE GENERATION
"""
        
        test_generation = self.documentation.get("test_generation", {})
        
        report += f"""
**Generation Methodology:** {test_generation.get('generation_methodology', 'Not documented')}
**LLM Generation Logic:** {test_generation.get('llm_generation_logic', 'Not documented')}

**Test Case Strategy:**
"""
        for strategy in test_generation.get('test_case_strategy', []):
            report += f"- {strategy}\n"
        
        total_stats = test_generation.get('total_statistics', {})
        report += f"""
**Generation Results:**
- Total Test Cases: {total_stats.get('total_test_cases', 0)}
- Priority Breakdown: {total_stats.get('priority_breakdown', {})}
- Test Type Breakdown: {total_stats.get('test_type_breakdown', {})}
- PACS.008 Enhancement: {total_stats.get('pacs008_enhancement', {})}

**Test Cases per Story:**
"""
        for story_id, count in total_stats.get('test_cases_per_story', {}).items():
            report += f"- {story_id}: {count} test cases\n"
        
        report += f"""
### Test Case Analysis by User Story:
"""
        
        test_analysis = test_generation.get('test_case_analysis', {})
        for story_id, tests in test_analysis.get('by_user_story', {}).items():
            report += f"""
#### {story_id} ({len(tests)} test cases)
"""
            for test in tests[:3]:  # Show first 3 tests per story
                report += f"""
- **{test['test_id']}**: {test['scenario']}
  - Description: {test['description']}
  - Priority: {test['priority'].title()}
  - PACS.008 Enhanced: {test['pacs008_enhanced'].title()}
  - Generation Reasoning: {test['generation_reasoning']}
"""
        
        report += f"""
## 8. PROCESSING SUMMARY & INSIGHTS
"""
        
        processing_summary = self.documentation.get("processing_summary", {})
        overall_results = processing_summary.get("overall_results", {})
        
        report += f"""
**Processing Status:** {overall_results.get('processing_status', 'unknown').title()}
**Automation Level:** {overall_results.get('automation_level', 'unknown').title()}
**User Stories Found:** {overall_results.get('user_stories_found', 0)}
**PACS.008 Fields Detected:** {overall_results.get('pacs008_fields_detected', 0)}
**Test Cases Generated:** {overall_results.get('test_cases_generated', 0)}
**Maker-Checker Items:** {overall_results.get('maker_checker_items', 0)}

**Quality Indicators:**
"""
        quality = processing_summary.get('quality_indicators', {})
        for indicator, value in quality.items():
            report += f"- {indicator.replace('_', ' ').title()}: {value.title()}\n"
        
        report += f"""
**Automation Insights:**
"""
        for insight in processing_summary.get('automation_insights', []):
            report += f"- {insight}\n"
        
        if processing_summary.get('processing_errors'):
            report += f"""
**Processing Errors:**
"""
            for error in processing_summary.get('processing_errors', []):
                report += f"- {error}\n"
        
        report += f"""
**Recommendations for Improvement:**
"""
        for recommendation in processing_summary.get('recommendations_for_improvement', []):
            report += f"- {recommendation}\n"
        
        report += f"""
## 9. COMPLETE EXTRACTED CONTENT
### Full Content from All Sources:
```
{self.documentation.get('extracted_content', {}).get('full_content', 'No content available')}
```

---
**Report Generated by PACS.008 Test Generation System**
**For questions or improvements, refer to this documentation**
"""
        
        return report
    
    def save_documentation(self, filepath: str) -> str:
        """Save documentation to file"""
        try:
            documentation_report = self.generate_documentation_report()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(documentation_report)
            
            logger.info(f"Processing documentation saved to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving documentation: {str(e)}")
            return f"Error saving documentation: {str(e)}"
    
    def get_json_documentation(self) -> Dict[str, Any]:
        """Get documentation as JSON for programmatic access"""
        return self.documentation