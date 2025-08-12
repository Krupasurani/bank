# src/ai_engine/enhanced_test_generator.py - COMPLETE DYNAMIC SYSTEM
"""
Complete Dynamic BFSI Testing System
Automates the entire workflow: Input Analysis → Field Detection → Maker-Checker → Test Generation
"""

import json
import re
import logging
from typing import Dict, List, Any, Optional
from ai_engine.test_generator import TestCaseGenerator

logger = logging.getLogger(__name__)

class CompleteBFSITestingSystem(TestCaseGenerator):
    """Complete automated BFSI testing system with dynamic user story detection and PACS.008 intelligence"""
    
    def __init__(self, api_key: str):
        """Initialize with your existing generator as base"""
        super().__init__(api_key)
        
        # Try to initialize PACS.008 detector
        try:
            from processors.pacs008_intelligent_detector import PACS008IntelligentDetector
            self.pacs008_detector = PACS008IntelligentDetector(api_key)
            self.pacs008_available = True
            logger.info("Complete BFSI testing system with PACS.008 intelligence initialized")
        except ImportError:
            self.pacs008_available = False
            logger.info("PACS.008 detector not available - using standard generation")
    
    def process_complete_bfsi_workflow(self, content: str, custom_instructions: str = "", 
                                     test_cases_per_story: int = 8) -> Dict[str, Any]:
        """
        Complete automated BFSI testing workflow
        Returns comprehensive results including user stories, fields, maker-checker items, and test cases
        """
        
        try:
            logger.info("Starting complete BFSI testing workflow...")
            
            # Step 1: Intelligent User Story Detection
            logger.info("Step 1: Detecting user stories with LLM intelligence...")
            user_stories = self._intelligent_user_story_detection(content)
            
            # Step 2: PACS.008 Field Detection (if available)
            pacs008_analysis = {}
            maker_checker_items = []
            
            if self.pacs008_available:
                logger.info("Step 2: Detecting PACS.008 fields...")
                pacs008_analysis = self.pacs008_detector.detect_pacs008_fields_in_input(content)
                maker_checker_items = pacs008_analysis.get('maker_checker_items', [])
            
            # Step 3: Generate comprehensive test cases for ALL user stories
            logger.info(f"Step 3: Generating {test_cases_per_story} test cases for each of {len(user_stories)} user stories...")
            all_test_cases = self._generate_comprehensive_test_cases(
                user_stories, pacs008_analysis, custom_instructions, test_cases_per_story
            )
            
            # Step 4: Prepare complete results
            workflow_results = {
                "workflow_status": "SUCCESS",
                "user_stories_detected": len(user_stories),
                "total_test_cases_generated": len(all_test_cases),
                "pacs008_fields_detected": len(pacs008_analysis.get('detected_fields', [])) if pacs008_analysis else 0,
                "maker_checker_items": len(maker_checker_items),
                
                # Detailed results
                "detected_user_stories": user_stories,
                "pacs008_analysis": pacs008_analysis,
                "maker_checker_items": maker_checker_items,
                "generated_test_cases": all_test_cases,
                
                # Summary metrics
                "summary": {
                    "user_stories_by_priority": self._categorize_user_stories(user_stories),
                    "test_cases_by_story": self._group_test_cases_by_story(all_test_cases),
                    "pacs008_coverage": self._calculate_pacs008_coverage(pacs008_analysis),
                    "testing_readiness": self._assess_testing_readiness(user_stories, pacs008_analysis, maker_checker_items)
                }
            }
            
            logger.info(f"Complete workflow finished: {len(user_stories)} stories → {len(all_test_cases)} test cases")
            return workflow_results
            
        except Exception as e:
            logger.error(f"Complete workflow failed: {str(e)}")
            return {
                "workflow_status": "ERROR",
                "error": str(e),
                "user_stories_detected": 0,
                "total_test_cases_generated": 0,
                "generated_test_cases": []
            }
    
    def _intelligent_user_story_detection(self, content: str) -> List[Dict[str, Any]]:
        """Use LLM to intelligently detect ALL user stories from input"""
        
        prompt = f"""
        You are a BFSI domain expert. Analyze this input and intelligently identify ALL user stories, requirements, or testable scenarios.

        INPUT CONTENT:
        {content[:3000]}  # Limit for token efficiency

        INSTRUCTIONS:
        1. Look for explicit user stories (As a... I want... So that...)
        2. Look for functional requirements that can be converted to user stories
        3. Look for business scenarios, processes, or workflows
        4. Look for acceptance criteria or business rules
        5. Look for payment flows, banking processes, or financial scenarios
        6. Even if not formally written as user stories, identify distinct testable functionalities

        IMPORTANT: 
        - Each identified item should represent a DISTINCT testable functionality
        - Don't split related functionality into multiple stories
        - Do combine related acceptance criteria under one story
        - Focus on BUSINESS VALUE and distinct USER NEEDS

        RESPOND WITH JSON ONLY:
        {{
          "identified_user_stories": [
            {{
              "story_id": "US001",
              "title": "Concise story title",
              "description": "Full story description or requirement text",
              "business_value": "Why this story is important",
              "complexity": "High/Medium/Low",
              "domain_area": "Payments/Authentication/Reporting/etc",
              "acceptance_criteria": ["AC1: ...", "AC2: ...", "AC3: ..."],
              "pacs008_relevance": "High/Medium/Low/None",
              "estimated_test_scenarios": "Number of test scenarios this story might need"
            }}
          ],
          "analysis_summary": {{
            "total_stories_identified": "number",
            "primary_domain_areas": ["area1", "area2"],
            "complexity_distribution": {{"High": 0, "Medium": 0, "Low": 0}},
            "pacs008_relevant_stories": "number"
          }}
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a BFSI requirements analysis expert. Respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=3000
            )
            
            result = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            
            if json_match:
                llm_result = json.loads(json_match.group())
                user_stories = llm_result.get("identified_user_stories", [])
                
                logger.info(f"LLM detected {len(user_stories)} user stories")
                return user_stories
            else:
                logger.warning("No valid JSON in user story detection response")
                return self._fallback_user_story_detection(content)
                
        except Exception as e:
            logger.error(f"Intelligent user story detection failed: {str(e)}")
            return self._fallback_user_story_detection(content)
    
    def _fallback_user_story_detection(self, content: str) -> List[Dict[str, Any]]:
        """Fallback user story detection using pattern matching"""
        
        user_stories = []
        
        # Pattern-based detection
        story_patterns = [
            r'(?:As\s+(?:a|an)\s+.+?I\s+want\s+.+?(?:so\s+that|in\s+order\s+to).+?)(?=As\s+(?:a|an)|$)',
            r'(?:User\s+Story\s*:?\s*.+?)(?=User\s+Story|$)',
            r'(?:Requirement\s*:?\s*.+?)(?=Requirement|$)',
            r'(?:Feature\s*:?\s*.+?)(?=Feature|$)',
        ]
        
        story_id = 1
        for pattern in story_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                if len(match.strip()) > 50:
                    user_stories.append({
                        "story_id": f"US{story_id:03d}",
                        "title": f"User Story {story_id}",
                        "description": match.strip(),
                        "business_value": "Business functionality requirement",
                        "complexity": "Medium",
                        "domain_area": "Banking",
                        "acceptance_criteria": ["Basic functionality must work as described"],
                        "pacs008_relevance": "Medium",
                        "estimated_test_scenarios": 8
                    })
                    story_id += 1
        
        # If no patterns found, create logical sections
        if not user_stories:
            sections = self._split_content_intelligently(content)
            for i, section in enumerate(sections, 1):
                if len(section.strip()) > 100:
                    user_stories.append({
                        "story_id": f"REQ{i:03d}",
                        "title": f"Requirement Section {i}",
                        "description": section.strip(),
                        "business_value": "Core business requirement",
                        "complexity": "Medium",
                        "domain_area": "BFSI",
                        "acceptance_criteria": ["Functionality must work as specified"],
                        "pacs008_relevance": "Low",
                        "estimated_test_scenarios": 6
                    })
        
        logger.info(f"Fallback detection found {len(user_stories)} user stories")
        return user_stories
    
    def _split_content_intelligently(self, content: str) -> List[str]:
        """Intelligently split content into logical sections"""
        
        # Try different splitting strategies
        sections = []
        
        # Strategy 1: Split by clear headers
        header_patterns = [
            r'\n(?=\d+\.\s+)',  # Numbered sections
            r'\n(?=[A-Z][A-Z\s]+:)',  # All caps headers
            r'\n(?=#{1,3}\s)',  # Markdown headers
            r'\n(?=\*\s+[A-Z])',  # Bulleted sections starting with capital
        ]
        
        for pattern in header_patterns:
            potential_sections = re.split(pattern, content)
            if len(potential_sections) > 1 and all(len(s.strip()) > 50 for s in potential_sections):
                sections = potential_sections
                break
        
        # Strategy 2: Split by paragraphs if headers didn't work
        if not sections:
            paragraphs = content.split('\n\n')
            sections = [p.strip() for p in paragraphs if len(p.strip()) > 100]
        
        # Strategy 3: If still no good sections, split by length
        if not sections:
            chunk_size = max(200, len(content) // 3)  # Aim for 3 chunks
            sections = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        
        return sections[:5]  # Limit to 5 sections maximum
    
    def _generate_comprehensive_test_cases(self, user_stories: List[Dict], pacs008_analysis: Dict, 
                                         custom_instructions: str, test_cases_per_story: int) -> List[Dict[str, Any]]:
        """Generate comprehensive test cases for ALL user stories with PACS.008 intelligence"""
        
        all_test_cases = []
        
        for story in user_stories:
            logger.info(f"Generating {test_cases_per_story} test cases for {story['story_id']}: {story['title']}")
            
            # Create story-specific instructions
            story_instructions = self._create_story_specific_instructions(
                story, pacs008_analysis, custom_instructions, test_cases_per_story
            )
            
            # Generate test cases for this story
            story_test_cases = self._generate_test_cases_for_single_story(
                story, story_instructions, test_cases_per_story
            )
            
            # Add metadata to test cases
            for test_case in story_test_cases:
                test_case["Story_Title"] = story["title"]
                test_case["Domain_Area"] = story.get("domain_area", "BFSI")
                test_case["Story_Complexity"] = story.get("complexity", "Medium")
                test_case["PACS008_Relevance"] = story.get("pacs008_relevance", "Low")
                
                if pacs008_analysis and pacs008_analysis.get('status') == 'SUCCESS':
                    test_case["PACS008_Enhanced"] = "Yes"
                    test_case["Enhancement_Type"] = "PACS008_Intelligent"
                else:
                    test_case["PACS008_Enhanced"] = "No"
                    test_case["Enhancement_Type"] = "Standard"
            
            all_test_cases.extend(story_test_cases)
            logger.info(f"Generated {len(story_test_cases)} test cases for {story['story_id']}")
        
        return all_test_cases
    
    def _create_story_specific_instructions(self, story: Dict, pacs008_analysis: Dict, 
                                          custom_instructions: str, test_cases_per_story: int) -> str:
        """Create enhanced instructions specific to each user story"""
        
        instructions = [custom_instructions] if custom_instructions else []
        
        # Add story-specific context
        instructions.append(f"USER STORY CONTEXT:")
        instructions.append(f"- Story: {story['title']}")
        instructions.append(f"- Domain: {story.get('domain_area', 'BFSI')}")
        instructions.append(f"- Complexity: {story.get('complexity', 'Medium')}")
        instructions.append(f"- Business Value: {story.get('business_value', 'Core functionality')}")
        
        # Add acceptance criteria
        acceptance_criteria = story.get('acceptance_criteria', [])
        if acceptance_criteria:
            instructions.append(f"- Acceptance Criteria: {', '.join(acceptance_criteria[:3])}")
        
        # Add PACS.008 intelligence if available
        if pacs008_analysis and pacs008_analysis.get('status') == 'SUCCESS':
            detected_fields = pacs008_analysis.get('detected_fields', [])
            if detected_fields:
                instructions.append(f"PACS.008 INTELLIGENCE:")
                instructions.append(f"- {len(detected_fields)} PACS.008 fields detected")
                
                # Group fields by category
                field_categories = self._categorize_pacs008_fields(detected_fields)
                for category, fields in field_categories.items():
                    if fields:
                        field_names = [f['field_name'] for f in fields[:2]]
                        instructions.append(f"- {category}: {', '.join(field_names)}")
                
                instructions.append("- Use realistic banking data based on detected fields")
                instructions.append("- Include field validation and business rule scenarios")
                instructions.append("- Focus on PACS.008 compliance and cross-field validation")
        
        # Add test generation strategy
        instructions.append(f"TEST GENERATION STRATEGY:")
        instructions.append(f"- Generate exactly {test_cases_per_story} test cases for this story")
        instructions.append(f"- Include positive scenarios (60%), negative scenarios (25%), edge cases (15%)")
        instructions.append(f"- Use BFSI domain expertise and realistic data")
        
        if story.get('complexity') == 'High':
            instructions.append("- Focus on complex integration and business rule scenarios")
        elif story.get('complexity') == 'Low':
            instructions.append("- Focus on basic functionality and simple validation")
        
        return "\n".join(instructions)
    
    def _categorize_pacs008_fields(self, detected_fields: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize PACS.008 fields for better instruction generation"""
        
        categories = {
            "Banking Agents": [],
            "Accounts": [],
            "Amounts": [],
            "Parties": [],
            "References": [],
            "Other": []
        }
        
        for field in detected_fields:
            field_key = field.get('field_key', '').lower()
            field_name = field.get('field_name', '').lower()
            
            if 'agent' in field_key or 'bic' in field_key:
                categories["Banking Agents"].append(field)
            elif 'account' in field_key or 'iban' in field_key:
                categories["Accounts"].append(field)
            elif 'amount' in field_key or 'currency' in field_key:
                categories["Amounts"].append(field)
            elif 'debtor' in field_key or 'creditor' in field_key or 'ultimate' in field_key:
                categories["Parties"].append(field)
            elif 'id' in field_key or 'reference' in field_key or 'remittance' in field_key:
                categories["References"].append(field)
            else:
                categories["Other"].append(field)
        
        return categories
    
    def _generate_test_cases_for_single_story(self, story: Dict, instructions: str, 
                                            num_cases: int) -> List[Dict[str, Any]]:
        """Generate test cases for a single user story using the parent class method"""
        
        # Convert our enhanced story format to the format expected by parent class
        story_for_generation = {
            "id": story["story_id"],
            "content": f"Title: {story['title']}\n\nDescription: {story['description']}\n\nBusiness Value: {story.get('business_value', '')}"
        }
        
        # Use parent class method with our enhanced instructions
        return self._generate_test_cases_for_story(story_for_generation, instructions, num_cases)
    
    def _categorize_user_stories(self, user_stories: List[Dict]) -> Dict[str, int]:
        """Categorize user stories by various attributes"""
        
        categories = {
            "by_complexity": {"High": 0, "Medium": 0, "Low": 0},
            "by_domain": {},
            "by_pacs008_relevance": {"High": 0, "Medium": 0, "Low": 0, "None": 0}
        }
        
        for story in user_stories:
            # Complexity
            complexity = story.get('complexity', 'Medium')
            categories["by_complexity"][complexity] = categories["by_complexity"].get(complexity, 0) + 1
            
            # Domain
            domain = story.get('domain_area', 'Unknown')
            categories["by_domain"][domain] = categories["by_domain"].get(domain, 0) + 1
            
            # PACS.008 relevance
            pacs008_rel = story.get('pacs008_relevance', 'None')
            categories["by_pacs008_relevance"][pacs008_rel] = categories["by_pacs008_relevance"].get(pacs008_rel, 0) + 1
        
        return categories
    
    def _group_test_cases_by_story(self, test_cases: List[Dict]) -> Dict[str, int]:
        """Group test cases by user story"""
        
        story_counts = {}
        for test_case in test_cases:
            story_id = test_case.get("User Story ID", "Unknown")
            story_counts[story_id] = story_counts.get(story_id, 0) + 1
        
        return story_counts
    
    def _calculate_pacs008_coverage(self, pacs008_analysis: Dict) -> Dict[str, Any]:
        """Calculate PACS.008 coverage metrics"""
        
        if not pacs008_analysis or pacs008_analysis.get('status') != 'SUCCESS':
            return {"status": "Not Available", "coverage": 0}
        
        summary = pacs008_analysis.get('summary', {})
        
        return {
            "status": "Available",
            "total_fields_detected": summary.get('total_detected', 0),
            "mandatory_fields_found": summary.get('mandatory_detected', 0),
            "completion_percentage": summary.get('completion_percentage', 0),
            "ready_for_testing": summary.get('ready_for_testing', False),
            "confidence_score": pacs008_analysis.get('confidence_score', 0)
        }
    
    def _assess_testing_readiness(self, user_stories: List[Dict], pacs008_analysis: Dict, 
                                maker_checker_items: List[Dict]) -> Dict[str, Any]:
        """Assess overall testing readiness"""
        
        readiness = {
            "overall_status": "Ready",
            "user_stories_ready": len(user_stories) > 0,
            "pacs008_analysis_complete": bool(pacs008_analysis and pacs008_analysis.get('status') == 'SUCCESS'),
            "maker_checker_pending": len(maker_checker_items),
            "recommendations": []
        }
        
        # Assess readiness and provide recommendations
        if len(user_stories) == 0:
            readiness["overall_status"] = "Not Ready"
            readiness["recommendations"].append("No user stories detected - please provide clearer requirements")
        
        if len(maker_checker_items) > 5:
            readiness["overall_status"] = "Review Required"
            readiness["recommendations"].append(f"{len(maker_checker_items)} fields require maker-checker validation")
        
        if pacs008_analysis and pacs008_analysis.get('status') == 'SUCCESS':
            summary = pacs008_analysis.get('summary', {})
            if not summary.get('ready_for_testing', True):
                readiness["overall_status"] = "PACS.008 Review Required"
                missing = len(summary.get('missing_mandatory', []))
                readiness["recommendations"].append(f"{missing} mandatory PACS.008 fields missing")
        
        if not readiness["recommendations"]:
            readiness["recommendations"].append("System ready for comprehensive test case generation")
        
        return readiness


# Integration function for Streamlit
def integrate_complete_bfsi_workflow(uploaded_files, api_key: str, custom_instructions: str, 
                                   test_cases_per_story: int = 8) -> Dict[str, Any]:
    """
    Complete integration function for Streamlit app
    Processes files and runs the complete BFSI testing workflow
    """
    
    try:
        from processors.document_processor import DocumentProcessor
        import tempfile
        import os
        from pathlib import Path
        
        # Initialize processors
        doc_processor = DocumentProcessor()
        bfsi_system = CompleteBFSITestingSystem(api_key)
        
        # Process all uploaded files
        all_content = []
        processed_files = []
        
        for uploaded_file in uploaded_files:
            # Save file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Process file
            result = doc_processor.process_file(tmp_file_path)
            if result.get('content'):
                all_content.append(result['content'])
                processed_files.append({
                    "file_name": uploaded_file.name,
                    "content_length": len(result['content']),
                    "file_type": result.get('file_type', 'unknown')
                })
            
            # Cleanup
            os.unlink(tmp_file_path)
        
        # Combine all content
        combined_content = '\n\n--- FILE SEPARATOR ---\n\n'.join(all_content)
        
        # Run complete BFSI workflow
        workflow_results = bfsi_system.process_complete_bfsi_workflow(
            combined_content, custom_instructions, test_cases_per_story
        )
        
        # Add file processing info
        workflow_results["processed_files"] = processed_files
        workflow_results["total_files_processed"] = len(processed_files)
        workflow_results["total_content_length"] = len(combined_content)
        
        return workflow_results
        
    except Exception as e:
        logger.error(f"Complete workflow integration failed: {str(e)}")
        return {
            "workflow_status": "ERROR",
            "error": str(e),
            "user_stories_detected": 0,
            "total_test_cases_generated": 0,
            "generated_test_cases": []
        }