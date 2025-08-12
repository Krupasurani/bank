# # src/ai_engine/dynamic_pacs008_test_generator.py
# """
# Complete Dynamic PACS.008 Test Generation System
# Automates the entire workflow: Document Analysis → Field Detection → Maker-Checker → Test Generation
# """

# import json
# import re
# import logging
# from typing import Dict, List, Any, Optional, Tuple
# from openai import OpenAI
# import time

# logger = logging.getLogger(__name__)

# class DynamicPACS008TestGenerator:
#     """Complete automation system for PACS.008 test case generation"""
    
#     def __init__(self, api_key: str):
#         self.client = OpenAI(api_key=api_key)
#         self.model = "gpt-4.1-mini-2025-04-14"
        
#         # PACS.008 domain knowledge
#         self.pacs008_knowledge = self._load_pacs008_knowledge()
        
#         logger.info("Dynamic PACS.008 Test Generation System initialized")
    
#     def _load_pacs008_knowledge(self) -> Dict[str, Any]:
#         """Load comprehensive PACS.008 domain knowledge"""
#         return {
#             "mandatory_fields": {
#                 "debtor_agent": {"name": "Debtor Agent BIC", "examples": ["DEUTDEFF", "CHASUS33"]},
#                 "creditor_agent": {"name": "Creditor Agent BIC", "examples": ["BNPAFRPP", "HSBCGB2L"]},
#                 "debtor_name": {"name": "Debtor Name", "examples": ["ABC Corporation", "John Smith"]},
#                 "creditor_name": {"name": "Creditor Name", "examples": ["XYZ Supplier", "Jane Doe"]},
#                 "debtor_account": {"name": "Debtor Account", "examples": ["DE89370400440532013000"]},
#                 "creditor_account": {"name": "Creditor Account", "examples": ["FR1420041010050500013M02606"]},
#                 "amount": {"name": "Payment Amount", "examples": ["5000.00", "1000.50"]},
#                 "currency": {"name": "Currency", "examples": ["EUR", "USD", "GBP"]},
#                 "instruction_id": {"name": "Instruction ID", "examples": ["INSTR20240801001"]}
#             },
#             "test_scenarios": {
#                 "maker_checker": [
#                     "Payment creation with maker/checker workflow",
#                     "Field validation and approval process",
#                     "Queue management and processing",
#                     "System integration and data flow"
#                 ],
#                 "processing_methods": ["SERIAL", "PARALLEL", "COVER"],
#                 "system_components": ["TPH system", "RLC queues", "Upstream/Downstream systems"],
#                 "user_roles": ["Ops User maker", "Ops User checker", "Admin"]
#             },
#             "business_rules": [
#                 "All banks must have established direct account relationships",
#                 "Nostro/Vostro agent configurations must be valid",
#                 "Cut-off times must be respected",
#                 "Exchange rates must be current",
#                 "RLC setup conditions must be met"
#             ]
#         }
    
#     def process_complete_workflow(self, content: str, num_test_cases_per_story: int = 8) -> Dict[str, Any]:
#         """
#         Complete workflow: Analysis → Detection → Maker-Checker → Test Generation
#         """
#         logger.info("Starting complete PACS.008 workflow automation...")
        
#         workflow_results = {
#             "step1_analysis": {},
#             "step2_user_stories": [],
#             "step3_pacs008_fields": {},
#             "step4_maker_checker": {},
#             "step5_test_cases": [],
#             "workflow_summary": {}
#         }
        
#         try:
#             # Step 1: Intelligent Content Analysis
#             logger.info("Step 1: Analyzing content for PACS.008 relevance...")
#             analysis_result = self._analyze_content_intelligence(content)
#             workflow_results["step1_analysis"] = analysis_result
            
#             # Step 2: Dynamic User Story Extraction
#             logger.info("Step 2: Extracting user stories with LLM intelligence...")
#             user_stories = self._extract_user_stories_intelligently(content, analysis_result)
#             workflow_results["step2_user_stories"] = user_stories
            
#             # Step 3: PACS.008 Field Detection for each story
#             logger.info("Step 3: Detecting PACS.008 fields for each user story...")
#             pacs008_fields = self._detect_pacs008_fields_per_story(user_stories, content)
#             workflow_results["step3_pacs008_fields"] = pacs008_fields
            
#             # Step 4: Maker-Checker Process
#             logger.info("Step 4: Preparing maker-checker validation...")
#             maker_checker_result = self._prepare_maker_checker_process(pacs008_fields)
#             workflow_results["step4_maker_checker"] = maker_checker_result
            
#             # Step 5: Dynamic Test Case Generation
#             logger.info("Step 5: Generating comprehensive test cases...")
#             test_cases = self._generate_dynamic_test_cases(
#                 user_stories, pacs008_fields, num_test_cases_per_story
#             )
#             workflow_results["step5_test_cases"] = test_cases
            
#             # Workflow Summary
#             workflow_results["workflow_summary"] = self._create_workflow_summary(workflow_results)
            
#             logger.info("Complete PACS.008 workflow automation completed successfully!")
#             return workflow_results
            
#         except Exception as e:
#             logger.error(f"Workflow automation error: {str(e)}")
#             workflow_results["error"] = str(e)
#             return workflow_results
    
#     def _analyze_content_intelligence(self, content: str) -> Dict[str, Any]:
#         """Intelligent analysis of content for PACS.008 relevance and context"""
        
#         prompt = f"""
# You are a PACS.008 domain expert. Analyze this content for banking payment relevance.

# CONTENT TO ANALYZE:
# {content[:2000]}

# ANALYSIS REQUIRED:
# 1. Is this content related to PACS.008 or banking payments?
# 2. What type of banking content is this? (requirements, user stories, specifications, etc.)
# 3. What banking concepts are mentioned?
# 4. What level of technical detail is present?
# 5. Are there specific banking systems mentioned?

# RESPOND WITH JSON:
# {{
#   "is_pacs008_relevant": true/false,
#   "content_type": "requirements|user_stories|specifications|procedures|other",
#   "banking_concepts": ["concept1", "concept2"],
#   "technical_level": "high|medium|basic",
#   "mentioned_systems": ["system1", "system2"],
#   "confidence_score": 0-100,
#   "recommended_approach": "enhanced|standard",
#   "key_indicators": ["indicator1", "indicator2"]
# }}
# """
        
#         try:
#             response = self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[
#                     {"role": "system", "content": "You are a PACS.008 banking expert. Respond with valid JSON only."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 temperature=0.1,
#                 max_tokens=1000
#             )
            
#             result = response.choices[0].message.content.strip()
#             json_match = re.search(r'\{.*\}', result, re.DOTALL)
            
#             if json_match:
#                 return json.loads(json_match.group())
#             else:
#                 return {"is_pacs008_relevant": False, "error": "Could not parse analysis"}
                
#         except Exception as e:
#             logger.error(f"Content analysis error: {str(e)}")
#             return {"is_pacs008_relevant": False, "error": str(e)}
    
#     def _extract_user_stories_intelligently(self, content: str, analysis: Dict) -> List[Dict[str, Any]]:
#         """Extract user stories using LLM intelligence based on content analysis"""
        
#         is_relevant = analysis.get("is_pacs008_relevant", False)
#         content_type = analysis.get("content_type", "other")
        
#         prompt = f"""
# You are a BFSI business analyst expert. Extract user stories from this content.

# CONTENT:
# {content}

# CONTENT ANALYSIS:
# - PACS.008 Relevant: {is_relevant}
# - Content Type: {content_type}
# - Banking Concepts: {analysis.get('banking_concepts', [])}

# EXTRACTION RULES:
# 1. Look for actual user stories (As a... I want... So that...)
# 2. Look for feature requirements that can be converted to user stories
# 3. Look for business processes that represent user needs
# 4. If content describes testing scenarios, extract the underlying user stories
# 5. Group related functionality into coherent user stories
# 6. Focus on banking/payment domain stories

# RESPOND WITH JSON:
# {{
#   "user_stories": [
#     {{
#       "id": "US001",
#       "title": "Brief title",
#       "story": "As a [user] I want [functionality] so that [benefit]",
#       "source_content": "Original text that led to this story",
#       "pacs008_relevance": "high|medium|low",
#       "story_type": "payment_processing|user_interface|validation|reporting|other",
#       "acceptance_criteria": ["AC1 description", "AC2 description"],
#       "estimated_test_scenarios": 5-15
#     }}
#   ],
#   "extraction_summary": {{
#     "total_stories": 3,
#     "pacs008_stories": 2,
#     "story_types": ["payment_processing", "user_interface"]
#   }}
# }}
# """
        
#         try:
#             response = self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[
#                     {"role": "system", "content": "You are a BFSI business analyst. Extract meaningful user stories. Respond with valid JSON only."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 temperature=0.2,
#                 max_tokens=2000
#             )
            
#             result = response.choices[0].message.content.strip()
#             json_match = re.search(r'\{.*\}', result, re.DOTALL)
            
#             if json_match:
#                 extracted = json.loads(json_match.group())
#                 return extracted.get("user_stories", [])
#             else:
#                 logger.warning("Could not parse user story extraction")
#                 return self._fallback_user_story_extraction(content)
                
#         except Exception as e:
#             logger.error(f"User story extraction error: {str(e)}")
#             return self._fallback_user_story_extraction(content)
    
#     def _fallback_user_story_extraction(self, content: str) -> List[Dict[str, Any]]:
#         """Fallback method for user story extraction"""
#         logger.info("Using fallback user story extraction")
        
#         # Simple pattern matching for common user story formats
#         patterns = [
#             r'As\s+(?:a|an)\s+(.+?)\s+I\s+want\s+(.+?)\s+(?:so\s+that|in\s+order\s+to)\s+(.+?)(?=\.|$)',
#             r'User\s+Story\s*:?\s*(.+?)(?=User\s+Story|$)',
#             r'Requirement\s*:?\s*(.+?)(?=Requirement|$)'
#         ]
        
#         stories = []
#         story_id = 1
        
#         for pattern in patterns:
#             matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
#             for match in matches:
#                 if isinstance(match, tuple):
#                     story_text = f"As a {match[0]}, I want {match[1]} so that {match[2]}"
#                 else:
#                     story_text = match
                
#                 if len(story_text.strip()) > 20:
#                     stories.append({
#                         "id": f"US{story_id:03d}",
#                         "title": f"User Story {story_id}",
#                         "story": story_text.strip(),
#                         "source_content": story_text.strip()[:200],
#                         "pacs008_relevance": "medium",
#                         "story_type": "payment_processing",
#                         "acceptance_criteria": ["Basic functionality", "Error handling"],
#                         "estimated_test_scenarios": 8
#                     })
#                     story_id += 1
        
#         # If no formal stories found, create one from content
#         if not stories:
#             stories.append({
#                 "id": "US001",
#                 "title": "Payment Processing Requirement",
#                 "story": "As a banking user, I want to process payments according to PACS.008 standards so that transactions are handled correctly",
#                 "source_content": content[:200],
#                 "pacs008_relevance": "high",
#                 "story_type": "payment_processing",
#                 "acceptance_criteria": ["Valid payment processing", "Error handling", "Compliance validation"],
#                 "estimated_test_scenarios": 8
#             })
        
#         return stories[:5]  # Limit to 5 user stories
    
#     def _detect_pacs008_fields_per_story(self, user_stories: List[Dict], full_content: str) -> Dict[str, Any]:
#         """Detect PACS.008 fields for each user story"""
        
#         story_field_mapping = {}
#         all_detected_fields = []
        
#         for story in user_stories:
#             story_id = story["id"]
#             story_content = story.get("source_content", "") + "\n" + story.get("story", "")
            
#             # Detect fields for this specific story
#             detected_fields = self._detect_fields_for_single_story(story_content, full_content)
            
#             story_field_mapping[story_id] = {
#                 "story_title": story["title"],
#                 "detected_fields": detected_fields,
#                 "field_count": len(detected_fields),
#                 "mandatory_fields": len([f for f in detected_fields if f.get("is_mandatory", False)]),
#                 "pacs008_relevance": story.get("pacs008_relevance", "medium")
#             }
            
#             all_detected_fields.extend(detected_fields)
        
#         return {
#             "story_field_mapping": story_field_mapping,
#             "all_detected_fields": all_detected_fields,
#             "total_unique_fields": len(set(f["field_name"] for f in all_detected_fields)),
#             "detection_summary": {
#                 "total_stories_processed": len(user_stories),
#                 "stories_with_pacs008": len([s for s in story_field_mapping.values() if s["field_count"] > 0]),
#                 "most_relevant_story": max(story_field_mapping.keys(), 
#                                          key=lambda x: story_field_mapping[x]["field_count"]) if story_field_mapping else None
#             }
#         }
    
#     def _detect_fields_for_single_story(self, story_content: str, context_content: str) -> List[Dict[str, Any]]:
#         """Detect PACS.008 fields for a single user story"""
        
#         # Create field reference
#         field_ref = []
#         for field_key, field_info in self.pacs008_knowledge["mandatory_fields"].items():
#             examples = ", ".join(field_info["examples"])
#             field_ref.append(f"- {field_key}: {field_info['name']} [Examples: {examples}]")
        
#         prompt = f"""
# You are a PACS.008 field detection expert. Analyze this user story for PACS.008 fields.

# USER STORY:
# {story_content}

# CONTEXT (for reference):
# {context_content[:1000]}

# PACS.008 FIELDS TO DETECT:
# {chr(10).join(field_ref)}

# DETECTION RULES:
# 1. Look for explicit field mentions
# 2. Infer fields from business context (e.g., "bank" = agent, "account" = debtor/creditor account)
# 3. Consider maker-checker workflow implications
# 4. Look for system integration points

# RESPOND WITH JSON:
# {{
#   "detected_fields": [
#     {{
#       "field_key": "debtor_agent",
#       "field_name": "Debtor Agent BIC",
#       "extracted_value": "DEUTDEFF or mentioned but not specified",
#       "confidence": "high|medium|low",
#       "detection_reason": "Why you detected this field",
#       "is_mandatory": true,
#       "business_context": "How this field relates to the user story"
#     }}
#   ]
# }}
# """
        
#         try:
#             response = self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[
#                     {"role": "system", "content": "You are a PACS.008 expert. Detect fields accurately. Respond with valid JSON only."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 temperature=0.1,
#                 max_tokens=1500
#             )
            
#             result = response.choices[0].message.content.strip()
#             json_match = re.search(r'\{.*\}', result, re.DOTALL)
            
#             if json_match:
#                 parsed = json.loads(json_match.group())
#                 return parsed.get("detected_fields", [])
#             else:
#                 return []
                
#         except Exception as e:
#             logger.error(f"Field detection error: {str(e)}")
#             return []
    
#     def _prepare_maker_checker_process(self, pacs008_fields: Dict) -> Dict[str, Any]:
#         """Prepare maker-checker validation process"""
        
#         maker_checker_items = []
        
#         for story_id, story_data in pacs008_fields.get("story_field_mapping", {}).items():
#             for field in story_data.get("detected_fields", []):
                
#                 # Determine if field needs validation
#                 needs_validation = (
#                     field.get("is_mandatory", False) or
#                     field.get("confidence", "low") == "low" or
#                     not field.get("extracted_value") or
#                     field.get("extracted_value", "").lower() in ["mentioned but not specified", "not specified"]
#                 )
                
#                 if needs_validation:
#                     maker_checker_items.append({
#                         "story_id": story_id,
#                         "field_key": field.get("field_key"),
#                         "field_name": field.get("field_name"),
#                         "extracted_value": field.get("extracted_value"),
#                         "confidence": field.get("confidence"),
#                         "is_mandatory": field.get("is_mandatory", False),
#                         "validation_reason": self._get_validation_reason(field),
#                         "maker_action": "Verify field accuracy and business relevance",
#                         "checker_action": "Validate against PACS.008 standards and approve",
#                         "business_impact": self._assess_business_impact(field)
#                     })
        
#         return {
#             "validation_items": maker_checker_items,
#             "summary": {
#                 "total_items": len(maker_checker_items),
#                 "mandatory_items": len([item for item in maker_checker_items if item["is_mandatory"]]),
#                 "high_priority_items": len([item for item in maker_checker_items if item["business_impact"] == "high"]),
#                 "stories_affected": len(set(item["story_id"] for item in maker_checker_items))
#             },
#             "validation_ready": len(maker_checker_items) > 0
#         }
    
#     def _get_validation_reason(self, field: Dict) -> str:
#         """Get reason why field needs validation"""
#         if field.get("is_mandatory") and not field.get("extracted_value"):
#             return "Mandatory field missing - critical for PACS.008 processing"
#         elif field.get("confidence") == "low":
#             return "Low confidence detection - needs verification"
#         elif not field.get("extracted_value"):
#             return "Field value not specified - needs maker input"
#         else:
#             return "Field detected but requires validation"
    
#     def _assess_business_impact(self, field: Dict) -> str:
#         """Assess business impact of field validation"""
#         if field.get("is_mandatory"):
#             return "high"
#         elif field.get("field_key") in ["debtor_agent", "creditor_agent", "amount"]:
#             return "high"
#         elif field.get("confidence") == "high":
#             return "medium"
#         else:
#             return "low"
    
#     def _generate_dynamic_test_cases(self, user_stories: List[Dict], pacs008_fields: Dict, 
#                                    num_cases_per_story: int) -> List[Dict[str, Any]]:
#         """Generate comprehensive test cases for each user story with PACS.008 intelligence"""
        
#         all_test_cases = []
        
#         for story in user_stories:
#             story_id = story["id"]
            
#             # Get PACS.008 context for this story
#             story_pacs008_data = pacs008_fields.get("story_field_mapping", {}).get(story_id, {})
#             detected_fields = story_pacs008_data.get("detected_fields", [])
            
#             # Generate test cases for this story
#             story_test_cases = self._generate_test_cases_for_story_with_context(
#                 story, detected_fields, num_cases_per_story
#             )
            
#             all_test_cases.extend(story_test_cases)
        
#         return all_test_cases
    
#     def _generate_test_cases_for_story_with_context(self, story: Dict, detected_fields: List[Dict], 
#                                                   num_cases: int) -> List[Dict[str, Any]]:
#         """Generate test cases for a single story with PACS.008 context"""
        
#         story_id = story["id"]
#         story_content = story["story"]
#         story_type = story.get("story_type", "payment_processing")
        
#         # Create PACS.008 context
#         pacs008_context = self._create_pacs008_test_context(detected_fields)
        
#         # Create domain-specific examples based on your client's feedback
#         domain_examples = self._get_domain_specific_examples(story_type)
        
#         prompt = f"""
# You are an expert BFSI test engineer specializing in PACS.008 payment systems. Generate comprehensive test cases.

# USER STORY:
# {story_content}

# STORY TYPE: {story_type}

# DETECTED PACS.008 FIELDS:
# {pacs008_context}

# DOMAIN EXAMPLES (for reference):
# {domain_examples}

# GENERATE EXACTLY {num_cases} TEST CASES for this user story.

# REQUIREMENTS:
# 1. All test cases must have SAME User Story ID: {story_id}
# 2. Include maker-checker workflow scenarios
# 3. Use realistic banking data from detected fields
# 4. Include system integration scenarios (TPH system, RLC queues, etc.)
# 5. Cover validation, processing, and approval workflows
# 6. Include both positive and negative scenarios
# 7. Focus on business rules and compliance

# RESPOND WITH ONLY JSON ARRAY:
# [
#   {{
#     "User Story ID": "{story_id}",
#     "Acceptance Criteria ID": "AC001",
#     "Scenario": "Payment creation with maker/checker",
#     "Test Case ID": "TC001",
#     "Test Case Description": "Verify successful creation of PACS.008 message with maker-checker workflow",
#     "Precondition": "All nostro/vostro agents configured, cut-off times set, exchange rates available",
#     "Steps": "1. Login as Ops User maker\\n2. Navigate to payment creation\\n3. Enter all required PACS.008 fields\\n4. Submit for approval\\n5. Login as checker\\n6. Review and approve",
#     "Expected Result": "Payment successfully created and approved, available in processing queue",
#     "Part of Regression": "Yes",
#     "Priority": "High"
#   }}
# ]
# """
        
#         try:
#             response = self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[
#                     {"role": "system", "content": f"You are a PACS.008 test expert. Generate exactly {num_cases} comprehensive test cases. Respond with valid JSON array only."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 temperature=0.2,
#                 max_tokens=4000
#             )
            
#             result = response.choices[0].message.content.strip()
#             json_match = re.search(r'\[.*\]', result, re.DOTALL)
            
#             if json_match:
#                 test_cases = json.loads(json_match.group())
                
#                 # Validate and enhance test cases
#                 enhanced_cases = self._enhance_test_cases_with_pacs008_intelligence(
#                     test_cases, story, detected_fields
#                 )
                
#                 return enhanced_cases
#             else:
#                 logger.warning(f"Could not parse test cases for {story_id}")
#                 return self._generate_fallback_test_cases(story, detected_fields, num_cases)
                
#         except Exception as e:
#             logger.error(f"Test generation error for {story_id}: {str(e)}")
#             return self._generate_fallback_test_cases(story, detected_fields, num_cases)
    
#     def _create_pacs008_test_context(self, detected_fields: List[Dict]) -> str:
#         """Create PACS.008 context for test generation"""
#         if not detected_fields:
#             return "No specific PACS.008 fields detected - use standard payment processing context"
        
#         context_lines = []
#         for field in detected_fields:
#             context_lines.append(
#                 f"- {field.get('field_name')}: {field.get('extracted_value', 'To be tested')} "
#                 f"(Confidence: {field.get('confidence', 'medium')})"
#             )
        
#         return "\n".join(context_lines)
    
#     def _get_domain_specific_examples(self, story_type: str) -> str:
#         """Get domain-specific examples based on client feedback"""
        
#         examples = {
#             "payment_processing": """
# Example Test Case Format:
# - Scenario: Payment creation with maker/checker
# - Description: Verify whether all fields are available for PACS.008 in the TPH system
# - Steps: 1. Login as Ops User maker, 2. View all fields like currency, amount, debit account number etc.
# - Expected: All relevant fields available (debtor name/address, debtor account, amount, currency, creditor, creditor agent, etc.)
#             """,
#             "user_interface": """
# Example Test Case Format:
# - Scenario: Field validation and display
# - Description: Verify maker user able to input all data for PACS.008 message creation
# - Prerequisites: All Nostro/Vostro agents, cut-off times, exchange rates, upstream/downstream systems connected
# - Expected: TPH system allows payment creation, defaults bank/agent/customer account per configuration
#             """,
#             "validation": """
# Example Test Case Format:
# - Scenario: Checker approval workflow
# - Description: Verify checker user able to see all data inputted by maker
# - Steps: Review maker inputs, validate against business rules, approve/reject
# - Expected: Checker can review and approve payment, transaction moves to processing queue
#             """
#         }
        
#         return examples.get(story_type, examples["payment_processing"])
    
#     def _enhance_test_cases_with_pacs008_intelligence(self, test_cases: List[Dict], 
#                                                     story: Dict, detected_fields: List[Dict]) -> List[Dict[str, Any]]:
#         """Enhance test cases with PACS.008 intelligence and client domain knowledge"""
        
#         enhanced_cases = []
        
#         for i, test_case in enumerate(test_cases):
#             # Ensure proper structure
#             enhanced_case = {
#                 "User Story ID": story["id"],
#                 "Acceptance Criteria ID": test_case.get("Acceptance Criteria ID", f"AC{(i//3)+1:03d}"),
#                 "Scenario": test_case.get("Scenario", f"Test Scenario {i+1}"),
#                 "Test Case ID": test_case.get("Test Case ID", f"TC{i+1:03d}"),
#                 "Test Case Description": test_case.get("Test Case Description", ""),
#                 "Precondition": test_case.get("Precondition", "System available and configured"),
#                 "Steps": test_case.get("Steps", "").replace("\\n", "\n"),
#                 "Expected Result": test_case.get("Expected Result", ""),
#                 "Part of Regression": test_case.get("Part of Regression", "Yes"),
#                 "Priority": test_case.get("Priority", "Medium")
#             }
            
#             # Add PACS.008 enhancement metadata
#             enhanced_case["PACS008_Enhanced"] = "Yes" if detected_fields else "No"
#             enhanced_case["Enhancement_Type"] = "PACS008_Domain_Intelligent"
#             enhanced_case["Detected_Fields_Count"] = len(detected_fields)
            
#             # Enhance with realistic banking data
#             enhanced_case = self._inject_realistic_banking_data(enhanced_case, detected_fields)
            
#             enhanced_cases.append(enhanced_case)
        
#         return enhanced_cases
    
#     def _inject_realistic_banking_data(self, test_case: Dict, detected_fields: List[Dict]) -> Dict[str, Any]:
#         """Inject realistic banking data based on detected fields"""
        
#         # Banking data pool
#         banking_data = {
#             "bics": ["DEUTDEFF", "BNPAFRPP", "HSBCGB2L", "CHASUS33", "CITIUS33"],
#             "ibans": ["DE89370400440532013000", "FR1420041010050500013M02606", "GB33BUKB20201555555555"],
#             "amounts": ["1000.00", "5000.50", "25000.00", "100.00"],
#             "currencies": ["EUR", "USD", "GBP", "CHF"],
#             "customer_names": ["ABC Corporation Ltd", "XYZ Trading Company", "Global Services Inc"]
#         }
        
#         steps = test_case.get("Steps", "")
#         expected = test_case.get("Expected Result", "")
        
#         # Replace generic placeholders with specific banking data
#         for field in detected_fields:
#             field_key = field.get("field_key", "")
#             extracted_value = field.get("extracted_value", "")
            
#             # Use extracted value if available, otherwise use realistic sample
#             if "agent" in field_key or "bic" in field_key:
#                 bic_value = extracted_value if extracted_value and extracted_value != "mentioned but not specified" else banking_data["bics"][0]
#                 steps = steps.replace("bank BIC", bic_value).replace("agent BIC", bic_value)
                
#             elif "account" in field_key or "iban" in field_key:
#                 iban_value = extracted_value if extracted_value and extracted_value != "mentioned but not specified" else banking_data["ibans"][0]
#                 steps = steps.replace("account number", iban_value).replace("IBAN", iban_value)
                
#             elif "amount" in field_key:
#                 amount_value = extracted_value if extracted_value and extracted_value != "mentioned but not specified" else banking_data["amounts"][0]
#                 steps = steps.replace("amount", f"amount: {amount_value}").replace("payment amount", amount_value)
                
#             elif "currency" in field_key:
#                 currency_value = extracted_value if extracted_value and extracted_value != "mentioned but not specified" else banking_data["currencies"][0]
#                 steps = steps.replace("currency", currency_value)
        
#         # Update the enhanced test case
#         test_case["Steps"] = steps
#         test_case["Expected Result"] = expected
        
#         return test_case
    
#     def _generate_fallback_test_cases(self, story: Dict, detected_fields: List[Dict], 
#                                     num_cases: int) -> List[Dict[str, Any]]:
#         """Generate fallback test cases based on client's domain examples"""
        
#         story_id = story["id"]
#         fallback_cases = []
        
#         # Base scenarios from client feedback
#         base_scenarios = [
#             {
#                 "scenario": "Payment creation with maker/checker",
#                 "description": "Verify whether all fields are available for PACS.008 in the TPH system",
#                 "precondition": "Menu, Navigation, fields, label should be available",
#                 "steps": "1. Login as Ops User maker\n2. View the all the fields like currency amount, debit account number etc",
#                 "expected": "All relevant fields available in TPH system to create a PACS.008\n1. debtor name and address\n2. debtor account\n3. amount\n4. currency\n5. creditor, creditor agent etc",
#                 "priority": "High",
#                 "regression": "Yes"
#             },
#             {
#                 "scenario": "Maker data input validation",
#                 "description": "Verify whether Maker user able to input all data for PACS.008 message creation",
#                 "precondition": "All Nostro/vostro agent, cut off time, exchange rate, upstream and downstream system are connected",
#                 "steps": "1. Login as Ops User maker\n2. Enter all required for PACS.008 creation",
#                 "expected": "TPH system should allow user create payment PACS.008 (yet to approve by checker)\nTPH system able to default bank/agent, customer account, as per setup/configuration\nTPH system able to fetch data upstream/downstream correctly",
#                 "priority": "High",
#                 "regression": "Yes"
#             },
#             {
#                 "scenario": "Checker approval process",
#                 "description": "Verify whether Checker user able to see all data in screen which are inputted by maker",
#                 "precondition": "All Nostro/vostro agent, cut off time, exchange rate, upstream and downstream system are connected",
#                 "steps": "1. Login as Ops User checker\n2. Navigate to approval queue\n3. Review maker inputs\n4. Approve/reject payment",
#                 "expected": "TPH system should allow the checker to check/approve the Payment",
#                 "priority": "High",
#                 "regression": "Yes"
#             },
#             {
#                 "scenario": "Queue management validation",
#                 "description": "Verify whether transaction available in Soft block queues after op checker approved the transaction",
#                 "precondition": "All RLC setup configuration available",
#                 "steps": "1. Login as Ops User maker\n2. Navigate to RLC queue\n3. Check transaction status",
#                 "expected": "Transaction should be available RLC as RLC Setup condition is met",
#                 "priority": "Medium",
#                 "regression": "Yes"
#             },
#             {
#                 "scenario": "PACS.008 message processing",
#                 "description": "Verify successful processing of PACS.008 message via SERIAL method",
#                 "precondition": "All banks in payment chain have established direct account relationships; valid payment data available",
#                 "steps": "1. Initiate PACS.008 message from Debtor Agent with amount 5000.00 EUR\n2. Ensure message contains all necessary payment information\n3. Send message to next bank in chain\n4. Each intermediate bank processes and forwards PACS.008 message\n5. Final Creditor Agent receives and processes the message",
#                 "expected": "Payment is successfully processed through all banks with correct settlement instructions and bookings at each step",
#                 "priority": "High",
#                 "regression": "Yes"
#             },
#             {
#                 "scenario": "Field validation and error handling",
#                 "description": "Verify system validation for mandatory PACS.008 fields",
#                 "precondition": "System is available and user is authenticated",
#                 "steps": "1. Login as Ops User maker\n2. Attempt to create payment with missing mandatory fields\n3. Submit for validation",
#                 "expected": "System displays appropriate validation errors for missing mandatory fields",
#                 "priority": "High",
#                 "regression": "Yes"
#             },
#             {
#                 "scenario": "System integration testing",
#                 "description": "Verify integration between TPH system and upstream/downstream systems",
#                 "precondition": "All system integrations configured and available",
#                 "steps": "1. Create PACS.008 payment in TPH system\n2. Verify data synchronization with upstream system\n3. Check downstream system processing",
#                 "expected": "Data flows correctly between all integrated systems",
#                 "priority": "Medium",
#                 "regression": "Yes"
#             },
#             {
#                 "scenario": "Business rule compliance",
#                 "description": "Verify compliance with banking business rules and regulations",
#                 "precondition": "All compliance rules configured in system",
#                 "steps": "1. Create payment that tests business rule boundaries\n2. Submit through maker-checker process\n3. Verify compliance validation",
#                 "expected": "System enforces all applicable business rules and compliance requirements",
#                 "priority": "High",
#                 "regression": "Yes"
#             }
#         ]
        
#         # Generate requested number of test cases
#         for i in range(num_cases):
#             base_index = i % len(base_scenarios)
#             base_scenario = base_scenarios[base_index]
            
#             tc_id = f"TC{i+1:03d}"
#             ac_id = f"AC{(i//3)+1:03d}"
            
#             # Customize scenario if using same base multiple times
#             scenario_suffix = f" - Variant {(i//len(base_scenarios))+1}" if i >= len(base_scenarios) else ""
            
#             test_case = {
#                 "User Story ID": story_id,
#                 "Acceptance Criteria ID": ac_id,
#                 "Scenario": base_scenario["scenario"] + scenario_suffix,
#                 "Test Case ID": tc_id,
#                 "Test Case Description": base_scenario["description"],
#                 "Precondition": base_scenario["precondition"],
#                 "Steps": base_scenario["steps"],
#                 "Expected Result": base_scenario["expected"],
#                 "Part of Regression": base_scenario["regression"],
#                 "Priority": base_scenario["priority"]
#             }
            
#             # Add PACS.008 enhancement metadata
#             test_case["PACS008_Enhanced"] = "Yes" if detected_fields else "Fallback"
#             test_case["Enhancement_Type"] = "Domain_Specific_Fallback"
#             test_case["Generation_Method"] = "Fallback_Domain_Examples"
            
#             fallback_cases.append(test_case)
        
#         return fallback_cases
    
#     def _create_workflow_summary(self, workflow_results: Dict) -> Dict[str, Any]:
#         """Create comprehensive workflow summary with safe error handling"""
        
#         # Safely extract data with defaults
#         analysis = workflow_results.get("step1_analysis") or {}
#         user_stories = workflow_results.get("step2_user_stories") or []
#         pacs008_fields = workflow_results.get("step3_pacs008_fields") or {}
#         maker_checker = workflow_results.get("step4_maker_checker") or {}
#         test_cases = workflow_results.get("step5_test_cases") or []
        
#         # Ensure user_stories is a list
#         if not isinstance(user_stories, list):
#             user_stories = []
        
#         # Ensure test_cases is a list
#         if not isinstance(test_cases, list):
#             test_cases = []
        
#         # Safe calculations
#         try:
#             total_stories = len(user_stories) if user_stories else 0
#             total_test_cases = len(test_cases) if test_cases else 0
            
#             pacs008_relevant_stories = 0
#             if user_stories:
#                 pacs008_relevant_stories = len([s for s in user_stories if isinstance(s, dict) and s.get("pacs008_relevance") != "low"])
            
#             story_types = []
#             if user_stories:
#                 story_types = list(set(s.get("story_type", "unknown") for s in user_stories if isinstance(s, dict)))
            
#             pacs008_enhanced_tests = 0
#             regression_tests = 0
#             if test_cases:
#                 pacs008_enhanced_tests = len([tc for tc in test_cases if isinstance(tc, dict) and tc.get("PACS008_Enhanced") == "Yes"])
#                 regression_tests = len([tc for tc in test_cases if isinstance(tc, dict) and tc.get("Part of Regression") == "Yes"])
            
#             coverage_per_story = (total_test_cases / total_stories) if total_stories > 0 else 0
            
#         except Exception as e:
#             logger.error(f"Error calculating workflow summary metrics: {str(e)}")
#             # Fallback values
#             total_stories = 0
#             total_test_cases = 0
#             pacs008_relevant_stories = 0
#             story_types = []
#             pacs008_enhanced_tests = 0
#             regression_tests = 0
#             coverage_per_story = 0
        
#         return {
#             "workflow_status": "completed",
#             "automation_intelligence": {
#                 "content_analysis": {
#                     "pacs008_relevant": analysis.get("is_pacs008_relevant", False),
#                     "content_type": analysis.get("content_type", "unknown"),
#                     "confidence_score": analysis.get("confidence_score", 0),
#                     "technical_level": analysis.get("technical_level", "medium")
#                 },
#                 "user_story_extraction": {
#                     "total_stories": total_stories,
#                     "pacs008_relevant_stories": pacs008_relevant_stories,
#                     "story_types": story_types
#                 },
#                 "field_detection": {
#                     "total_unique_fields": pacs008_fields.get("total_unique_fields", 0) if isinstance(pacs008_fields, dict) else 0,
#                     "stories_with_fields": pacs008_fields.get("detection_summary", {}).get("stories_with_pacs008", 0) if isinstance(pacs008_fields, dict) else 0,
#                     "field_coverage": "comprehensive" if pacs008_fields.get("total_unique_fields", 0) > 5 else "basic"
#                 },
#                 "maker_checker": {
#                     "validation_items": len(maker_checker.get("validation_items", [])) if isinstance(maker_checker, dict) else 0,
#                     "mandatory_items": maker_checker.get("summary", {}).get("mandatory_items", 0) if isinstance(maker_checker, dict) else 0,
#                     "validation_ready": maker_checker.get("validation_ready", False) if isinstance(maker_checker, dict) else False
#                 },
#                 "test_generation": {
#                     "total_test_cases": total_test_cases,
#                     "pacs008_enhanced": pacs008_enhanced_tests,
#                     "coverage_per_story": coverage_per_story,
#                     "regression_tests": regression_tests
#                 }
#             },
#             "business_value": {
#                 "automation_achieved": True,
#                 "domain_expertise_applied": True,
#                 "maker_checker_integrated": True,
#                 "pacs008_intelligence_used": analysis.get("is_pacs008_relevant", False),
#                 "test_coverage": "comprehensive" if total_test_cases > 10 else "basic"
#             },
#             "next_steps": [
#                 "Review maker-checker validation items",
#                 "Execute generated test cases in test environment", 
#                 "Validate test results against business requirements",
#                 "Update test automation framework with new scenarios"
#             ],
#             "quality_indicators": {
#                 "field_detection_accuracy": "high" if pacs008_fields.get("total_unique_fields", 0) > 3 else "medium",
#                 "test_case_relevance": "high" if pacs008_enhanced_tests > 5 else "medium",
#                 "business_alignment": "high" if maker_checker.get("validation_ready", False) else "medium"
#             }
#         }

# # Integration class for Streamlit
# class StreamlitPACS008Integration:
#     """Integration layer for Streamlit UI"""
    
#     def __init__(self, api_key: str):
#         self.generator = DynamicPACS008TestGenerator(api_key)
    
#     def process_uploaded_files(self, uploaded_files, custom_instructions: str, 
#                              num_test_cases_per_story: int) -> Dict[str, Any]:
#         """Process uploaded files and return complete workflow results"""
        
#         # Combine content from all uploaded files
#         all_content = []
        
#         for uploaded_file in uploaded_files:
#             # In a real implementation, you'd use DocumentProcessor here
#             # For now, assuming text content
#             try:
#                 content = uploaded_file.getvalue().decode('utf-8')
#                 all_content.append(content)
#             except:
#                 # Handle binary files or use DocumentProcessor
#                 all_content.append(f"Content from {uploaded_file.name}")
        
#         combined_content = "\n\n--- Next Document ---\n\n".join(all_content)
        
#         # Add custom instructions context
#         if custom_instructions:
#             combined_content += f"\n\nCustom Instructions: {custom_instructions}"
        
#         # Run complete workflow
#         workflow_results = self.generator.process_complete_workflow(
#             combined_content, num_test_cases_per_story
#         )
        
#         return workflow_results
    
#     def get_maker_checker_items(self, workflow_results: Dict) -> List[Dict[str, Any]]:
#         """Extract maker-checker items for UI display"""
#         return workflow_results.get("step4_maker_checker", {}).get("validation_items", [])
    
#     def get_test_cases_for_export(self, workflow_results: Dict) -> List[Dict[str, Any]]:
#         """Get test cases formatted for export"""
#         return workflow_results.get("step5_test_cases", [])
    
#     def get_pacs008_analysis_summary(self, workflow_results: Dict) -> Dict[str, Any]:
#         """Get PACS.008 analysis summary for UI display"""
#         return {
#             "content_analysis": workflow_results.get("step1_analysis", {}),
#             "user_stories": workflow_results.get("step2_user_stories", []),
#             "field_detection": workflow_results.get("step3_pacs008_fields", {}),
#             "workflow_summary": workflow_results.get("workflow_summary", {})
#         }




# src/ai_engine/dynamic_pacs008_test_generator.py
"""
Complete Dynamic PACS.008 Test Generation System
Automates the entire workflow: Document Analysis → Field Detection → Maker-Checker → Test Generation
"""

import json
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from openai import OpenAI
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class DynamicPACS008TestGenerator:
    """Complete automation system for PACS.008 test case generation"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4.1-mini-2025-04-14"
        
        # PACS.008 domain knowledge
        self.pacs008_knowledge = self._load_pacs008_knowledge()
        
        # Initialize documentation generator
        from utils.processing_documentation_generator import ProcessingDocumentationGenerator
        self.doc_generator = ProcessingDocumentationGenerator()
        
        logger.info("Dynamic PACS.008 Test Generation System initialized")
    
    def _load_pacs008_knowledge(self) -> Dict[str, Any]:
        """Load comprehensive PACS.008 domain knowledge"""
        return {
            "mandatory_fields": {
                "debtor_agent": {"name": "Debtor Agent BIC", "examples": ["DEUTDEFF", "CHASUS33"]},
                "creditor_agent": {"name": "Creditor Agent BIC", "examples": ["BNPAFRPP", "HSBCGB2L"]},
                "debtor_name": {"name": "Debtor Name", "examples": ["ABC Corporation", "John Smith"]},
                "creditor_name": {"name": "Creditor Name", "examples": ["XYZ Supplier", "Jane Doe"]},
                "debtor_account": {"name": "Debtor Account", "examples": ["DE89370400440532013000"]},
                "creditor_account": {"name": "Creditor Account", "examples": ["FR1420041010050500013M02606"]},
                "amount": {"name": "Payment Amount", "examples": ["5000.00", "1000.50"]},
                "currency": {"name": "Currency", "examples": ["EUR", "USD", "GBP"]},
                "instruction_id": {"name": "Instruction ID", "examples": ["INSTR20240801001"]}
            },
            "test_scenarios": {
                "maker_checker": [
                    "Payment creation with maker/checker workflow",
                    "Field validation and approval process",
                    "Queue management and processing",
                    "System integration and data flow"
                ],
                "processing_methods": ["SERIAL", "PARALLEL", "COVER"],
                "system_components": ["TPH system", "RLC queues", "Upstream/Downstream systems"],
                "user_roles": ["Ops User maker", "Ops User checker", "Admin"]
            },
            "business_rules": [
                "All banks must have established direct account relationships",
                "Nostro/Vostro agent configurations must be valid",
                "Cut-off times must be respected",
                "Exchange rates must be current",
                "RLC setup conditions must be met"
            ]
        }
    
    def process_complete_workflow(self, content: str, num_test_cases_per_story: int = 8, 
                                files_info: List[Dict] = None) -> Dict[str, Any]:
        """
        Complete workflow: Analysis → Detection → Maker-Checker → Test Generation + Documentation
        """
        logger.info("Starting complete PACS.008 workflow automation...")
        
        # Initialize files info if not provided
        if files_info is None:
            files_info = [{"name": "content", "size_mb": len(content)/(1024*1024), "type": "text", "status": "processed"}]
        
        # Document input analysis
        self.doc_generator.add_input_analysis(files_info, content)
        self.doc_generator.add_extracted_content(content, [f["name"] for f in files_info])
        
        workflow_results = {
            "step1_analysis": {},
            "step2_user_stories": [],
            "step3_pacs008_fields": {},
            "step4_maker_checker": {},
            "step5_test_cases": [],
            "workflow_summary": {},
            "processing_errors": [],
            "documentation": {}
        }
        
        try:
            # Step 1: Intelligent Content Analysis
            logger.info("Step 1: Analyzing content for PACS.008 relevance...")
            try:
                analysis_result = self._analyze_content_intelligence(content)
                workflow_results["step1_analysis"] = analysis_result
            except Exception as e:
                logger.error(f"Step 1 error: {str(e)}")
                workflow_results["processing_errors"].append(f"Content analysis error: {str(e)}")
                workflow_results["step1_analysis"] = {"is_pacs008_relevant": False, "error": str(e)}
            
            # Step 2: Dynamic User Story Extraction
            logger.info("Step 2: Extracting user stories with LLM intelligence...")
            try:
                analysis_result = workflow_results["step1_analysis"]
                user_stories = self._extract_user_stories_intelligently(content, analysis_result)
                # Ensure user_stories is always a list
                if not isinstance(user_stories, list):
                    user_stories = []
                workflow_results["step2_user_stories"] = user_stories
                
                # Document user stories extraction
                extraction_method = "LLM intelligent extraction"
                extraction_reasoning = "LLM analyzed content for formal user story patterns and converted requirements to stories"
                self.doc_generator.add_user_stories_extraction(user_stories, extraction_method, extraction_reasoning)
                
            except Exception as e:
                logger.error(f"Step 2 error: {str(e)}")
                workflow_results["processing_errors"].append(f"User story extraction error: {str(e)}")
                workflow_results["step2_user_stories"] = []
            
            # Step 3: PACS.008 Field Detection for each story
            logger.info("Step 3: Detecting PACS.008 fields for each user story...")
            try:
                user_stories = workflow_results["step2_user_stories"]
                if user_stories:
                    pacs008_fields = self._detect_pacs008_fields_per_story(user_stories, content)
                else:
                    pacs008_fields = {"story_field_mapping": {}, "all_detected_fields": [], "total_unique_fields": 0}
                workflow_results["step3_pacs008_fields"] = pacs008_fields
                
                # Document PACS.008 analysis and field detection
                analysis_result = workflow_results["step1_analysis"]
                self.doc_generator.add_pacs008_analysis(analysis_result, pacs008_fields)
                self.doc_generator.add_field_detection_details(pacs008_fields)
                
            except Exception as e:
                logger.error(f"Step 3 error: {str(e)}")
                workflow_results["processing_errors"].append(f"Field detection error: {str(e)}")
                workflow_results["step3_pacs008_fields"] = {"story_field_mapping": {}, "all_detected_fields": [], "total_unique_fields": 0}
            
            # Step 4: Maker-Checker Process
            logger.info("Step 4: Preparing maker-checker validation...")
            try:
                pacs008_fields = workflow_results["step3_pacs008_fields"]
                maker_checker_result = self._prepare_maker_checker_process(pacs008_fields)
                workflow_results["step4_maker_checker"] = maker_checker_result
                
                # Document maker-checker logic
                self.doc_generator.add_maker_checker_logic(maker_checker_result)
                
            except Exception as e:
                logger.error(f"Step 4 error: {str(e)}")
                workflow_results["processing_errors"].append(f"Maker-checker preparation error: {str(e)}")
                workflow_results["step4_maker_checker"] = {"validation_items": [], "summary": {}, "validation_ready": False}
            
            # Step 5: Dynamic Test Case Generation
            logger.info("Step 5: Generating comprehensive test cases...")
            try:
                user_stories = workflow_results["step2_user_stories"]
                pacs008_fields = workflow_results["step3_pacs008_fields"]
                
                if user_stories:
                    test_cases = self._generate_dynamic_test_cases(
                        user_stories, pacs008_fields, num_test_cases_per_story
                    )
                    # Ensure test_cases is always a list
                    if not isinstance(test_cases, list):
                        test_cases = []
                else:
                    # Generate fallback test cases if no user stories
                    test_cases = self._generate_fallback_test_cases_from_content(content, num_test_cases_per_story)
                
                workflow_results["step5_test_cases"] = test_cases
                
                # Document test generation logic
                generation_params = {
                    "num_test_cases_per_story": num_test_cases_per_story,
                    "total_user_stories": len(user_stories),
                    "pacs008_fields_available": len(pacs008_fields.get("all_detected_fields", [])) > 0,
                    "generation_method": "PACS008_enhanced" if pacs008_fields.get("total_unique_fields", 0) > 0 else "standard"
                }
                self.doc_generator.add_test_generation_logic(test_cases, generation_params)
                
            except Exception as e:
                logger.error(f"Step 5 error: {str(e)}")
                workflow_results["processing_errors"].append(f"Test case generation error: {str(e)}")
                workflow_results["step5_test_cases"] = []
            
            # Workflow Summary
            try:
                workflow_results["workflow_summary"] = self._create_workflow_summary(workflow_results)
                
                # Document processing summary
                self.doc_generator.add_processing_summary(workflow_results)
                
            except Exception as e:
                logger.error(f"Summary creation error: {str(e)}")
                workflow_results["processing_errors"].append(f"Summary creation error: {str(e)}")
                workflow_results["workflow_summary"] = {"workflow_status": "completed_with_errors"}
            
            # Generate final documentation
            try:
                workflow_results["documentation"] = {
                    "report_text": self.doc_generator.generate_documentation_report(),
                    "json_data": self.doc_generator.get_json_documentation(),
                    "generation_timestamp": datetime.now().isoformat()
                }
                logger.info("Complete processing documentation generated")
            except Exception as e:
                logger.error(f"Documentation generation error: {str(e)}")
                workflow_results["documentation"] = {"error": str(e)}
            
            # Log completion status
            if workflow_results["processing_errors"]:
                logger.warning(f"PACS.008 workflow completed with {len(workflow_results['processing_errors'])} errors")
            else:
                logger.info("Complete PACS.008 workflow automation completed successfully!")
            
            return workflow_results
            
        except Exception as e:
            logger.error(f"Critical workflow automation error: {str(e)}")
            workflow_results["critical_error"] = str(e)
            workflow_results["workflow_summary"] = {"workflow_status": "failed", "error": str(e)}
            return workflow_results
    
    def _generate_fallback_test_cases_from_content(self, content: str, num_cases: int) -> List[Dict[str, Any]]:
        """Generate fallback test cases when user story extraction fails"""
        logger.info(f"Generating {num_cases} fallback test cases from raw content")
        
        fallback_story = {
            "id": "REQ001",
            "title": "Banking Requirements",
            "story": "As a banking user, I want to process payments according to banking standards so that transactions are handled correctly",
            "source_content": content[:200],
            "pacs008_relevance": "medium",
            "story_type": "payment_processing",
            "acceptance_criteria": ["Valid payment processing", "Error handling", "Compliance validation"],
            "estimated_test_scenarios": num_cases
        }
        
        return self._generate_fallback_test_cases(fallback_story, [], num_cases)
    
    def _analyze_content_intelligence(self, content: str) -> Dict[str, Any]:
        """Intelligent analysis of content for PACS.008 relevance and context"""
        
        prompt = f"""
You are a PACS.008 domain expert. Analyze this content for banking payment relevance.

CONTENT TO ANALYZE:
{content[:2000]}

ANALYSIS REQUIRED:
1. Is this content related to PACS.008 or banking payments?
2. What type of banking content is this? (requirements, user stories, specifications, etc.)
3. What banking concepts are mentioned?
4. What level of technical detail is present?
5. Are there specific banking systems mentioned?

RESPOND WITH JSON:
{{
  "is_pacs008_relevant": true/false,
  "content_type": "requirements|user_stories|specifications|procedures|other",
  "banking_concepts": ["concept1", "concept2"],
  "technical_level": "high|medium|basic",
  "mentioned_systems": ["system1", "system2"],
  "confidence_score": 0-100,
  "recommended_approach": "enhanced|standard",
  "key_indicators": ["indicator1", "indicator2"]
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a PACS.008 banking expert. Respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            result = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"is_pacs008_relevant": False, "error": "Could not parse analysis"}
                
        except Exception as e:
            logger.error(f"Content analysis error: {str(e)}")
            return {"is_pacs008_relevant": False, "error": str(e)}
    
    def _extract_user_stories_intelligently(self, content: str, analysis: Dict) -> List[Dict[str, Any]]:
        """Extract user stories using LLM intelligence based on content analysis"""
        
        is_relevant = analysis.get("is_pacs008_relevant", False)
        content_type = analysis.get("content_type", "other")
        
        prompt = f"""
You are a BFSI business analyst expert. Extract user stories from this content.

CONTENT:
{content}

CONTENT ANALYSIS:
- PACS.008 Relevant: {is_relevant}
- Content Type: {content_type}
- Banking Concepts: {analysis.get('banking_concepts', [])}

EXTRACTION RULES:
1. Look for actual user stories (As a... I want... So that...)
2. Look for feature requirements that can be converted to user stories
3. Look for business processes that represent user needs
4. If content describes testing scenarios, extract the underlying user stories
5. Group related functionality into coherent user stories
6. Focus on banking/payment domain stories

RESPOND WITH JSON:
{{
  "user_stories": [
    {{
      "id": "US001",
      "title": "Brief title",
      "story": "As a [user] I want [functionality] so that [benefit]",
      "source_content": "Original text that led to this story",
      "pacs008_relevance": "high|medium|low",
      "story_type": "payment_processing|user_interface|validation|reporting|other",
      "acceptance_criteria": ["AC1 description", "AC2 description"],
      "estimated_test_scenarios": 5-15
    }}
  ],
  "extraction_summary": {{
    "total_stories": 3,
    "pacs008_stories": 2,
    "story_types": ["payment_processing", "user_interface"]
  }}
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a BFSI business analyst. Extract meaningful user stories. Respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )
            
            result = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            
            if json_match:
                extracted = json.loads(json_match.group())
                return extracted.get("user_stories", [])
            else:
                logger.warning("Could not parse user story extraction")
                return self._fallback_user_story_extraction(content)
                
        except Exception as e:
            logger.error(f"User story extraction error: {str(e)}")
            return self._fallback_user_story_extraction(content)
    
    def _fallback_user_story_extraction(self, content: str) -> List[Dict[str, Any]]:
        """Fallback method for user story extraction"""
        logger.info("Using fallback user story extraction")
        
        # Simple pattern matching for common user story formats
        patterns = [
            r'As\s+(?:a|an)\s+(.+?)\s+I\s+want\s+(.+?)\s+(?:so\s+that|in\s+order\s+to)\s+(.+?)(?=\.|$)',
            r'User\s+Story\s*:?\s*(.+?)(?=User\s+Story|$)',
            r'Requirement\s*:?\s*(.+?)(?=Requirement|$)'
        ]
        
        stories = []
        story_id = 1
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                if isinstance(match, tuple):
                    story_text = f"As a {match[0]}, I want {match[1]} so that {match[2]}"
                else:
                    story_text = match
                
                if len(story_text.strip()) > 20:
                    stories.append({
                        "id": f"US{story_id:03d}",
                        "title": f"User Story {story_id}",
                        "story": story_text.strip(),
                        "source_content": story_text.strip()[:200],
                        "pacs008_relevance": "medium",
                        "story_type": "payment_processing",
                        "acceptance_criteria": ["Basic functionality", "Error handling"],
                        "estimated_test_scenarios": 8
                    })
                    story_id += 1
        
        # If no formal stories found, create one from content
        if not stories:
            stories.append({
                "id": "US001",
                "title": "Payment Processing Requirement",
                "story": "As a banking user, I want to process payments according to PACS.008 standards so that transactions are handled correctly",
                "source_content": content[:200],
                "pacs008_relevance": "high",
                "story_type": "payment_processing",
                "acceptance_criteria": ["Valid payment processing", "Error handling", "Compliance validation"],
                "estimated_test_scenarios": 8
            })
        
        return stories[:5]  # Limit to 5 user stories
    
    def _detect_pacs008_fields_per_story(self, user_stories: List[Dict], full_content: str) -> Dict[str, Any]:
        """Detect PACS.008 fields for each user story"""
        
        story_field_mapping = {}
        all_detected_fields = []
        
        for story in user_stories:
            story_id = story["id"]
            story_content = story.get("source_content", "") + "\n" + story.get("story", "")
            
            # Detect fields for this specific story
            detected_fields = self._detect_fields_for_single_story(story_content, full_content)
            
            story_field_mapping[story_id] = {
                "story_title": story["title"],
                "detected_fields": detected_fields,
                "field_count": len(detected_fields),
                "mandatory_fields": len([f for f in detected_fields if f.get("is_mandatory", False)]),
                "pacs008_relevance": story.get("pacs008_relevance", "medium")
            }
            
            all_detected_fields.extend(detected_fields)
        
        return {
            "story_field_mapping": story_field_mapping,
            "all_detected_fields": all_detected_fields,
            "total_unique_fields": len(set(f["field_name"] for f in all_detected_fields)),
            "detection_summary": {
                "total_stories_processed": len(user_stories),
                "stories_with_pacs008": len([s for s in story_field_mapping.values() if s["field_count"] > 0]),
                "most_relevant_story": max(story_field_mapping.keys(), 
                                         key=lambda x: story_field_mapping[x]["field_count"]) if story_field_mapping else None
            }
        }
    
    def _detect_fields_for_single_story(self, story_content: str, context_content: str) -> List[Dict[str, Any]]:
        """Detect PACS.008 fields for a single user story"""
        
        # Create field reference
        field_ref = []
        for field_key, field_info in self.pacs008_knowledge["mandatory_fields"].items():
            examples = ", ".join(field_info["examples"])
            field_ref.append(f"- {field_key}: {field_info['name']} [Examples: {examples}]")
        
        prompt = f"""
You are a PACS.008 field detection expert. Analyze this user story for PACS.008 fields.

USER STORY:
{story_content}

CONTEXT (for reference):
{context_content[:1000]}

PACS.008 FIELDS TO DETECT:
{chr(10).join(field_ref)}

DETECTION RULES:
1. Look for explicit field mentions
2. Infer fields from business context (e.g., "bank" = agent, "account" = debtor/creditor account)
3. Consider maker-checker workflow implications
4. Look for system integration points

RESPOND WITH JSON:
{{
  "detected_fields": [
    {{
      "field_key": "debtor_agent",
      "field_name": "Debtor Agent BIC",
      "extracted_value": "DEUTDEFF or mentioned but not specified",
      "confidence": "high|medium|low",
      "detection_reason": "Why you detected this field",
      "is_mandatory": true,
      "business_context": "How this field relates to the user story"
    }}
  ]
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a PACS.008 expert. Detect fields accurately. Respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            result = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            
            if json_match:
                parsed = json.loads(json_match.group())
                return parsed.get("detected_fields", [])
            else:
                return []
                
        except Exception as e:
            logger.error(f"Field detection error: {str(e)}")
            return []
    
    def _prepare_maker_checker_process(self, pacs008_fields: Dict) -> Dict[str, Any]:
        """Prepare maker-checker validation process"""
        
        maker_checker_items = []
        
        for story_id, story_data in pacs008_fields.get("story_field_mapping", {}).items():
            for field in story_data.get("detected_fields", []):
                
                # Determine if field needs validation
                needs_validation = (
                    field.get("is_mandatory", False) or
                    field.get("confidence", "low") == "low" or
                    not field.get("extracted_value") or
                    field.get("extracted_value", "").lower() in ["mentioned but not specified", "not specified"]
                )
                
                if needs_validation:
                    maker_checker_items.append({
                        "story_id": story_id,
                        "field_key": field.get("field_key"),
                        "field_name": field.get("field_name"),
                        "extracted_value": field.get("extracted_value"),
                        "confidence": field.get("confidence"),
                        "is_mandatory": field.get("is_mandatory", False),
                        "validation_reason": self._get_validation_reason(field),
                        "maker_action": "Verify field accuracy and business relevance",
                        "checker_action": "Validate against PACS.008 standards and approve",
                        "business_impact": self._assess_business_impact(field)
                    })
        
        return {
            "validation_items": maker_checker_items,
            "summary": {
                "total_items": len(maker_checker_items),
                "mandatory_items": len([item for item in maker_checker_items if item["is_mandatory"]]),
                "high_priority_items": len([item for item in maker_checker_items if item["business_impact"] == "high"]),
                "stories_affected": len(set(item["story_id"] for item in maker_checker_items))
            },
            "validation_ready": len(maker_checker_items) > 0
        }
    
    def _get_validation_reason(self, field: Dict) -> str:
        """Get reason why field needs validation"""
        if field.get("is_mandatory") and not field.get("extracted_value"):
            return "Mandatory field missing - critical for PACS.008 processing"
        elif field.get("confidence") == "low":
            return "Low confidence detection - needs verification"
        elif not field.get("extracted_value"):
            return "Field value not specified - needs maker input"
        else:
            return "Field detected but requires validation"
    
    def _assess_business_impact(self, field: Dict) -> str:
        """Assess business impact of field validation"""
        if field.get("is_mandatory"):
            return "high"
        elif field.get("field_key") in ["debtor_agent", "creditor_agent", "amount"]:
            return "high"
        elif field.get("confidence") == "high":
            return "medium"
        else:
            return "low"
    
    def _generate_dynamic_test_cases(self, user_stories: List[Dict], pacs008_fields: Dict, 
                                   num_cases_per_story: int) -> List[Dict[str, Any]]:
        """Generate comprehensive test cases for each user story with PACS.008 intelligence"""
        
        all_test_cases = []
        
        for story in user_stories:
            story_id = story["id"]
            
            # Get PACS.008 context for this story
            story_pacs008_data = pacs008_fields.get("story_field_mapping", {}).get(story_id, {})
            detected_fields = story_pacs008_data.get("detected_fields", [])
            
            # Generate test cases for this story
            story_test_cases = self._generate_test_cases_for_story_with_context(
                story, detected_fields, num_cases_per_story
            )
            
            all_test_cases.extend(story_test_cases)
        
        return all_test_cases
    
    def _generate_test_cases_for_story_with_context(self, story: Dict, detected_fields: List[Dict], 
                                                  num_cases: int) -> List[Dict[str, Any]]:
        """Generate test cases for a single story with PACS.008 context"""
        
        story_id = story["id"]
        story_content = story["story"]
        story_type = story.get("story_type", "payment_processing")
        
        # Create PACS.008 context
        pacs008_context = self._create_pacs008_test_context(detected_fields)
        
        # Create domain-specific examples based on your client's feedback
        domain_examples = self._get_domain_specific_examples(story_type)
        
        prompt = f"""
You are an expert BFSI test engineer specializing in PACS.008 payment systems. Generate comprehensive test cases.

USER STORY:
{story_content}

STORY TYPE: {story_type}

DETECTED PACS.008 FIELDS:
{pacs008_context}

DOMAIN EXAMPLES (for reference):
{domain_examples}

GENERATE EXACTLY {num_cases} TEST CASES for this user story.

REQUIREMENTS:
1. All test cases must have SAME User Story ID: {story_id}
2. Include maker-checker workflow scenarios
3. Use realistic banking data from detected fields
4. Include system integration scenarios (TPH system, RLC queues, etc.)
5. Cover validation, processing, and approval workflows
6. Include both positive and negative scenarios
7. Focus on business rules and compliance

RESPOND WITH ONLY JSON ARRAY:
[
  {{
    "User Story ID": "{story_id}",
    "Acceptance Criteria ID": "AC001",
    "Scenario": "Payment creation with maker/checker",
    "Test Case ID": "TC001",
    "Test Case Description": "Verify successful creation of PACS.008 message with maker-checker workflow",
    "Precondition": "All nostro/vostro agents configured, cut-off times set, exchange rates available",
    "Steps": "1. Login as Ops User maker\\n2. Navigate to payment creation\\n3. Enter all required PACS.008 fields\\n4. Submit for approval\\n5. Login as checker\\n6. Review and approve",
    "Expected Result": "Payment successfully created and approved, available in processing queue",
    "Part of Regression": "Yes",
    "Priority": "High"
  }}
]
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"You are a PACS.008 test expert. Generate exactly {num_cases} comprehensive test cases. Respond with valid JSON array only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=4000
            )
            
            result = response.choices[0].message.content.strip()
            json_match = re.search(r'\[.*\]', result, re.DOTALL)
            
            if json_match:
                test_cases = json.loads(json_match.group())
                
                # Validate and enhance test cases
                enhanced_cases = self._enhance_test_cases_with_pacs008_intelligence(
                    test_cases, story, detected_fields
                )
                
                return enhanced_cases
            else:
                logger.warning(f"Could not parse test cases for {story_id}")
                return self._generate_fallback_test_cases(story, detected_fields, num_cases)
                
        except Exception as e:
            logger.error(f"Test generation error for {story_id}: {str(e)}")
            return self._generate_fallback_test_cases(story, detected_fields, num_cases)
    
    def _create_pacs008_test_context(self, detected_fields: List[Dict]) -> str:
        """Create PACS.008 context for test generation"""
        if not detected_fields:
            return "No specific PACS.008 fields detected - use standard payment processing context"
        
        context_lines = []
        for field in detected_fields:
            context_lines.append(
                f"- {field.get('field_name')}: {field.get('extracted_value', 'To be tested')} "
                f"(Confidence: {field.get('confidence', 'medium')})"
            )
        
        return "\n".join(context_lines)
    
    def _get_domain_specific_examples(self, story_type: str) -> str:
        """Get domain-specific examples based on client feedback"""
        
        examples = {
            "payment_processing": """
Example Test Case Format:
- Scenario: Payment creation with maker/checker
- Description: Verify whether all fields are available for PACS.008 in the TPH system
- Steps: 1. Login as Ops User maker, 2. View all fields like currency, amount, debit account number etc.
- Expected: All relevant fields available (debtor name/address, debtor account, amount, currency, creditor, creditor agent, etc.)
            """,
            "user_interface": """
Example Test Case Format:
- Scenario: Field validation and display
- Description: Verify maker user able to input all data for PACS.008 message creation
- Prerequisites: All Nostro/Vostro agents, cut-off times, exchange rates, upstream/downstream systems connected
- Expected: TPH system allows payment creation, defaults bank/agent/customer account per configuration
            """,
            "validation": """
Example Test Case Format:
- Scenario: Checker approval workflow
- Description: Verify checker user able to see all data inputted by maker
- Steps: Review maker inputs, validate against business rules, approve/reject
- Expected: Checker can review and approve payment, transaction moves to processing queue
            """
        }
        
        return examples.get(story_type, examples["payment_processing"])
    
    def _enhance_test_cases_with_pacs008_intelligence(self, test_cases: List[Dict], 
                                                    story: Dict, detected_fields: List[Dict]) -> List[Dict[str, Any]]:
        """Enhance test cases with PACS.008 intelligence and client domain knowledge"""
        
        enhanced_cases = []
        
        for i, test_case in enumerate(test_cases):
            # Ensure proper structure
            enhanced_case = {
                "User Story ID": story["id"],
                "Acceptance Criteria ID": test_case.get("Acceptance Criteria ID", f"AC{(i//3)+1:03d}"),
                "Scenario": test_case.get("Scenario", f"Test Scenario {i+1}"),
                "Test Case ID": test_case.get("Test Case ID", f"TC{i+1:03d}"),
                "Test Case Description": test_case.get("Test Case Description", ""),
                "Precondition": test_case.get("Precondition", "System available and configured"),
                "Steps": test_case.get("Steps", "").replace("\\n", "\n"),
                "Expected Result": test_case.get("Expected Result", ""),
                "Part of Regression": test_case.get("Part of Regression", "Yes"),
                "Priority": test_case.get("Priority", "Medium")
            }
            
            # Add PACS.008 enhancement metadata
            enhanced_case["PACS008_Enhanced"] = "Yes" if detected_fields else "No"
            enhanced_case["Enhancement_Type"] = "PACS008_Domain_Intelligent"
            enhanced_case["Detected_Fields_Count"] = len(detected_fields)
            
            # Enhance with realistic banking data
            enhanced_case = self._inject_realistic_banking_data(enhanced_case, detected_fields)
            
            enhanced_cases.append(enhanced_case)
        
        return enhanced_cases
    
    def _inject_realistic_banking_data(self, test_case: Dict, detected_fields: List[Dict]) -> Dict[str, Any]:
        """Inject realistic banking data based on detected fields"""
        
        # Banking data pool
        banking_data = {
            "bics": ["DEUTDEFF", "BNPAFRPP", "HSBCGB2L", "CHASUS33", "CITIUS33"],
            "ibans": ["DE89370400440532013000", "FR1420041010050500013M02606", "GB33BUKB20201555555555"],
            "amounts": ["1000.00", "5000.50", "25000.00", "100.00"],
            "currencies": ["EUR", "USD", "GBP", "CHF"],
            "customer_names": ["ABC Corporation Ltd", "XYZ Trading Company", "Global Services Inc"]
        }
        
        steps = test_case.get("Steps", "")
        expected = test_case.get("Expected Result", "")
        
        # Replace generic placeholders with specific banking data
        for field in detected_fields:
            field_key = field.get("field_key", "")
            extracted_value = field.get("extracted_value", "")
            
            # Use extracted value if available, otherwise use realistic sample
            if "agent" in field_key or "bic" in field_key:
                bic_value = extracted_value if extracted_value and extracted_value != "mentioned but not specified" else banking_data["bics"][0]
                steps = steps.replace("bank BIC", bic_value).replace("agent BIC", bic_value)
                
            elif "account" in field_key or "iban" in field_key:
                iban_value = extracted_value if extracted_value and extracted_value != "mentioned but not specified" else banking_data["ibans"][0]
                steps = steps.replace("account number", iban_value).replace("IBAN", iban_value)
                
            elif "amount" in field_key:
                amount_value = extracted_value if extracted_value and extracted_value != "mentioned but not specified" else banking_data["amounts"][0]
                steps = steps.replace("amount", f"amount: {amount_value}").replace("payment amount", amount_value)
                
            elif "currency" in field_key:
                currency_value = extracted_value if extracted_value and extracted_value != "mentioned but not specified" else banking_data["currencies"][0]
                steps = steps.replace("currency", currency_value)
        
        # Update the enhanced test case
        test_case["Steps"] = steps
        test_case["Expected Result"] = expected
        
        return test_case
    
    def _generate_fallback_test_cases(self, story: Dict, detected_fields: List[Dict], 
                                    num_cases: int) -> List[Dict[str, Any]]:
        """Generate fallback test cases based on client's domain examples"""
        
        story_id = story["id"]
        fallback_cases = []
        
        # Base scenarios from client feedback
        base_scenarios = [
            {
                "scenario": "Payment creation with maker/checker",
                "description": "Verify whether all fields are available for PACS.008 in the TPH system",
                "precondition": "Menu, Navigation, fields, label should be available",
                "steps": "1. Login as Ops User maker\n2. View the all the fields like currency amount, debit account number etc",
                "expected": "All relevant fields available in TPH system to create a PACS.008\n1. debtor name and address\n2. debtor account\n3. amount\n4. currency\n5. creditor, creditor agent etc",
                "priority": "High",
                "regression": "Yes"
            },
            {
                "scenario": "Maker data input validation",
                "description": "Verify whether Maker user able to input all data for PACS.008 message creation",
                "precondition": "All Nostro/vostro agent, cut off time, exchange rate, upstream and downstream system are connected",
                "steps": "1. Login as Ops User maker\n2. Enter all required for PACS.008 creation",
                "expected": "TPH system should allow user create payment PACS.008 (yet to approve by checker)\nTPH system able to default bank/agent, customer account, as per setup/configuration\nTPH system able to fetch data upstream/downstream correctly",
                "priority": "High",
                "regression": "Yes"
            },
            {
                "scenario": "Checker approval process",
                "description": "Verify whether Checker user able to see all data in screen which are inputted by maker",
                "precondition": "All Nostro/vostro agent, cut off time, exchange rate, upstream and downstream system are connected",
                "steps": "1. Login as Ops User checker\n2. Navigate to approval queue\n3. Review maker inputs\n4. Approve/reject payment",
                "expected": "TPH system should allow the checker to check/approve the Payment",
                "priority": "High",
                "regression": "Yes"
            },
            {
                "scenario": "Queue management validation",
                "description": "Verify whether transaction available in Soft block queues after op checker approved the transaction",
                "precondition": "All RLC setup configuration available",
                "steps": "1. Login as Ops User maker\n2. Navigate to RLC queue\n3. Check transaction status",
                "expected": "Transaction should be available RLC as RLC Setup condition is met",
                "priority": "Medium",
                "regression": "Yes"
            },
            {
                "scenario": "PACS.008 message processing",
                "description": "Verify successful processing of PACS.008 message via SERIAL method",
                "precondition": "All banks in payment chain have established direct account relationships; valid payment data available",
                "steps": "1. Initiate PACS.008 message from Debtor Agent with amount 5000.00 EUR\n2. Ensure message contains all necessary payment information\n3. Send message to next bank in chain\n4. Each intermediate bank processes and forwards PACS.008 message\n5. Final Creditor Agent receives and processes the message",
                "expected": "Payment is successfully processed through all banks with correct settlement instructions and bookings at each step",
                "priority": "High",
                "regression": "Yes"
            },
            {
                "scenario": "Field validation and error handling",
                "description": "Verify system validation for mandatory PACS.008 fields",
                "precondition": "System is available and user is authenticated",
                "steps": "1. Login as Ops User maker\n2. Attempt to create payment with missing mandatory fields\n3. Submit for validation",
                "expected": "System displays appropriate validation errors for missing mandatory fields",
                "priority": "High",
                "regression": "Yes"
            },
            {
                "scenario": "System integration testing",
                "description": "Verify integration between TPH system and upstream/downstream systems",
                "precondition": "All system integrations configured and available",
                "steps": "1. Create PACS.008 payment in TPH system\n2. Verify data synchronization with upstream system\n3. Check downstream system processing",
                "expected": "Data flows correctly between all integrated systems",
                "priority": "Medium",
                "regression": "Yes"
            },
            {
                "scenario": "Business rule compliance",
                "description": "Verify compliance with banking business rules and regulations",
                "precondition": "All compliance rules configured in system",
                "steps": "1. Create payment that tests business rule boundaries\n2. Submit through maker-checker process\n3. Verify compliance validation",
                "expected": "System enforces all applicable business rules and compliance requirements",
                "priority": "High",
                "regression": "Yes"
            }
        ]
        
        # Generate requested number of test cases
        for i in range(num_cases):
            base_index = i % len(base_scenarios)
            base_scenario = base_scenarios[base_index]
            
            tc_id = f"TC{i+1:03d}"
            ac_id = f"AC{(i//3)+1:03d}"
            
            # Customize scenario if using same base multiple times
            scenario_suffix = f" - Variant {(i//len(base_scenarios))+1}" if i >= len(base_scenarios) else ""
            
            test_case = {
                "User Story ID": story_id,
                "Acceptance Criteria ID": ac_id,
                "Scenario": base_scenario["scenario"] + scenario_suffix,
                "Test Case ID": tc_id,
                "Test Case Description": base_scenario["description"],
                "Precondition": base_scenario["precondition"],
                "Steps": base_scenario["steps"],
                "Expected Result": base_scenario["expected"],
                "Part of Regression": base_scenario["regression"],
                "Priority": base_scenario["priority"]
            }
            
            # Add PACS.008 enhancement metadata
            test_case["PACS008_Enhanced"] = "Yes" if detected_fields else "Fallback"
            test_case["Enhancement_Type"] = "Domain_Specific_Fallback"
            test_case["Generation_Method"] = "Fallback_Domain_Examples"
            
            fallback_cases.append(test_case)
        
        return fallback_cases
    
    def _create_workflow_summary(self, workflow_results: Dict) -> Dict[str, Any]:
        """Create comprehensive workflow summary with safe error handling"""
        
        # Safely extract data with defaults
        analysis = workflow_results.get("step1_analysis") or {}
        user_stories = workflow_results.get("step2_user_stories") or []
        pacs008_fields = workflow_results.get("step3_pacs008_fields") or {}
        maker_checker = workflow_results.get("step4_maker_checker") or {}
        test_cases = workflow_results.get("step5_test_cases") or []
        
        # Ensure user_stories is a list
        if not isinstance(user_stories, list):
            user_stories = []
        
        # Ensure test_cases is a list
        if not isinstance(test_cases, list):
            test_cases = []
        
        # Safe calculations
        try:
            total_stories = len(user_stories) if user_stories else 0
            total_test_cases = len(test_cases) if test_cases else 0
            
            pacs008_relevant_stories = 0
            if user_stories:
                pacs008_relevant_stories = len([s for s in user_stories if isinstance(s, dict) and s.get("pacs008_relevance") != "low"])
            
            story_types = []
            if user_stories:
                story_types = list(set(s.get("story_type", "unknown") for s in user_stories if isinstance(s, dict)))
            
            pacs008_enhanced_tests = 0
            regression_tests = 0
            if test_cases:
                pacs008_enhanced_tests = len([tc for tc in test_cases if isinstance(tc, dict) and tc.get("PACS008_Enhanced") == "Yes"])
                regression_tests = len([tc for tc in test_cases if isinstance(tc, dict) and tc.get("Part of Regression") == "Yes"])
            
            coverage_per_story = (total_test_cases / total_stories) if total_stories > 0 else 0
            
        except Exception as e:
            logger.error(f"Error calculating workflow summary metrics: {str(e)}")
            # Fallback values
            total_stories = 0
            total_test_cases = 0
            pacs008_relevant_stories = 0
            story_types = []
            pacs008_enhanced_tests = 0
            regression_tests = 0
            coverage_per_story = 0
        
        return {
            "workflow_status": "completed",
            "automation_intelligence": {
                "content_analysis": {
                    "pacs008_relevant": analysis.get("is_pacs008_relevant", False),
                    "content_type": analysis.get("content_type", "unknown"),
                    "confidence_score": analysis.get("confidence_score", 0),
                    "technical_level": analysis.get("technical_level", "medium")
                },
                "user_story_extraction": {
                    "total_stories": total_stories,
                    "pacs008_relevant_stories": pacs008_relevant_stories,
                    "story_types": story_types
                },
                "field_detection": {
                    "total_unique_fields": pacs008_fields.get("total_unique_fields", 0) if isinstance(pacs008_fields, dict) else 0,
                    "stories_with_fields": pacs008_fields.get("detection_summary", {}).get("stories_with_pacs008", 0) if isinstance(pacs008_fields, dict) else 0,
                    "field_coverage": "comprehensive" if pacs008_fields.get("total_unique_fields", 0) > 5 else "basic"
                },
                "maker_checker": {
                    "validation_items": len(maker_checker.get("validation_items", [])) if isinstance(maker_checker, dict) else 0,
                    "mandatory_items": maker_checker.get("summary", {}).get("mandatory_items", 0) if isinstance(maker_checker, dict) else 0,
                    "validation_ready": maker_checker.get("validation_ready", False) if isinstance(maker_checker, dict) else False
                },
                "test_generation": {
                    "total_test_cases": total_test_cases,
                    "pacs008_enhanced": pacs008_enhanced_tests,
                    "coverage_per_story": coverage_per_story,
                    "regression_tests": regression_tests
                }
            },
            "business_value": {
                "automation_achieved": True,
                "domain_expertise_applied": True,
                "maker_checker_integrated": True,
                "pacs008_intelligence_used": analysis.get("is_pacs008_relevant", False),
                "test_coverage": "comprehensive" if total_test_cases > 10 else "basic"
            },
            "next_steps": [
                "Review maker-checker validation items",
                "Execute generated test cases in test environment", 
                "Validate test results against business requirements",
                "Update test automation framework with new scenarios"
            ],
            "quality_indicators": {
                "field_detection_accuracy": "high" if pacs008_fields.get("total_unique_fields", 0) > 3 else "medium",
                "test_case_relevance": "high" if pacs008_enhanced_tests > 5 else "medium",
                "business_alignment": "high" if maker_checker.get("validation_ready", False) else "medium"
            }
        }

# Integration class for Streamlit
class StreamlitPACS008Integration:
    """Integration layer for Streamlit UI"""
    
    def __init__(self, api_key: str):
        self.generator = DynamicPACS008TestGenerator(api_key)
    
    def process_uploaded_files(self, uploaded_files, custom_instructions: str, 
                             num_test_cases_per_story: int) -> Dict[str, Any]:
        """Process uploaded files and return complete workflow results"""
        
        # Combine content from all uploaded files
        all_content = []
        
        for uploaded_file in uploaded_files:
            # In a real implementation, you'd use DocumentProcessor here
            # For now, assuming text content
            try:
                content = uploaded_file.getvalue().decode('utf-8')
                all_content.append(content)
            except:
                # Handle binary files or use DocumentProcessor
                all_content.append(f"Content from {uploaded_file.name}")
        
        combined_content = "\n\n--- Next Document ---\n\n".join(all_content)
        
        # Add custom instructions context
        if custom_instructions:
            combined_content += f"\n\nCustom Instructions: {custom_instructions}"
        
        # Run complete workflow
        workflow_results = self.generator.process_complete_workflow(
            combined_content, num_test_cases_per_story
        )
        
        return workflow_results
    
    def get_maker_checker_items(self, workflow_results: Dict) -> List[Dict[str, Any]]:
        """Extract maker-checker items for UI display"""
        return workflow_results.get("step4_maker_checker", {}).get("validation_items", [])
    
    def get_test_cases_for_export(self, workflow_results: Dict) -> List[Dict[str, Any]]:
        """Get test cases formatted for export"""
        return workflow_results.get("step5_test_cases", [])
    
    def get_pacs008_analysis_summary(self, workflow_results: Dict) -> Dict[str, Any]:
        """Get PACS.008 analysis summary for UI display"""
        return {
            "content_analysis": workflow_results.get("step1_analysis", {}),
            "user_stories": workflow_results.get("step2_user_stories", []),
            "field_detection": workflow_results.get("step3_pacs008_fields", {}),
            "workflow_summary": workflow_results.get("workflow_summary", {})
        }