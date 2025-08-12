# # src/ai_engine/test_generator.py - DYNAMIC INTELLIGENT VERSION
# import json
# import re
# from typing import Dict, List, Any, Optional
# import logging
# from openai import OpenAI
# import time

# logger = logging.getLogger(__name__)

# class TestCaseGenerator:
#     """Dynamic AI-powered test case generation with intelligent user story detection"""
    
#     def __init__(self, api_key: str):
#         self.client = OpenAI(api_key=api_key)
#         self.model = "gpt-4.1-mini-2025-04-14"
        
#     def generate_test_cases(self, content: str, custom_instructions: str = "") -> List[Dict[str, Any]]:
#         """Intelligently detect user stories and generate specified number of test cases for each"""
#         try:
#             # Clean and prepare content
#             cleaned_content = self._clean_content(content)
            
#             # Extract the number of test cases per story from instructions
#             num_cases_per_story = self._extract_test_case_count(custom_instructions)
#             logger.info(f"Target: {num_cases_per_story} test cases per user story")
            
#             # PHASE 1: Intelligent User Story Detection using LLM
#             user_stories = self._intelligent_user_story_detection(cleaned_content)
#             logger.info(f"Detected {len(user_stories)} user stories using LLM intelligence")
            
#             if not user_stories:
#                 # Fallback: treat entire content as one comprehensive requirement
#                 user_stories = [{"id": "REQ001", "title": "Comprehensive Requirements", "content": cleaned_content[:3000]}]
#                 logger.info("No distinct user stories found, treating as single comprehensive requirement")
            
#             all_test_cases = []
            
#             # PHASE 2: Generate test cases for each detected user story
#             for i, story in enumerate(user_stories, 1):
#                 logger.info(f"Generating {num_cases_per_story} test cases for User Story {i}: {story['title']}")
                
#                 story_test_cases = self._generate_test_cases_for_story(story, custom_instructions, num_cases_per_story)
                
#                 if story_test_cases:
#                     logger.info(f"Successfully generated {len(story_test_cases)} test cases for {story['id']}")
#                     all_test_cases.extend(story_test_cases)
#                 else:
#                     logger.warning(f"Failed to generate test cases for {story['id']}, using fallback")
#                     fallback_cases = self._fallback_test_case_generation(story, num_cases_per_story)
#                     all_test_cases.extend(fallback_cases)
            
#             # PHASE 3: Post-process and validate
#             validated_test_cases = self._validate_and_enhance_test_cases(all_test_cases)
            
#             logger.info(f"Final result: {len(validated_test_cases)} test cases generated for {len(user_stories)} user stories")
#             return validated_test_cases
            
#         except Exception as e:
#             logger.error(f"Error in dynamic test case generation: {str(e)}")
#             # Return fallback cases even on error
#             return self._emergency_fallback_generation(content, custom_instructions)
    
#     def _intelligent_user_story_detection(self, content: str) -> List[Dict[str, str]]:
#         """Use LLM to intelligently detect and extract user stories from content"""
        
#         detection_prompt = f"""
# You are an expert business analyst. Analyze the provided content and intelligently detect all user stories, requirements, or functional specifications.

# CONTENT TO ANALYZE:
# {content}

# INSTRUCTIONS:
# 1. Identify ALL distinct user stories, requirements, or functional specifications in the content
# 2. Look for patterns like:
#    - "As a [user], I want [goal] so that [benefit]"
#    - "The system shall/should/must..."
#    - Numbered requirements (1. User login, 2. Payment processing, etc.)
#    - Feature descriptions that represent distinct functionality
#    - Business scenarios or use cases
#    - Acceptance criteria that represent separate features

# 3. Each user story should represent a DISTINCT piece of functionality
# 4. Don't split one feature into multiple stories unnecessarily
# 5. But DO separate clearly different features/requirements

# 6. For each detected user story, provide:
#    - A unique ID (US001, US002, etc.)
#    - A clear, concise title
#    - The full content/description of that requirement

# RESPOND WITH ONLY JSON:
# {{
#   "detected_stories": [
#     {{
#       "id": "US001",
#       "title": "User Authentication",
#       "content": "Full description of the user authentication requirement...",
#       "confidence": "High",
#       "reasoning": "Clear authentication requirement found"
#     }},
#     {{
#       "id": "US002", 
#       "title": "Payment Processing",
#       "content": "Full description of payment processing requirement...",
#       "confidence": "High",
#       "reasoning": "Distinct payment functionality identified"
#     }}
#   ],
#   "total_stories": 2,
#   "analysis_summary": "Found 2 distinct user stories covering authentication and payments"
# }}

# Be intelligent - if content has multiple distinct features, identify them all. If it's one cohesive requirement, return one story.
# """
        
#         try:
#             response = self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[
#                     {"role": "system", "content": "You are an expert business analyst specializing in requirement analysis. Respond with ONLY valid JSON."},
#                     {"role": "user", "content": detection_prompt}
#                 ],
#                 temperature=0.2,
#                 max_tokens=2000
#             )
            
#             response_text = response.choices[0].message.content.strip()
#             logger.info(f"LLM User Story Detection Response: {len(response_text)} characters")
            
#             # Extract JSON from response
#             json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
#             if json_match:
#                 detection_result = json.loads(json_match.group())
#                 detected_stories = detection_result.get('detected_stories', [])
                
#                 logger.info(f"LLM detected {len(detected_stories)} user stories")
#                 logger.info(f"Analysis summary: {detection_result.get('analysis_summary', 'N/A')}")
                
#                 return detected_stories
#             else:
#                 logger.warning("No valid JSON found in LLM user story detection response")
#                 return []
                
#         except Exception as e:
#             logger.error(f"LLM user story detection failed: {str(e)}")
#             # Fallback to pattern-based detection
#             return self._pattern_based_user_story_extraction(content)
    
#     def _pattern_based_user_story_extraction(self, content: str) -> List[Dict[str, str]]:
#         """Fallback pattern-based user story extraction"""
#         logger.info("Using pattern-based user story extraction as fallback")
        
#         user_stories = []
        
#         # Pattern 1: Formal user stories
#         user_story_pattern = r'(?i)(?:As\s+(?:a|an)\s+.+?I\s+want\s+.+?(?:so\s+that|in\s+order\s+to).+?)(?=As\s+(?:a|an)|$|\n\n)'
#         matches = re.findall(user_story_pattern, content, re.DOTALL)
#         for i, match in enumerate(matches, 1):
#             if len(match.strip()) > 30:
#                 user_stories.append({
#                     "id": f"US{i:03d}",
#                     "title": f"User Story {i}",
#                     "content": match.strip(),
#                     "confidence": "Medium",
#                     "reasoning": "Pattern-based detection of formal user story"
#                 })
        
#         # Pattern 2: Shall/Should/Must statements
#         requirement_patterns = [
#             r'(?i)(?:The\s+system\s+(?:shall|should|must).+?)(?=The\s+system\s+(?:shall|should|must)|$|\n\n)',
#             r'(?i)(?:User\s+(?:shall|should|must|can|will).+?)(?=User\s+(?:shall|should|must|can|will)|$|\n\n)',
#             r'(?i)(?:Application\s+(?:shall|should|must).+?)(?=Application\s+(?:shall|should|must)|$|\n\n)'
#         ]
        
#         story_counter = len(user_stories) + 1
#         for pattern in requirement_patterns:
#             matches = re.findall(pattern, content, re.DOTALL)
#             for match in matches:
#                 if len(match.strip()) > 30:
#                     user_stories.append({
#                         "id": f"US{story_counter:03d}",
#                         "title": f"Requirement {story_counter}",
#                         "content": match.strip(),
#                         "confidence": "Medium", 
#                         "reasoning": "Pattern-based detection of requirement statement"
#                     })
#                     story_counter += 1
        
#         # Pattern 3: Numbered requirements
#         numbered_pattern = r'(?i)(?:\d+\.\s*.+?)(?=\d+\.\s*|$|\n\n)'
#         matches = re.findall(numbered_pattern, content, re.DOTALL)
#         for match in matches:
#             if len(match.strip()) > 50 and story_counter <= 10:  # Limit to prevent too many
#                 user_stories.append({
#                     "id": f"US{story_counter:03d}",
#                     "title": f"Numbered Requirement {story_counter}",
#                     "content": match.strip(),
#                     "confidence": "Low",
#                     "reasoning": "Pattern-based detection of numbered requirement"
#                 })
#                 story_counter += 1
        
#         # If still no stories found, look for clear sections
#         if not user_stories:
#             sections = self._split_into_logical_sections(content)
#             for i, section in enumerate(sections[:5], 1):  # Max 5 sections
#                 if len(section.strip()) > 100:
#                     user_stories.append({
#                         "id": f"REQ{i:03d}",
#                         "title": f"Requirement Section {i}",
#                         "content": section.strip(),
#                         "confidence": "Low",
#                         "reasoning": "Content section identified as potential requirement"
#                     })
        
#         logger.info(f"Pattern-based extraction found {len(user_stories)} user stories")
#         return user_stories
    
#     def _split_into_logical_sections(self, content: str) -> List[str]:
#         """Split content into logical sections for analysis"""
#         sections = []
        
#         # Try different splitting strategies
        
#         # Strategy 1: Split by headers (###, ##, etc.)
#         header_sections = re.split(r'\n\s*#{1,3}\s+', content)
#         if len(header_sections) > 1:
#             sections.extend([s.strip() for s in header_sections if len(s.strip()) > 50])
        
#         # Strategy 2: Split by numbered items
#         if not sections:
#             numbered_sections = re.split(r'\n\s*\d+\.\s+', content)
#             if len(numbered_sections) > 1:
#                 sections.extend([s.strip() for s in numbered_sections if len(s.strip()) > 50])
        
#         # Strategy 3: Split by double newlines (paragraphs)
#         if not sections:
#             paragraph_sections = content.split('\n\n')
#             sections.extend([p.strip() for p in paragraph_sections if len(p.strip()) > 100])
        
#         # Strategy 4: If content is very long, split by sentences/logical breaks
#         if not sections and len(content) > 2000:
#             # Split by periods followed by capital letters (sentence boundaries)
#             sentence_groups = re.split(r'\.\s+(?=[A-Z])', content)
#             current_group = []
#             for sentence in sentence_groups:
#                 current_group.append(sentence)
#                 group_text = '. '.join(current_group)
#                 if len(group_text) > 300:  # Group sentences into ~300 char sections
#                     sections.append(group_text.strip())
#                     current_group = []
            
#             # Add remaining sentences
#             if current_group:
#                 sections.append('. '.join(current_group).strip())
        
#         return sections[:8]  # Limit to max 8 sections to avoid too many stories
    
#     def _extract_test_case_count(self, custom_instructions: str) -> int:
#         """Extract the desired number of test cases per story from instructions"""
#         # Look for various patterns
#         patterns = [
#             r'exactly\s+(\d+)\s+test\s+cases?\s+per\s+story',
#             r'(\d+)\s+test\s+cases?\s+per\s+story',
#             r'generate\s+(\d+)\s+test\s+cases?\s+per',
#             r'(\d+)\s+test\s+cases?\s+each',
#             r'(\d+)\s+tests?\s+per\s+story',
#             r'(\d+)\s+tests?\s+per\s+requirement',
#             r'exactly\s+(\d+)\s+test\s+cases?',
#             r'generate\s+(\d+)\s+test\s+cases?',
#             r'(\d+)\s+test\s+cases?'
#         ]
        
#         for pattern in patterns:
#             match = re.search(pattern, custom_instructions.lower())
#             if match:
#                 count = int(match.group(1))
#                 logger.info(f"Extracted test case count: {count} per story")
#                 return count
        
#         # Default based on common UI slider values
#         logger.info("No specific test case count found, using default: 8")
#         return 8
    
#     def _generate_test_cases_for_story(self, story: Dict[str, str], custom_instructions: str, num_cases: int = 8) -> List[Dict[str, Any]]:
#         """Generate the specified number of test cases for a single user story"""
        
#         story_content = story.get('content', '')
#         story_id = story.get('id', 'US001')
#         story_title = story.get('title', 'User Story')
        
#         prompt = f"""
# You are an expert BFSI (Banking, Financial Services, Insurance) test engineer with deep domain knowledge.

# USER STORY/REQUIREMENT TO TEST:
# ID: {story_id}
# Title: {story_title}
# Content: {story_content}

# CUSTOM INSTRUCTIONS: {custom_instructions}

# CRITICAL REQUIREMENTS:
# 1. Generate EXACTLY {num_cases} comprehensive test cases for this ONE user story
# 2. All test cases MUST have the same User Story ID: {story_id}
# 3. Respond with ONLY a valid JSON array - no explanations, no markdown, no code blocks

# TEST CASE STRUCTURE (generate {num_cases} of these):
# - User Story ID: {story_id} (SAME for all {num_cases} test cases)
# - Acceptance Criteria ID: AC001, AC002, AC003, etc. (group 2-3 test cases per AC)
# - Scenario: Unique scenario name for each test case
# - Test Case ID: TC001, TC002, TC003, etc. (sequential)
# - Test Case Description: Clear, specific description
# - Precondition: Prerequisites for test execution  
# - Steps: Detailed numbered steps (use \\n for line breaks)
# - Expected Result: Clear expected outcome
# - Part of Regression: "Yes" for critical paths, "No" for edge cases
# - Priority: "High" for core functionality, "Medium" for validations, "Low" for edge cases

# COVERAGE REQUIREMENTS FOR ALL {num_cases} TEST CASES:
# ✅ Positive scenarios (happy path) - 30-40%
# ✅ Negative scenarios (error handling) - 30-40% 
# ✅ Boundary/edge cases - 20-30%
# ✅ Integration scenarios - if applicable
# ✅ Security validations - if applicable
# ✅ Performance considerations - if applicable

# BFSI DOMAIN REQUIREMENTS:
# - Use realistic banking data: IBANs (DE89370400440532013000, GB33BUKB20201555555555)
# - Use realistic BICs: DEUTDEFF, CHASUS33, BNPAFRPP, HSBCGB2L
# - Use realistic amounts: 0.01, 100.50, 1000.00, 50000.00, 999999.99
# - Use realistic currencies: EUR, USD, GBP, CHF
# - Include authentication, authorization, and audit scenarios
# - Cover regulatory compliance where applicable

# EXAMPLE OUTPUT FORMAT (respond with EXACTLY this structure for {num_cases} test cases):
# [
#   {{
#     "User Story ID": "{story_id}",
#     "Acceptance Criteria ID": "AC001",
#     "Scenario": "Valid Transaction Processing",
#     "Test Case ID": "TC001", 
#     "Test Case Description": "Verify successful processing with valid inputs",
#     "Precondition": "User is authenticated and has sufficient balance",
#     "Steps": "1. Navigate to transaction page\\n2. Enter amount: 1000.00\\n3. Enter IBAN: DE89370400440532013000\\n4. Submit transaction\\n5. Confirm with OTP",
#     "Expected Result": "Transaction processed successfully with confirmation",
#     "Part of Regression": "Yes",
#     "Priority": "High"
#   }},
#   {{
#     "User Story ID": "{story_id}",
#     "Acceptance Criteria ID": "AC001",
#     "Scenario": "Invalid Amount Validation",
#     "Test Case ID": "TC002",
#     "Test Case Description": "Verify error handling for invalid amount",
#     "Precondition": "User is authenticated",
#     "Steps": "1. Navigate to transaction page\\n2. Enter invalid amount: -100\\n3. Attempt to submit",
#     "Expected Result": "Error message displayed: 'Amount must be positive'",
#     "Part of Regression": "Yes", 
#     "Priority": "High"
#   }}
# ]

# REMEMBER: 
# - Generate EXACTLY {num_cases} test cases
# - ALL test cases must have User Story ID: {story_id}
# - Each test case must be unique and test different aspects
# - Cover positive, negative, and edge case scenarios
# - Use realistic BFSI domain data
# """
        
#         try:
#             response = self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[
#                     {"role": "system", "content": f"You are an expert BFSI test engineer. Generate exactly {num_cases} test cases for the provided user story. Respond ONLY with valid JSON array."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 temperature=0.2,
#                 max_tokens=4500  # Increased for more test cases
#             )
            
#             response_text = response.choices[0].message.content.strip()
#             logger.info(f"Generated response for {story_id}: {len(response_text)} characters")
            
#             # Parse JSON response
#             test_cases = self._extract_test_cases_from_response(response_text, story_id)
            
#             if test_cases:
#                 logger.info(f"Successfully parsed {len(test_cases)} test cases for {story_id}")
                
#                 # Validate we got the right number
#                 if len(test_cases) < num_cases:
#                     logger.warning(f"Got {len(test_cases)} test cases, expected {num_cases}. Generating additional cases.")
#                     additional_needed = num_cases - len(test_cases)
#                     additional_cases = self._generate_additional_test_cases(story, additional_needed, len(test_cases) + 1)
#                     test_cases.extend(additional_cases)
                
#                 return test_cases[:num_cases]  # Ensure exact count
#             else:
#                 logger.error(f"Failed to parse test cases for {story_id}")
#                 return []
                
#         except Exception as e:
#             logger.error(f"Error generating test cases for {story_id}: {str(e)}")
#             return []
    
#     def _extract_test_cases_from_response(self, response_text: str, story_id: str) -> List[Dict[str, Any]]:
#         """Extract test cases from LLM response with multiple fallback methods"""
        
#         # Method 1: Direct JSON array
#         json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
#         if json_match:
#             try:
#                 test_cases = json.loads(json_match.group())
#                 return test_cases
#             except json.JSONDecodeError:
#                 pass
        
#         # Method 2: JSON in code blocks
#         code_block_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response_text, re.DOTALL)
#         if code_block_match:
#             try:
#                 test_cases = json.loads(code_block_match.group(1))
#                 return test_cases
#             except json.JSONDecodeError:
#                 pass
        
#         # Method 3: Extract individual JSON objects
#         test_cases = []
#         object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
#         json_objects = re.findall(object_pattern, response_text, re.DOTALL)
        
#         for obj_str in json_objects:
#             try:
#                 obj = json.loads(obj_str)
#                 if 'Test Case ID' in obj or 'Test Case Description' in obj:
#                     obj['User Story ID'] = story_id  # Ensure correct story ID
#                     test_cases.append(obj)
#             except json.JSONDecodeError:
#                 continue
        
#         if test_cases:
#             return test_cases
        
#         # Method 4: Try to repair JSON
#         repaired_json = self._repair_json_response(response_text)
#         if repaired_json:
#             try:
#                 return json.loads(repaired_json)
#             except json.JSONDecodeError:
#                 pass
        
#         return []
    
#     def _generate_additional_test_cases(self, story: Dict[str, str], num_additional: int, start_tc_id: int) -> List[Dict[str, Any]]:
#         """Generate additional test cases to meet the target count"""
#         logger.info(f"Generating {num_additional} additional test cases for {story['id']}")
        
#         additional_cases = []
#         scenarios = [
#             ("Boundary Value Testing", "Verify system handles boundary values correctly", "Medium", "No"),
#             ("Concurrent Access Testing", "Verify system handles multiple simultaneous requests", "Medium", "Yes"),
#             ("Data Integrity Testing", "Verify data integrity is maintained during operations", "High", "Yes"), 
#             ("Performance Testing", "Verify system performance under normal load", "Low", "No"),
#             ("Security Testing", "Verify security measures are enforced", "High", "Yes"),
#             ("Error Recovery Testing", "Verify system recovers gracefully from errors", "Medium", "No"),
#             ("Integration Testing", "Verify integration with external systems", "Medium", "Yes"),
#             ("Audit Trail Testing", "Verify audit trail is properly maintained", "Medium", "Yes")
#         ]
        
#         for i in range(num_additional):
#             scenario_idx = i % len(scenarios)
#             scenario_name, description, priority, regression = scenarios[scenario_idx]
            
#             tc_id = f"TC{start_tc_id + i:03d}"
#             ac_id = f"AC{((start_tc_id + i - 1) // 3) + 1:03d}"  # Group every 3 test cases
            
#             additional_case = {
#                 "User Story ID": story['id'],
#                 "Acceptance Criteria ID": ac_id,
#                 "Scenario": f"{scenario_name} {i+1}" if num_additional > len(scenarios) else scenario_name,
#                 "Test Case ID": tc_id,
#                 "Test Case Description": f"{description} for {story.get('title', 'this requirement')}",
#                 "Precondition": "System is available and user has appropriate permissions",
#                 "Steps": f"1. Setup {scenario_name.lower()} conditions\\n2. Execute the scenario\\n3. Observe system behavior\\n4. Verify results",
#                 "Expected Result": f"System handles {scenario_name.lower()} appropriately with expected behavior",
#                 "Part of Regression": regression,
#                 "Priority": priority
#             }
            
#             additional_cases.append(additional_case)
        
#         return additional_cases
    
#     def _fallback_test_case_generation(self, story: Dict[str, str], num_cases: int = 8) -> List[Dict[str, Any]]:
#         """Generate comprehensive fallback test cases when LLM fails"""
#         logger.info(f"Generating {num_cases} fallback test cases for {story['id']}")
        
#         fallback_scenarios = [
#             ("Valid Input Processing", "Verify successful processing with valid inputs", "1. Enter valid data\\n2. Submit request\\n3. Verify processing", "Request processed successfully", "High", "Yes"),
#             ("Invalid Input Validation", "Verify error handling for invalid inputs", "1. Enter invalid data\\n2. Submit request\\n3. Verify error message", "Appropriate error message displayed", "High", "Yes"),
#             ("Empty Field Validation", "Verify validation for empty required fields", "1. Leave required fields empty\\n2. Attempt submission\\n3. Verify validation", "Validation error displayed for empty fields", "High", "Yes"),
#             ("Boundary Value Testing", "Verify handling of boundary values", "1. Input boundary values\\n2. Submit request\\n3. Verify handling", "Boundary values handled correctly", "Medium", "No"),
#             ("Maximum Length Testing", "Verify maximum field length validation", "1. Input maximum allowed characters\\n2. Submit request\\n3. Verify acceptance", "Maximum length input accepted", "Medium", "No"),
#             ("Special Character Handling", "Verify special character processing", "1. Input special characters\\n2. Submit request\\n3. Verify handling", "Special characters handled appropriately", "Medium", "No"),
#             ("Authentication Testing", "Verify authentication requirements", "1. Access without authentication\\n2. Verify access denied\\n3. Login and retry", "Authentication required and enforced", "High", "Yes"),
#             ("Authorization Testing", "Verify authorization controls", "1. Login with insufficient privileges\\n2. Attempt operation\\n3. Verify access denied", "Authorization controls enforced", "High", "Yes"),
#             ("Data Persistence Testing", "Verify data is saved correctly", "1. Input data\\n2. Submit and save\\n3. Retrieve data\\n4. Verify integrity", "Data saved and retrieved correctly", "Medium", "Yes"),
#             ("Error Recovery Testing", "Verify system recovery from errors", "1. Induce system error\\n2. Observe system response\\n3. Verify recovery", "System recovers gracefully from error", "Medium", "No"),
#             ("Performance Testing", "Verify acceptable response time", "1. Execute operation\\n2. Measure response time\\n3. Verify within limits", "Operation completes within acceptable time", "Low", "No"),
#             ("Concurrency Testing", "Verify concurrent user handling", "1. Multiple users access simultaneously\\n2. Execute operations\\n3. Verify integrity", "System handles concurrent access correctly", "Medium", "No")
#         ]
        
#         fallback_cases = []
        
#         for i in range(num_cases):
#             scenario_idx = i % len(fallback_scenarios)
#             scenario, description, steps, expected, priority, regression = fallback_scenarios[scenario_idx]
            
#             # Add variation for repeated scenarios
#             if i >= len(fallback_scenarios):
#                 scenario = f"{scenario} - Variant {(i // len(fallback_scenarios)) + 1}"
#                 description = f"{description} (Additional variant)"
            
#             tc_id = f"TC{i+1:03d}"
#             ac_id = f"AC{(i // 3) + 1:03d}"  # Group every 3 test cases
            
#             fallback_case = {
#                 "User Story ID": story['id'],
#                 "Acceptance Criteria ID": ac_id,
#                 "Scenario": scenario,
#                 "Test Case ID": tc_id,
#                 "Test Case Description": description,
#                 "Precondition": "System is available and user has appropriate access",
#                 "Steps": steps,
#                 "Expected Result": expected,
#                 "Part of Regression": regression,
#                 "Priority": priority
#             }
            
#             fallback_cases.append(fallback_case)
        
#         return fallback_cases
    
#     def _emergency_fallback_generation(self, content: str, instructions: str) -> List[Dict[str, Any]]:
#         """Emergency fallback when everything else fails"""
#         logger.warning("Using emergency fallback test case generation")
        
#         num_cases = self._extract_test_case_count(instructions)
        
#         emergency_story = {
#             "id": "EMRG001",
#             "title": "Emergency Requirement Analysis", 
#             "content": content[:500] + "..."
#         }
        
#         return self._fallback_test_case_generation(emergency_story, num_cases)
    
#     def _repair_json_response(self, response_text: str) -> str:
#         """Attempt to repair malformed JSON"""
#         try:
#             start_idx = response_text.find('[')
#             if start_idx == -1:
#                 return ""
            
#             end_idx = response_text.rfind(']')
#             if end_idx == -1:
#                 return ""
            
#             json_str = response_text[start_idx:end_idx + 1]
            
#             # Fix common issues
#             json_str = json_str.replace("'", '"')
#             json_str = re.sub(r'(?<!\\)\\n', '\\\\n', json_str)
#             json_str = re.sub(r',\s*}', '}', json_str)
#             json_str = re.sub(r',\s*]', ']', json_str)
            
#             return json_str
#         except:
#             return ""
    
#     def _clean_content(self, content: str) -> str:
#         """Clean and normalize content for processing"""
#         content = re.sub(r'\s+', ' ', content)
#         content = re.sub(r'[^\w\s\-\.\,\:\;\(\)\[\]\"\'\/\@\#\$\%\&\*\+\=\<\>\?]', ' ', content)
        
#         if len(content) > 12000:  # Increased limit for better analysis
#             content = content[:12000] + "..."
        
#         return content.strip()
    
#     def _validate_and_enhance_test_cases(self, test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         """Validate and enhance all generated test cases"""
#         validated_cases = []
#         required_fields = [
#             "User Story ID", "Acceptance Criteria ID", "Scenario", "Test Case ID",
#             "Test Case Description", "Precondition", "Steps", "Expected Result",
#             "Part of Regression", "Priority"
#         ]
        
#         story_counters = {}  # Track counters per story
        
#         for case in test_cases:
#             try:
#                 validated_case = {}
                
#                 # Ensure all required fields exist
#                 for field in required_fields:
#                     validated_case[field] = case.get(field, "").strip()
                
#                 story_id = validated_case.get("User Story ID", "US001")
                
#                 # Initialize counters for new story
#                 if story_id not in story_counters:
#                     story_counters[story_id] = {"tc_counter": 1, "ac_counter": 1}
                
#                 # Auto-generate missing IDs
#                 if not validated_case["Test Case ID"]:
#                     validated_case["Test Case ID"] = f"TC{story_counters[story_id]['tc_counter']:03d}"
                
#                 if not validated_case["Acceptance Criteria ID"]:
#                     validated_case["Acceptance Criteria ID"] = f"AC{story_counters[story_id]['ac_counter']:03d}"
                
#                 # Validate critical fields
#                 if len(validated_case["Test Case Description"]) < 10:
#                     validated_case["Test Case Description"] = f"Test case for {story_id} - scenario {story_counters[story_id]['tc_counter']}"
                
#                 if len(validated_case["Steps"]) < 10:
#                     validated_case["Steps"] = "1. Setup test conditions\\n2. Execute test scenario\\n3. Verify results"
                
#                 # Ensure proper values
#                 if validated_case["Part of Regression"] not in ["Yes", "No"]:
#                     validated_case["Part of Regression"] = "No"
                
#                 if validated_case["Priority"] not in ["High", "Medium", "Low"]:
#                     validated_case["Priority"] = "Medium"
                
#                 if not validated_case["Scenario"]:
#                     validated_case["Scenario"] = f"Test Scenario {story_counters[story_id]['tc_counter']}"
                
#                 if not validated_case["Expected Result"]:
#                     validated_case["Expected Result"] = "Expected behavior occurs as defined"
                
#                 if not validated_case["Precondition"]:
#                     validated_case["Precondition"] = "System is available and user has appropriate access"
                
#                 validated_cases.append(validated_case)
                
#                 # Update counters
#                 story_counters[story_id]["tc_counter"] += 1
#                 if story_counters[story_id]["tc_counter"] % 3 == 1:  # New AC every 3 test cases
#                     story_counters[story_id]["ac_counter"] += 1
                    
#             except Exception as e:
#                 logger.warning(f"Skipping invalid test case: {str(e)}")
#                 continue
        
#         # Log final statistics
#         total_stories = len(story_counters)
#         total_test_cases = len(validated_cases)
#         logger.info(f"Validation complete: {total_test_cases} test cases across {total_stories} user stories")
        
#         for story_id, counters in story_counters.items():
#             story_test_count = len([tc for tc in validated_cases if tc.get("User Story ID") == story_id])
#             logger.info(f"  {story_id}: {story_test_count} test cases")
        
#         return validated_cases
    
#     def enhance_with_custom_instructions(self, test_cases: List[Dict[str, Any]], instructions: str) -> List[Dict[str, Any]]:
#         """Enhance test cases based on custom instructions"""
#         if not instructions or not test_cases:
#             return test_cases
        
#         enhancement_prompt = f"""
# Based on these custom instructions: "{instructions}"
# Enhance the following test cases accordingly.

# Current test cases sample:
# {json.dumps(test_cases[:3], indent=2)}

# Total test cases to enhance: {len(test_cases)}

# Enhancement instructions could include:
# - "focus on cross-border payments" -> Add international banking scenarios
# - "include compliance testing" -> Add regulatory compliance scenarios
# - "emphasize security" -> Add security-focused test cases
# - "performance testing" -> Add performance validation scenarios

# Provide enhancement suggestions in JSON format:
# {{
#   "enhancements": [
#     {{
#       "type": "add_scenarios",
#       "description": "Add cross-border payment scenarios",
#       "affected_test_cases": ["TC001", "TC002"],
#       "new_steps": "Enhanced steps with international elements"
#     }}
#   ]
# }}
# """
        
#         try:
#             response = self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[
#                     {"role": "system", "content": "You are a test enhancement expert. Provide specific enhancement suggestions."},
#                     {"role": "user", "content": enhancement_prompt}
#                 ],
#                 temperature=0.3,
#                 max_tokens=1000
#             )
            
#             response_text = response.choices[0].message.content.strip()
#             json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
#             if json_match:
#                 enhancement_data = json.loads(json_match.group())
#                 enhanced_cases = self._apply_enhancements(test_cases, enhancement_data)
#                 return enhanced_cases
#             else:
#                 return test_cases
                
#         except Exception as e:
#             logger.error(f"Enhancement error: {str(e)}")
#             return test_cases
    
#     def _apply_enhancements(self, test_cases: List[Dict[str, Any]], enhancement_data: Dict) -> List[Dict[str, Any]]:
#         """Apply enhancement suggestions to test cases"""
#         enhanced_cases = test_cases.copy()
        
#         try:
#             enhancements = enhancement_data.get("enhancements", [])
            
#             for enhancement in enhancements:
#                 enhancement_type = enhancement.get("type")
#                 affected_cases = enhancement.get("affected_test_cases", [])
                
#                 if enhancement_type == "add_scenarios":
#                     # Add new scenario elements to specified test cases
#                     new_steps = enhancement.get("new_steps", "")
#                     for case in enhanced_cases:
#                         if case.get("Test Case ID") in affected_cases and new_steps:
#                             current_steps = case.get("Steps", "")
#                             case["Steps"] = f"{current_steps}\\n{new_steps}"
                
#                 elif enhancement_type == "modify_priority":
#                     # Modify priority of specified test cases
#                     new_priority = enhancement.get("new_priority", "Medium")
#                     for case in enhanced_cases:
#                         if case.get("Test Case ID") in affected_cases:
#                             case["Priority"] = new_priority
                
#                 elif enhancement_type == "add_preconditions":
#                     # Add additional preconditions
#                     additional_precondition = enhancement.get("additional_precondition", "")
#                     for case in enhanced_cases:
#                         if case.get("Test Case ID") in affected_cases and additional_precondition:
#                             current_precondition = case.get("Precondition", "")
#                             case["Precondition"] = f"{current_precondition}; {additional_precondition}"
            
#         except Exception as e:
#             logger.error(f"Error applying enhancements: {str(e)}")
        
#         return enhanced_cases

# # Usage example and testing
# if __name__ == "__main__":
#     # Test the dynamic system
#     generator = TestCaseGenerator("your-openai-api-key")
    
#     sample_content = """
#     User Story 1: As a bank customer, I want to transfer money to another account 
#     so that I can pay my bills online.
    
#     Acceptance Criteria:
#     - User must be authenticated
#     - Transfer amount must be positive
#     - Recipient account must be valid
#     - User must have sufficient balance
    
#     User Story 2: As a bank employee, I want to view customer transaction history
#     so that I can provide better customer service.
    
#     Acceptance Criteria:
#     - Employee must have appropriate permissions
#     - Transaction history must be complete
#     - Data must be properly formatted
    
#     The system shall also provide real-time notifications for all transactions.
#     The application must comply with PCI-DSS requirements.
#     """
    
#     # Test with different instructions
#     instructions = "Generate exactly 6 test cases per user story. Focus on security and compliance testing."
    
#     test_cases = generator.generate_test_cases(sample_content, instructions)
    
#     print(f"\n=== DYNAMIC TEST GENERATION RESULTS ===")
#     print(f"Generated {len(test_cases)} test cases")
    
#     # Group by user story
#     story_groups = {}
#     for case in test_cases:
#         story_id = case.get("User Story ID", "Unknown")
#         if story_id not in story_groups:
#             story_groups[story_id] = []
#         story_groups[story_id].append(case)
    
#     print(f"Across {len(story_groups)} user stories:")
#     for story_id, cases in story_groups.items():
#         print(f"  {story_id}: {len(cases)} test cases")
    
#     # Show sample test cases
#     print(f"\nSample test cases:")
#     for case in test_cases[:2]:
#         print(f"  {case.get('Test Case ID')}: {case.get('Test Case Description')}")
#         print(f"    Story: {case.get('User Story ID')}")
#         print(f"    Priority: {case.get('Priority')}")
#         print()



# src/ai_engine/test_generator.py - ENHANCED VERSION with BETTER PROMPTS
import json
import re
from typing import Dict, List, Any, Optional
import logging
from openai import OpenAI
import time

logger = logging.getLogger(__name__)

class TestCaseGenerator:
    """Enhanced AI-powered test case generation with PACS.008 domain expertise"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4.1-mini-2025-04-14"
        
    def generate_test_cases(self, content: str, custom_instructions: str = "") -> List[Dict[str, Any]]:
        """Enhanced test case generation with better prompts and domain focus"""
        try:
            # Clean and prepare content
            cleaned_content = self._clean_content(content)
            
            # Extract the number of test cases per story from instructions
            num_cases_per_story = self._extract_test_case_count(custom_instructions)
            logger.info(f"Target: {num_cases_per_story} test cases per user story")
            
            # PHASE 1: Enhanced User Story Detection
            user_stories = self._enhanced_user_story_detection(cleaned_content)
            logger.info(f"Detected {len(user_stories)} user stories using enhanced intelligence")
            
            if not user_stories:
                # Fallback: treat entire content as one comprehensive requirement
                user_stories = [{"id": "REQ001", "title": "Banking Requirements", "content": cleaned_content[:3000]}]
                logger.info("No distinct user stories found, treating as single banking requirement")
            
            all_test_cases = []
            
            # PHASE 2: Generate domain-specific test cases for each user story
            for i, story in enumerate(user_stories, 1):
                logger.info(f"Generating {num_cases_per_story} PACS.008 expert test cases for User Story {i}: {story['title']}")
                
                story_test_cases = self._generate_pacs008_expert_test_cases(story, custom_instructions, num_cases_per_story)
                
                if story_test_cases:
                    logger.info(f"Successfully generated {len(story_test_cases)} test cases for {story['id']}")
                    all_test_cases.extend(story_test_cases)
                else:
                    logger.warning(f"Failed to generate test cases for {story['id']}, using domain fallback")
                    fallback_cases = self._domain_fallback_test_cases(story, num_cases_per_story)
                    all_test_cases.extend(fallback_cases)
            
            # PHASE 3: Post-process and validate
            validated_test_cases = self._validate_and_enhance_test_cases(all_test_cases)
            
            logger.info(f"Final result: {len(validated_test_cases)} PACS.008 expert test cases generated for {len(user_stories)} user stories")
            return validated_test_cases
            
        except Exception as e:
            logger.error(f"Error in enhanced test case generation: {str(e)}")
            return self._emergency_domain_fallback(content, custom_instructions)
    
    def _enhanced_user_story_detection(self, content: str) -> List[Dict[str, str]]:
        """Enhanced user story detection with PACS.008 domain intelligence"""
        
        detection_prompt = f"""
You are a PACS.008 banking expert and business analyst. Analyze this content and intelligently identify ALL user stories, requirements, or testable banking scenarios.

CONTENT TO ANALYZE:
{content}

INSTRUCTIONS - BANKING DOMAIN FOCUS:
1. Identify ALL distinct user stories, requirements, or functional specifications
2. Look for PACS.008/banking patterns:
   - Payment processing workflows
   - Maker-checker processes  
   - Agent/bank relationships
   - Cross-border payment scenarios
   - Settlement and routing logic
   - Field validation requirements

3. Convert requirements to user stories format when needed
4. Focus on BANKING USER PERSONAS:
   - Ops User (maker)
   - Ops User (checker)
   - Bank customer
   - Compliance officer
   - System administrator

5. Each story should represent a DISTINCT banking functionality
6. Group related payment processing features logically

RESPOND WITH ONLY JSON:
{{
  "detected_stories": [
    {{
      "id": "US001",
      "title": "PACS.008 Payment Creation with Maker-Checker",
      "content": "As an Ops User maker, I want to create PACS.008 payments with all required fields so that payments can be processed through correspondent banking networks",
      "confidence": "High",
      "reasoning": "Clear payment processing requirement with maker-checker workflow",
      "banking_context": "PACS.008 payment initiation and validation"
    }},
    {{
      "id": "US002", 
      "title": "Payment Approval and Queue Management",
      "content": "As an Ops User checker, I want to review and approve payments so that only validated transactions proceed to settlement",
      "confidence": "High",
      "reasoning": "Approval workflow for payment processing",
      "banking_context": "Maker-checker approval and queue management"
    }}
  ],
  "total_stories": 2,
  "analysis_summary": "Found 2 distinct PACS.008 banking stories covering payment creation and approval workflows"
}}

Focus on REAL banking scenarios, not generic software requirements.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a PACS.008 banking expert specializing in correspondent banking and payment processing. Focus on real banking workflows and scenarios. Respond with ONLY valid JSON."},
                    {"role": "user", "content": detection_prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )
            
            response_text = response.choices[0].message.content.strip()
            logger.info(f"Enhanced User Story Detection Response: {len(response_text)} characters")
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                detection_result = json.loads(json_match.group())
                detected_stories = detection_result.get('detected_stories', [])
                
                logger.info(f"Enhanced detection found {len(detected_stories)} banking user stories")
                logger.info(f"Analysis summary: {detection_result.get('analysis_summary', 'N/A')}")
                
                return detected_stories
            else:
                logger.warning("No valid JSON in enhanced user story detection response")
                return []
                
        except Exception as e:
            logger.error(f"Enhanced user story detection failed: {str(e)}")
            return self._pattern_based_user_story_extraction(content)
    
    def _generate_pacs008_expert_test_cases(self, story: Dict[str, str], custom_instructions: str, num_cases: int = 8) -> List[Dict[str, Any]]:
        """Generate PACS.008 expert-level test cases with domain knowledge"""
        
        story_content = story.get('content', '')
        story_id = story.get('id', 'US001')
        story_title = story.get('title', 'Banking Requirement')
        banking_context = story.get('banking_context', 'PACS.008 payment processing')
        
        prompt = f"""
You are a Senior PACS.008 Test Engineer with deep expertise in correspondent banking, cross-border payments, and ISO 20022 standards.

USER STORY TO TEST:
ID: {story_id}
Title: {story_title}
Content: {story_content}
Banking Context: {banking_context}

CUSTOM INSTRUCTIONS: {custom_instructions}

CRITICAL REQUIREMENTS:
1. Generate EXACTLY {num_cases} comprehensive PACS.008 test cases
2. All test cases MUST have the same User Story ID: {story_id}
3. Use REAL BANKING DOMAIN EXPERTISE - not generic testing
4. Focus on CORRESPONDENT BANKING scenarios
5. Include realistic PACS.008 field values
6. Respond with ONLY a valid JSON array

PACS.008 DOMAIN EXPERTISE TO APPLY:
- Agent chain processing (Debtor Agent → Intermediary → Creditor Agent)
- Settlement methods (INDA/INGA/CLRG/COVE)
- Nostro/Vostro account relationships
- Cross-border compliance (CBPR+ rules)
- Cut-off times and business day logic
- Currency conversion and exchange rates
- Maker-checker approval workflows
- TPH system integration
- RLC queue management

REALISTIC BANKING DATA TO USE:
- BICs: DEUTDEFF (Deutsche Bank), BNPAFRPP (BNP Paribas), HSBCGB2L (HSBC), CITIUS33 (Citi), SBININBB (SBI)
- IBANs: DE89370400440532013000, FR1420041010050500013M02606, GB33BUKB20201555555555
- Amounts: 1000.00, 25000.50, 565000.00, 999999.99
- Currencies: EUR, USD, GBP, CHF
- Customer Names: "ABC Corporation Ltd", "Global Trading Inc", "International Services SA"

TEST COVERAGE STRATEGY:
✅ Positive scenarios (40%): Successful payment processing end-to-end
✅ Negative scenarios (30%): Invalid data, missing fields, compliance violations
✅ Business rule scenarios (20%): Cut-off times, limits, agent relationships 
✅ Integration scenarios (10%): TPH system, upstream/downstream interfaces

EXAMPLE TEST CASE FORMAT:
{{
  "User Story ID": "{story_id}",
  "Acceptance Criteria ID": "AC001",
  "Scenario": "Cross-border PACS.008 Payment via SERIAL Method",
  "Test Case ID": "TC001", 
  "Test Case Description": "Verify successful processing of EUR 25000.50 payment from Deutsche Bank (DEUTDEFF) to BNP Paribas (BNPAFRPP) using SERIAL settlement method",
  "Precondition": "Nostro account relationship established between DEUTDEFF and BNPAFRPP. Cut-off times configured. Exchange rates available.",
  "Steps": "1. Login as Ops User maker\\n2. Create PACS.008 payment: Debtor=ABC Corporation Ltd, Amount=EUR 25000.50, Debtor Agent=DEUTDEFF, Creditor Agent=BNPAFRPP\\n3. Set settlement method to INDA (Agent accounts)\\n4. Submit for approval\\n5. Login as Ops User checker\\n6. Review payment details in approval queue\\n7. Verify CBPR+ compliance\\n8. Approve payment",
  "Expected Result": "Payment successfully created, approved, and available in TPH processing queue. PACS.008 message generated with all mandatory fields. Settlement instructions prepared for SERIAL processing.",
  "Part of Regression": "Yes",
  "Priority": "High"
}}

GENERATE EXACTLY {num_cases} TEST CASES WITH THIS LEVEL OF PACS.008 EXPERTISE:
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"You are a Senior PACS.008 Test Engineer with deep correspondent banking expertise. Generate exactly {num_cases} expert-level test cases using real banking scenarios and data. Respond ONLY with valid JSON array."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=4500
            )
            
            response_text = response.choices[0].message.content.strip()
            logger.info(f"Generated PACS.008 expert response for {story_id}: {len(response_text)} characters")
            
            # Parse JSON response
            test_cases = self._extract_test_cases_from_response(response_text, story_id)
            
            if test_cases:
                logger.info(f"Successfully parsed {len(test_cases)} expert test cases for {story_id}")
                
                # Validate we got the right number
                if len(test_cases) < num_cases:
                    logger.warning(f"Got {len(test_cases)} test cases, expected {num_cases}. Generating additional domain cases.")
                    additional_needed = num_cases - len(test_cases)
                    additional_cases = self._generate_additional_pacs008_test_cases(story, additional_needed, len(test_cases) + 1)
                    test_cases.extend(additional_cases)
                
                return test_cases[:num_cases]  # Ensure exact count
            else:
                logger.error(f"Failed to parse test cases for {story_id}")
                return []
                
        except Exception as e:
            logger.error(f"Error generating PACS.008 expert test cases for {story_id}: {str(e)}")
            return []
    
    def _generate_additional_pacs008_test_cases(self, story: Dict[str, str], num_additional: int, start_tc_id: int) -> List[Dict[str, Any]]:
        """Generate additional PACS.008 domain-specific test cases"""
        logger.info(f"Generating {num_additional} additional PACS.008 test cases for {story['id']}")
        
        additional_cases = []
        
        # PACS.008 domain scenarios from client feedback
        pacs008_scenarios = [
            {
                "scenario": "PACS.008 Field Validation in TPH System",
                "description": "Verify all PACS.008 mandatory fields are available and validated in TPH system",
                "steps": "1. Login as Ops User maker\\n2. Navigate to PACS.008 payment creation\\n3. Verify all fields available: debtor name/address, debtor account, amount, currency, creditor, creditor agent\\n4. Validate field formats per ISO 20022",
                "expected": "All relevant PACS.008 fields available in TPH system with proper validation",
                "priority": "High",
                "regression": "Yes"
            },
            {
                "scenario": "Maker Input Validation for PACS.008 Creation", 
                "description": "Verify Ops User maker can input all required data for PACS.008 message creation",
                "steps": "1. Login as Ops User maker\\n2. Enter all required PACS.008 fields with valid data\\n3. Verify system accepts input\\n4. Submit for checker approval",
                "expected": "TPH system allows user to create payment PACS.008 (pending checker approval). System defaults bank/agent/customer account per configuration. System fetches upstream/downstream data correctly.",
                "priority": "High", 
                "regression": "Yes"
            },
            {
                "scenario": "Checker Review and Approval Workflow",
                "description": "Verify Ops User checker can review and approve PACS.008 payments created by maker",
                "steps": "1. Login as Ops User checker\\n2. Navigate to approval queue\\n3. Review all data inputted by maker\\n4. Verify PACS.008 compliance\\n5. Approve payment",
                "expected": "TPH system allows checker to review and approve the PACS.008 payment. All maker inputs visible and editable if needed.",
                "priority": "High",
                "regression": "Yes"
            },
            {
                "scenario": "RLC Queue Processing After Approval",
                "description": "Verify transaction appears in soft block queues after ops checker approval",
                "steps": "1. Complete maker-checker approval process\\n2. Navigate to RLC queue management\\n3. Verify transaction status and queue placement",
                "expected": "Transaction available in RLC queue as RLC setup conditions are met. Queue processing logic applied correctly.",
                "priority": "Medium",
                "regression": "Yes"
            },
            {
                "scenario": "SERIAL Method Processing Validation",
                "description": "Verify PACS.008 message processing via SERIAL settlement method",
                "steps": "1. Create PACS.008 with SERIAL settlement method\\n2. Process through correspondent bank chain\\n3. Verify each hop maintains message integrity\\n4. Validate settlement instructions",
                "expected": "Payment processed successfully through SERIAL method with correct routing and settlement",
                "priority": "High",
                "regression": "Yes"
            },
            {
                "scenario": "Cross-Border Compliance Validation",
                "description": "Verify CBPR+ compliance rules applied to cross-border PACS.008 payments",
                "steps": "1. Create cross-border payment exceeding thresholds\\n2. Verify compliance checks triggered\\n3. Validate regulatory screening\\n4. Confirm reporting requirements",
                "expected": "All applicable compliance rules enforced. Regulatory reporting generated. AML/sanctions screening completed.",
                "priority": "High",
                "regression": "Yes"
            },
            {
                "scenario": "Agent Relationship Validation",
                "description": "Verify nostro/vostro agent relationships are validated before processing", 
                "steps": "1. Attempt payment between banks without established relationship\\n2. Verify system validation\\n3. Test with valid agent relationships\\n4. Confirm routing logic",
                "expected": "System validates agent relationships. Payments rejected if no relationship exists. Valid relationships enable processing.",
                "priority": "Medium",
                "regression": "Yes"
            },
            {
                "scenario": "Cut-off Time and Business Day Processing",
                "description": "Verify cut-off times and business day rules applied to PACS.008 processing",
                "steps": "1. Submit payment before cut-off time\\n2. Submit payment after cut-off time\\n3. Test weekend/holiday processing\\n4. Verify value dating logic",
                "expected": "Cut-off times enforced correctly. Weekend/holiday rules applied. Value dating calculated per business rules.",
                "priority": "Medium", 
                "regression": "No"
            }
        ]
        
        for i in range(num_additional):
            scenario_idx = i % len(pacs008_scenarios)
            base_scenario = pacs008_scenarios[scenario_idx]
            
            tc_id = f"TC{start_tc_id + i:03d}"
            ac_id = f"AC{((start_tc_id + i - 1) // 3) + 1:03d}"  # Group every 3 test cases
            
            # Add variation for repeated scenarios
            scenario_suffix = f" - Variant {(i // len(pacs008_scenarios)) + 1}" if i >= len(pacs008_scenarios) else ""
            
            additional_case = {
                "User Story ID": story['id'],
                "Acceptance Criteria ID": ac_id,
                "Scenario": base_scenario["scenario"] + scenario_suffix,
                "Test Case ID": tc_id,
                "Test Case Description": base_scenario["description"],
                "Precondition": "All nostro/vostro agents configured. Cut-off times set. Exchange rates available. TPH system operational.",
                "Steps": base_scenario["steps"],
                "Expected Result": base_scenario["expected"],
                "Part of Regression": base_scenario["regression"],
                "Priority": base_scenario["priority"]
            }
            
            additional_cases.append(additional_case)
        
        return additional_cases
    
    def _domain_fallback_test_cases(self, story: Dict[str, str], num_cases: int = 8) -> List[Dict[str, Any]]:
        """Generate PACS.008 domain fallback test cases based on client examples"""
        logger.info(f"Generating {num_cases} PACS.008 domain fallback test cases for {story['id']}")
        
        story_id = story['id']
        
        # Client's exact domain examples
        client_domain_scenarios = [
            {
                "scenario": "PACS.008 Field Availability Verification",
                "description": "Verify whether all fields are available for PACS.008 in the TPH system",
                "precondition": "Menu, Navigation, fields, label should be available",
                "steps": "1. Login as Ops User maker\\n2. View all the fields like currency, amount, debit account number etc.",
                "expected": "All relevant fields available in TPH system to create a PACS.008:\\n1. debtor name and address\\n2. debtor account\\n3. amount\\n4. currency\\n5. creditor, creditor agent etc",
                "priority": "High",
                "regression": "Yes"
            },
            {
                "scenario": "Maker Data Input for PACS.008 Creation",
                "description": "Verify whether Maker user able to input all data for PACS.008 message creation",
                "precondition": "All Nostro/vostro agent, cut off time, exchange rate, upstream and downstream system are connected",
                "steps": "1. Login as Ops User maker\\n2. Enter all required for PACS.008 creation",
                "expected": "TPH system should allow user create payment PACS.008 (yet to approve by checker)\\nTPH system able to default bank/agent, customer account, as per setup/configuration\\nTPH system able to fetch data upstream/downstream correctly",
                "priority": "High",
                "regression": "Yes"
            },
            {
                "scenario": "Checker Approval and Review Process",
                "description": "Verify whether Checker user able to see all data in screen which are inputted by maker",
                "precondition": "All Nostro/vostro agent, cut off time, exchange rate, upstream and downstream system are connected",
                "steps": "1. Login as Ops User checker\\n2. Navigate to approval queue\\n3. Review maker inputs\\n4. Approve/reject payment",
                "expected": "TPH system should allow the checker to check/approve the Payment",
                "priority": "High",
                "regression": "Yes"
            },
            {
                "scenario": "Transaction Queue Management Validation",
                "description": "Verify whether transaction available in Soft block queues after op checker approved the transaction",
                "precondition": "All RLC setup configuration available",
                "steps": "1. Login as Ops User maker\\n2. Navigate to RLC queue\\n3. Check transaction status",
                "expected": "Transaction should be available RLC as RLC Setup condition is met",
                "priority": "Medium",
                "regression": "Yes"
            },
            {
                "scenario": "End-to-End PACS.008 Payment Processing",
                "description": "Verify complete PACS.008 payment flow from creation to settlement",
                "precondition": "All banks have established direct account relationships. Valid payment data available.",
                "steps": "1. Initiate PACS.008 message from Debtor Agent with amount EUR 565000\\n2. Process through correspondent banking network\\n3. Complete settlement via SERIAL method",
                "expected": "Payment is successfully processed through all banks with correct settlement instructions and bookings at each step",
                "priority": "High",
                "regression": "Yes"
            },
            {
                "scenario": "PACS.008 Field Format Validation",
                "description": "Verify system validation for mandatory PACS.008 field formats per ISO 20022",
                "precondition": "System is available and user is authenticated",
                "steps": "1. Login as Ops User maker\\n2. Attempt to create payment with invalid field formats\\n3. Submit for validation",
                "expected": "System displays appropriate validation errors for invalid field formats per ISO 20022 standards",
                "priority": "High", 
                "regression": "Yes"
            },
            {
                "scenario": "Cross-Border Payment Routing Validation",
                "description": "Verify cross-border payment routing through correspondent banks",
                "precondition": "All system integrations configured and correspondent relationships established",
                "steps": "1. Create cross-border PACS.008 payment\\n2. Verify routing logic\\n3. Check correspondent bank processing\\n4. Validate settlement method selection",
                "expected": "Payment routed correctly through correspondent network with appropriate settlement method",
                "priority": "Medium",
                "regression": "Yes"
            },
            {
                "scenario": "Business Rule and Compliance Validation",
                "description": "Verify compliance with CBPR+ business rules and regulatory requirements",
                "precondition": "All compliance rules configured in system",
                "steps": "1. Create payment that tests business rule boundaries\\n2. Submit through maker-checker process\\n3. Verify compliance validation\\n4. Check regulatory reporting",
                "expected": "System enforces all applicable business rules and compliance requirements. Regulatory reports generated.",
                "priority": "High",
                "regression": "Yes"
            }
        ]
        
        fallback_cases = []
        
        for i in range(num_cases):
            scenario_idx = i % len(client_domain_scenarios)
            base_scenario = client_domain_scenarios[scenario_idx]
            
            tc_id = f"TC{i+1:03d}"
            ac_id = f"AC{(i // 3) + 1:03d}"  # Group every 3 test cases
            
            # Add variation for repeated scenarios
            scenario_suffix = f" - Variant {(i // len(client_domain_scenarios)) + 1}" if i >= len(client_domain_scenarios) else ""
            
            fallback_case = {
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
            
            fallback_cases.append(fallback_case)
        
        return fallback_cases
    
    def _emergency_domain_fallback(self, content: str, instructions: str) -> List[Dict[str, Any]]:
        """Emergency PACS.008 domain fallback when everything fails"""
        logger.warning("Using emergency PACS.008 domain fallback test case generation")
        
        num_cases = self._extract_test_case_count(instructions)
        
        emergency_story = {
            "id": "PACS001",
            "title": "Emergency PACS.008 Payment Processing",
            "content": "As a banking system, I want to process PACS.008 payments according to ISO 20022 standards so that cross-border transactions are handled correctly"
        }
        
        return self._domain_fallback_test_cases(emergency_story, num_cases)
    
    def _extract_test_case_count(self, custom_instructions: str) -> int:
        """Extract the desired number of test cases per story from instructions"""
        # Look for various patterns
        patterns = [
            r'exactly\s+(\d+)\s+test\s+cases?\s+per\s+story',
            r'(\d+)\s+test\s+cases?\s+per\s+story',
            r'generate\s+(\d+)\s+test\s+cases?\s+per',
            r'(\d+)\s+test\s+cases?\s+each',
            r'(\d+)\s+tests?\s+per\s+story',
            r'(\d+)\s+tests?\s+per\s+requirement',
            r'exactly\s+(\d+)\s+test\s+cases?',
            r'generate\s+(\d+)\s+test\s+cases?',
            r'(\d+)\s+test\s+cases?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, custom_instructions.lower())
            if match:
                count = int(match.group(1))
                logger.info(f"Extracted test case count: {count} per story")
                return count
        
        # Default based on common UI slider values
        logger.info("No specific test case count found, using default: 8")
        return 8
    
    def _extract_test_cases_from_response(self, response_text: str, story_id: str) -> List[Dict[str, Any]]:
        """Extract test cases from LLM response with multiple fallback methods"""
        
        # Method 1: Direct JSON array
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            try:
                test_cases = json.loads(json_match.group())
                return test_cases
            except json.JSONDecodeError:
                pass
        
        # Method 2: JSON in code blocks
        code_block_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response_text, re.DOTALL)
        if code_block_match:
            try:
                test_cases = json.loads(code_block_match.group(1))
                return test_cases
            except json.JSONDecodeError:
                pass
        
        # Method 3: Extract individual JSON objects
        test_cases = []
        object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        json_objects = re.findall(object_pattern, response_text, re.DOTALL)
        
        for obj_str in json_objects:
            try:
                obj = json.loads(obj_str)
                if 'Test Case ID' in obj or 'Test Case Description' in obj:
                    obj['User Story ID'] = story_id  # Ensure correct story ID
                    test_cases.append(obj)
            except json.JSONDecodeError:
                continue
        
        if test_cases:
            return test_cases
        
        return []
    
    def _pattern_based_user_story_extraction(self, content: str) -> List[Dict[str, str]]:
        """Enhanced pattern-based user story extraction with banking focus"""
        logger.info("Using enhanced pattern-based user story extraction")
        
        user_stories = []
        
        # Enhanced patterns for banking content
        story_patterns = [
            r'(?i)(?:As\s+(?:a|an)\s+.+?I\s+want\s+.+?(?:so\s+that|in\s+order\s+to).+?)(?=As\s+(?:a|an)|$|\n\n)',
            r'(?i)(?:User\s+Story\s*:?\s*.+?)(?=User\s+Story|$)',
            r'(?i)(?:Requirement\s*:?\s*.+?)(?=Requirement|$)',
            r'(?i)(?:Feature\s*:?\s*.+?)(?=Feature|$)',
            r'(?i)(?:Scenario\s*:?\s*.+?)(?=Scenario|$)'
        ]
        
        story_id = 1
        for pattern in story_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                if len(match.strip()) > 50:
                    # Determine banking context
                    banking_context = "PACS.008 payment processing"
                    if any(word in match.lower() for word in ["maker", "checker", "approval"]):
                        banking_context = "Maker-checker workflow"
                    elif any(word in match.lower() for word in ["agent", "correspondent", "nostro"]):
                        banking_context = "Correspondent banking"
                    elif any(word in match.lower() for word in ["compliance", "aml", "regulatory"]):
                        banking_context = "Compliance and regulatory"
                    
                    user_stories.append({
                        "id": f"US{story_id:03d}",
                        "title": f"Banking User Story {story_id}",
                        "content": match.strip(),
                        "confidence": "Medium",
                        "reasoning": "Pattern-based detection of banking user story",
                        "banking_context": banking_context
                    })
                    story_id += 1
        
        # If no formal stories found, create logical banking sections
        if not user_stories:
            sections = self._split_content_intelligently(content)
            for i, section in enumerate(sections, 1):
                if len(section.strip()) > 100:
                    user_stories.append({
                        "id": f"REQ{i:03d}",
                        "title": f"Banking Requirement Section {i}",
                        "content": section.strip(),
                        "confidence": "Low",
                        "reasoning": "Content section identified as potential banking requirement",
                        "banking_context": "General banking requirement"
                    })
        
        logger.info(f"Enhanced pattern-based extraction found {len(user_stories)} user stories")
        return user_stories[:5]  # Limit to 5 user stories
    
    def _split_content_intelligently(self, content: str) -> List[str]:
        """Intelligently split content into logical sections"""
        
        sections = []
        
        # Banking-specific splitting strategies
        banking_patterns = [
            r'\n(?=\d+\.\s+)',  # Numbered sections
            r'\n(?=[A-Z][A-Z\s]+:)',  # All caps headers
            r'\n(?=#{1,3}\s)',  # Markdown headers
            r'\n(?=\*\s+[A-Z])',  # Bulleted sections starting with capital
            r'\n(?=As\s+(?:a|an)\s+)',  # User story starts
            r'\n(?=User\s+Story)',  # User story headers
            r'\n(?=Requirement)',  # Requirement headers
        ]
        
        for pattern in banking_patterns:
            potential_sections = re.split(pattern, content)
            if len(potential_sections) > 1 and all(len(s.strip()) > 50 for s in potential_sections):
                sections = potential_sections
                break
        
        # If still no good sections, split by paragraphs
        if not sections:
            paragraphs = content.split('\n\n')
            sections = [p.strip() for p in paragraphs if len(p.strip()) > 100]
        
        # If still no sections, split by length
        if not sections:
            chunk_size = max(300, len(content) // 3)  # Aim for 3 chunks
            sections = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        
        return sections[:5]  # Limit to 5 sections maximum
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content for processing"""
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'[^\w\s\-\.\,\:\;\(\)\[\]\"\'\/\@\#\$\%\&\*\+\=\<\>\?]', ' ', content)
        
        if len(content) > 15000:  # Increased limit for better analysis
            content = content[:15000] + "..."
        
        return content.strip()
    
    def _validate_and_enhance_test_cases(self, test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and enhance all generated test cases with banking domain knowledge"""
        validated_cases = []
        required_fields = [
            "User Story ID", "Acceptance Criteria ID", "Scenario", "Test Case ID",
            "Test Case Description", "Precondition", "Steps", "Expected Result",
            "Part of Regression", "Priority"
        ]
        
        story_counters = {}  # Track counters per story
        
        for case in test_cases:
            try:
                validated_case = {}
                
                # Ensure all required fields exist
                for field in required_fields:
                    validated_case[field] = case.get(field, "").strip()
                
                story_id = validated_case.get("User Story ID", "US001")
                
                # Initialize counters for new story
                if story_id not in story_counters:
                    story_counters[story_id] = {"tc_counter": 1, "ac_counter": 1}
                
                # Auto-generate missing IDs
                if not validated_case["Test Case ID"]:
                    validated_case["Test Case ID"] = f"TC{story_counters[story_id]['tc_counter']:03d}"
                
                if not validated_case["Acceptance Criteria ID"]:
                    validated_case["Acceptance Criteria ID"] = f"AC{story_counters[story_id]['ac_counter']:03d}"
                
                # Enhance with banking domain defaults
                if len(validated_case["Test Case Description"]) < 10:
                    validated_case["Test Case Description"] = f"Verify PACS.008 payment processing for {story_id} - scenario {story_counters[story_id]['tc_counter']}"
                
                if len(validated_case["Steps"]) < 20:
                    validated_case["Steps"] = "1. Login as Ops User maker\\n2. Create PACS.008 payment with required fields\\n3. Submit for checker approval\\n4. Login as checker and approve\\n5. Verify processing queue"
                
                # Ensure proper banking values
                if validated_case["Part of Regression"] not in ["Yes", "No"]:
                    validated_case["Part of Regression"] = "Yes"  # Default to regression for banking
                
                if validated_case["Priority"] not in ["High", "Medium", "Low"]:
                    validated_case["Priority"] = "High"  # Default to high for PACS.008
                
                if not validated_case["Scenario"]:
                    validated_case["Scenario"] = f"PACS.008 Banking Scenario {story_counters[story_id]['tc_counter']}"
                
                if not validated_case["Expected Result"]:
                    validated_case["Expected Result"] = "PACS.008 payment processed successfully per banking standards and compliance requirements"
                
                if not validated_case["Precondition"]:
                    validated_case["Precondition"] = "TPH system operational. Nostro/vostro agents configured. Cut-off times set. User authenticated with appropriate permissions."
                
                validated_cases.append(validated_case)
                
                # Update counters
                story_counters[story_id]["tc_counter"] += 1
                if story_counters[story_id]["tc_counter"] % 3 == 1:  # New AC every 3 test cases
                    story_counters[story_id]["ac_counter"] += 1
                    
            except Exception as e:
                logger.warning(f"Skipping invalid test case: {str(e)}")
                continue
        
        # Log final statistics
        total_stories = len(story_counters)
        total_test_cases = len(validated_cases)
        logger.info(f"Banking validation complete: {total_test_cases} PACS.008 test cases across {total_stories} user stories")
        
        for story_id, counters in story_counters.items():
            story_test_count = len([tc for tc in validated_cases if tc.get("User Story ID") == story_id])
            logger.info(f"  {story_id}: {story_test_count} PACS.008 test cases")
        
        return validated_cases