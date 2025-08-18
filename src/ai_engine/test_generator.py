

# src/ai_engine/test_generator.py - FIXED FOR MAKER-CHECKER WORKFLOWS
"""
FIXED: Test Case Generator with Explicit Maker-Checker Validation Focus
Addresses client feedback: Test Case Description must have validation pertaining to maker and checker process
"""

import json
import re
from typing import Dict, List, Any, Optional
import logging
from openai import OpenAI
import time

logger = logging.getLogger(__name__)

class TestCaseGenerator:
    """FIXED: AI-powered test case generation with explicit maker-checker workflow validation"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4.1-mini-2025-04-14"
        
    def generate_test_cases(self, content: str, custom_instructions: str = "") -> List[Dict[str, Any]]:
        """FIXED: Generate test cases with explicit maker-checker validation focus"""
        try:
            # Extract target number of test cases
            num_cases_per_story = self._extract_test_case_count(custom_instructions)
            logger.info(f"FIXED: Generating {num_cases_per_story} maker-checker focused test cases per story")
            
            # Detect user stories with maker-checker context
            user_stories = self._detect_user_stories_with_maker_checker_focus(content)
            logger.info(f"FIXED: Detected {len(user_stories)} user stories with maker-checker workflows")
            
            if not user_stories:
                user_stories = [{"id": "REQ001", "title": "Banking Requirements with Maker-Checker", "content": content[:3000]}]
            
            all_test_cases = []
            
            # Generate maker-checker focused test cases for each story
            for i, story in enumerate(user_stories, 1):
                logger.info(f"FIXED: Generating maker-checker validation test cases for Story {i}: {story['title']}")
                
                story_test_cases = self._generate_maker_checker_focused_test_cases(
                    story, custom_instructions, num_cases_per_story
                )
                
                if story_test_cases:
                    logger.info(f"FIXED: Generated {len(story_test_cases)} maker-checker test cases for {story['id']}")
                    all_test_cases.extend(story_test_cases)
                else:
                    logger.warning(f"FIXED: Using fallback maker-checker tests for {story['id']}")
                    fallback_cases = self._create_maker_checker_fallback_tests(story, num_cases_per_story)
                    all_test_cases.extend(fallback_cases)
            
            # Validate and enhance test cases for maker-checker workflows
            validated_test_cases = self._validate_maker_checker_test_cases(all_test_cases)
            
            logger.info(f"FIXED: Generated {len(validated_test_cases)} maker-checker focused test cases")
            return validated_test_cases
            
        except Exception as e:
            logger.error(f"FIXED: Error in maker-checker test generation: {str(e)}")
            return self._emergency_maker_checker_fallback(content, custom_instructions)
    
    def _detect_user_stories_with_maker_checker_focus(self, content: str) -> List[Dict[str, str]]:
        """FIXED: Detect user stories with specific focus on maker-checker workflows"""
        
        prompt = f"""
You are a PACS.008 banking expert specializing in maker-checker workflows. Analyze this content and identify user stories that require maker-checker validation processes.

CONTENT TO ANALYZE:
{content}

CRITICAL FOCUS: MAKER-CHECKER WORKFLOWS
1. Look for payment creation and approval processes
2. Identify scenarios requiring dual authorization
3. Focus on PACS.008 field validation workflows
4. Emphasize maker input → checker approval → system processing

MAKER-CHECKER PERSONAS TO USE:
- Ops User (maker): Creates payments, inputs data, submits for approval
- Ops User (checker): Reviews data, validates fields, approves/rejects payments
- Compliance officer: Validates regulatory compliance
- System administrator: Manages workflow configurations

RESPOND WITH JSON ONLY:
{{
  "detected_stories": [
    {{
      "id": "US001",
      "title": "PACS.008 Payment Creation with Maker-Checker Validation",
      "content": "As an Ops User maker, I want to create PACS.008 payments with all required fields so that the payment can be submitted to Ops User checker for validation and approval before processing",
      "maker_checker_focus": "Payment creation and field validation by maker, followed by checker approval workflow",
      "validation_requirements": ["Field accuracy verification", "Business rule compliance", "Regulatory validation"],
      "workflow_type": "creation_approval"
    }},
    {{
      "id": "US002", 
      "title": "Checker Approval and Validation Process",
      "content": "As an Ops User checker, I want to review and validate all PACS.008 payment details created by makers so that only compliant and accurate payments proceed to processing",
      "maker_checker_focus": "Checker validation of maker inputs with detailed field-by-field verification",
      "validation_requirements": ["Maker input verification", "Field format validation", "Business logic checks"],
      "workflow_type": "approval_validation"
    }}
  ],
  "workflow_focus": "maker_checker_validation"
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a PACS.008 banking expert focused on maker-checker workflows. Extract user stories that explicitly require maker-checker validation. Respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )
            
            result = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            
            if json_match:
                detection_result = json.loads(json_match.group())
                detected_stories = detection_result.get('detected_stories', [])
                
                logger.info(f"FIXED: Detected {len(detected_stories)} maker-checker focused user stories")
                return detected_stories
            else:
                logger.warning("FIXED: No valid JSON in maker-checker story detection")
                return self._create_default_maker_checker_stories(content)
                
        except Exception as e:
            logger.error(f"FIXED: Maker-checker story detection failed: {str(e)}")
            return self._create_default_maker_checker_stories(content)
    
    def _generate_maker_checker_focused_test_cases(self, story: Dict[str, str], 
                                                 custom_instructions: str, num_cases: int = 8) -> List[Dict[str, Any]]:
        """FIXED: Generate test cases with explicit maker-checker validation in descriptions"""
        
        story_content = story.get('content', '')
        story_id = story.get('id', 'US001')
        story_title = story.get('title', 'Maker-Checker Workflow')
        maker_checker_focus = story.get('maker_checker_focus', 'Payment validation workflow')
        validation_requirements = story.get('validation_requirements', [])
        
        prompt = f"""
You are a Senior PACS.008 Test Engineer specializing in maker-checker workflows. Generate {num_cases} test cases where EVERY test case description explicitly includes maker-checker validation processes.

USER STORY CONTEXT:
ID: {story_id}
Title: {story_title}
Content: {story_content}
Maker-Checker Focus: {maker_checker_focus}
Validation Requirements: {', '.join(validation_requirements)}

CUSTOM INSTRUCTIONS: {custom_instructions}

CRITICAL REQUIREMENT - TEST DESCRIPTIONS MUST INCLUDE:
1. EXPLICIT MAKER ACTIONS: "Ops User maker creates/inputs/submits..."
2. EXPLICIT CHECKER ACTIONS: "Ops User checker reviews/validates/approves..."
3. FIELD VALIDATION PROCESS: "Checker validates field accuracy, format, business rules..."
4. APPROVAL WORKFLOW: "System requires checker approval before processing..."

EXAMPLE CORRECT TEST DESCRIPTION:
"Verify that when Ops User maker creates PACS.008 payment for USD 565000 from Al Ahli Bank of Kuwait to BNP Paribas, the system requires Ops User checker to review and validate all mandatory fields (amount, debtor agent, creditor agent) before approving payment for processing"

MAKER-CHECKER TEST SCENARIOS TO COVER:
- Maker creates payment → Checker validates → Approval workflow
- Field validation by checker with specific PACS.008 requirements
- Checker rejection scenarios with maker re-work
- Authority limit validation requiring checker approval
- Compliance validation by checker before processing
- Audit trail creation for maker-checker workflow

BANKING DATA TO USE:
- Amounts: USD 565000, EUR 25000, USD 1,000,000
- Banks: Al Ahli Bank of Kuwait, BNP Paribas, Deutsche Bank
- Customers: Corporate Customer, ABC Corporation, Corporation Y

RESPOND WITH EXACTLY {num_cases} TEST CASES:
[
  {{
    "User Story ID": "{story_id}",
    "Acceptance Criteria ID": "AC001",
    "Scenario": "Maker-Checker Payment Creation Workflow",
    "Test Case ID": "TC001",
    "Test Case Description": "Verify that when Ops User maker creates PACS.008 payment for USD 565000, the system requires Ops User checker to validate all payment fields and approve before processing can proceed",
    "Precondition": "TPH system operational. Maker and checker users authenticated with appropriate permissions. Nostro relationships established.",
    "Steps": "1. Login as Ops User maker\\n2. Create payment: Amount=USD 565000, From=Al Ahli Bank of Kuwait, To=BNP Paribas\\n3. Enter all required PACS.008 fields\\n4. Submit for checker approval\\n5. Login as Ops User checker\\n6. Review all maker inputs\\n7. Validate field accuracy and compliance\\n8. Approve payment",
    "Expected Result": "Payment created by maker successfully enters checker approval queue. Checker can review all fields, validate accuracy, and approve. Only after checker approval does payment proceed to processing.",
    "Part of Regression": "Yes",
    "Priority": "High"
  }}
]

ENSURE EVERY TEST DESCRIPTION EXPLICITLY MENTIONS MAKER-CHECKER VALIDATION PROCESS.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"You are a PACS.008 maker-checker workflow expert. Generate {num_cases} test cases where EVERY description explicitly includes maker and checker validation processes. Focus on dual authorization workflows."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=4500
            )
            
            result = response.choices[0].message.content.strip()
            json_match = re.search(r'\[.*\]', result, re.DOTALL)
            
            if json_match:
                test_cases = json.loads(json_match.group())
                
                # FIXED: Validate that each test case includes maker-checker validation
                validated_cases = self._ensure_maker_checker_validation_in_descriptions(test_cases, story)
                
                logger.info(f"FIXED: Generated {len(validated_cases)} maker-checker focused test cases")
                return validated_cases
            else:
                logger.warning(f"FIXED: Could not parse maker-checker test cases for {story_id}")
                return self._create_maker_checker_fallback_tests(story, num_cases)
                
        except Exception as e:
            logger.error(f"FIXED: Maker-checker test generation failed for {story_id}: {str(e)}")
            return self._create_maker_checker_fallback_tests(story, num_cases)
    
    def _ensure_maker_checker_validation_in_descriptions(self, test_cases: List[Dict], story: Dict) -> List[Dict]:
        """FIXED: Ensure every test case description explicitly includes maker-checker validation"""
        
        validated_cases = []
        
        for i, test_case in enumerate(test_cases):
            description = test_case.get("Test Case Description", "")
            
            # Check if description includes maker-checker validation
            has_maker = any(term in description.lower() for term in ["maker creates", "maker inputs", "ops user maker"])
            has_checker = any(term in description.lower() for term in ["checker validates", "checker reviews", "checker approves", "ops user checker"])
            has_validation = any(term in description.lower() for term in ["validate", "review", "approval", "verify fields"])
            
            if has_maker and has_checker and has_validation:
                # Description is good - keep as is
                validated_cases.append(test_case)
            else:
                # FIXED: Enhance description to include maker-checker validation
                enhanced_description = self._enhance_description_with_maker_checker(description, story, i+1)
                test_case["Test Case Description"] = enhanced_description
                validated_cases.append(test_case)
        
        return validated_cases
    
    def _enhance_description_with_maker_checker(self, original_description: str, story: Dict, test_num: int) -> str:
        """FIXED: Enhance test description to explicitly include maker-checker validation"""
        
        # Extract key elements
        story_id = story.get('id', 'US001')
        
        # Banking data for realistic scenarios
        amounts = ["USD 565000", "EUR 25000", "USD 1,000,000"]
        banks = ["Al Ahli Bank of Kuwait", "BNP Paribas", "Deutsche Bank"]
        
        amount = amounts[test_num % len(amounts)]
        bank_from = banks[0]
        bank_to = banks[1]
        
        # Create maker-checker focused description
        if "payment" in original_description.lower():
            enhanced = f"Verify that when Ops User maker creates PACS.008 payment for {amount} from {bank_from} to {bank_to}, "
            enhanced += "the system requires Ops User checker to review and validate all payment fields (amount, debtor agent, creditor agent, debtor name) "
            enhanced += "and approve the payment before it can proceed to processing. Checker must verify field accuracy and business rule compliance."
        
        elif "field" in original_description.lower():
            enhanced = f"Verify that Ops User checker can validate all PACS.008 fields created by Ops User maker for {amount} payment, "
            enhanced += "including field format validation, business logic checks, and compliance verification before approving for processing."
        
        elif "approval" in original_description.lower():
            enhanced = f"Verify that {amount} payment created by Ops User maker requires Ops User checker approval workflow, "
            enhanced += "where checker must review all maker inputs, validate against PACS.008 standards, and explicitly approve before system processes payment."
        
        else:
            # Generic maker-checker enhancement
            enhanced = f"Verify that Ops User maker can create {amount} PACS.008 payment and Ops User checker can validate all maker inputs, "
            enhanced += "verify field accuracy and compliance, and approve payment through proper maker-checker workflow before processing."
        
        return enhanced
    
    def _create_maker_checker_fallback_tests(self, story: Dict, num_cases: int) -> List[Dict[str, Any]]:
        """FIXED: Create fallback test cases with explicit maker-checker workflows"""
        
        story_id = story.get('id', 'US001')
        
        # Pre-defined maker-checker focused test scenarios
        maker_checker_scenarios = [
            {
                "scenario": "Maker-Checker Payment Creation Workflow",
                "description": "Verify that when Ops User maker creates PACS.008 payment for USD 565000 from Al Ahli Bank of Kuwait to BNP Paribas, the system requires Ops User checker to validate all mandatory fields (amount, debtor agent, creditor agent) and approve before processing",
                "steps": "1. Login as Ops User maker\\n2. Create payment: Amount=USD 565000, From=Al Ahli Bank of Kuwait, To=BNP Paribas\\n3. Enter all PACS.008 fields\\n4. Submit for approval\\n5. Login as Ops User checker\\n6. Review payment details\\n7. Validate field accuracy\\n8. Approve payment",
                "expected": "Payment created by maker enters checker queue. Checker validates all fields and approves. Payment proceeds to processing only after checker approval.",
                "priority": "High"
            },
            {
                "scenario": "Checker Field Validation and Approval Process",
                "description": "Verify that Ops User checker can validate all PACS.008 fields (debtor name, creditor name, amount, currency) created by Ops User maker for EUR 25000 payment and approve or reject based on field accuracy and compliance",
                "steps": "1. Ops User maker creates EUR 25000 payment\\n2. Submits for checker approval\\n3. Login as Ops User checker\\n4. Access approval queue\\n5. Review each PACS.008 field\\n6. Validate format and business rules\\n7. Approve payment",
                "expected": "Checker can access all maker inputs, validate each field against PACS.008 standards, and approve payment. System records checker validation and approval actions.",
                "priority": "High"
            },
            {
                "scenario": "Maker Authority Limit with Checker Override",
                "description": "Verify that when Ops User maker attempts to create payment exceeding authority limit (USD 1,000,000), the system requires Ops User checker with higher authority to validate and approve the high-value payment",
                "steps": "1. Login as Ops User maker\\n2. Create USD 1,000,000 payment\\n3. System detects authority limit breach\\n4. Routes to checker for approval\\n5. Login as authorized checker\\n6. Review high-value payment\\n7. Validate and approve",
                "expected": "System prevents maker from processing high-value payment independently. Checker with appropriate authority can validate and approve. Audit trail records authority override.",
                "priority": "High"
            },
            {
                "scenario": "Checker Rejection and Maker Rework Process",
                "description": "Verify that when Ops User checker rejects payment due to invalid fields, the payment returns to Ops User maker for correction, and the reworked payment requires fresh checker validation and approval",
                "steps": "1. Maker creates payment with invalid BIC\\n2. Submits for approval\\n3. Checker reviews and identifies invalid field\\n4. Rejects with comments\\n5. Maker receives rejection\\n6. Corrects BIC field\\n7. Resubmits for checker approval",
                "expected": "Checker can reject payment with specific reasons. Maker receives rejection notification and can correct fields. Corrected payment requires new checker approval.",
                "priority": "High"
            },
            {
                "scenario": "Dual Authorization for Cross-Border Payments",
                "description": "Verify that cross-border PACS.008 payments created by Ops User maker require Ops User checker validation of correspondent banking details, compliance requirements, and regulatory checks before approval",
                "steps": "1. Maker creates cross-border payment\\n2. Enters correspondent bank details\\n3. Submits for checker approval\\n4. Checker validates correspondent relationships\\n5. Checks compliance requirements\\n6. Verifies regulatory compliance\\n7. Approves payment",
                "expected": "Cross-border payments require enhanced checker validation. Checker verifies correspondent banking setup, compliance rules, and regulatory requirements before approval.",
                "priority": "Medium"
            },
            {
                "scenario": "Compliance Validation in Maker-Checker Workflow",
                "description": "Verify that Ops User checker validates PACS.008 payment compliance (AML, sanctions screening, regulatory requirements) for payments created by Ops User maker before granting final approval",
                "steps": "1. Maker creates high-value payment\\n2. Submits for approval\\n3. Checker accesses compliance validation\\n4. Reviews AML requirements\\n5. Checks sanctions screening\\n6. Validates regulatory compliance\\n7. Approves with compliance sign-off",
                "expected": "Checker can perform comprehensive compliance validation. System provides compliance tools and checks. Approval includes compliance confirmation.",
                "priority": "High"
            },
            {
                "scenario": "Maker Input Validation with Field-Level Checking",
                "description": "Verify that Ops User checker can validate each individual PACS.008 field (amount format, BIC validation, IBAN verification) inputted by Ops User maker and provide field-specific approval or correction requests",
                "steps": "1. Maker inputs all PACS.008 fields\\n2. Submits for validation\\n3. Checker reviews each field individually\\n4. Validates amount format\\n5. Verifies BIC codes\\n6. Checks IBAN validity\\n7. Approves or requests corrections",
                "expected": "Checker can validate each field individually. System highlights field validation results. Checker can approve payment or request specific field corrections.",
                "priority": "Medium"
            },
            {
                "scenario": "Audit Trail for Maker-Checker Activities",
                "description": "Verify that all Ops User maker and Ops User checker activities in PACS.008 payment processing are recorded in audit trail, including creation, validation, approval, and rejection actions with timestamps",
                "steps": "1. Maker creates payment (recorded)\\n2. Submits for approval (recorded)\\n3. Checker accesses payment (recorded)\\n4. Checker validates fields (recorded)\\n5. Checker approves payment (recorded)\\n6. Review audit trail",
                "expected": "Complete audit trail captures all maker and checker actions with timestamps, user IDs, and action details. Audit trail is accessible for compliance reporting.",
                "priority": "Medium"
            }
        ]
        
        fallback_cases = []
        
        for i in range(num_cases):
            scenario_idx = i % len(maker_checker_scenarios)
            scenario = maker_checker_scenarios[scenario_idx]
            
            tc_id = f"TC{i+1:03d}"
            ac_id = f"AC{(i // 3) + 1:03d}"
            
            # Add variation for repeated scenarios
            scenario_suffix = f" - Variant {(i // len(maker_checker_scenarios)) + 1}" if i >= len(maker_checker_scenarios) else ""
            
            fallback_case = {
                "User Story ID": story_id,
                "Acceptance Criteria ID": ac_id,
                "Scenario": scenario["scenario"] + scenario_suffix,
                "Test Case ID": tc_id,
                "Test Case Description": scenario["description"],
                "Precondition": "TPH system operational. Maker and checker users authenticated. Nostro/vostro relationships configured. Authority limits set.",
                "Steps": scenario["steps"],
                "Expected Result": scenario["expected"],
                "Part of Regression": "Yes",
                "Priority": scenario["priority"]
            }
            
            fallback_cases.append(fallback_case)
        
        return fallback_cases
    
    def _create_default_maker_checker_stories(self, content: str) -> List[Dict[str, str]]:
        """FIXED: Create default maker-checker focused user stories"""
        
        return [
            {
                "id": "US001",
                "title": "PACS.008 Payment Creation with Maker-Checker Validation",
                "content": "As an Ops User maker, I want to create PACS.008 payments with all required fields so that Ops User checker can validate and approve payments before processing",
                "maker_checker_focus": "Payment creation by maker followed by comprehensive checker validation",
                "validation_requirements": ["Field accuracy", "Business rule compliance", "Authority validation"],
                "workflow_type": "creation_approval"
            },
            {
                "id": "US002",
                "title": "Checker Approval and Field Validation Workflow",
                "content": "As an Ops User checker, I want to validate all PACS.008 payment fields created by makers so that only accurate and compliant payments proceed to processing",
                "maker_checker_focus": "Detailed field validation and approval process by checker",
                "validation_requirements": ["Field format validation", "Business logic verification", "Compliance checks"],
                "workflow_type": "approval_validation"
            },
            {
                "id": "US003",
                "title": "Compliance Validation in Maker-Checker Process",
                "content": "As a Compliance officer, I want to validate PACS.008 payments through maker-checker workflow so that regulatory compliance is maintained before payment processing",
                "maker_checker_focus": "Compliance validation within maker-checker workflow",
                "validation_requirements": ["Regulatory compliance", "AML validation", "Sanctions screening"],
                "workflow_type": "compliance_validation"
            }
        ]
    
    def _validate_maker_checker_test_cases(self, test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """FIXED: Validate that all test cases include maker-checker validation"""
        
        validated_cases = []
        required_fields = [
            "User Story ID", "Acceptance Criteria ID", "Scenario", "Test Case ID",
            "Test Case Description", "Precondition", "Steps", "Expected Result",
            "Part of Regression", "Priority"
        ]
        
        for case in test_cases:
            try:
                validated_case = {}
                
                # Ensure all required fields exist
                for field in required_fields:
                    validated_case[field] = case.get(field, "").strip()
                
                # FIXED: Ensure description includes maker-checker validation
                description = validated_case["Test Case Description"]
                if not self._has_maker_checker_validation(description):
                    description = self._add_maker_checker_to_description(description)
                    validated_case["Test Case Description"] = description
                
                # FIXED: Ensure steps include maker-checker workflow
                steps = validated_case["Steps"]
                if not self._has_maker_checker_steps(steps):
                    steps = self._add_maker_checker_to_steps(steps)
                    validated_case["Steps"] = steps
                
                # FIXED: Ensure expected result includes maker-checker validation
                expected = validated_case["Expected Result"]
                if not self._has_maker_checker_validation(expected):
                    expected = self._add_maker_checker_to_expected(expected)
                    validated_case["Expected Result"] = expected
                
                # Set defaults for banking scenarios
                if not validated_case["Priority"]:
                    validated_case["Priority"] = "High"
                if not validated_case["Part of Regression"]:
                    validated_case["Part of Regression"] = "Yes"
                
                validated_cases.append(validated_case)
                
            except Exception as e:
                logger.warning(f"FIXED: Skipping invalid test case: {str(e)}")
                continue
        
        logger.info(f"FIXED: Validated {len(validated_cases)} maker-checker test cases")
        return validated_cases
    
    def _has_maker_checker_validation(self, text: str) -> bool:
        """Check if text includes maker-checker validation terms"""
        text_lower = text.lower()
        
        maker_terms = ["maker creates", "maker inputs", "ops user maker", "maker submits"]
        checker_terms = ["checker validates", "checker reviews", "checker approves", "ops user checker", "checker verifies"]
        validation_terms = ["validation", "approval", "review", "verify"]
        
        has_maker = any(term in text_lower for term in maker_terms)
        has_checker = any(term in text_lower for term in checker_terms)
        has_validation = any(term in text_lower for term in validation_terms)
        
        return has_maker and has_checker and has_validation
    
    def _has_maker_checker_steps(self, steps: str) -> bool:
        """Check if steps include maker-checker workflow"""
        steps_lower = steps.lower()
        
        return ("login as" in steps_lower and "maker" in steps_lower and 
                "checker" in steps_lower and "approve" in steps_lower)
    
    def _add_maker_checker_to_description(self, description: str) -> str:
        """Add maker-checker validation to description"""
        if "payment" in description.lower():
            return f"Verify that when Ops User maker creates PACS.008 payment, the system requires Ops User checker to validate all payment fields and approve before processing. {description}"
        else:
            return f"Verify that Ops User maker and Ops User checker workflow validates the requirement: {description}"
    
    def _add_maker_checker_to_steps(self, steps: str) -> str:
        """Add maker-checker workflow to steps"""
        if "login" not in steps.lower():
            enhanced_steps = "1. Login as Ops User maker\\n2. Create/input required data\\n3. Submit for checker approval\\n"
            enhanced_steps += "4. Login as Ops User checker\\n5. Review maker inputs\\n6. Validate accuracy\\n7. Approve"
            return enhanced_steps
        return steps
    
    def _add_maker_checker_to_expected(self, expected: str) -> str:
        """Add maker-checker validation to expected result"""
        return f"Maker successfully creates/inputs data. Checker can review all maker inputs, validate accuracy, and approve. {expected}"
    
    def _emergency_maker_checker_fallback(self, content: str, instructions: str) -> List[Dict[str, Any]]:
        """Emergency fallback with maker-checker focus"""
        logger.warning("FIXED: Using emergency maker-checker fallback")
        
        num_cases = self._extract_test_case_count(instructions)
        
        emergency_story = {
            "id": "PACS001",
            "title": "Emergency PACS.008 Maker-Checker Workflow",
            "content": "Maker-checker validation workflow for PACS.008 payment processing"
        }
        
        return self._create_maker_checker_fallback_tests(emergency_story, num_cases)
    
    def _extract_test_case_count(self, custom_instructions: str) -> int:
        """Extract number of test cases from instructions"""
        patterns = [
            r'exactly\s+(\d+)\s+test\s+cases?\s+per\s+story',
            r'(\d+)\s+test\s+cases?\s+per\s+story',
            r'generate\s+(\d+)\s+test\s+cases?',
            r'(\d+)\s+test\s+cases?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, custom_instructions.lower())
            if match:
                return int(match.group(1))
        
        return 8  # Default
