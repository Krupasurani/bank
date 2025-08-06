# src/ai_engine/enhanced_test_generator.py
"""
Enhanced Test Case Generator with PACS.008 Intelligence
Extends your existing test generator with banking domain intelligence
"""

import json
import re
import logging
from typing import Dict, List, Any, Optional
from ai_engine.test_generator import TestCaseGenerator

logger = logging.getLogger(__name__)

class EnhancedTestCaseGenerator(TestCaseGenerator):
    """Enhanced version of your existing test generator with PACS.008 intelligence"""
    
    def __init__(self, api_key: str):
        """Initialize with your existing generator as base"""
        super().__init__(api_key)
        
        # Try to initialize PACS.008 detector
        try:
            from processors.pacs008_intelligent_detector import PACS008LLMDetector
            self.pacs008_detector = PACS008LLMDetector(api_key)
            self.pacs008_available = True
            logger.info("Enhanced test generator with PACS.008 intelligence initialized")
        except ImportError:
            self.pacs008_available = False
            logger.info("PACS.008 detector not available - using standard generation")
    
    def generate_enhanced_test_cases(self, content: str, custom_instructions: str = "", pacs008_pdf_content: str = None) -> List[Dict[str, Any]]:
        """
        Generate test cases with PACS.008 intelligence if available
        Falls back to your existing generation if PACS.008 not available
        """
        
        try:
            # Check if we should use PACS.008 enhancement
            if self.pacs008_available:
                
                # Update PACS.008 knowledge if PDF content provided
                if pacs008_pdf_content:
                    self.pacs008_detector.pacs008_knowledge = pacs008_pdf_content
                
                # Run PACS.008 analysis
                pacs008_analysis = self.pacs008_detector.detect_pacs008_fields(content)
                
                # If PACS.008 content detected, use enhanced generation
                if pacs008_analysis.get('is_pacs008_related', False):
                    logger.info("PACS.008 content detected - using enhanced generation")
                    return self._generate_pacs008_enhanced_tests(content, custom_instructions, pacs008_analysis)
            
            # Use your existing standard generation
            logger.info("Using standard test case generation")
            return super().generate_test_cases(content, custom_instructions)
            
        except Exception as e:
            logger.error(f"Enhanced generation failed: {str(e)}")
            # Fallback to your existing generation
            return super().generate_test_cases(content, custom_instructions)
    
    def _generate_pacs008_enhanced_tests(self, content: str, custom_instructions: str, pacs008_analysis: Dict) -> List[Dict[str, Any]]:
        """Generate PACS.008-enhanced test cases"""
        
        detected_fields = pacs008_analysis.get('detected_fields', [])
        test_recommendations = pacs008_analysis.get('test_recommendations', [])
        
        # Create enhanced instructions
        enhanced_instructions = self._create_enhanced_instructions(
            custom_instructions, detected_fields, test_recommendations
        )
        
        # Generate enhanced test cases using LLM
        enhanced_test_cases = self._generate_intelligent_tests(content, enhanced_instructions, detected_fields)
        
        # Generate some standard test cases too
        standard_test_cases = super().generate_test_cases(content, custom_instructions)
        
        # Combine and prioritize
        all_test_cases = self._combine_test_cases(enhanced_test_cases, standard_test_cases)
        
        logger.info(f"Generated {len(all_test_cases)} enhanced test cases ({len(enhanced_test_cases)} PACS.008-specific)")
        
        return all_test_cases
    
    def _create_enhanced_instructions(self, custom_instructions: str, detected_fields: List[Dict], recommendations: List[str]) -> str:
        """Create enhanced generation instructions"""
        
        instructions = [custom_instructions] if custom_instructions else []
        
        # Add PACS.008-specific instructions
        instructions.append("PACS.008 ENHANCEMENT APPLIED:")
        instructions.append(f"- {len(detected_fields)} PACS.008 fields detected in requirements")
        
        # Add field-specific instructions
        field_types = {}
        for field in detected_fields:
            field_name = field.get('field_name', '')
            if 'agent' in field_name.lower():
                field_types.setdefault('agents', []).append(field_name)
            elif 'amount' in field_name.lower():
                field_types.setdefault('amounts', []).append(field_name)
            elif 'account' in field_name.lower():
                field_types.setdefault('accounts', []).append(field_name)
            elif any(party in field_name.lower() for party in ['debtor', 'creditor', 'customer']):
                field_types.setdefault('parties', []).append(field_name)
        
        if field_types.get('agents'):
            instructions.append(f"- Focus on banking agent validation: {', '.join(field_types['agents'][:3])}")
        if field_types.get('accounts'):
            instructions.append(f"- Include account validation scenarios: {', '.join(field_types['accounts'][:2])}")
        if field_types.get('amounts'):
            instructions.append(f"- Test payment amount scenarios: {', '.join(field_types['amounts'][:2])}")
        
        # Add recommendations
        instructions.extend(recommendations[:3])
        
        # Add banking-specific guidance
        instructions.extend([
            "- Use realistic banking data (BICs like DEUTDEFF, IBANs like DE89370400440532013000)",
            "- Include both positive and negative validation scenarios",
            "- Focus on business rules and regulatory compliance",
            "- Generate cross-border payment scenarios where applicable"
        ])
        
        return "\n".join(instructions)
    
    def _generate_intelligent_tests(self, content: str, enhanced_instructions: str, detected_fields: List[Dict]) -> List[Dict[str, Any]]:
        """Generate intelligent PACS.008-specific test cases"""
        
        # Create field context for test generation
        field_context = []
        for field in detected_fields[:5]:  # Limit for token efficiency
            field_context.append(f"- {field.get('field_name')}: {field.get('extracted_value', 'To be tested')} (Confidence: {field.get('confidence')})")
        
        prompt = f"""
        Generate PACS.008-enhanced test cases based on the detected banking fields.

        ORIGINAL CONTENT:
        {content[:1500]}

        DETECTED PACS.008 FIELDS:
        {chr(10).join(field_context)}

        ENHANCED INSTRUCTIONS:
        {enhanced_instructions}

        REQUIREMENTS:
        1. Generate 6-8 test cases that are specifically relevant to the detected PACS.008 fields
        2. Use realistic banking data based on what was detected
        3. Focus on business scenarios, not just technical validation
        4. Include field validation, business rules, and integration scenarios
        5. Make test cases practical and executable

        Generate test cases with these EXACT fields:
        - User Story ID: Extract from content or generate (US001, US002, etc.)
        - Acceptance Criteria ID: Generate (AC001, AC002, etc.)
        - Scenario: Business-focused scenario name
        - Test Case ID: Generate (TC001, TC002, etc.)
        - Test Case Description: Clear description focusing on PACS.008 aspects
        - Precondition: Include PACS.008-specific preconditions
        - Steps: Detailed steps using realistic banking data from detected fields
        - Expected Result: PACS.008-aware expected outcomes
        - Part of Regression: Yes for mandatory fields, No for edge cases
        - Priority: High for mandatory fields, Medium/Low for optional

        FOCUS ON:
        - Field format validation for detected fields
        - Business rule compliance
        - Cross-field validation and dependencies
        - Realistic banking scenarios

        Respond with ONLY a JSON array of test cases:
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert PACS.008 test engineer. Generate business-relevant test cases based on detected fields. Respond with ONLY valid JSON array."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=3500
            )
            
            result = response.choices[0].message.content.strip()
            json_match = re.search(r'\[.*\]', result, re.DOTALL)
            
            if json_match:
                test_cases = json.loads(json_match.group())
                
                # Add PACS.008 enhancement markers
                for test_case in test_cases:
                    test_case['PACS008_Enhanced'] = 'Yes'
                    test_case['Enhancement_Type'] = 'PACS008_Intelligent'
                
                return test_cases
            else:
                logger.warning("Could not extract JSON from PACS.008 test generation")
                return []
                
        except Exception as e:
            logger.error(f"PACS.008 intelligent test generation failed: {str(e)}")
            return []
    
    def _combine_test_cases(self, enhanced_tests: List[Dict], standard_tests: List[Dict]) -> List[Dict[str, Any]]:
        """Combine PACS.008-enhanced and standard test cases"""
        
        all_tests = []
        
        # Add PACS.008-enhanced test cases first (higher priority)
        for test in enhanced_tests:
            test['Generation_Method'] = 'PACS008_Enhanced'
            all_tests.append(test)
        
        # Add complementary standard test cases (avoid duplicates)
        added_standard = 0
        for standard_test in standard_tests:
            # Simple duplicate check
            is_duplicate = any(
                self._tests_are_similar(standard_test.get('Test Case Description', ''), 
                                       enhanced_test.get('Test Case Description', ''))
                for enhanced_test in enhanced_tests
            )
            
            if not is_duplicate and added_standard < 4:  # Limit standard tests
                standard_test['Generation_Method'] = 'Standard'
                standard_test['PACS008_Enhanced'] = 'No'
                all_tests.append(standard_test)
                added_standard += 1
        
        return all_tests
    
    def _tests_are_similar(self, desc1: str, desc2: str) -> bool:
        """Check if two test descriptions are similar"""
        if not desc1 or not desc2:
            return False
        
        # Simple similarity check
        words1 = set(desc1.lower().split())
        words2 = set(desc2.lower().split())
        
        if len(words1) == 0 or len(words2) == 0:
            return False
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union) if union else 0
        return similarity > 0.5  # 50% similarity threshold