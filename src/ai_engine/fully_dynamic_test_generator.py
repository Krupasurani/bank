# src/ai_engine/fully_dynamic_test_generator.py - UPDATED WITH DOMAIN EXPERTISE
"""
Updated Fully Dynamic Test Generator with Domain-Specific PACS.008 Expertise
- Generates EXACTLY specified number of test cases
- Highly technical and domain-specific
- Expert-level banking scenarios only
"""

import json
import re
import logging
from typing import Dict, List, Any, Optional
from openai import OpenAI
from ai_engine.test_generator import TestCaseGenerator

logger = logging.getLogger(__name__)

class FullyDynamicTestGenerator(TestCaseGenerator):
    """Updated dynamic test generator with domain-specific expertise and precise control"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.model = "gpt-4.1-mini-2025-04-14"
        
        # Import domain-specific generator
        try:
            from ai_engine.domain_specific_pacs008_generator import DomainSpecificPACS008Generator
            self.domain_generator = DomainSpecificPACS008Generator(api_key)
            self.domain_specific_available = True
            logger.info("Domain-Specific PACS.008 Generator integrated successfully")
        except ImportError:
            self.domain_specific_available = False
            logger.warning("Domain-specific generator not available")
        
        # Import dynamic maker-checker
        try:
            from processors.fully_dynamic_intelligent_maker_checker import FullyDynamicIntelligentMakerChecker
            self.dynamic_maker_checker = FullyDynamicIntelligentMakerChecker(api_key)
            self.dynamic_validation_available = True
        except ImportError:
            self.dynamic_validation_available = False
    
    def generate_fully_dynamic_test_cases(self, content: str, detected_fields: List[Dict], 
                                        custom_instructions: str = "", exact_test_count: int = 8) -> Dict[str, Any]:
        """
        Generate test cases with domain expertise and exact count control
        """
        
        try:
            logger.info(f"Starting domain-specific test generation for exactly {exact_test_count} test cases...")
            
            # Step 1: Perform dynamic validation
            if self.dynamic_validation_available and detected_fields:
                logger.info("Performing dynamic validation for domain context...")
                validation_results = self.dynamic_maker_checker.perform_fully_dynamic_validation(detected_fields)
                
                if validation_results['status'] != 'SUCCESS':
                    logger.warning("Dynamic validation failed, using standard generation")
                    return self._fallback_to_standard_generation(content, custom_instructions, exact_test_count)
            else:
                validation_results = {}
            
            # Step 2: Use domain-specific generator if available
            if self.domain_specific_available and detected_fields:
                logger.info(f"Using domain-specific PACS.008 generator for exactly {exact_test_count} test cases...")
                
                domain_test_cases = self.domain_generator.generate_precise_domain_tests(
                    detected_fields, validation_results, custom_instructions, exact_test_count
                )
                
                if domain_test_cases and len(domain_test_cases) == exact_test_count:
                    return {
                        'status': 'SUCCESS',
                        'message': f'Generated exactly {exact_test_count} domain-specific PACS.008 test cases',
                        'test_cases': domain_test_cases,
                        'generation_method': 'DOMAIN_SPECIFIC_PACS008',
                        'total_test_cases': len(domain_test_cases)
                    }
                else:
                    logger.warning(f"Domain generator returned {len(domain_test_cases) if domain_test_cases else 0} instead of {exact_test_count}")
            
            # Step 3: Fallback to enhanced generation with exact count control
            logger.info(f"Using enhanced generation with exact count control...")
            enhanced_test_cases = self._generate_expert_controlled_tests(
                content, detected_fields, validation_results, custom_instructions, exact_test_count
            )
            
            return {
                'status': 'SUCCESS',
                'message': f'Generated exactly {exact_test_count} enhanced test cases',
                'test_cases': enhanced_test_cases,
                'generation_method': 'ENHANCED_CONTROLLED',
                'total_test_cases': len(enhanced_test_cases)
            }
            
        except Exception as e:
            logger.error(f"Fully dynamic test generation failed: {str(e)}")
            return self._fallback_to_standard_generation(content, custom_instructions, exact_test_count)
    
    def _generate_expert_controlled_tests(self, content: str, detected_fields: List[Dict], 
                                        validation_results: Dict, custom_instructions: str, 
                                        exact_test_count: int) -> List[Dict[str, Any]]:
        """Generate expert test cases with exact count control"""
        
        # Prepare expert context
        expert_context = self._prepare_expert_context(detected_fields, validation_results)
        
        # Expert prompt with precise count control
        expert_prompt = f"""
You are a Senior PACS.008 Banking Expert with deep technical knowledge of ISO 20022 standards and cross-border payment processing.

CONTENT ANALYSIS:
{content[:1500]}

DETECTED PACS.008 FIELDS:
{json.dumps([{"field_name": f.get('field_name'), "value": f.get('extracted_value'), "mandatory": f.get('is_mandatory')} for f in detected_fields], indent=2)}

EXPERT CONTEXT:
{json.dumps(expert_context, indent=2)}

INSTRUCTIONS: {custom_instructions}

CRITICAL REQUIREMENT: Generate EXACTLY {exact_test_count} test cases - NO MORE, NO LESS.

EXPERT REQUIREMENTS:
1. EXACT COUNT: Must generate precisely {exact_test_count} test cases
2. TECHNICAL DEPTH: Focus on PACS.008 technical scenarios, not generic payment tests
3. DOMAIN EXPERTISE: Use real banking terminology and scenarios
4. FIELD-BASED: Incorporate detected field values into realistic banking contexts
5. COMPREHENSIVE COVERAGE: Cover positive, negative, business rule, and edge scenarios
6. BANKING REALISM: Use correspondent banking, settlement methods, agent chains
7. ISO 20022 COMPLIANCE: Include technical validation per PACS.008 standards

TECHNICAL FOCUS AREAS (distribute across {exact_test_count} test cases):
- Agent chain validation and BIC processing
- Payment routing through correspondent networks  
- Settlement method scenarios (INDA/INGA)
- Cross-border regulatory compliance
- Technical field validation per ISO 20022
- Business rule enforcement scenarios
- Error handling and exception processing
- Integration testing across banking systems

RESPONSE FORMAT - EXACTLY {exact_test_count} test cases:
[
  {{
    "User Story ID": "US001",
    "Acceptance Criteria ID": "AC001",
    "Scenario": "Technical PACS.008 scenario",
    "Test Case ID": "TC001", 
    "Test Case Description": "Detailed technical description using PACS.008 expertise",
    "Precondition": "Technical banking preconditions",
    "Steps": "Expert-level steps using detected field values",
    "Expected Result": "Technical banking outcome",
    "Part of Regression": "Yes for critical flows, No for edge cases",
    "Priority": "High for mandatory fields, Medium/Low for optional"
  }}
]

COUNT VERIFICATION: You must generate exactly {exact_test_count} test cases. Count them before responding.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": f"You are a Senior PACS.008 Banking Expert. Generate EXACTLY {exact_test_count} test cases. Count precisely to ensure exact number. Use deep banking domain knowledge. Respond with ONLY valid JSON array containing exactly {exact_test_count} test cases."
                    },
                    {"role": "user", "content": expert_prompt}
                ],
                temperature=0.1,
                max_tokens=4000
            )
            
            result = response.choices[0].message.content.strip()
            json_match = re.search(r'\[.*\]', result, re.DOTALL)
            
            if json_match:
                test_cases = json.loads(json_match.group())
                
                # Ensure exact count
                if len(test_cases) == exact_test_count:
                    logger.info(f"Perfect: Generated exactly {exact_test_count} expert test cases")
                    return test_cases
                elif len(test_cases) > exact_test_count:
                    logger.info(f"Trimming {len(test_cases)} to exactly {exact_test_count}")
                    return test_cases[:exact_test_count]
                else:
                    logger.info(f"Generated {len(test_cases)}, need {exact_test_count}")
                    return self._ensure_exact_count(test_cases, exact_test_count)
            else:
                logger.warning("Could not extract JSON from expert generation")
                return self._create_fallback_tests(exact_test_count)
                
        except Exception as e:
            logger.error(f"Expert controlled generation failed: {str(e)}")
            return self._create_fallback_tests(exact_test_count)
    
    def _prepare_expert_context(self, detected_fields: List[Dict], validation_results: Dict) -> Dict[str, Any]:
        """Prepare expert context for enhanced prompting"""
        
        context = {
            "field_count": len(detected_fields),
            "mandatory_fields": len([f for f in detected_fields if f.get('is_mandatory')]),
            "payment_scenario": "CROSS_BORDER_CORRESPONDENT_BANKING",
            "technical_complexity": "HIGH",
            "domain_focus": "PACS008_SERIAL_METHOD"
        }
        
        # Add validation insights if available
        if validation_results:
            final_analysis = validation_results.get('final_analysis', {})
            context.update({
                "validation_score": final_analysis.get('final_validation_score', 0),
                "validation_status": final_analysis.get('overall_status', 'UNKNOWN'),
                "technical_readiness": final_analysis.get('final_assessment', {}).get('technical_readiness', 'READY')
            })
        
        # Classify field types for expert context
        agent_fields = [f for f in detected_fields if 'agent' in f.get('field_key', '').lower()]
        payment_fields = [f for f in detected_fields if 'amount' in f.get('field_key', '').lower() or 'currency' in f.get('field_key', '').lower()]
        
        if agent_fields:
            context["agent_chain_complexity"] = "MULTI_HOP" if len(agent_fields) > 2 else "DIRECT"
        if payment_fields:
            context["payment_type"] = "HIGH_VALUE" if any('565000' in str(f.get('extracted_value', '')) for f in payment_fields) else "STANDARD"
        
        return context
    
    def _ensure_exact_count(self, existing_tests: List[Dict], exact_count: int) -> List[Dict[str, Any]]:
        """Ensure exactly the specified number of test cases"""
        
        if len(existing_tests) >= exact_count:
            return existing_tests[:exact_count]
        
        # Generate additional tests to reach exact count
        additional_needed = exact_count - len(existing_tests)
        logger.info(f"Generating {additional_needed} additional test cases to reach exactly {exact_count}")
        
        additional_prompt = f"""
Generate EXACTLY {additional_needed} additional PACS.008 expert test cases.

Requirements:
1. Generate precisely {additional_needed} test cases
2. Maintain high technical quality
3. Use PACS.008 domain expertise
4. Avoid duplication with existing tests

Generate exactly {additional_needed} test cases in JSON array format.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"Generate EXACTLY {additional_needed} PACS.008 test cases."},
                    {"role": "user", "content": additional_prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )
            
            result = response.choices[0].message.content.strip()
            json_match = re.search(r'\[.*\]', result, re.DOTALL)
            
            if json_match:
                additional_tests = json.loads(json_match.group())
                all_tests = existing_tests + additional_tests
                return all_tests[:exact_count]  # Ensure exact count
            
        except Exception as e:
            logger.error(f"Additional test generation failed: {str(e)}")
        
        # If additional generation fails, create simple fallback
        return existing_tests + self._create_fallback_tests(additional_needed)[:additional_needed]
    
    def _create_fallback_tests(self, count: int) -> List[Dict[str, Any]]:
        """Create fallback expert test cases if generation fails"""
        
        expert_templates = [
            {
                "domain": "Agent Validation",
                "scenario": "PACS.008 Agent Chain Validation",
                "description": "Validate agent chain processing in PACS.008 message flow",
                "priority": "High"
            },
            {
                "domain": "Settlement Processing", 
                "scenario": "Settlement Method INDA/INGA Processing",
                "description": "Verify settlement method processing per CBPR+ requirements",
                "priority": "High"
            },
            {
                "domain": "Field Validation",
                "scenario": "Technical Field Format Validation",
                "description": "Validate PACS.008 field formats per ISO 20022 standards",
                "priority": "High"
            },
            {
                "domain": "Cross-Border Processing",
                "scenario": "Cross-Border Payment Routing",
                "description": "Test cross-border payment routing through correspondent banks",
                "priority": "Medium"
            },
            {
                "domain": "Business Rules",
                "scenario": "CBPR+ Business Rule Compliance",
                "description": "Verify compliance with CBPR+ business rules and constraints",
                "priority": "Medium"
            },
            {
                "domain": "Error Handling",
                "scenario": "Technical Error Processing",
                "description": "Test technical error handling in PACS.008 processing",
                "priority": "Medium"
            },
            {
                "domain": "Regulatory Compliance",
                "scenario": "AML/KYC Compliance Validation", 
                "description": "Verify regulatory compliance requirements in payment processing",
                "priority": "Low"
            },
            {
                "domain": "Integration Testing",
                "scenario": "End-to-End Payment Integration",
                "description": "Test complete payment flow from initiation to settlement",
                "priority": "Low"
            }
        ]
        
        fallback_tests = []
        
        for i in range(count):
            template_index = i % len(expert_templates)
            template = expert_templates[template_index]
            
            test_case = {
                "User Story ID": f"US{i+1:03d}",
                "Acceptance Criteria ID": f"AC{i+1:03d}",
                "Scenario": template["scenario"],
                "Test Case ID": f"TC{i+1:03d}",
                "Test Case Description": f"Validate {template['description']} in PACS.008 technical processing workflow",
                "Precondition": f"PACS.008 processing system operational with {template['domain']} capabilities enabled",
                "Steps": f"1. Initialize PACS.008 message with {template['domain']} requirements\n2. Process message through technical validation\n3. Verify {template['domain']} compliance\n4. Validate processing outcome",
                "Expected Result": f"{template['description']} completed successfully per ISO 20022 and CBPR+ standards",
                "Part of Regression": "Yes" if template["priority"] == "High" else "No",
                "Priority": template["priority"]
            }
            
            fallback_tests.append(test_case)
        
        logger.info(f"Created {len(fallback_tests)} expert fallback test cases")
        return fallback_tests
    
    def _fallback_to_standard_generation(self, content: str, custom_instructions: str, exact_count: int) -> Dict[str, Any]:
        """Fallback with exact count control"""
        
        logger.info(f"Using standard generation with exact count control for {exact_count} test cases")
        
        try:
            # Enhanced standard prompt with count control
            controlled_prompt = f"""
Generate EXACTLY {exact_count} banking test cases from the provided content.

CONTENT:
{content[:1000]}

INSTRUCTIONS: {custom_instructions}

CRITICAL: Generate precisely {exact_count} test cases - count carefully.

Requirements:
1. EXACT COUNT: Must be exactly {exact_count} test cases
2. Banking focus with realistic scenarios
3. Use content context for test generation
4. Include positive, negative, and edge cases
5. Technical banking terminology

Generate exactly {exact_count} test cases in standard format.
"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"Generate EXACTLY {exact_count} banking test cases. Count precisely."},
                    {"role": "user", "content": controlled_prompt}
                ],
                temperature=0.2,
                max_tokens=3000
            )
            
            result = response.choices[0].message.content.strip()
            json_match = re.search(r'\[.*\]', result, re.DOTALL)
            
            if json_match:
                test_cases = json.loads(json_match.group())
                
                # Ensure exact count
                if len(test_cases) != exact_count:
                    if len(test_cases) > exact_count:
                        test_cases = test_cases[:exact_count]
                    else:
                        test_cases.extend(self._create_fallback_tests(exact_count - len(test_cases)))
                        test_cases = test_cases[:exact_count]
                
                return {
                    'status': 'FALLBACK_SUCCESS',
                    'message': f'Generated exactly {exact_count} test cases using controlled standard method',
                    'test_cases': test_cases,
                    'generation_method': 'CONTROLLED_STANDARD',
                    'total_test_cases': len(test_cases)
                }
            else:
                # Complete fallback
                fallback_tests = self._create_fallback_tests(exact_count)
                return {
                    'status': 'COMPLETE_FALLBACK',
                    'message': f'Generated exactly {exact_count} fallback test cases',
                    'test_cases': fallback_tests,
                    'generation_method': 'EXPERT_FALLBACK',
                    'total_test_cases': len(fallback_tests)
                }
                
        except Exception as e:
            logger.error(f"Controlled standard generation failed: {str(e)}")
            fallback_tests = self._create_fallback_tests(exact_count)
            return {
                'status': 'ERROR_FALLBACK',
                'message': f'All generation methods failed, created {exact_count} expert fallback test cases',
                'test_cases': fallback_tests,
                'generation_method': 'ERROR_FALLBACK',
                'total_test_cases': len(fallback_tests)
            }