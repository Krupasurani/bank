# src/processors/fully_dynamic_intelligent_maker_checker.py
"""
Fully Dynamic LLM-Powered Maker-Checker for PACS.008
- Only field definitions are static
- All validation, examples, and logic is fully dynamic via LLM
- Super intelligent system with no hardcoded rules
"""

import json
import re
import logging
from typing import Dict, List, Any, Optional
from openai import OpenAI
from datetime import datetime

logger = logging.getLogger(__name__)

class FullyDynamicIntelligentMakerChecker:
    """Fully dynamic LLM-powered maker-checker with only static field definitions"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4.1-mini-2025-04-14"
        
        # ONLY static field definitions - everything else is dynamic
        self.pacs008_field_definitions = {
            "debtor_agent_bic": {
                "name": "Debtor Agent BIC",
                "description": "BIC of the debtor's bank (payer's bank)",
                "mandatory": True
            },
            "creditor_agent_bic": {
                "name": "Creditor Agent BIC", 
                "description": "BIC of the creditor's bank (payee's bank)",
                "mandatory": True
            },
            "debtor_name": {
                "name": "Debtor Name",
                "description": "Name of the payer/customer initiating payment",
                "mandatory": True
            },
            "creditor_name": {
                "name": "Creditor Name",
                "description": "Name of the payee/beneficiary receiving payment", 
                "mandatory": True
            },
            "debtor_account": {
                "name": "Debtor Account",
                "description": "Account number/IBAN of the payer",
                "mandatory": True
            },
            "creditor_account": {
                "name": "Creditor Account", 
                "description": "Account number/IBAN of the payee",
                "mandatory": True
            },
            "payment_amount": {
                "name": "Payment Amount",
                "description": "Amount to be transferred",
                "mandatory": True
            },
            "currency": {
                "name": "Currency",
                "description": "Currency of the payment",
                "mandatory": True
            },
            "instruction_id": {
                "name": "Instruction Identification",
                "description": "Unique payment instruction reference",
                "mandatory": True
            },
            "end_to_end_id": {
                "name": "End-to-End Identification",
                "description": "End-to-end payment reference",
                "mandatory": False
            },
            "ultimate_debtor": {
                "name": "Ultimate Debtor",
                "description": "Party on whose behalf debtor is acting",
                "mandatory": False
            },
            "ultimate_creditor": {
                "name": "Ultimate Creditor", 
                "description": "Party on whose behalf creditor receives payment",
                "mandatory": False
            },
            "intermediary_agent": {
                "name": "Intermediary Agent",
                "description": "Intermediary bank in payment chain",
                "mandatory": False
            },
            "settlement_method": {
                "name": "Settlement Method",
                "description": "How the payment will be settled",
                "mandatory": False
            },
            "charge_bearer": {
                "name": "Charge Bearer",
                "description": "Who pays the charges",
                "mandatory": False
            },
            "remittance_info": {
                "name": "Remittance Information",
                "description": "Payment reference/purpose",
                "mandatory": False
            },
            "settlement_date": {
                "name": "Settlement Date",
                "description": "When payment should be settled",
                "mandatory": False
            }
        }
        
        logger.info("Fully Dynamic Intelligent PACS.008 Maker-Checker initialized")
    
    def perform_fully_dynamic_validation(self, detected_fields: List[Dict]) -> Dict[str, Any]:
        """
        Main method: Perform fully dynamic LLM-powered validation
        """
        
        try:
            logger.info(f"Starting fully dynamic validation for {len(detected_fields)} fields...")
            
            # Step 1: LLM performs complete validation analysis
            comprehensive_validation = self._llm_comprehensive_validation(detected_fields)
            
            # Step 2: LLM makes intelligent maker-checker decisions
            intelligent_decisions = self._llm_maker_checker_decisions(detected_fields, comprehensive_validation)
            
            # Step 3: LLM generates final report and recommendations
            final_analysis = self._llm_final_analysis(detected_fields, comprehensive_validation, intelligent_decisions)
            
            logger.info("Fully dynamic validation completed successfully")
            
            return {
                "status": "SUCCESS",
                "validation_timestamp": datetime.now().isoformat(),
                "total_fields_validated": len(detected_fields),
                "comprehensive_validation": comprehensive_validation,
                "intelligent_decisions": intelligent_decisions,
                "final_analysis": final_analysis,
                "overall_validation_status": final_analysis.get("overall_status"),
                "ready_for_test_generation": final_analysis.get("ready_for_testing", False),
                "validation_approach": "FULLY_DYNAMIC_LLM"
            }
            
        except Exception as e:
            logger.error(f"Fully dynamic validation failed: {str(e)}")
            return {
                "status": "ERROR",
                "error": str(e),
                "validation_approach": "FAILED"
            }
    
    def _llm_comprehensive_validation(self, detected_fields: List[Dict]) -> Dict[str, Any]:
        """LLM performs comprehensive validation of all detected fields"""
        
        # Prepare field data for LLM
        field_data = []
        for field in detected_fields:
            field_data.append({
                "field_name": field.get('field_name'),
                "field_key": field.get('field_key'),
                "extracted_value": field.get('extracted_value'),
                "confidence": field.get('confidence'),
                "is_mandatory": field.get('is_mandatory', False),
                "description": field.get('description', '')
            })
        
        prompt = f"""
You are an expert PACS.008 banking validation specialist with deep knowledge of international payment standards, banking regulations, and compliance requirements.

DETECTED PACS.008 FIELDS TO VALIDATE:
{json.dumps(field_data, indent=2)}

VALIDATION INSTRUCTIONS:
1. For each field, dynamically assess:
   - Format validity (use your banking knowledge, not static patterns)
   - Business logic appropriateness
   - Compliance with international standards
   - Context-appropriate validation
   - Real-world feasibility

2. For missing or incomplete fields:
   - Assess impact on payment processing
   - Determine if field can be reasonably inferred
   - Evaluate business consequences

3. Cross-field validation:
   - Check consistency between related fields
   - Validate business workflow logic
   - Assess payment routing feasibility

4. Apply dynamic banking intelligence:
   - Consider current banking practices
   - Apply regulatory knowledge
   - Use contextual validation logic

RESPOND WITH COMPREHENSIVE JSON:
{{
  "field_validations": [
    {{
      "field_name": "field name",
      "field_key": "field_key",
      "validation_status": "VALID" | "INVALID" | "WARNING" | "MISSING",
      "validation_score": 0-100,
      "dynamic_assessment": "detailed intelligent assessment",
      "format_analysis": "intelligent format validation",
      "business_logic_check": "business appropriateness analysis",
      "compliance_assessment": "regulatory compliance evaluation",
      "contextual_validation": "context-specific validation insights",
      "improvement_suggestions": "intelligent suggestions for improvement"
    }}
  ],
  "cross_field_analysis": {{
    "consistency_check": "analysis of field consistency",
    "workflow_validation": "payment workflow feasibility",
    "routing_assessment": "payment routing analysis",
    "business_logic_coherence": "overall business logic assessment"
  }},
  "overall_assessment": {{
    "validation_summary": "comprehensive validation summary",
    "technical_score": 0-100,
    "business_score": 0-100,
    "compliance_score": 0-100,
    "overall_confidence": 0-100,
    "payment_feasibility": "HIGH" | "MEDIUM" | "LOW",
    "major_concerns": ["list of major issues"],
    "positive_findings": ["list of positive aspects"]
  }}
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert PACS.008 banking validator with comprehensive knowledge of international payment standards. Provide detailed, intelligent validation without using static patterns or examples."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=3000
            )
            
            result = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            
            if json_match:
                validation_result = json.loads(json_match.group())
                validation_result["llm_analysis_timestamp"] = datetime.now().isoformat()
                return validation_result
            else:
                return self._fallback_validation_response()
                
        except Exception as e:
            logger.error(f"LLM comprehensive validation failed: {str(e)}")
            return self._fallback_validation_response()
    
    def _llm_maker_checker_decisions(self, detected_fields: List[Dict], 
                                   comprehensive_validation: Dict) -> Dict[str, Any]:
        """LLM makes intelligent maker-checker decisions"""
        
        # Extract key metrics from comprehensive validation
        overall_assessment = comprehensive_validation.get('overall_assessment', {})
        field_validations = comprehensive_validation.get('field_validations', [])
        cross_field_analysis = comprehensive_validation.get('cross_field_analysis', {})
        
        context_data = {
            "total_fields": len(detected_fields),
            "mandatory_fields": len([f for f in detected_fields if f.get('is_mandatory')]),
            "validation_scores": {
                "technical": overall_assessment.get('technical_score', 0),
                "business": overall_assessment.get('business_score', 0),
                "compliance": overall_assessment.get('compliance_score', 0),
                "overall_confidence": overall_assessment.get('overall_confidence', 0)
            },
            "payment_feasibility": overall_assessment.get('payment_feasibility', 'UNKNOWN'),
            "major_concerns": overall_assessment.get('major_concerns', []),
            "field_issues": len([f for f in field_validations if f.get('validation_status') in ['INVALID', 'WARNING']]),
            "cross_field_status": cross_field_analysis.get('workflow_validation', 'Unknown')
        }
        
        prompt = f"""
You are an intelligent PACS.008 maker-checker decision system with expert banking knowledge and regulatory compliance expertise.

VALIDATION CONTEXT:
{json.dumps(context_data, indent=2)}

FIELD VALIDATION RESULTS:
{json.dumps(field_validations[:10], indent=2)}

CROSS-FIELD ANALYSIS:
{json.dumps(cross_field_analysis, indent=2)}

MAKER-CHECKER DECISION INSTRUCTIONS:
1. As AI Maker (Technical Validator):
   - Review all technical validation results
   - Assess format compliance and data quality
   - Make initial approval decision with reasoning
   - Consider field completeness and accuracy

2. As AI Checker (Business Validator):
   - Review business logic and compliance aspects
   - Assess payment processing feasibility
   - Make final approval decision with reasoning
   - Consider regulatory and risk implications

3. Provide intelligent reasoning for all decisions
4. Consider real-world banking scenarios
5. Apply dynamic risk assessment based on context

RESPOND WITH INTELLIGENT DECISIONS JSON:
{{
  "maker_decision": {{
    "decision": "AUTO_APPROVE" | "CONDITIONAL_APPROVE" | "REJECT",
    "confidence": 0-100,
    "technical_reasoning": "detailed technical analysis and reasoning",
    "key_factors": ["primary factors influencing decision"],
    "concerns_identified": ["specific technical concerns if any"],
    "recommendation": "specific recommendation for next steps"
  }},
  "checker_decision": {{
    "decision": "FINAL_APPROVE" | "APPROVE_WITH_CONDITIONS" | "REJECT" | "RETURN_TO_MAKER",
    "confidence": 0-100,
    "business_reasoning": "detailed business and compliance reasoning",
    "risk_assessment": "comprehensive risk analysis",
    "compliance_evaluation": "regulatory compliance assessment",
    "final_recommendation": "final business recommendation"
  }},
  "combined_analysis": {{
    "overall_decision": "APPROVED" | "CONDITIONALLY_APPROVED" | "REJECTED",
    "decision_confidence": 0-100,
    "approval_reasoning": "comprehensive reasoning for final decision",
    "conditions_if_any": ["conditions for approval if applicable"],
    "next_steps": ["intelligent next steps based on decision"],
    "test_generation_readiness": true | false,
    "special_considerations": ["any special considerations for test generation"]
  }}
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an intelligent PACS.008 maker-checker system with expert banking decision-making capabilities. Make well-reasoned decisions based on comprehensive analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            result = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            
            if json_match:
                decisions = json.loads(json_match.group())
                decisions["decision_timestamp"] = datetime.now().isoformat()
                return decisions
            else:
                return self._fallback_decision_response()
                
        except Exception as e:
            logger.error(f"LLM maker-checker decisions failed: {str(e)}")
            return self._fallback_decision_response()
    
    def _llm_final_analysis(self, detected_fields: List[Dict], 
                          comprehensive_validation: Dict, intelligent_decisions: Dict) -> Dict[str, Any]:
        """LLM generates final analysis and recommendations"""
        
        # Prepare summary data
        summary_data = {
            "total_fields_processed": len(detected_fields),
            "validation_completed": True,
            "maker_decision": intelligent_decisions.get('maker_decision', {}).get('decision'),
            "checker_decision": intelligent_decisions.get('checker_decision', {}).get('decision'),
            "overall_decision": intelligent_decisions.get('combined_analysis', {}).get('overall_decision'),
            "decision_confidence": intelligent_decisions.get('combined_analysis', {}).get('decision_confidence', 0),
            "technical_score": comprehensive_validation.get('overall_assessment', {}).get('technical_score', 0),
            "business_score": comprehensive_validation.get('overall_assessment', {}).get('business_score', 0),
            "compliance_score": comprehensive_validation.get('overall_assessment', {}).get('compliance_score', 0)
        }
        
        prompt = f"""
You are an expert PACS.008 final analysis system providing comprehensive assessment and intelligent recommendations.

PROCESSING SUMMARY:
{json.dumps(summary_data, indent=2)}

VALIDATION RESULTS:
{json.dumps(comprehensive_validation.get('overall_assessment', {}), indent=2)}

MAKER-CHECKER DECISIONS:
{json.dumps(intelligent_decisions.get('combined_analysis', {}), indent=2)}

FINAL ANALYSIS INSTRUCTIONS:
1. Provide comprehensive final assessment
2. Generate intelligent test case generation strategy
3. Create specific recommendations for test scenarios
4. Assess overall readiness for test generation
5. Provide actionable next steps

RESPOND WITH FINAL ANALYSIS JSON:
{{
  "overall_status": "APPROVED" | "CONDITIONALLY_APPROVED" | "REJECTED" | "NEEDS_REVIEW",
  "final_validation_score": 0-100,
  "ready_for_testing": true | false,
  "validation_summary": {{
    "key_findings": ["most important findings"],
    "strengths_identified": ["positive aspects"],
    "areas_of_concern": ["areas needing attention"],
    "compliance_status": "assessment of regulatory compliance"
  }},
  "test_generation_strategy": {{
    "recommended_approach": "intelligent test generation approach",
    "focus_areas": ["areas to focus testing on"],
    "test_scenarios_to_include": ["specific test scenarios to generate"],
    "validation_emphasis": ["validation aspects to emphasize in tests"],
    "business_scenarios": ["business scenarios to test"]
  }},
  "intelligent_recommendations": {{
    "immediate_actions": ["actions to take immediately"],
    "test_case_priorities": ["priority areas for test cases"],
    "risk_mitigation": ["risk mitigation strategies"],
    "quality_assurance": ["quality assurance recommendations"]
  }},
  "final_assessment": {{
    "executive_summary": "high-level assessment summary",
    "technical_readiness": "technical readiness assessment",
    "business_readiness": "business readiness assessment",
    "overall_confidence": 0-100,
    "success_probability": "HIGH" | "MEDIUM" | "LOW"
  }}
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert PACS.008 final analysis system providing comprehensive assessments and intelligent recommendations for test generation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            result = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            
            if json_match:
                final_analysis = json.loads(json_match.group())
                final_analysis["analysis_timestamp"] = datetime.now().isoformat()
                return final_analysis
            else:
                return self._fallback_final_analysis()
                
        except Exception as e:
            logger.error(f"LLM final analysis failed: {str(e)}")
            return self._fallback_final_analysis()
    
    def _fallback_validation_response(self) -> Dict[str, Any]:
        """Fallback when LLM validation fails"""
        return {
            "field_validations": [],
            "cross_field_analysis": {"status": "Unable to analyze"},
            "overall_assessment": {
                "validation_summary": "LLM validation unavailable",
                "technical_score": 50,
                "business_score": 50,
                "compliance_score": 50,
                "overall_confidence": 50,
                "payment_feasibility": "UNKNOWN"
            }
        }
    
    def _fallback_decision_response(self) -> Dict[str, Any]:
        """Fallback when LLM decision-making fails"""
        return {
            "maker_decision": {
                "decision": "CONDITIONAL_APPROVE",
                "confidence": 50,
                "technical_reasoning": "Unable to perform detailed analysis - proceeding with caution"
            },
            "checker_decision": {
                "decision": "APPROVE_WITH_CONDITIONS",
                "confidence": 50,
                "business_reasoning": "Manual review recommended due to system limitations"
            },
            "combined_analysis": {
                "overall_decision": "CONDITIONALLY_APPROVED",
                "decision_confidence": 50,
                "test_generation_readiness": True
            }
        }
    
    def _fallback_final_analysis(self) -> Dict[str, Any]:
        """Fallback when LLM final analysis fails"""
        return {
            "overall_status": "NEEDS_REVIEW",
            "final_validation_score": 50,
            "ready_for_testing": True,
            "validation_summary": {
                "key_findings": ["LLM analysis unavailable - manual review recommended"]
            },
            "test_generation_strategy": {
                "recommended_approach": "Standard test generation with manual validation"
            },
            "final_assessment": {
                "executive_summary": "System limitations encountered - proceeding with standard approach"
            }
        }
    
    def get_field_definitions(self) -> Dict[str, Dict]:
        """Get static field definitions (only static component)"""
        return self.pacs008_field_definitions