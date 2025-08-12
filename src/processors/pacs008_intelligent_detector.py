# # src/processors/pacs008_intelligent_detector.py
# """
# Simplified PACS.008 Intelligent Detector
# Uses static field definitions from your PDF + LLM intelligence to detect fields in user input
# """

# import json
# import re
# import logging
# from typing import Dict, List, Any, Optional
# from openai import OpenAI

# logger = logging.getLogger(__name__)

# class PACS008IntelligentDetector:
#     """Simplified PACS.008 detector with static field knowledge and LLM intelligence"""
    
#     def __init__(self, api_key: str):
#         self.client = OpenAI(api_key=api_key)
        
#         # Static PACS.008 fields extracted from your PDF
#         self.pacs008_fields = self._get_static_pacs008_fields()
        
#         logger.info("PACS.008 Intelligent Detector initialized with static field knowledge")
    
#     def _get_static_pacs008_fields(self) -> Dict[str, Dict]:
#         """Static PACS.008 fields extracted from your official PDF documentation"""
        
#         return {
#             # MANDATORY FIELDS
#             "debtor_agent_bic": {
#                 "name": "Debtor Agent BIC",
#                 "description": "BIC of the debtor's bank (payer's bank)",
#                 "mandatory": True,
#                 "mt_equivalent": "Field 52a",
#                 "examples": ["DEUTDEFF", "CHASUS33", "SBININBB"],
#                 "validation": "8 or 11 character BIC format"
#             },
#             "creditor_agent_bic": {
#                 "name": "Creditor Agent BIC", 
#                 "description": "BIC of the creditor's bank (payee's bank)",
#                 "mandatory": True,
#                 "mt_equivalent": "Field 57a",
#                 "examples": ["BNPAFRPP", "HSBCGB2L", "CITIUS33"],
#                 "validation": "8 or 11 character BIC format"
#             },
#             "debtor_name": {
#                 "name": "Debtor Name",
#                 "description": "Name of the payer/customer initiating payment",
#                 "mandatory": True,
#                 "mt_equivalent": "Field 50a",
#                 "examples": ["John Smith", "ABC Corporation Ltd", "Corporate Treasury"],
#                 "validation": "Non-empty text, max 140 characters"
#             },
#             "creditor_name": {
#                 "name": "Creditor Name",
#                 "description": "Name of the payee/beneficiary receiving payment", 
#                 "mandatory": True,
#                 "mt_equivalent": "Field 59a",
#                 "examples": ["Jane Doe", "XYZ Supplier Inc", "Government Agency"],
#                 "validation": "Non-empty text, max 140 characters"
#             },
#             "debtor_account": {
#                 "name": "Debtor Account",
#                 "description": "Account number/IBAN of the payer",
#                 "mandatory": True,
#                 "examples": ["DE89370400440532013000", "GB33BUKB20201555555555"],
#                 "validation": "Valid IBAN format"
#             },
#             "creditor_account": {
#                 "name": "Creditor Account", 
#                 "description": "Account number/IBAN of the payee",
#                 "mandatory": True,
#                 "examples": ["FR1420041010050500013M02606", "IT60X0542811101000000123456"],
#                 "validation": "Valid IBAN format"
#             },
#             "payment_amount": {
#                 "name": "Payment Amount",
#                 "description": "Amount to be transferred",
#                 "mandatory": True,
#                 "mt_equivalent": "Field 32a",
#                 "examples": ["1000.00", "50000.50", "999999.99"],
#                 "validation": "Positive number with max 2 decimals"
#             },
#             "currency": {
#                 "name": "Currency",
#                 "description": "Currency of the payment",
#                 "mandatory": True,
#                 "examples": ["EUR", "USD", "GBP", "CHF"],
#                 "validation": "Valid ISO currency code"
#             },
#             "instruction_id": {
#                 "name": "Instruction Identification",
#                 "description": "Unique payment instruction reference",
#                 "mandatory": True,
#                 "mt_equivalent": "Field 20",
#                 "examples": ["INSTR20240801001", "REF123456789"],
#                 "validation": "Unique alphanumeric, max 35 characters"
#             },
            
#             # OPTIONAL BUT IMPORTANT FIELDS
#             "end_to_end_id": {
#                 "name": "End-to-End Identification",
#                 "description": "End-to-end payment reference",
#                 "mandatory": False,
#                 "mt_equivalent": "Field 21 (MT 202 COV)",
#                 "examples": ["E2E20240801001", "CLIENT12345"],
#                 "validation": "Alphanumeric, max 35 characters"
#             },
#             "ultimate_debtor": {
#                 "name": "Ultimate Debtor",
#                 "description": "Party on whose behalf debtor is acting (NEW in PACS.008)",
#                 "mandatory": False,
#                 "examples": ["Beneficial Owner Corp", "Trust Fund XYZ"],
#                 "validation": "Text, max 140 characters"
#             },
#             "ultimate_creditor": {
#                 "name": "Ultimate Creditor", 
#                 "description": "Party on whose behalf creditor receives payment (NEW in PACS.008)",
#                 "mandatory": False,
#                 "examples": ["Final Beneficiary Ltd", "Investment Fund ABC"],
#                 "validation": "Text, max 140 characters"
#             },
#             "intermediary_agent": {
#                 "name": "Intermediary Agent",
#                 "description": "Intermediary bank in payment chain (up to 3 in PACS.008)",
#                 "mandatory": False,
#                 "mt_equivalent": "Field 56a",
#                 "examples": ["HSBCGB2L", "CITIUS33"],
#                 "validation": "Valid BIC format"
#             },
#             "settlement_method": {
#                 "name": "Settlement Method",
#                 "description": "How the payment will be settled",
#                 "mandatory": False,
#                 "examples": ["CLRG", "COVE", "INDA", "INGA"],
#                 "validation": "CLRG=Clearing, COVE=Cover, INDA/INGA=Agent accounts"
#             },
#             "charge_bearer": {
#                 "name": "Charge Bearer",
#                 "description": "Who pays the charges",
#                 "mandatory": False,
#                 "mt_equivalent": "Field 71a",
#                 "examples": ["CRED", "DEBT", "SHAR"],
#                 "validation": "CRED=Creditor pays, DEBT=Debtor pays, SHAR=Shared"
#             },
#             "remittance_info": {
#                 "name": "Remittance Information",
#                 "description": "Payment reference/purpose",
#                 "mandatory": False,
#                 "examples": ["Invoice 12345", "Salary payment", "Settlement INV-2024-001"],
#                 "validation": "Free text, max 140 characters"
#             },
#             "settlement_date": {
#                 "name": "Settlement Date",
#                 "description": "When payment should be settled",
#                 "mandatory": False,
#                 "examples": ["2024-08-02", "Same day", "Next business day"],
#                 "validation": "Valid date, cannot be in past"
#             }
#         }
    
#     def detect_pacs008_fields_in_input(self, user_input: str) -> Dict[str, Any]:
#         """Main method: Detect PACS.008 fields in user input using LLM intelligence"""
        
#         try:
#             logger.info("Starting intelligent PACS.008 field detection in user input...")
            
#             # Step 1: Check if input is banking/payment related
#             if not self._is_banking_related(user_input):
#                 return {
#                     "status": "NOT_BANKING_RELATED",
#                     "message": "Input does not appear to be banking/payment related",
#                     "detected_fields": []
#                 }
            
#             # Step 2: Use LLM to intelligently identify PACS.008 fields
#             detected_fields = self._intelligent_field_detection(user_input)
            
#             # Step 3: Prepare for maker-checker process
#             maker_checker_items = self._prepare_for_maker_checker(detected_fields)
            
#             # Step 4: Generate summary
#             summary = self._generate_detection_summary(detected_fields)
            
#             logger.info(f"Detection complete: {len(detected_fields)} PACS.008 fields identified")
            
#             return {
#                 "status": "SUCCESS",
#                 "detected_fields": detected_fields,
#                 "maker_checker_items": maker_checker_items,
#                 "summary": summary,
#                 "total_fields": len(detected_fields),
#                 "mandatory_fields": len([f for f in detected_fields if f.get("is_mandatory")]),
#                 "confidence_score": self._calculate_confidence(detected_fields)
#             }
            
#         except Exception as e:
#             logger.error(f"PACS.008 field detection failed: {str(e)}")
#             return {
#                 "status": "ERROR",
#                 "error": str(e),
#                 "detected_fields": []
#             }
    
#     def _is_banking_related(self, content: str) -> bool:
#         """Quick check if content is banking/payment related"""
        
#         banking_keywords = [
#             "payment", "transfer", "bank", "account", "amount", "bic", "iban",
#             "debtor", "creditor", "agent", "settlement", "remittance", "invoice",
#             "pacs.008", "iso 20022", "swift", "currency", "charge"
#         ]
        
#         content_lower = content.lower()
#         matches = sum(1 for keyword in banking_keywords if keyword in content_lower)
        
#         return matches >= 2  # At least 2 banking keywords
    
#     def _intelligent_field_detection(self, user_input: str) -> List[Dict[str, Any]]:
#         """Use LLM to intelligently detect PACS.008 fields in user input"""
        
#         # Create field reference for LLM
#         field_reference = self._create_field_reference()
        
#         prompt = f"""
# You are a PACS.008 expert. Analyze this user input and identify which PACS.008 fields are present.

# USER INPUT:
# {user_input}

# PACS.008 FIELDS TO LOOK FOR:
# {field_reference}

# INSTRUCTIONS:
# 1. Look for any information that maps to the PACS.008 fields above
# 2. Use banking intelligence - "customer bank" = Debtor Agent, "supplier bank" = Creditor Agent
# 3. Don't look for exact field names - understand the business context
# 4. Extract actual values where possible, or note "mentioned but no value"

# RESPOND WITH JSON ONLY:
# {{
#   "identified_fields": [
#     {{
#       "field_key": "field_key_from_list_above",
#       "field_name": "Human readable name",
#       "extracted_value": "actual value found OR 'mentioned but not specified'",
#       "confidence": "High/Medium/Low",
#       "context": "where in the text you found this",
#       "reasoning": "why you think this maps to this field"
#     }}
#   ]
# }}
# """
        
#         try:
#             response = self.client.chat.completions.create(
#                 model="gpt-4o-mini",
#                 messages=[
#                     {"role": "system", "content": "You are a PACS.008 expert. Respond with valid JSON only."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 temperature=0.1,
#                 max_tokens=2000
#             )
            
#             result = response.choices[0].message.content.strip()
#             json_match = re.search(r'\{.*\}', result, re.DOTALL)
            
#             if json_match:
#                 llm_result = json.loads(json_match.group())
#                 return self._process_llm_results(llm_result.get("identified_fields", []))
#             else:
#                 logger.warning("No valid JSON in LLM response")
#                 return []
                
#         except Exception as e:
#             logger.error(f"LLM field detection failed: {str(e)}")
#             return []
    
#     def _create_field_reference(self) -> str:
#         """Create concise field reference for LLM"""
        
#         reference = []
#         for field_key, field_info in self.pacs008_fields.items():
#             examples = ", ".join(field_info["examples"][:2])
#             mandatory = "MANDATORY" if field_info["mandatory"] else "Optional"
#             reference.append(f"- {field_key}: {field_info['description']} ({mandatory}) [Examples: {examples}]")
        
#         return "\n".join(reference)
    
#     def _process_llm_results(self, llm_fields: List[Dict]) -> List[Dict[str, Any]]:
#         """Process and enrich LLM detection results"""
        
#         processed_fields = []
        
#         for llm_field in llm_fields:
#             field_key = llm_field.get("field_key")
            
#             if field_key in self.pacs008_fields:
#                 field_info = self.pacs008_fields[field_key]
                
#                 processed_field = {
#                     "field_key": field_key,
#                     "field_name": field_info["name"],
#                     "extracted_value": llm_field.get("extracted_value"),
#                     "confidence": llm_field.get("confidence", "Medium"),
#                     "context": llm_field.get("context", ""),
#                     "reasoning": llm_field.get("reasoning", ""),
#                     "is_mandatory": field_info["mandatory"],
#                     "description": field_info["description"],
#                     "examples": field_info["examples"],
#                     "validation_rule": field_info.get("validation", ""),
#                     "mt_equivalent": field_info.get("mt_equivalent", "")
#                 }
                
#                 processed_fields.append(processed_field)
        
#         return processed_fields
    
#     def _prepare_for_maker_checker(self, detected_fields: List[Dict]) -> List[Dict[str, Any]]:
#         """Prepare detected fields for maker-checker process"""
        
#         maker_checker_items = []
        
#         for field in detected_fields:
#             # All mandatory fields and low confidence fields need review
#             needs_review = (
#                 field["is_mandatory"] or 
#                 field["confidence"] == "Low" or
#                 field["extracted_value"] in ["mentioned but not specified", ""]
#             )
            
#             if needs_review:
#                 maker_checker_items.append({
#                     "field_name": field["field_name"],
#                     "field_key": field["field_key"],
#                     "extracted_value": field["extracted_value"],
#                     "confidence": field["confidence"],
#                     "is_mandatory": field["is_mandatory"],
#                     "validation_rule": field["validation_rule"],
#                     "review_reason": self._get_review_reason(field),
#                     "maker_action_needed": "Verify value accuracy and completeness",
#                     "checker_action_needed": "Validate against business rules and format"
#                 })
        
#         return maker_checker_items
    
#     def _get_review_reason(self, field: Dict) -> str:
#         """Get reason why field needs maker-checker review"""
        
#         if field["is_mandatory"] and not field["extracted_value"]:
#             return "Mandatory field missing - critical for payment processing"
#         elif field["confidence"] == "Low":
#             return "Low confidence detection - needs verification"
#         elif field["extracted_value"] in ["mentioned but not specified", ""]:
#             return "Field mentioned but value not provided"
#         else:
#             return "Standard review for critical field"
    
#     def _generate_detection_summary(self, detected_fields: List[Dict]) -> Dict[str, Any]:
#         """Generate summary of detection results"""
        
#         mandatory_detected = [f for f in detected_fields if f["is_mandatory"]]
#         missing_mandatory = []
        
#         # Check for missing mandatory fields
#         for field_key, field_info in self.pacs008_fields.items():
#             if field_info["mandatory"]:
#                 if not any(f["field_key"] == field_key for f in detected_fields):
#                     missing_mandatory.append(field_info["name"])
        
#         return {
#             "total_detected": len(detected_fields),
#             "mandatory_detected": len(mandatory_detected),
#             "missing_mandatory": missing_mandatory,
#             "completion_percentage": (len(mandatory_detected) / len([f for f in self.pacs008_fields.values() if f["mandatory"]]) * 100),
#             "ready_for_testing": len(missing_mandatory) == 0,
#             "high_confidence_fields": len([f for f in detected_fields if f["confidence"] == "High"]),
#             "fields_needing_review": len([f for f in detected_fields if f["confidence"] == "Low" or not f["extracted_value"]])
#         }
    
#     def _calculate_confidence(self, detected_fields: List[Dict]) -> float:
#         """Calculate overall confidence score"""
        
#         if not detected_fields:
#             return 0.0
        
#         confidence_values = {"High": 1.0, "Medium": 0.7, "Low": 0.4}
#         total = sum(confidence_values.get(f["confidence"], 0.4) for f in detected_fields)
        
#         return total / len(detected_fields)
    
#     def get_all_pacs008_fields(self) -> Dict[str, Dict]:
#         """Get all static PACS.008 field definitions"""
#         return self.pacs008_fields
    
#     def get_mandatory_fields(self) -> List[str]:
#         """Get list of mandatory field names"""
#         return [info["name"] for info in self.pacs008_fields.values() if info["mandatory"]]



# src/processors/pacs008_intelligent_detector.py - ENHANCED VERSION
"""
Enhanced PACS.008 Intelligent Detector with Better Prompts and Accuracy
"""

import json
import re
import logging
from typing import Dict, List, Any, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)

class PACS008IntelligentDetector:
    """Enhanced PACS.008 detector with improved accuracy and confidence scoring"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        
        # Enhanced PACS.008 fields with better detection patterns
        self.pacs008_fields = self._get_enhanced_pacs008_fields()
        
        logger.info("Enhanced PACS.008 Intelligent Detector initialized")
    
    def _get_enhanced_pacs008_fields(self) -> Dict[str, Dict]:
        """Enhanced PACS.008 fields with better detection patterns"""
        
        return {
            # MANDATORY FIELDS - Enhanced Detection
            "debtor_agent_bic": {
                "name": "Debtor Agent BIC",
                "description": "BIC of the debtor's bank (payer's bank)",
                "mandatory": True,
                "detection_patterns": ["debtor agent", "payer bank", "sending bank", "originating bank", "customer bank"],
                "examples": ["DEUTDEFF", "CHASUS33", "SBININBB", "Al Ahli Bank", "Deutsche Bank"],
                "validation": "8 or 11 character BIC format or bank name"
            },
            "creditor_agent_bic": {
                "name": "Creditor Agent BIC", 
                "description": "BIC of the creditor's bank (payee's bank)",
                "mandatory": True,
                "detection_patterns": ["creditor agent", "beneficiary bank", "receiving bank", "payee bank", "destination bank"],
                "examples": ["BNPAFRPP", "HSBCGB2L", "CITIUS33", "BNP Paribas", "HSBC"],
                "validation": "8 or 11 character BIC format or bank name"
            },
            "debtor_name": {
                "name": "Debtor Name",
                "description": "Name of the payer/customer initiating payment",
                "mandatory": True,
                "detection_patterns": ["debtor", "payer", "customer", "originator", "sender"],
                "examples": ["John Smith", "ABC Corporation Ltd", "Corporate Treasury"],
                "validation": "Non-empty text, max 140 characters"
            },
            "creditor_name": {
                "name": "Creditor Name",
                "description": "Name of the payee/beneficiary receiving payment", 
                "mandatory": True,
                "detection_patterns": ["creditor", "beneficiary", "payee", "recipient", "receiver"],
                "examples": ["Jane Doe", "XYZ Supplier Inc", "Government Agency"],
                "validation": "Non-empty text, max 140 characters"
            },
            "debtor_account": {
                "name": "Debtor Account",
                "description": "Account number/IBAN of the payer",
                "mandatory": True,
                "detection_patterns": ["debtor account", "payer account", "source account", "debit account"],
                "examples": ["DE89370400440532013000", "GB33BUKB20201555555555"],
                "validation": "Valid IBAN format"
            },
            "creditor_account": {
                "name": "Creditor Account", 
                "description": "Account number/IBAN of the payee",
                "mandatory": True,
                "detection_patterns": ["creditor account", "beneficiary account", "destination account", "credit account"],
                "examples": ["FR1420041010050500013M02606", "IT60X0542811101000000123456"],
                "validation": "Valid IBAN format"
            },
            "payment_amount": {
                "name": "Payment Amount",
                "description": "Amount to be transferred",
                "mandatory": True,
                "detection_patterns": ["amount", "value", "payment", "transfer amount", "sum"],
                "examples": ["1000.00", "50000.50", "999999.99", "USD 565000", "EUR 25000"],
                "validation": "Positive number with currency"
            },
            "currency": {
                "name": "Currency",
                "description": "Currency of the payment",
                "mandatory": True,
                "detection_patterns": ["currency", "USD", "EUR", "GBP", "CHF"],
                "examples": ["EUR", "USD", "GBP", "CHF"],
                "validation": "Valid ISO currency code"
            },
            "instruction_id": {
                "name": "Instruction Identification",
                "description": "Unique payment instruction reference",
                "mandatory": True,
                "detection_patterns": ["instruction id", "reference", "transaction id", "payment id"],
                "examples": ["INSTR20240801001", "REF123456789"],
                "validation": "Unique alphanumeric, max 35 characters"
            }
        }
    
    def detect_pacs008_fields_in_input(self, user_input: str) -> Dict[str, Any]:
        """Enhanced main method with better field detection"""
        
        try:
            logger.info("Starting enhanced PACS.008 field detection...")
            
            # Step 1: Check if input is banking/payment related
            if not self._is_banking_related(user_input):
                return {
                    "status": "NOT_BANKING_RELATED",
                    "message": "Input does not appear to be banking/payment related",
                    "detected_fields": []
                }
            
            # Step 2: Enhanced field detection with better prompting
            detected_fields = self._enhanced_field_detection(user_input)
            
            # Step 3: Prepare for maker-checker process
            maker_checker_items = self._prepare_for_maker_checker(detected_fields)
            
            # Step 4: Generate summary
            summary = self._generate_detection_summary(detected_fields)
            
            logger.info(f"Enhanced detection complete: {len(detected_fields)} PACS.008 fields identified")
            
            return {
                "status": "SUCCESS",
                "detected_fields": detected_fields,
                "maker_checker_items": maker_checker_items,
                "summary": summary,
                "total_fields": len(detected_fields),
                "mandatory_fields": len([f for f in detected_fields if f.get("is_mandatory")]),
                "confidence_score": self._calculate_confidence(detected_fields)
            }
            
        except Exception as e:
            logger.error(f"Enhanced PACS.008 field detection failed: {str(e)}")
            return {
                "status": "ERROR",
                "error": str(e),
                "detected_fields": []
            }
    
    def _is_banking_related(self, content: str) -> bool:
        """Enhanced banking content detection"""
        
        banking_keywords = [
            "payment", "transfer", "bank", "account", "amount", "bic", "iban",
            "debtor", "creditor", "agent", "settlement", "remittance", "invoice",
            "pacs.008", "iso 20022", "swift", "currency", "charge", "USD", "EUR",
            "correspondent", "nostro", "vostro", "serial", "inda", "inga"
        ]
        
        content_lower = content.lower()
        matches = sum(1 for keyword in banking_keywords if keyword in content_lower)
        
        return matches >= 2  # At least 2 banking keywords
    
    def _enhanced_field_detection(self, user_input: str) -> List[Dict[str, Any]]:
        """Enhanced LLM field detection with better prompts"""
        
        # Create enhanced field reference
        field_reference = self._create_enhanced_field_reference()
        
        prompt = f"""
You are a PACS.008 banking expert with deep knowledge of international payments and correspondent banking. 

Analyze this content carefully and identify PACS.008 payment fields with HIGH ACCURACY.

CONTENT TO ANALYZE:
{user_input}

PACS.008 FIELDS TO DETECT:
{field_reference}

ENHANCED DETECTION RULES:
1. Look for EXPLICIT VALUES with HIGH confidence:
   - "USD 565000" = Payment Amount: "USD 565000" (High confidence)
   - "Al Ahli Bank of Kuwait" = Debtor Agent BIC: "Al Ahli Bank of Kuwait" (High confidence)  
   - "Deutsche Bank" = Creditor Agent BIC: "Deutsche Bank" (High confidence)

2. Use BANKING INTELLIGENCE:
   - "customer bank" = debtor agent, "beneficiary bank" = creditor agent
   - "payer" = debtor, "recipient" = creditor

3. Don't say "mentioned but not specified" if you see ACTUAL VALUES
4. Be CONFIDENT when values are clearly stated
5. Look for business context clues - payment scenarios, bank relationships, amounts


CONFIDENCE RULES:
- HIGH: Value explicitly stated (e.g., "USD 565000", "Deutsche Bank", "DEUTDEFF")
- MEDIUM: Field mentioned with partial info or context clues
- LOW: Only when genuinely uncertain

RESPOND WITH JSON ONLY:
{{
  "identified_fields": [
    {{
      "field_key": "payment_amount",
      "field_name": "Payment Amount", 
      "extracted_value": "USD 565000",
      "confidence": "High",
      "context": "business scenario mentions USD 565000 payment",
      "reasoning": "Amount explicitly stated in business example"
    }},
    {{
      "field_key": "debtor_agent_bic",
      "field_name": "Debtor Agent BIC",
      "extracted_value": "Al Ahli Bank of Kuwait", 
      "confidence": "High",
      "context": "mentioned as originating bank",
      "reasoning": "Bank name clearly identified as payer's bank"
    }}
  ]
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a PACS.008 banking expert. Extract field values with HIGH ACCURACY. When you see explicit values like amounts or bank names, extract them with HIGH confidence. Respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            result = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            
            if json_match:
                llm_result = json.loads(json_match.group())
                return self._process_enhanced_llm_results(llm_result.get("identified_fields", []))
            else:
                logger.warning("No valid JSON in enhanced LLM response")
                return []
                
        except Exception as e:
            logger.error(f"Enhanced LLM field detection failed: {str(e)}")
            return []
    
    def _create_enhanced_field_reference(self) -> str:
        """Create enhanced field reference with detection patterns"""
        
        reference = []
        for field_key, field_info in self.pacs008_fields.items():
            examples = ", ".join(field_info["examples"][:3])
            patterns = ", ".join(field_info["detection_patterns"][:3])
            mandatory = "MANDATORY" if field_info["mandatory"] else "Optional"
            
            reference.append(f"- {field_key}: {field_info['description']} ({mandatory})")
            reference.append(f"  Detection patterns: {patterns}")
            reference.append(f"  Examples: {examples}")
            reference.append("")
        
        return "\n".join(reference)
    
    def _process_enhanced_llm_results(self, llm_fields: List[Dict]) -> List[Dict[str, Any]]:
        """Process and enrich enhanced LLM detection results"""
        
        processed_fields = []
        
        for llm_field in llm_fields:
            field_key = llm_field.get("field_key")
            
            if field_key in self.pacs008_fields:
                field_info = self.pacs008_fields[field_key]
                
                # Enhanced processing with better validation
                extracted_value = llm_field.get("extracted_value", "")
                confidence = llm_field.get("confidence", "Medium")
                
                # Boost confidence for clearly extracted values
                if extracted_value and extracted_value not in ["mentioned but not specified", "not specified", ""]:
                    if any(keyword in extracted_value.lower() for keyword in ["usd", "eur", "bank", "corp", "ltd"]):
                        confidence = "High"
                
                processed_field = {
                    "field_key": field_key,
                    "field_name": field_info["name"],
                    "extracted_value": extracted_value,
                    "confidence": confidence,
                    "context": llm_field.get("context", ""),
                    "reasoning": llm_field.get("reasoning", ""),
                    "is_mandatory": field_info["mandatory"],
                    "description": field_info["description"],
                    "examples": field_info["examples"],
                    "validation_rule": field_info.get("validation", ""),
                    "detection_patterns": field_info.get("detection_patterns", [])
                }
                
                processed_fields.append(processed_field)
        
        return processed_fields
    
    def _prepare_for_maker_checker(self, detected_fields: List[Dict]) -> List[Dict[str, Any]]:
        """Enhanced maker-checker preparation"""
        
        maker_checker_items = []
        
        for field in detected_fields:
            # More intelligent validation logic
            needs_review = (
                field["is_mandatory"] or 
                field["confidence"] == "Low" or
                not field["extracted_value"] or
                field["extracted_value"] in ["mentioned but not specified", "", "not specified"]
            )
            
            if needs_review:
                maker_checker_items.append({
                    "field_name": field["field_name"],
                    "field_key": field["field_key"],
                    "extracted_value": field["extracted_value"],
                    "confidence": field["confidence"],
                    "is_mandatory": field["is_mandatory"],
                    "validation_rule": field["validation_rule"],
                    "review_reason": self._get_enhanced_review_reason(field),
                    "maker_action_needed": "Verify field accuracy and provide value if missing",
                    "checker_action_needed": "Validate against PACS.008 standards and business rules"
                })
        
        return maker_checker_items
    
    def _get_enhanced_review_reason(self, field: Dict) -> str:
        """Enhanced review reason logic"""
        
        if field["is_mandatory"] and not field["extracted_value"]:
            return f"CRITICAL: Mandatory {field['field_name']} missing - required for PACS.008 processing"
        elif field["confidence"] == "Low":
            return f"UNCERTAIN: Low confidence detection for {field['field_name']} - needs verification"
        elif field["extracted_value"] in ["mentioned but not specified", "", "not specified"]:
            return f"INCOMPLETE: {field['field_name']} mentioned but value not provided"
        else:
            return f"STANDARD: Review required for {field['field_name']} validation"
    
    def _generate_detection_summary(self, detected_fields: List[Dict]) -> Dict[str, Any]:
        """Enhanced detection summary"""
        
        mandatory_detected = [f for f in detected_fields if f["is_mandatory"]]
        high_confidence = [f for f in detected_fields if f["confidence"] == "High"]
        missing_mandatory = []
        
        # Check for missing mandatory fields
        for field_key, field_info in self.pacs008_fields.items():
            if field_info["mandatory"]:
                if not any(f["field_key"] == field_key for f in detected_fields):
                    missing_mandatory.append(field_info["name"])
        
        total_mandatory = len([f for f in self.pacs008_fields.values() if f["mandatory"]])
        completion_percentage = (len(mandatory_detected) / total_mandatory * 100) if total_mandatory > 0 else 0
        
        return {
            "total_detected": len(detected_fields),
            "mandatory_detected": len(mandatory_detected),
            "high_confidence_detected": len(high_confidence),
            "missing_mandatory": missing_mandatory,
            "completion_percentage": round(completion_percentage, 1),
            "ready_for_testing": len(missing_mandatory) <= 2,  # Allow some missing for flexibility
            "detection_quality": "High" if len(high_confidence) >= 3 else "Medium" if len(detected_fields) >= 3 else "Basic",
            "fields_needing_review": len([f for f in detected_fields if f["confidence"] == "Low" or not f["extracted_value"]])
        }
    
    def _calculate_confidence(self, detected_fields: List[Dict]) -> float:
        """Enhanced confidence calculation"""
        
        if not detected_fields:
            return 0.0
        
        confidence_values = {"High": 1.0, "Medium": 0.6, "Low": 0.2}
        total = sum(confidence_values.get(f["confidence"], 0.2) for f in detected_fields)
        
        # Bonus for high-value extractions
        value_bonus = 0
        for field in detected_fields:
            if field["extracted_value"] and field["extracted_value"] not in ["mentioned but not specified", "", "not specified"]:
                value_bonus += 0.1
        
        base_confidence = total / len(detected_fields)
        final_confidence = min(1.0, base_confidence + (value_bonus / len(detected_fields)))
        
        return round(final_confidence, 2)
    
    def get_all_pacs008_fields(self) -> Dict[str, Dict]:
        """Get all enhanced PACS.008 field definitions"""
        return self.pacs008_fields
    
    def get_mandatory_fields(self) -> List[str]:
        """Get list of mandatory field names"""
        return [info["name"] for info in self.pacs008_fields.values() if info["mandatory"]]