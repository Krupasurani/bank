# # main.py
# """
# ITASSIST - Intelligent Test Case Generator
# Main application entry point

# Run with: streamlit run main.py
# """

# import os
# import sys
# import logging
# from pathlib import Path

# # Add src directory to Python path
# src_path = Path(__file__).parent / "src"
# sys.path.insert(0, str(src_path))

# # Configure logging (fix Unicode issue on Windows)
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('itassist.log', encoding='utf-8'),
#         logging.StreamHandler()
#     ]
# )

# logger = logging.getLogger(__name__)

# def check_dependencies():
#     """Check if all required dependencies are installed"""
#     required_packages = [
#         ('streamlit', 'streamlit'),
#         ('openai', 'openai'), 
#         ('docx', 'python-docx'),
#         ('PyPDF2', 'PyPDF2'),
#         ('openpyxl', 'openpyxl'), 
#         ('pandas', 'pandas'),
#         ('pytesseract', 'pytesseract'),
#         ('cv2', 'opencv-python'),
#         ('PIL', 'Pillow')
#     ]
    
#     missing_packages = []
    
#     for import_name, package_name in required_packages:
#         try:
#             __import__(import_name)
#             logger.info(f"OK {package_name} loaded successfully")
#         except ImportError:
#             missing_packages.append(package_name)
#             logger.error(f"X {package_name} not found")
    
#     if missing_packages:
#         logger.error(f"Missing required packages: {', '.join(missing_packages)}")
#         logger.error("Please install missing packages using: pip install -r requirements.txt")
#         return False
    
#     logger.info("OK All dependencies checked and loaded successfully")
#     return True

# def setup_directories():
#     """Create necessary directories"""
#     directories = ['temp', 'outputs', 'logs']
    
#     for directory in directories:
#         dir_path = Path(directory)
#         dir_path.mkdir(exist_ok=True)
#         logger.info(f"Directory created/verified: {dir_path}")

# def check_environment():
#     """Check environment setup"""
#     logger.info("Checking environment setup...")
    
#     # Check Python version
#     python_version = sys.version_info
#     if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
#         logger.error("Python 3.8+ is required")
#         return False
    
#     logger.info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
#     # Check OCR dependencies
#     try:
#         import pytesseract
#         import cv2
#         logger.info("OCR dependencies loaded successfully")
#     except ImportError as e:
#         logger.warning(f"OCR dependencies not fully available: {e}")
#         logger.warning("Some image processing features may not work")
    
#     return True

# def main():
#     """Main application entry point"""
    
#     print("🤖 ITASSIST - Intelligent Test Case Generator")
#     print("=" * 50)
    
#     # Check environment
#     if not check_environment():
#         sys.exit(1)
    
#     # Check dependencies (skip on error for now)
#     try:
#         check_dependencies()
#     except Exception as e:
#         logger.warning(f"Dependency check failed: {e}")
#         logger.warning("Continuing anyway...")
    
#     # Setup directories
#     setup_directories()
    
#     logger.info("Starting ITASSIST application...")
    
#     try:
#         # Import and run the Streamlit app
#         from ui.streamlit_app import main as streamlit_main
#         streamlit_main()
        
#     except ImportError as e:
#         logger.error(f"Failed to import Streamlit app: {e}")
#         logger.error("Make sure all dependencies are installed")
#         sys.exit(1)
    
#     except Exception as e:
#         logger.error(f"Application error: {e}")
#         sys.exit(1)

# if __name__ == "__main__":
#     main()






# main.py
"""
ITASSIST - Enhanced Intelligent Test Case Generator with PACS.008 Intelligence
Main application entry point

Run with: streamlit run main.py
"""

import os
import sys
import logging
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Configure logging (fix Unicode issue on Windows)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('itassist.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        ('streamlit', 'streamlit'),
        ('openai', 'openai'), 
        ('docx', 'python-docx'),
        ('PyPDF2', 'PyPDF2'),
        ('openpyxl', 'openpyxl'), 
        ('pandas', 'pandas'),
        ('pytesseract', 'pytesseract'),
        ('cv2', 'opencv-python'),
        ('PIL', 'Pillow')
    ]
    
    missing_packages = []
    
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
            logger.info(f"OK {package_name} loaded successfully")
        except ImportError:
            missing_packages.append(package_name)
            logger.error(f"X {package_name} not found")
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.error("Please install missing packages using: pip install -r requirements.txt")
        return False
    
    logger.info("OK All dependencies checked and loaded successfully")
    return True

def setup_directories():
    """Create necessary directories"""
    directories = ['temp', 'outputs', 'logs']
    
    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(exist_ok=True)
        logger.info(f"Directory created/verified: {dir_path}")

def check_environment():
    """Check environment setup"""
    logger.info("Checking environment setup...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        logger.error("Python 3.8+ is required")
        return False
    
    logger.info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check OCR dependencies
    try:
        import pytesseract
        import cv2
        logger.info("OCR dependencies loaded successfully")
    except ImportError as e:
        logger.warning(f"OCR dependencies not fully available: {e}")
        logger.warning("Some image processing features may not work")
    
    # Check PACS.008 enhancement availability
    try:
        from processors.pacs008_llm_detector import PACS008LLMDetector
        logger.info("🏦 PACS.008 LLM intelligence available")
    except ImportError:
        logger.info("📄 PACS.008 enhancements not available - using standard mode")
    
    return True

def main():
    """Main application entry point"""
    
    print("🏦 ITASSIST - Enhanced with PACS.008 Intelligence")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Check dependencies (skip on error for now)
    try:
        check_dependencies()
    except Exception as e:
        logger.warning(f"Dependency check failed: {e}")
        logger.warning("Continuing anyway...")
    
    # Setup directories
    setup_directories()
    
    logger.info("Starting ITASSIST application...")
    
    try:
        # Import and run the Streamlit app
        from ui.streamlit_app import main as streamlit_main
        streamlit_main()
        
    except ImportError as e:
        logger.error(f"Failed to import Streamlit app: {e}")
        logger.error("Make sure all dependencies are installed")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()