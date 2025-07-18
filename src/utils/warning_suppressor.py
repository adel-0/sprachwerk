"""
Centralized warning suppression for sprachwerk
Consolidates all warning suppressions to eliminate code duplication
"""

import os
import warnings
import logging
from src.core.config import get_cache_dir

def configure_torch_tf32():
    """Configure TF32 settings if torch is available"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    except ImportError:
        pass

def setup_logging_suppressions():
    """Apply all suppressions early in application startup"""
    # Set environment variables early
    os.environ.setdefault('PYTHONWARNINGS', 'ignore::UserWarning')
    os.environ.setdefault('SPEECHBRAIN_LOG_LEVEL', 'WARNING')
    os.environ.setdefault('SPEECHBRAIN_CACHE', os.path.join(get_cache_dir(), 'speechbrain'))
    
    # Suppress TF32 warnings globally
    warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")
    warnings.filterwarnings("ignore", message=".*TF32.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*TensorFloat-32.*")
    
    # Suppress deprecated warnings
    warnings.filterwarnings("ignore", message=".*std.*degrees of freedom.*")
    warnings.filterwarnings("ignore", message=".*correction should be strictly less than.*")
    
    # Suppress deprecated torchaudio warnings
    warnings.filterwarnings("ignore", message=".*torchaudio.*deprecated.*")
    warnings.filterwarnings("ignore", message=".*AudioMetaData.*moved.*")
    
    # Suppress speechbrain warnings
    warnings.filterwarnings("ignore", message=".*speechbrain.pretrained.*deprecated.*")
    warnings.filterwarnings("ignore", message=".*Module 'speechbrain.pretrained' was deprecated.*")
    
    # Suppress torch.cuda.amp warnings
    warnings.filterwarnings("ignore", message=".*torch.cuda.amp.custom_fwd.*deprecated.*")
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.custom_fwd.*")
    
    # Suppress ctranslate2 pkg_resources warning
    warnings.filterwarnings("ignore", message=".*pkg_resources.*deprecated.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")
    
    # SpeechBrain and its dependencies sometimes emit logs and warnings even when log level is set.
    # We must aggressively suppress all output from speechbrain and its submodules, including INFO, DEBUG, WARNING, and even custom handlers.
    import sys
    import types
    # Suppress all warnings from speechbrain and its submodules
    warnings.filterwarnings("ignore", module="speechbrain.*")
    # Suppress all logging from speechbrain and its submodules
    for logger_name in [
        'speechbrain',
        'speechbrain.utils.fetching',
        'speechbrain.utils.parameter_transfer',
        'speechbrain.utils.checkpoints',
    ]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)
        logger.disabled = True
        # Remove all handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
    # Patch sys.stderr and sys.stdout for speechbrain if needed
    class DevNull:
        def write(self, _): pass
        def flush(self): pass
    # Optionally, patch warnings.showwarning to a no-op for speechbrain
    orig_showwarning = warnings.showwarning
    def silent_showwarning(*args, **kwargs):
        if args and hasattr(args[0], 'module') and args[0].module and args[0].module.startswith('speechbrain'):
            return
        return orig_showwarning(*args, **kwargs)
    warnings.showwarning = silent_showwarning
    
    # Configure speechbrain logging
    speechbrain_logger = logging.getLogger('speechbrain')
    speechbrain_logger.setLevel(logging.CRITICAL)
    speechbrain_logger.disabled = True
    
    checkpoint_logger = logging.getLogger('speechbrain.utils.checkpoints')
    checkpoint_logger.setLevel(logging.CRITICAL)
    checkpoint_logger.disabled = True
    
    # Configure TF32 settings
    configure_torch_tf32() 