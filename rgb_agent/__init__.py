"""
ArcGym: extensions for ARC-AGI-3 puzzle solving.
"""

# Make registry import optional - it requires verl which may not be available
try:
    from .registry import register_rgb_agent_components
    __all__ = ["register_rgb_agent_components"]
except ImportError as e:
    import logging
    logging.getLogger(__name__).debug(f"Registry not available: {e}")
    __all__ = []
