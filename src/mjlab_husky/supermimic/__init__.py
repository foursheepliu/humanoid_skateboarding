"""Optional learned-dynamics training backend for the original skating task.

Keep this package import light so the default PPO entry point does not import
or initialize the world model, replay storage, or AMP expert loader.
"""
