"""Conversation AI Config File.

This file stores local configuration variables, like API keys and paths where
training and test data is stored. This file should not be checked in.

You should copy this file to `convai_config.py` in the same directory and edit
values in it there.
"""

# Set this to your www.comet.ml API key to track experiments there.
# If set to None, all comet_* vars are ignored.
# These can be overriden by command line argumwents.
comet_api_key = None
comet_team_name=''
comet_project_name=''

cloud_storage_path = ''
