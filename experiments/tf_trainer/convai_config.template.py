# coding=utf-8
# Copyright 2018 The Conversation-AI.github.io Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
