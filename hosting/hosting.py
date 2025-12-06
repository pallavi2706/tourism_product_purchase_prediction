from huggingface_hub import HfApi, login
import os

token = os.environ.get("HF_TOKEN")
if not token:
    raise ValueError("HF_TOKEN not found in environment variables")

login(token=token)

api = HfApi(token=token)

api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id="pal27/tourism",
    repo_type="space",
    path_in_repo=""
)
