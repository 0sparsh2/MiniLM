import sys
import os
from huggingface_hub import HfApi, login

def main():
    token = None
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    token = line.strip().split("=", 1)[1]
    
    if not token and len(sys.argv) > 1:
        token = sys.argv[1]
        
    if not token:
        print("Usage: python3 push_sparse_to_hf.py <HUGGINGFACE_TOKEN> or set HF_TOKEN in .env")
        sys.exit(1)
        
    repo_id = "0sparsh2/MiniLM"
    
    print(f"Logging into HuggingFace Hub...", flush=True)
    login(token=token)
    
    api = HfApi()
    
    print(f"Creating repository {repo_id} if it doesn't exist...", flush=True)
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
    except Exception as e:
        print(f"Could not create/confirm repo: {e}")
        
    # Define files to upload
    files_to_upload = [
        ("../MiniLM/bitnet_sparse_instruct_15k.pt", "bitnet_sparse_instruct_15k.pt"),
        ("../MiniLM/model.py", "model.py"),
        ("../MiniLM/README_SPARSE.md", "README.md"),
    ]
    
    for local_path, repo_path in files_to_upload:
        abs_local_path = os.path.abspath(local_path)
        if not os.path.exists(abs_local_path):
            print(f"Error: Local file {abs_local_path} does not exist!")
            sys.exit(1)
            
        print(f"Uploading {repo_path} from {abs_local_path}...", flush=True)
        try:
            api.upload_file(
                path_or_fileobj=abs_local_path,
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"Successfully uploaded {repo_path}", flush=True)
        except Exception as e:
            print(f"Failed to upload {repo_path}: {e}")
            sys.exit(1)
            
    print(f"All files successfully pushed to https://huggingface.co/{repo_id}", flush=True)

if __name__ == "__main__":
    main()
