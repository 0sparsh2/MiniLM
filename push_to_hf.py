import sys
from huggingface_hub import HfApi, login

def main():
    import os
    token = None
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    token = line.strip().split("=", 1)[1]
    
    if not token and len(sys.argv) > 1:
        token = sys.argv[1]
        
    if not token:
        print("Usage: python3 push_to_hf.py <HUGGINGFACE_TOKEN> or set HF_TOKEN in .env")
        sys.exit(1)
    repo_id = "0sparsh2/BitNet-TinyStories-V2"
    
    print(f"Logging into HuggingFace Hub...", flush=True)
    login(token=token)
    
    api = HfApi()
    
    print(f"Creating repository {repo_id} (if it doesn't exist)...", flush=True)
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
    except Exception as e:
        print(f"Could not create repo: {e}")
        
    print("Uploading bitnet_tied.pt...", flush=True)
    api.upload_file(
        path_or_fileobj="bitnet_tied.pt",
        path_in_repo="bitnet_tied.pt",
        repo_id=repo_id,
        repo_type="model",
    )
    
    print("Uploading README.md (Model Card)...", flush=True)
    api.upload_file(
        path_or_fileobj="README_HF.md",
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    
    print("Uploading bitnet_test.py (Architecture source)...", flush=True)
    api.upload_file(
        path_or_fileobj="bitnet_test.py",
        path_in_repo="bitnet_test.py",
        repo_id=repo_id,
        repo_type="model",
    )
    
    print(f"Done! Model pushed to https://huggingface.co/{repo_id}", flush=True)

if __name__ == "__main__":
    main()
