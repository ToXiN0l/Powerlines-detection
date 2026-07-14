from huggingface_hub import login, upload_folder

# (optional) Login with your Hugging Face credentials
# login()

# Push your dataset files
upload_folder(folder_path="yolo_dataset/", repo_id="ToXiN0/Powerlines-detection", repo_type="dataset")
