step1: download dependencies from requirement.txt
step2: python train.py --data ./dataset --download_kaggle --kaggle_name cashbowman/ai-generated-images-vs-real-images --epochs 8 --batch 32 --lr 1e-4 --pretrained
step3: streamlit run app.py
step4: python fix_dataset_structure.py (if alredy fixed then dont do)
step6: python train_timm.py --data ./dataset --model_name tf_efficientnet_b3 --pretrained --epochs 12 --batch 16 --input_size 300 --model_out best_model.pth --mixup
step7:streamlit run app_ensemble.py
