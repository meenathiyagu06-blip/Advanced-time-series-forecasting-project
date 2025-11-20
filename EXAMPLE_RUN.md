# Example run (minimal)

1. Create data:
   `python data_generator.py --out data/synthetic.csv --n_steps 1500`
2. Train hybrid model:
   `python train.py --data data/synthetic.csv --save_dir outputs --input_len 56 --output_len 24 --epochs 5`
3. Evaluate only using saved artifacts:
   `python train.py --data data/synthetic.csv --save_dir outputs --evaluate_only`
