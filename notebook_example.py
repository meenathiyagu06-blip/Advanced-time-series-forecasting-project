# Quick example: generate data and run a single training epoch
import os
from data_generator import generate
if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    df = generate(n_steps=800)
    df.to_csv('data/synthetic.csv', index=False)
    print('Saved data/data/synthetic.csv. Now run: python train.py --data data/synthetic.csv --epochs 1 --save_dir outputs')
