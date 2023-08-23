import argparse
import logging
import traceback
import torch

from src.learning import Downstream

torch.set_num_threads(16)


parser = argparse.ArgumentParser(description='classifier Training')
parser.add_argument('--device', '-d', type=str, 
                    help='cpu or GPU Device Number', default='cpu')
parser.add_argument('--bs', type=int, 
                    help='Batch Size', default=2048)
parser.add_argument('--lr', type=float, 
                    help='Learning Rate', default=0.0003)
parser.add_argument('--max-epoch', '-e', type=int, 
                    help='Max Epochs for Each Site', default=1000)
parser.add_argument('--valtime', '-v', type=int, 
                    help='Validation Interval', default=1)
parser.add_argument('--patience', '-p', type=int, 
                    help='Early Stopping Patience', default=20)
parser.add_argument('--es', type=int, 
                    help='early stopping option', default=1)
parser.add_argument('--seed', type=int, 
                    help='Random Seed', default=54)
parser.add_argument('--hid-dim', type=int, 
                    help='hidden dimension', default=256)

                    
                    
if __name__ == "__main__":
    try:
        args = parser.parse_args()
        Downstream(args)
        
    except:
        logging.error(traceback.format_exc())