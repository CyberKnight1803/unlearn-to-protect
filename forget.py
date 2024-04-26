import argparse 
import lightning as L 
from lightning.pytorch.loggers import WandbLogger
from lightning import seed_everything

from src.datamodule import ContrastiveDataModule
from src.model import ContrastiveLearning
from src.config import Config

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--exp', 
        type=str,
        required=True,
        help="Set Experiment"
    )
    
    parser.add_argument(
        "--model", 
        type=str,
        default=Config.MODEL_NAME, 
        help='Set model'
    )
    
    parser.add_argument(
        "--model_weights",
        type=str,
        default=Config.DEFAULT_FEATURE_EXTRACTOR_WEIGHTS,
        help='Set path to weight file'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=Config.NUM_EPOCHS,
        help='Set epochs'
    )
    
    parser.add_argument( 
        '--lr',
        type=float,
        default=Config.LEARNING_RATE,
        help="Set learning rate"
    )
    
    parser.add_argument(
        '--wd', 
        type=float, 
        default=Config.WEIGHT_DECAY,
        help='Set weight decay'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=Config.BATCH_SIZE,
        help='Set batch_size'
    )
    
    
    args = parser.parse_args()
    
    # Global seed
    seed_everything(Config.GLOBAL_SEED)
    
    dm = ContrastiveDataModule(
        batch_size=args.batch_size,
    )
    
    dm.setup()
    
    names_f, imgs_f, labels_f, names_p_r, imgs_p_r, labels_p_r, names_n_r, imgs_n_r, labels_n_r = next(iter(dm.train_dataloader()))
    
    print(imgs_f.size(), labels_f.size())
    print(imgs_p_r.size(), labels_p_r.size())
    print(imgs_n_r.size(), labels_n_r.size())
    
    print(labels_f)
    print(labels_p_r)
    print(labels_n_r)