import argparse 
import torch
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
        "--fe_weights",
        type=str,
        default=Config.DEFAULT_FEATURE_EXTRACTOR_WEIGHTS,
        help='Set path to weight file'
    )
    
    parser.add_argument(
        "--c_weights",
        type=str,
        default=Config.DEFAULT_CLASSIFIER_WEIGHTS,
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
    
    model = ContrastiveLearning(
        feature_extractor_path=args.fe_weights,
        classifier_path=args.c_weights,
        learning_rate=args.lr, 
        weight_decay=args.wd
    )

    logger = WandbLogger(
        name=args.exp,
        save_dir='./logs',
        project='unlearning'
    )
    
    # Trainer 
    trainer = L.Trainer(
        # callbacks=[checkpoint_callback],
        accelerator='gpu',
        logger=logger,
        max_epochs=args.epochs,
        enable_checkpointing=True,
        log_every_n_steps=10
    )
    
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
    
    torch.save(model.feature_extractor, f'./checkpoints/feature_extractor/{args.model}-{args.exp}.pt')
    torch.save(model.classifier, f'./checkpoints/classifier/{args.model}-{args.exp}.pt')