class Config:
    GLOBAL_SEED = 42
    
    MODEL_NAME = 'resnet-50'
    
    # HYPERPARAMS
    NUM_EPOCHS = 20
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-2
    WEIGHT_DECAY = 1e-4
    DROPOUT_RATE = 1e-3 
    ACCELERATOR = 'cuda'
    FEATURE_SIZE = 1024
    
    # Dataset 
    PATH_DATASET = "./CACD2000"
    DATA_FILES = {
        'train': './celeb-data/train.csv',
        'test': './celeb-data/test.csv'
    }
    
    RETAIN_DATA_FILES = {
        'retain': './celeb-data/retain.csv',
        'forget': './celeb-data/forget.csv'
    }
    
    NUM_CLASSES = 5
    