class Config:
    GLOBAL_SEED = 42
    
    MODEL_NAME = 'resnet-18'
    
    # HYPERPARAMS
    NUM_EPOCHS = 20
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-2
    WEIGHT_DECAY = 1e-4
    DROPOUT_RATE = 1e-3 
    ACCELERATOR = 'cuda'
    FEATURE_SIZE = 1024
    TEMPERATURE = 0.1
    
    # Dataset 
    PATH_DATASET = "./CACD2000"
    DATA_FILES = {
        'train': './celeb-data/retain.csv',
        'test': './celeb-data/test.csv'
    }
    
    RETAIN_DATA_FILES = {
        'retain': './celeb-data/retain.csv',
        'forget': './celeb-data/forget.csv'
    }
    
    NUM_CLASSES = 5
    
    # Contrastive 
    DEFAULT_FEATURE_EXTRACTOR_WEIGHTS = "./checkpoints/feature_extractor/resnet-18-r18-b64.pt"
    DEFAULT_CLASSIFIER_WEIGHTS = "./checkpoints/classifier/resnet-18-r18-b64.pt"
    