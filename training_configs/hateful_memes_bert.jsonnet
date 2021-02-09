local text_model_name = "bert-base-uncased";

local datadir = "/home/moshe/repos/hateful-memes/data/";

local train_data = datadir + "train.jsonl";
local dev_data = datadir + "dev_seen.jsonl";
local test_data = datadir + "test_seen.jsonl";

local num_gpus = 0;

{
    "train_data_path": train_data,
    "validation_data_path": dev_data,
    "dataset_reader": {
        "type": "memereader",
        "image_dir": datadir + "img",
        "feature_cache_dir": datadir + "/feature_cache_torchvision",
        "image_loader": "torch",
        "image_featurizer": "resnet_backbone",
        "region_detector": "faster_rcnn",
        "image_processing_batch_size": 16,
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": text_model_name
        },
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": text_model_name,
                "namespace": "tokens"
            }
        },
        "source_max_tokens": 512,
        "uncased": true,
//         "max_instances": 10, // DEBUG setting
        "truncate_long_sequences": true, // if false, will skip long sequences
    },
    "data_loader": {
        "batch_size": 8,
        "shuffle": true
    },
    [if num_gpus > 1 then "distributed"]: {
        "cuda_devices": std.range(0, num_gpus - 1)
        #"cuda_devices": std.repeat([-1], num_gpus)  # Use this for debugging on CPU
    },
    "model": {
        "type": "hatefulmememodel",
        "text_model_name": text_model_name
    },
    "trainer": {
        "num_epochs": 60,
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 5e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "correct_bias": true
        },
//        "tensorboard_writer": {
//            "summary_interval": 4,
//            "should_log_learning_rate": true
//        },
        "grad_norm": 1.0,
        "validation_metric": "+accuracy"
    }
}
