local text_model_name = "bert-base-uncased";
local train_data = "/home/moshe/repos/hateful-memes/data/train.jsonl";
local dev_data = "/home/moshe/repos/hateful-memes/data/dev_seen.jsonl";
local test_data = "/home/moshe/repos/hateful-memes/data/test_seen.jsonl";

{
    "train_data_path": train_data,
    "validation_data_path": dev_data,
    "dataset_reader": {
        "type": "memereader",
        "source_tokenizer": {
            "type": "pretrained_transformer",
            "model_name": text_model_name
        },
        "source_token_indexers": {
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
