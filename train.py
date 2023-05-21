import pandas as pd
import logging
from seq2seq_model import Seq2SeqModel
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

train_data = pd.read_csv("./water_io_w2_10-shot/train.csv", sep=',').values.tolist()
train_df = pd.DataFrame(train_data, columns=["input_text", "target_text"])

eval_data = pd.read_csv("./water_io_w2_10-shot/dev_10-shot.csv", sep=',').values.tolist()
eval_df = pd.DataFrame(eval_data, columns=["input_text", "target_text"])

model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 256,
    # "max_seq_length": 1024,
    "train_batch_size": 8,
    # "train_batch_size": 1,
    "num_train_epochs": 20,
    # "num_train_epochs": 1,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": False,
    "evaluate_during_training": True,
    "evaluate_generated_text": True,
    "evaluate_during_training_verbose": True,
    "use_multiprocessing": False,
    "max_length": 256,
    # "max_length": 1024,
    "manual_seed": 4,
    "save_steps": 11898,
    # "save_steps": 5000,
    "gradient_accumulation_steps": 1,
    "output_dir": "./exp/water_io_w2_10-shot",
    "best_model_dir":"./outputs_water_io_w2_10-shot/best_model",
    # "use_early_stopping" : True,
    # "early_stopping_patience" : 5
}

# Initialize model
model = Seq2SeqModel(
    encoder_decoder_type="bart",
    encoder_decoder_name="facebook/bart-large",
    # encoder_decoder_name='outputs_water2/best_model',
    args=model_args,
    # use_cuda=False,
)


# Train the model
model.train_model(train_df, eval_data=eval_df)

# Evaluate the model
results = model.eval_model(eval_df)

# Use the model for prediction

# print(model.predict(["Japan began the defence of their Asian Cup title with a lucky 2-1 win against Syria in a Group C championship match on Friday."]))
# print(model.predict(["replace entity to label : The widely extensive Bunter sandstone ( Lower Triassic ) in SW Germany represents an important aquifer system which call be subdivided into an unconfined zone ( Black Forest ) and a confined zone covered by carbonates with evaporites and continental beds . As a result of erosion , valley incision and significant long-term groundwater circulation in the unconfined zone , most of the carbonate cement and high TDS formation water of the Bunter has been leached ."]))
# print(model.predict(["The widely extensive Bunter sandstone ( Lower Triassic ) in SW Germany represents an important aquifer system which call be subdivided into an unconfined zone ( Black Forest ) and a confined zone covered by carbonates with evaporites and continental beds . As a result of erosion , valley incision and significant long-term groundwater circulation in the unconfined zone , most of the carbonate cement and high TDS formation water of the Bunter has been leached ."]))