"""
## References

## FIXME: you.com code has bugs (traceback at bottom)
- https://you.com/search?q=python+question+answering+transformers+bert
- https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering
"""
from transformers import BertForQuestionAnswering, BertTokenizer, TrainingArguments, Trainer
from transformers.data.processors.squad import SquadV2Processor

from nlpia2.constants import SRC_DATA_DIR

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

processor = SquadV2Processor()
train_examples = processor.get_train_examples(data_dir=SRC_DATA_DIR / 'big' / 'squad_v2')


# Initialize the BERT model for question answering
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

# Set up the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=2e-5,
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    evaluation_strategy='epoch',
    save_total_limit=2,
    optim='adam',
)

# Set up the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_examples,
)

# Train the model
trainer.train()


"""
>>> from nlpia2.ch09.train_BERT_QA import *
100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 442/442 [01:01<00:00,  7.15it/s]
Downloading: 100%|████████████████████████████████████████████████████████████████████████████████████| 440M/440M [00:34<00:00, 12.6MB/s]
Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertForQuestionAnswering: ['cls.predictions.transform.dense.weight', 'cls.predictions.bias', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.dense.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.decoder.weight', 'cls.seq_relationship.bias', 'cls.seq_relationship.weight']
- This IS expected if you are initializing BertForQuestionAnswering from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertForQuestionAnswering from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of BertForQuestionAnswering were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['qa_outputs.weight', 'qa_outputs.bias']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
/home/hobs/code/tangibleai/nlpia2/.venv/lib/python3.9/site-packages/transformers/optimization.py:306: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
  warnings.warn(
***** Running training *****
  Num examples = 130319
  Num Epochs = 2
  Instantaneous batch size per device = 16
  Total train batch size (w. parallel, distributed & accumulation) = 32
  Gradient Accumulation steps = 2
  Total optimization steps = 8144
  Number of trainable parameters = 108893186
  0%|                                                                                                           | 0/8144 [00:00<?, ?it/s]---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[30], line 1
----> 1 from nlpia2.ch09.train_BERT_QA import *

File ~/code/tangibleai/nlpia2/src/nlpia2/ch09/train_BERT_QA.py:34
     27 trainer = Trainer(
     28     model=model,
     29     args=training_args,
     30     train_dataset=train_examples,
     31 )
     33 # Train the model
---> 34 trainer.train()

File ~/code/tangibleai/nlpia2/.venv/lib/python3.9/site-packages/transformers/trainer.py:1543, in Trainer.train(self, resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
   1538     self.model_wrapped = self.model
   1540 inner_training_loop = find_executable_batch_size(
   1541     self._inner_training_loop, self._train_batch_size, args.auto_find_batch_size
   1542 )
-> 1543 return inner_training_loop(
   1544     args=args,
   1545     resume_from_checkpoint=resume_from_checkpoint,
   1546     trial=trial,
   1547     ignore_keys_for_eval=ignore_keys_for_eval,
   1548 )

File ~/code/tangibleai/nlpia2/.venv/lib/python3.9/site-packages/transformers/trainer.py:1765, in Trainer._inner_training_loop(self, batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval)
   1762     self._load_rng_state(resume_from_checkpoint)
   1764 step = -1
-> 1765 for step, inputs in enumerate(epoch_iterator):
   1766 
   1767     # Skip past any already trained steps if resuming training
   1768     if steps_trained_in_current_epoch > 0:
   1769         steps_trained_in_current_epoch -= 1

File ~/code/tangibleai/nlpia2/.venv/lib/python3.9/site-packages/torch/utils/data/dataloader.py:628, in _BaseDataLoaderIter.__next__(self)
    625 if self._sampler_iter is None:
    626     # TODO(https://github.com/pytorch/pytorch/issues/76750)
    627     self._reset()  # type: ignore[call-arg]
--> 628 data = self._next_data()
    629 self._num_yielded += 1
    630 if self._dataset_kind == _DatasetKind.Iterable and \
    631         self._IterableDataset_len_called is not None and \
    632         self._num_yielded > self._IterableDataset_len_called:

File ~/code/tangibleai/nlpia2/.venv/lib/python3.9/site-packages/torch/utils/data/dataloader.py:671, in _SingleProcessDataLoaderIter._next_data(self)
    669 def _next_data(self):
    670     index = self._next_index()  # may raise StopIteration
--> 671     data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    672     if self._pin_memory:
    673         data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)

File ~/code/tangibleai/nlpia2/.venv/lib/python3.9/site-packages/torch/utils/data/_utils/fetch.py:61, in _MapDatasetFetcher.fetch(self, possibly_batched_index)
     59 else:
     60     data = self.dataset[possibly_batched_index]
---> 61 return self.collate_fn(data)

File ~/code/tangibleai/nlpia2/.venv/lib/python3.9/site-packages/transformers/trainer_utils.py:700, in RemoveColumnsCollator.__call__(self, features)
    698 def __call__(self, features: List[dict]):
    699     features = [self._remove_columns(feature) for feature in features]
--> 700     return self.data_collator(features)

File ~/code/tangibleai/nlpia2/.venv/lib/python3.9/site-packages/transformers/data/data_collator.py:70, in default_data_collator(features, return_tensors)
     64 # In this function we'll make the assumption that all `features` in the batch
     65 # have the same attributes.
     66 # So we will look at the first element as a proxy for what attributes exist
     67 # on the whole batch.
     69 if return_tensors == "pt":
---> 70     return torch_default_data_collator(features)
     71 elif return_tensors == "tf":
     72     return tf_default_data_collator(features)

File ~/code/tangibleai/nlpia2/.venv/lib/python3.9/site-packages/transformers/data/data_collator.py:136, in torch_default_data_collator(features)
    134             batch[k] = torch.tensor(np.stack([f[k] for f in features]))
    135         else:
--> 136             batch[k] = torch.tensor([f[k] for f in features])
    138 return batch

ValueError: too many dimensions 'str'
"""