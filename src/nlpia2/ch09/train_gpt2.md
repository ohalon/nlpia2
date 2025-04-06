# [fine-tuning-GPT2](https://github.com/itsuncheng/fine-tuning-GPT2
)

This repo contains the code for the Medium Article: [Fine-tuning GPT2 for Text Generation Using Pytorch](https://towardsdatascience.com/fine-tuning-gpt2-for-text-generation-using-pytorch-2ee61a4f1ba7).

The `run_language_modeling.py` and `run_generation.py` are originally from Huggingface with tiny modifications.

```
wget https://www.cs.cmu.edu/~dbamman/data/booksummaries.tar.gz
tar -xvzf booksummaries.tar.gz
mkdir data
mv booksummaries* data
```

## [Intro](https://towardsdatascience.com/fine-tuning-gpt2-for-text-generation-using-pytorch-2ee61a4f1ba7)

The past few years have been especially booming in the world of NLP. This is mainly due to one of the most important breakthroughs of NLP in the modern decade — [**Transformers**](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf). If you haven’t read my previous article on [**BERT for text classification**](https://towardsdatascience.com/bert-text-classification-using-pytorch-723dfb8b6b5b), go ahead and take a look! Another popular transformer that we will talk about today is [**GPT2**](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). Developed by OpenAI, GPT2 is a large-scale transformer-based language model that is pre-trained on a large corpus of text: 8 million high-quality webpages. It results in competitive performance on multiple language tasks using only the pre-trained knowledge without explicitly training on them. GPT2 is really useful for language generation tasks as it is an autoregressive language model.

Here in today’s article, we will dive deeply into how to implement another popular transformer, GPT2, to write interesting and creative stories! Specifically, we will test the ability of GPT2 to write creative book summaries using the [CMU Books Summary Dataset](http://www.cs.cmu.edu/~dbamman/booksummaries.html). We will be using the [Huggingface](https://huggingface.co/) repository for building our model and generating the texts.

The **entire codebase** for this article can be viewed [here](https://github.com/itsuncheng/fine-tuning-GPT2).

# Step 1: Prepare Dataset

Before building the model, we need to download and preprocess the dataset first.

We are using The CMU Books Summary Dataset, which contains 16,559 books extracted from Wikipedia along with the metadata including title, author, publication date, genres, and plot summary. Download the dataset [here](http://www.cs.cmu.edu/~dbamman/booksummaries.html). Here is what the dataset looks like:


For data preprocessing, we first split the entire dataset into the train, validation, and test datasets with the train-valid-test ratio: 70–20–10. We add a bos token <BOS> to the start of each summary and eos token <EOS> to the end of each summary for later training purposes. We finally save the summaries into .txt files, getting train.txt, valid.txt, test.txt.

You can get the preprocessing notebook [here](https://github.com/itsuncheng/fine-tuning-GPT2/blob/master/preprocessing.ipynb).

# Step 2: Download libraries

To build and train GPT2, we need to install the Huggingface library, as well as its repository.

Install Huggingface library:

```
pip install transformers
```

Clone Huggingface repo:

```
git clone github.com/huggingface/transformers
```

If you want to see visualizations of your model and hyperparameters during training, you can also choose to install tensorboard or wandb:

```
pip install tensorboardpip install wandb; wandb login
```

# Step 3: Fine-tune GPT2

Before training, we should set the bos token and eos token as defined earlier in our datasets.

We should also set the pad token because we will be using _LineByLineDataset_, which will essentially treat each line in the dataset as distinct examples. In _transformers/example/language-modeling/run-language-modelling.py_, we should append the following code for the model before training:

```
special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>', 'pad_token': '<PAD>'}num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)model.resize_token_embeddings(len(tokenizer))
```

After running this code, the special tokens will be added to the tokenizer and the model will resize its embedding to fit with the modified tokenizer.

For training, we define some parameters first and then run the language modeling script:

```
cd transformers/example/language-modelingN=gpu_numOUTPUT_DIR=/path/to/modelTRAIN_FILE=/path/to/dataset/train.txtVALID_FILE=/path/to/dataset/valid.txtCUDA_VISIBLE_DEVICES=$N python run_language_modeling.py \--output_dir=$OUTPUT_DIR \--model_type=gpt2 \--model_name_or_path=gpt2 \--do_train \--train_data_file=$TRAIN_FILE \--do_eval \--eval_data_file=$VALID_FILE \--per_device_train_batch_size=2 \--per_device_eval_batch_size=2 \--line_by_line \--evaluate_during_training \--learning_rate 5e-5 \--num_train_epochs=5
```

We set per\_device\_train\_batch\_size=2 and per\_device\_eval\_batch\_size=2 because of the GPU constraints. Feel free to use a batch size that fits your GPU. We use line\_by\_line, which tells our model to treat each line in our dataset as an individual example, as explained earlier. Evaluate\_during\_training runs evaluation on the evaluation dataset after each `logging_steps`, which is defaulted to 500.

In case you want to continue training from the last checkpoint, you can run:

```
CUDA_VISIBLE_DEVICES=$N python run_language_modeling.py \--output_dir=$OUTPUT_DIR \--model_type=gpt2 \--model_name_or_path=$OUTPUT_DIR \--do_train \--train_data_file=$TRAIN_FILE \--do_eval \--eval_data_file=$VALID_FILE \--per_device_train_batch_size=2 \--per_device_eval_batch_size=2 \--line_by_line \--evaluate_during_training \--learning_rate 5e-5 \--num_train_epochs=5 \--overwrite_output_dir
```

# (Optional ) Step 4: Evaluate Perplexity on Test Dataset

This step is optional depending on whether you want to evaluate the performance of your trained GPT2. You can do this by evaluating perplexity on the test dataset.

```
TEST_FILE=/path/to/dataset/test.txtCUDA_VISIBLE_DEVICES=$N python run_language_modeling.py \--output_dir=$OUTPUT_DIR \--model_type=gpt2 \--model_name_or_path=$OUTPUT_DIR \--do_eval \--eval_data_file=$TEST_FILE \--per_device_eval_batch_size=2 \--line_by_line
```

Here, in my case, we attained a loss of 2.46 and a perplexity of 11.70 after training for 5 epochs:

![](https://miro.medium.com/v2/resize:fit:700/0*5NDcfkSit80colx2)

Image by author

# Step 5: Generate Text

Before generating texts using our trained model, we first enable special tokens in our prompt by setting `add_special_tokens=True` in the _transformers/examples/text-generation/run\_generation.py_:

```
encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=True, return_tensors=”pt”)
```

Then, we are ready to generate some text! Start generating by:

```
cd transformers/examples/text-generationK=k_for_top-k_sampling_decoderCUDA_VISIBLE_DEVICES=$N python run_generation.py \--model_type gpt2 \--model_name_or_path $OUTPUT_DIR \--length 300 \--prompt "<BOS>" \--stop_token "<EOS>" \--k $K \--num_return_sequences 5
```

We feed in the prompt “<BOS>” as the input, which represents the beginning of each example and stops the model from generating once the “<EOS>” token is generated. This way, our GPT2 will learn to generate a full example of the summary from the beginning to the end, leveraging what it learned of the bos token and eos token during training. In addition, we are using the top-k sampling decoder which has been proven to be very effective in generating irrepetitive and better texts. k=50 is a good value to start off with. Huggingface also supports other decoding methods, including greedy search, beam search, and top-p sampling decoder. For more information, look into the [docstring](https://huggingface.co/transformers/main_classes/model.html?highlight=generate#transformers.TFPreTrainedModel.generate) of `model.generate`.

Here are a few examples of the generated texts with k=50.

> The protagonist is an Englishman, William Lark, who has been sent on an adventure with the British Government on a mission to the Arctic. The novel tells the story of how his friends and family are being sold into slavery in the small Norwegian town of Shok…
> 
> A new world is awakening, and the humans of the planet Vorta must work together to save it from destruction. The New Earth is now populated by three species. The first are the humans who are a bit older, the second are the Vorta, and the third are the humans with dark blue eyes…
> 
> The novel begins in the year 2143, when a group of “dungeons”, or witches, decide to break the spell that prevents the power of the dead by consuming the souls of those who died to them. They use the bodies to help the dying, as well as to raise the dead themselves…

You can see more generated examples [here](https://github.com/itsuncheng/fine-tuning-GPT2/blob/master/generated_summaries.txt).

# Conclusion

In this article, we showed how to implement one of the most popular transformer models, GPT2, to create interesting texts. GPT2’s large-scale pre-trained dataset and architecture allows it to produce coherent and fluent pieces of writing. Although GPT2’s texts are still distinguishable from those written by humans, this is proof that creativity by machines is only going upwards from now. For more info, you can take a look at the [official paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) or OpenAI’s [blog](https://openai.com/blog/better-language-models/) on GPT2.

This article only showed how to generate text that is determined by AI. If you are wondering whether it’s possible to control the text being generated (and it’s possible!), take a read at the following article I wrote 😊.

[

## Controlling Text Generation for Language Models

### Hands-on approach to control style and content of machine-generated text

towardsdatascience.com



](https://towardsdatascience.com/controlling-text-generation-from-language-models-6334935e80cf)

# References

\[1\] A. Vaswani, N. Shazeer, N. Parmar, etc., [Attention Is All You Need](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) (2017), 31st Conference on Neural Information Processing Systems

\[2\] A. Radford, J. Wu, R. Child, etc., [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (2019), OpenAI

