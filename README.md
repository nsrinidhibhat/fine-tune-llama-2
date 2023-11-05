# Fine-tuning Llama-2

This project aims to fine-tune the Llama-2 language model using Hugging Face's Transformers library. By following these steps, you can fine-tune the model and use it for inference.

---
## Prerequisites

Before getting started, make sure you have the following:

- Hugging Face API token (HF token)
- Python installed on your system
  
---
## Setup

1. Clone this repository to your local machine.

2. Create a `.env` file in the project directory and add your Hugging Face API token:
   ```HUGGING_FACE_API_KEY = "your_HF_API_key"```<br>
   The code for training (train.py) has the code to pick this API key up.<br>


*PS:* Google Colab has added a new Secrets function to store your API keys. Add Name, Value to the Secrets, and run the following:
  ```python
  from google.colab import userdata
  userdata.get('secretName')
  ```

3. Install the required packages using the following command:

   ```bash
   pip install -r requirements.txt
   ```

*PS:* Sometimes the code might throw an error for not finding the right version of bitsandbytes downloaded. Try debugging with the following installation: <br>
  
  ```shell
  !pip install -i https://test.pypi.org/simple/ bitsandbytes
  ```


---
## Fine-tuning Llama-2

To fine-tune Llama-2, run the following command:

```shell
!python -c "from your_module import Train; train_llm = Train(); train_llm.train_llama()"
```
This command will fine-tune the model and save it to the model_ft folder.

Custom Data Ingestion
To ingest your own data for fine-tuning, you'll need to modify the code in your script. I have provided one example here:

```python
#Reading the file
data = pd.read_excel("your_dataset.xlsx")

# Convert the pandas DataFrame to Hugging Face's Dataset
hf_dataset = Dataset.from_pandas(data)

```
Happy fine-tuning!

---
## Inference

To perform inference using the fine-tuned Llama-2 model, follow these steps:

1. Ensure you've successfully fine-tuned Llama-2 as explained in the Fine-tuning Llama-2 section.

2. Run the inference script, `infer.py`, with the following command:

   ```shell
   !python infer.py
   ```
