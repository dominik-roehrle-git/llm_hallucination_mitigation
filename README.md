This is the repository for the research:  
**Addressing Hallucination in Large Language Models: Detection and  
Mitigation Techniques for Claim-Based Verification**

Please read the documentation first:  
[View PDF](https://github.com/dominik-roehrle-git/llm_hallucinations/blob/main/docs/Masterarbeit_Dominik_R%C3%B6hrle.pdf)

To download LLMs create an account on [Hugging Face](https://huggingface.co/).  
The access to Llama has to be requested first: [Hugging Face Llama](https://huggingface.co/meta-llama) and [Meta Llama Downloads](https://llama.meta.com/llama-downloads).  
Also, the [BART-Large-MNLI model](https://huggingface.co/facebook/bart-large-mnli) has to be downloaded.

To install PyTorch, go to [pytorch.org](https://pytorch.org/) and follow the instructions (2.5.1 cu118), then install the other packages with:

    pip install -r requirements.txt

**Note for Windows Users:**  
The library `bitsandbytes` does not work on Windows. Remove it from `requirements.txt` first and then:  
Visit: [GitHub Issue for bitsandbytes Windows](https://github.com/d8ahazard/sd_dreambooth_extension/issues/7)  
and type:

    pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl

Also for Windows you need to install:

    pip install unicodedata2==15.1.0

if you want to generate evidence.

The repository is divided into three parts:
- **Probes**
- **Mitigation**
- **Entity Analysis**

---

## Probes

### To create the dataset for the probes (in `Probes/`):

- **Download the datasets:**  
  [Raw FEVER and HoVer datasets + already generated files](https://1drv.ms/u/s!AhVn8Lx_iIapipk3hFU1JRpMosF--g?e=llFsno)

- **Generate Evidence:**

    python generate_evidence.py

  *(Creates evidence text from HoVer and FEVER with an LLM.)*

- **Generate Mini-Facts and Sentences:**

    python generate_mini_facts_and_sentences.py

  *(Creates the labelled mini-facts from the evidence, generates the sentences and removes sentences that are not aligned with BART-large-MNLI.)*

- **Generate Embeddings:**

    python generate_embeddings.py

  *(Generates the embeddings of mini-facts and sentences.)*

- **Split and Balance the Dataset:**

    python train_test_split_balance.py

  *(Splits the datasets and balances them.)*

- **Train the Probes:**

    python TrainProbes.py

  *(Trains the probes with a feedforward neural network.)*

- **Evaluation:**  
  Go to `model_test.ipynb` to run the evaluation.

### To run the results from the existing probes:

- **Download the dataset with embeddings (8 GB):**  
  [Download Dataset](https://1drv.ms/u/s!AhVn8Lx_iIapipkfm8s7rjAHUdhQLQ?e=nX4arh)  
  *(Make sure the folders inside the zip are inside `Probes/`. The embeddings are indexed from the last hidden layer: `-1, -8, -16, -24` and the first layer (`1`), corresponding to `32, 25, 17, 9, and 1`.)*

- **Download the probes (`.pth` files):**  
  [Download Probes](https://1drv.ms/f/s!AhVn8Lx_iIapipkgKEms9E6yilRC5A?e=rwuBT8)  
  *(Make sure the folder is inside `Probes/`.)*

- **Evaluation:**  
  Go to `model_test.ipynb` to run the evaluation.

### To run the evaluation for the baselines:

- **Download the baseline data:**  
  [Download Baselines Data](https://1drv.ms/u/s!AhVn8Lx_iIapipk2ydzSyocV3ZZhJg?e=YLcEo7)  
  *(Make sure the folders are inside `Probes/`.)*

- **Evaluation:**  
  Go to `evaluate_probs.ipynb`.

---

## Mitigation

### To recreate the dataset for finetuning (in `Mitigation/`):

- **Download the Finetuning Dataset:**  
  [Download Labelled Mini-Facts with Associated Evidence](https://1drv.ms/f/s!AhVn8Lx_iIapipk6-VBvpNUn3pPbIw?e=NpvXF6)  
  *(Make sure the files are inside `Mitigation/`.)*

- **Pre-generate Corrections with OpenAI:**

    python generate_corrections.py

- **Split the Dataset:**

    python train_test_split.py

  *(Splits the dataset into train, dev, and test.)*

- **Create Train Dataset:**

    python create_train_dataset.py

  *(For train and dev: removes samples that are not aligned with BART-Large-MNLI and balances the dataset.)*

- **Finetuning:**  
  Go to `finetuning.ipynb` to finetune the LLM.

- **Evaluation:**

    python evaluate.py

  *(Evaluates the LLMs and the finetuned LLM with BART-Large-MNLI.)*

### To get the already finetuned LLM (make sure to change the path to your base model in `adapter_config.json`):

- **Download the Pre-Finetuned LLM:**  
  [Download Finetuned LLM](https://1drv.ms/u/s!AhVn8Lx_iIapiplTSTcnfV0ZfQz2-g?e=9fUSoi)

### To get the already created datasets for finetuning:

- **Download the Finetuning Datasets:**  
  [Download Finetuning Datasets](https://1drv.ms/u/s!AhVn8Lx_iIapipk9fqq85l-4SFZepg?e=5fA4mM)  
  *(Make sure every folder of the zip is inside `Mitigation/`.)*

---

### To run the application scenario (in `Mitigation/`):

- **Download the Application Folder:**  
  [Download Application Folder](https://1drv.ms/f/s!AhVn8Lx_iIapiplAmL3VAzH2Sxp0Rw?e=zQcpOZ)  
  *(Make sure the folder `application` is inside `Mitigation/`.)*

- **Run the Application:**

    python application.py

  *(Make sure to select a probe and the fine-tuned LLM.)*

- **Evaluation:**  
  Go to `evaluate_application.ipynb` to evaluate the approaches.

---

## Entity Analysis

### To recreate the real train samples:

- Go to `gen_real_samples.ipynb`

### To run the evaluation of the probes on the test datasets (in `Entity Analysis/`):

- **Download the Test Datasets:**  
  [Download Test Datasets](https://1drv.ms/u/s!AhVn8Lx_iIapiplQt1lFdOiEcJmxZw?e=DNyGw8)  
  *(Make sure every folder is inside `Entity Analysis`.)*

- **Evaluation:**  
  Run `model_test.ipynb` to test the probes on the test datasets.  
  Run `results.ipynb` to plot the probability distributions.

### To get the already created dataset:

- **Download the Entity Analysis Dataset:**  
  [Download Entity Analysis Dataset](https://1drv.ms/u/s!AhVn8Lx_iIapiplROt5adNLKBiQ-xw?e=a6TfhQ)

---

**Note:**  
Some coding was conducted with the help of GitHub Copilot, inspirations for finetuning were taken from [this Kaggle notebook](https://www.kaggle.com/code/zivicmilos/llm-finetuning), for token SAR from [this repository](https://github.com/jinhaoduan/SAR), and for the probes from [this repository](https://github.com/balevinstein/Probes).









