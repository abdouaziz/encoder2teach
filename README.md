# Description 
We explore the potential for language models, such as BERT, to teach Wav2vec2 representation learning. Creating audio data for automatic speech recognition (ASR) can be challenging, especially when large quantities of data are needed. While textual data is easier to collect, language models have demonstrated impressive results in learning contextual representations that are useful for a range of applications.

The central question of this study is whether language models can effectively teach models like Wav2vec2 to learn representations. The proposed approach involves freezing a pre-trained language model and comparing its output representation with a student model that will learn to read the representation.

# Installation
First, clone the repository and install the requirements.

```bash
pip install -r requirements.txt
```

# Module Usage

```python
import Trainer

# Initialize Trainer
trainer = Trainer(
    model_name="bert-base-uncased",
    dataset_name="patrickvonplaten/librispeech_asr_dummy",
    batch_size=4,
    epocs=1,
    learning_rate=2e-5,
    report_to=False,
)

# Train Model
trainer.train()

# Save Model
trainer.save("model.pt")
```