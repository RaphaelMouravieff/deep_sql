
## 🔧 Installation

> 📌 **Reproducing results?** Just copy-paste this to get started with TabStruct.

```bash
# Clone the repo
git clone https://github.com/RaphaelMouravieff/TabStruct.git TabStruct
cd TabStruct

# Set up the environment
conda create -n tabstruct python=3.11.11 -y
conda activate tabstruct

# Install dependencies
pip install -r requirements.txt
```


**Connect to the GPU Node**
```bash ssh gpu ``` 
**Request a GPU with Slurm**
```bash 
srun -p hard --gpus-per-node=1 --constraint=A6000 --pty bash 
```
**Activate the Conda Environment**
```bash 
conda activate [env] 
``` 
**Starting Ollama**
```bash 
ollama serve & 
``` 
**Running the Code Agents**: Create the data
```bash 
cd deep_sql/scripts 
bash Agents/run.sh 
```
**Clean/Increase the dataset generated**: Clean the generated data + merged files
```bash
cd deep_sql/scripts 
bash Datasets/prepare.sh 
```
**Pre-train the clean dataset**: Pre-trained the model on the dataset
```bash 
cd deep_sql/scripts 
bash Train/ptrain.sh 
```
**Fine-tuned the clean dataset**: Fine-tuning on wikitablequestions
```bash
cd deep_sql/scripts 
bash Train/fine_tuned.sh 
```

**Uni-test Library**: Check if multi chunk work to save library, one common vector store and multiple .json. 
```bash
cd deep_sql
python -m uni_test.library_multi_chunk
```

**Uni-test answer check**: Check vectore_store_content (step1= 96766)
```bash
cd deep_sql
python -m uni_test.vectore_store_content --vector_store_path data/library/vector_store_step_copy
```

**Uni-test likelihood**: Create the likelihood threeshold
```bash
cd deep_sql
python -m uni_test.find_likelihood_threeshold 
```