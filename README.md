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
