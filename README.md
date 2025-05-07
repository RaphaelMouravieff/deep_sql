# Deep SQL Guide 🚀

### 🔧 Slurm Setup: Connecting to GPU Node

Before running any code, you need to connect to the GPU cluster and activate the appropriate environment.

##### 1
 Connect to the GPU Node
```bash
ssh gpu
```
#### Request a GPU with Slurm
```bash
srun -p hard --gpus-per-node=1 --constraint=A6000 --pty bash
```
📌 Explanation:
	•	srun → Launches a Slurm job interactively.
	•	-p hard → Specifies the partition (hard).
	•	--gpus-per-node=1 → Requests 1 GPU.
	•	--constraint=A6000 → Ensures allocation of an A6000 GPU (you can try A5000).
	•	--pty bash → Starts an interactive bash session.

#### Activate the Conda Environment
```bash
conda activate [env]
```
🔹 Replace [env] with the name of your Conda environment.

⸻

### 🖥️ Starting Ollama

Ollama is required for the model. Start the Ollama server in the background:
```bash
ollama serve &
```

🔹 The & runs the server in the background so you can continue using the terminal.

⸻

### 🚀 Running the Code

Now you’re ready to run Deep SQL!

🔹 Running the Script Interactively

If you want to monitor execution in real-time, run:

```bash
cd deep_sql/scripts
bash Agents/run.sh
```

⸻

### Clean/Increase the dataset generated

```bash
cd deep_sql/scripts
bash Datasets/prepare.sh
```



### Pre-train the clean dataset

```bash
cd deep_sql/scripts
bash Train/ptrain.sh
```


### Fine-tuned the clean dataset

```bash
cd deep_sql/scripts
bash Train/fine_tuned.sh
```