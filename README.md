# GFPC-General-Format-Progressive-Learned-Image-Compression

## <div align="center">Quick Start Examples</div>

<details open>
<summary>Install</summary>


Clone repo and
install [requirements.txt](https://github.com/DuBianJun-007/GFPC-General-Format-Progressive-Learned-Image-Compression/blob/main/requirements.txt)
in a
[**Python>=3.8.0**](https://www.python.org/) environment, including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/DuBianJun-007/GFPC-General-Format-Progressive-Learned-Image-Compression.git  # clone
cd GFPC-General-Format-Progressive-Learned-Image-Compression
pip install -r requirements.txt  # install
```

</details>

<details open>
<summary>Model download</summary>

Downloaded model, put into mod folder

| lambda | Link                                                                                         |
|--------|----------------------------------------------------------------------------------------------|
| 0.09   | [0.09](https://drive.google.com/file/d/1vGWt-qF_DEKKhi-mtSpO7S_zkUCXUBvQ/view?usp=sharing)   |
| 0.07   | [0.07](https://drive.google.com/file/d/1F250henyzJ4CHo_lLmbNL1pi44MvLeYI/view?usp=sharing)   |
| 0.045  | [0.045](https://drive.google.com/file/d/1-IsjdzaQYWhcuP1ZfCiHX0Sw5uILIBf9/view?usp=sharing)  |
| 0.03   | [0.03](https://drive.google.com/file/d/1hQIzHrTZUXazDdw8uraAoaJfLyOYUpk8/view?usp=sharing)   |
| 0.015  | [0.015](https://drive.google.com/file/d/1JKwX9YvmSaZ68V0c-fzGn2NQ0ovRo-ld/view?usp=sharing)  |
| 0.0075 | [0.0075](https://drive.google.com/file/d/1Ks4DwXiU3vAZ6DE6VeIy_2BpIP_7Qqs9/view?usp=sharing) |
| 0.0038 | [0.0038](https://drive.google.com/file/d/1QGBgZaCwVqUVcKp0R8vSlG62ir13bbIJ/view?usp=sharing) |
| 0.002  | [0.002](https://drive.google.com/file/d/11wwxVzyMjCH2GgFg4LavoJbp3YiQO_dr/view?usp=sharing)  |

<details open>
<summary>Inference</summary>

Sample inference test under the specified dataset, the test dataset should be put to the _dataset_val_ file:

```bash
#Defaults to the Kodak dataset
python Inference_GFPC.py 
#Specify a directory
python Inference_GFPC.py --dataset XXX
```

Requirements Package：
```
numpy~=1.26.1
pillow~=10.1.0
compressai~=1.1.5
opencv-python~=4.8.1.78
simpleitk~=2.3.0
```

</details>


<details open>
<summary>Training</summary>

The training data should be placed in the _dataset_train_ folder, directory structure:

```
--dataset_train
  --train
  --test
```

Sample training execution:

```bash
#Starting a new training program.
python train_GFPC.py --lambda 0.09 --batch-size 8 
#Continuing training in a checkpoint
python train_GFPC.py --lambda 0.09 --batch-size 8 --checkpoint XX
```

_checkpoint_ directory structure:

```
--checkpoint
  --best_checkpoint
  --checkpoint    
  --log  
  --updataModel
```

After training is complete, the model needs to be updated:

```bash
python updata.py --run XX  # XX is the checkpoint number, e.g. 01
```
The updated model is stored in the updataModel folder for the checkpoint.

Some experimental results:
<div style="display: flex;">
    <div style="flex: 1; margin: 5px;">
        <img src="PSNR-BPP-Kodak.png" alt="Image 1" style="max-width: 100%; height: auto;">
    </div>
    <div style="flex: 1; margin: 5px;">
        <img src="MSSSIM-BPP-Kodak.png" alt="Image 2" style="max-width: 100%; height: auto;">
    </div>
</div>



</details>