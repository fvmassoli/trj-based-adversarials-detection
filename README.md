# Detection of Face Recognition Adversarial Attacks

This repository contains the code relative to the paper "[Detection of Face Recognition Adversarial Attacks](https://www.sciencedirect.com/science/article/pii/S1077314220301296)" by Fabio Valerio Massoli (ISTI - CNR), Fabio Carrara (ISTI - CNR), Giuseppe Amato (ISTI - CNR), and Fabrizio Falchi (ISTI - CNR).

It reports a new technique to detect adversarial attacks against a face recognition system. 

Moreover, we showed that our technique can also be used to detect adversrial attacks in more generic settings.


## Adversarial Detection 

Proposed detection approach

<p align="center">
<img src="https://github.com/fvmassoli/trj-based-adversarials-detection/blob/master/images/img1.png"  alt="Detection technique">
</p>


Comparison of the distances, from class centroids, of adversarial attacks generated with label-focused attacks and with deep features attacks for which we used a kNN algorithm as guidance for the adversarial instance generation.

<p align="center">
<img src="https://github.com/fvmassoli/trj-based-adversarials-detection/blob/master/images/img2.png"  alt="" width="500" height="350">
</p>

## Model checkpoints

The checkpoints are relative to models reported in **Table 1** of the paper

| Configuration (Targeted attacks) | BIM | CW | MI-FGSM | Macro-AUC |
| --- | --- | --- | --- | --- |
| [LSTM + M + L_2](https://drive.google.com/file/d/1T-k6sTbcwDQV5GhV9UpQ5KhKMuTqbJr-/view?usp=sharing) | 0.977 | 0.871 | 0.986 | 0.944 |
| [LSTM + C + L_2](https://drive.google.com/file/d/1dggQ7G2kYw0D8P7qE_IUJHStP2sEKYHZ/view?usp=sharing) | 0.970 | 0.857 | 0.982 | 0.936 |
| [LSTM + M + cos](https://drive.google.com/file/d/1AjGGi9cePIfjwZ2xNiCaQqmzNAUqWVuy/view?usp=sharing) | 0.986 | 0.904 | 0.991 | 0.960 |
| [LSTM + C + cos](https://drive.google.com/file/d/1KFZHKGfCp_QNc9nCRms97E-Kzc3Nxxgh/view?usp=sharing) | 0.968 | 0.895 | 0.981 | 0.948 |

| Configuration (Unargeted attacks) | BIM | CW | MI-FGSM | Macro-AUC | 
| --- | --- | --- | --- | --- |
| [LSTM + M + L_2](https://drive.google.com/file/d/1J0_U-jf_HkyDDtBja53G1UlbSANH6Ecg/view?usp=sharing) | 0.878 | 0.615 | 0.889 | 0.794 |
| [LSTM + C + L_2](https://drive.google.com/file/d/1s5LKjIzfwdlS5Aa0ayuHexw4nGwQlC_Q/view?usp=sharing) | 0.863 | 0.596 | 0.869 | 0.776 |
| [LSTM + M + cos](https://drive.google.com/file/d/1eRgWg6ePD_-dUQNk3aFct4NSGuWEgJzg/view?usp=sharing) | 0.929 | 0.599 | 0.930 | 0.819 |
| [LSTM + C + cos](https://drive.google.com/file/d/1HrRvNg5-SoxJ-K7FKs5yOpk6L6kq_yFI/view?usp=sharing) | 0.884 | 0.568 | 0.886 | 0.779 |



## How to run the code

### Train and test the detector on MNIST and CIFAR-10 datasets.

#### Run the following commands inside the ```detect_attack_benchmark_datasets``` folder

1. Features extraction and class representatives evaluation

        python exctract_features_eval_class_representatives.py -ex -ma small
        python exctract_features_eval_class_representatives.py -ex -ma wide

2. Generate adversarial instances
(example of PGD attack)

        python generate_adversarials.py -an pgd -e <epsilon_value> -ni <number_of_steps> -ma <model_architecture> -cr <attack_criterion>

3. Detector training

        python train_detector.py 

4. Detector test

        python train_detector.py -tt -dck <detector_checkpoint>
  


### White box attacks on MNIST and VGGFace2

#### Run the following commands inside the ```white_box_attacks``` folder

1. White box attacks with the vggface2 dataset

        python3 white_box_attacks.py -atk deep -ds vggface2

1. White box attacks with the MNIST dataset

        python3 white_box_attacks.py -atk cw2 -ds mnist


## Reference
For all the details about the detection technique and the experimental results, please have a look at the [paper](https://www.sciencedirect.com/science/article/pii/S1077314220301296).

To cite our work, please use the following form

```
@article{massoli2020detection,
  title={Detection of Face Recognition adversarial attacks},
  author={Massoli, Fabio Valerio and Carrara, Fabio and Amato, Giuseppe and Falchi, Fabrizio},
  journal={Computer Vision and Image Understanding},
  pages={103103},
  year={2020},
  publisher={Elsevier}
}
```

## Contacts 
If you have any question about our work, please contact [Dr. Fabio Valerio Massoli](mailto:fabio.massoli@isti.cnr.it). 


Have fun! :-D
