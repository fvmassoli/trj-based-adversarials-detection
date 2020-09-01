# Code Usage


### Train and test the detector on the VGGFace2 dataset.

### Run the following commands inside the ```face_attacks``` folder



### Train and test the detector on MNIST and CIFAR-10 datasets.

### Run the following commands inside the ```detect_attack_benchmark_datasets``` folder

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

### Run the following commands inside the ```white_box_attacks``` folder

1. White box attacks with the vggface2 dataset

        python3 white_box_attacks.py -atk deep -ds vggface2

1. White box attacks with the MNIST dataset

        python3 white_box_attacks.py -atk cw2 -ds mnist
