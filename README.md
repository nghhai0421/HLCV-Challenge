# HLCV Image Classification Challenge

Implement and train an image classifier in **challenge\_train.py**. This README
contains the setup instructions, challenge rules, submission workflow and final
reproducibility requirements.

* [Overview and registration](#overview-and-registration)
* [Provided files and setup](#provided-files-and-setup)
* [Training and local evaluation](#training-and-local-evaluation)
* [Challenge rules](#challenge-rules)
* [Submitting and viewing results](#submitting-and-viewing-results)
* [Final reproducibility package](#final-reproducibility-package)

## Overview and registration

Teams have one or two members and share one submission allowance. Connect to the
University of Mannheim VPN and open the
[submission portal](https://kampen.informatik.uni-mannheim.de).

Select **Register a team** on the login page to open an email to the instructor.
Provide your team name and each member's full name and university email address;
a second member is optional. The instructor will send your Team ID and access
code. Both members use the same team access. Keep the code private and contact
the instructor if it is lost or shared.

After signing in, select **Download materials** under **Course materials** to
get the starter scripts, training and validation images, and this README.

## Provided files and setup

|File or folder|Purpose|
|-|-|
|`challenge\_train.py`|Editable starter with `Trainer.create\_model()`, `Trainer.create\_transform()` and `Trainer.save\_model()`.|
|`challenge\_test.py`|Fixed local evaluator. Keep this file unchanged.|
|`dataset/train`|Training images organized by class.|
|`dataset/val`|Public validation images organized by class.|
|`classes.json`|Class names in prediction-index order.|
|`requirements.txt`|Course package versions.|

Use the class-directory names as supplied. Both public splits use the same class
order as the submission server. The hidden test split is not included.

Use Python 3.12 with the package versions in **requirements.txt**. You may train
on a GPU; install a PyTorch build appropriate for your hardware using the course
environment instructions. Your submitted model must also run in the server's
[evaluation environment](#evaluation-environment).

## Training and local evaluation

The starter creates a small model and saves random initial weights; it does not
implement a training loop. To try it, open a terminal in this folder and run:

```sh
python challenge\_train.py
python challenge\_test.py
```

The first command creates **model.torch** with untrained starter weights. The
second evaluates it on the available train and val folders and writes
**result\_YYYY-MM-DD.txt**. Expect low accuracy until you implement training.
**The starter overwrites model.torch**, so do not run it over a trained checkpoint
you want to keep.

Implement the training loop using **dataset/train** and **dataset/val**, for
example with `torchvision.datasets.ImageFolder`. Implement
`Trainer.create\_model()` and `Trainer.create\_transform()` using the supplied
interfaces, and save the trained model state dictionary with
`Trainer.save\_model(model)`.

Keep training under the `if \_\_name\_\_ == "\_\_main\_\_":` guard. The server imports
your script for inference, so importing it must not start training, prompt for
input or require local training files. The checkpoint must support
`weights\_only=True` loading on the CPU. Follow the
[complete inference pipeline rule](#parameter-limit-for-the-complete-inference-pipeline)
when designing both the model and its transforms.

The fixed local evaluator retains its original report format. Portal reports
show overall and class-wise accuracy; their format may differ.

## Challenge rules

### Allowances and limits

|Item|Limit|
|-|-|
|Weekly test|One submission per team per calendar week; resets Monday at 00:00, Europe/Berlin.|
|Validation|Up to three requests per hour.|
|Pending work|One queued or running evaluation per team.|
|Training script|1 MB (1,000,000 bytes).|
|Checkpoint|10 MiB.|
|Evaluation time|Up to one hour after evaluation starts; queue time is additional.|
|Memory|2 GiB, with no extra swap.|
|CPU|0.75 CPU; evaluation is CPU-only.|

A failed weekly test also consumes the allowance. Contact the instructor if
there was a server problem. Validation does not consume the weekly test allowance.

### Parameter limit for the complete inference pipeline

The complete inference pipeline—from the input image to the final class
prediction—may use at most **10,000 model parameters in total**.

This includes the classifier and any models used for preprocessing, feature
extraction or other intermediate computations, including models called inside
`create\_transform()`. Trainable, frozen and pretrained parameters all count
toward the limit.

Storing model weights in buffers, arrays or constants, or calling a model outside
`create\_model()`, does not exempt those parameters from the limit.

### Evaluation environment

The evaluation server uses Python 3.12, PyTorch 2.8.0 CPU, torchvision 0.23.0 CPU,
NumPy 2.1.2, Pillow 11.0.0 and tqdm 4.67.1. The complete package list is published
with the course materials. Evaluation has no internet access. Do not download
weights or install packages from your submitted script.

Server inference uses batches of 64 and zero data-loader workers. Predictions
must be independent of batch size, image order, filenames and other submissions.

### Deadline and submission availability

Your dashboard shows the challenge deadline in **Europe/Berlin** time. Complete
validation early enough to submit the weekly test before that time. Starting an
upload before the deadline is not enough: the server must accept it before the
cutoff. Both validation uploads and weekly submissions close at the deadline.
Already accepted evaluations may finish afterward.

After the weekly allowance is used, **Weekly reset in** shows the countdown to
the next reset. The dashboard shows **Final submission week** if the challenge
ends before another allowance becomes available.

## Submitting and viewing results

### Upload, validate and submit

1. Optionally give the submission a name (up to 80 characters), then upload
**challenge\_train.py** and a saved checkpoint (`.torch`, `.pt`, `.pth` or `.ckpt`).
2. Select **Run validation** to evaluate the files on the public validation split.
3. After validation succeeds, select that run under **Choose your weekly model**,
confirm the weekly submission and submit.
4. Refresh the history to see whether the run is queued, running, completed or
failed. A completed run shows its accuracy and a **Download .txt** link.

The weekly test uses exactly the files from the selected validation and copies
its current name. If either file changes, upload both and validate again.

### History and submission names

History shows 20 runs per page, newest first.

Use **Rename** or **Name submission** to label a run; leave the field blank to
remove its name. Renaming changes only that run and its future downloads, without
affecting files, scores or the weekly allowance. Submission names are visible
only to your team and instructors.

### Leaderboard and reports

The landing-page leaderboard shows each active team's name and best completed
weekly-test accuracy in the current challenge. Validation scores do not count;
teams without a completed test have no rank, and equal scores share a rank.
Other submission details and downloadable reports are visible only to the team
and instructors.

Use **Download .txt** beside a completed run. Weekly-test reports include the
submission name, overall and per-class accuracy percentages, and file hashes.
Class-wise accuracy is the fraction of true examples in that class predicted
correctly. Public-validation reports additionally include example counts, the
integer-count confusion matrix and Evaluation duration. Each report covers the
split evaluated by that run.

## Final reproducibility package

Every team must submit a reproducibility package at the end of the challenge.
It must allow the instructor to run the training procedure used to produce the
team's final submitted model and evaluate the resulting checkpoint. Providing
only a checkpoint or inference code does not satisfy this requirement.

Include:

* The complete training code and any supporting scripts needed to prepare data,
train the model, save the checkpoint and evaluate it.
* The final **challenge\_train.py**, model checkpoint and corresponding portal
submission ID, so the package can be matched to the submitted model.
* The training configuration: data splits, preprocessing and augmentation,
model configuration, initialization, optimizer, learning-rate schedule,
batch size, number of epochs and checkpoint-selection procedure.
* Random seeds and any settings used to make training reproducible.
* The Python and package versions. GPU training is allowed; the resulting model
must still satisfy the server evaluation rules in this guide.
* A README with the exact commands to prepare the environment, locate the course
data, run training and evaluate the resulting checkpoint, plus the expected
validation result. Document all data and other resources used during training.

The instructor may use the package to reproduce training and check compliance
with the challenge rules. The submission channel and due date for this package
will be provided by the instructor.

