# Deep Learning project

## Missions

You must find a way to restore/decrypt images!
 
Two challenges are given.  

1. **Image restoration**
* Dataset 1A: images only need to be restored, examples of damaged and restored version of images are provided.
* Dataset 1B: images have to be restored using advanced techniques, examples of damaged and restored version of images are also provided.
2. **Image decryption**
* Dataset 2: images are clearly encrypted... but some of them have been cracked!!! Examples of encrypted images and corresponding original images are provided.

Both train and test sets are provided. 

## Rules 
* You must use Deep Learning techniques.
* Results must be reproductible, use `torch.manual_seed(1234)`. Training must also be reproductible.
* Team size <= 4
* The following evaluation metric will be used.
```
def eval_metric(img, pred):
     return torch.abs(img - pred).sum() 
```
* Do not use the test set during training... The test set cannot be used to train or select you model. test set + `eval_metric` can be used to compare your results with other teams.
* For each challenge, a bonus will be given to the best team (the one maximizing `eval_metric` on the test set). Best team bonus will be `+2/#number of team members` (challenge 1, average between Dataset 1A, and 1B will be made). 

You should submit: 
* Trained models.
* A report detailing the methodology, tested architectures, results, illustration of predictions on the test set, as well as discussions. You must report the `eval_metric` results for the test sets (max 20 pages). 
