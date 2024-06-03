# NAC TCN Code

Basic code for training models is in each directory. Please contact alexandermehta@outlook.com with any questions. Note we use pytorch lightning. 

Please **cite the following papers** if you choose to use the code provided:

**Our Paper:**
```bibtex
@inproceedings{Mehta_2023, series={ICVIP 2023},
   title={NAC-TCN: Temporal Convolutional Networks with Causal Dilated Neighborhood Attention for Emotion Understanding},
   url={http://dx.doi.org/10.1145/3639390.3639392},
   DOI={10.1145/3639390.3639392},
   booktitle={Proceedings of the 2023 7th International Conference on Video and Image Processing},
   publisher={ACM},
   author={Mehta, Alexander and Yang, William},
   year={2023},
   month=dec, collection={ICVIP 2023} }

```


**The libraries we used/modified:**
Haasani 2023 for neighborhood attention lib, Nguyen 2022 for affwild2 base training code (used to verify prior papers results and our own model), TCN original paper
```bibtex
@inproceedings{hassani2023neighborhood,
	title        = {Neighborhood Attention Transformer},
	author       = {Ali Hassani and Steven Walton and Jiachen Li and Shen Li and Humphrey Shi},
	year         = 2023,
        booktitle    = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}
}
@article{hassani2022dilated,
	title        = {Dilated Neighborhood Attention Transformer},
	author       = {Ali Hassani and Humphrey Shi},
	year         = 2022,
	url          = {https://arxiv.org/abs/2209.15001},
	eprint       = {2209.15001},
	archiveprefix = {arXiv},
	primaryclass = {cs.CV}
}

@InProceedings{Nguyen_2022_CVPR,
    author    = {Nguyen, Hong-Hai and Huynh, Van-Thong and Kim, Soo-Hyung},
    title     = {An Ensemble Approach for Facial Behavior Analysis In-the-Wild Video},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2022},
    pages     = {2512-2517}
}
@article{bai2018empirical,
  title={An empirical evaluation of generic convolutional and recurrent networks for sequence modeling},
  author={Bai, Shaojie and Kolter, J Zico and Koltun, Vladlen},
  journal={arXiv preprint arXiv:1803.01271},
  year={2018}
}

```

