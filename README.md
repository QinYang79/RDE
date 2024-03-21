## Introduction
PyTorch implementation for [Noisy-Correspondence Learning for Text-to-Image Person Re-identification](./src/RDE_main.pdf) (CVPR 2024). The solution to [the noisy correspondence problem](https://github.com/QinYang79/Noisy-Correspondence-Summary)  in TIReID.

### RDE framework
<img src="./src/frame.png"  width="720"  />

## Requirements and Datasets
- Same as [IRRA](https://github.com/anosorae/IRRA)

### Noise index
If you want to experiment with the same noise index as in the paper, the noise index files can be found in [2024-CVPR-RDE/noiseindex](https://github.com/QinYang79/RDE/tree/main/2024-CVPR-RDE/noiseindex).


## Training and Evaluation

### Training new models

```
sh run_rde.sh
```

### Evaluation
Modify the  ```sub``` in the ```test.py``` file and run it.
```
python test.py
```

 

### Experiment Results:
<img src="./src/results.png"  width="720" />


## Citation
If RDE is useful for your research, please cite the following paper:
```
@inproceedings{cvpr23crossmodal,
  title={Noisy-Correspondence Learning for Text-to-Image Person Re-identification},
  author={Qin, Yang and Chen, Yingke and Peng, Dezhong and Peng, Xi and Zhou, Joey Tianyi and Hu, Peng},
  booktitle={IEEE International Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024},
}
```
```
@inproceedings{qin2022deep,
  title={Deep evidential learning with noisy correspondence for cross-modal retrieval},
  author={Qin, Yang and Peng, Dezhong and Peng, Xi and Wang, Xu and Hu, Peng},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={4948--4956},
  year={2022}
}

```

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)

## Acknowledgements
The code is based on [IRRA](https://github.com/anosorae/IRRA) licensed under Apache 2.0.