<div align="center">

# Flash Bi-directional Linear Attention

</div>

The aim of this repository is to implement **bi-directional linear attention** for **non-causal** modeling using Triton.

<div align="center">
  <img width="600" alt="image" src="https://private-user-images.githubusercontent.com/74758580/388149648-76612bb6-0f46-4c36-9502-bdf95e437b66.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzIxMTU3MzcsIm5iZiI6MTczMjExNTQzNywicGF0aCI6Ii83NDc1ODU4MC8zODgxNDk2NDgtNzY2MTJiYjYtMGY0Ni00YzM2LTk1MDItYmRmOTVlNDM3YjY2LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDExMjAlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMTIwVDE1MTAzN1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWVlYzFlYmEzNjYzYTc5ZWZkNWQ1MjdiZjkyNjEwNWVlZTVjOTQ5YTc3NDEwYzAxZTBmOGQ3NDY2ZWFiMTM4MzQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.McRWG9yxZZqf6eC9gFtLrehk2Hn3sSz58mzWG3R2Yxc">
</div>



This project is currently maintained by an individual and remains a work in progress. As the maintainer is still in the early stages of learning Triton, many implementations may not be optimal. **Contributions and suggestions are welcome!**


# Models
Roughly sorted according to the timeline supported in FBi-LA

| Date    | Model     | Title                                                                                                     |                                  Paper                                   |                                            Code                                             |                                                  FLA impl                                                   |
| :------ | :-------- | :-------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------: |
| 2024-11 | Linfusion    | LinFusion: 1 GPU, 1 Minute, 16K Image                                   |                [arxiv](https://arxiv.org/abs/2409.02097)                 |                [official](https://github.com/Huage001/LinFusion)                | [code](https://github.com/hp-l33/flash-bidirectional-linear-attention/blob/main/fbi_la/layers/linfusion/attention.py) |
| 2024-11 | MLLA       | Demystify Mamba in Vision: A Linear Attention Perspective                                      |                [arxiv](https://arxiv.org/abs/2405.16605)                 |                [official](https://github.com/LeapLabTHU/MLLA)                |         [code](https://github.com/hp-l33/flash-bidirectional-linear-attention/blob/main/fbi_la/layers/mlla/attention.py)          |
| 2024-11 | Focused-LA     | FLatten Transformer: Vision Transformer using Focused Linear Attention                                                               | [arxiv](https://arxiv.org/abs/2308.00442) |                     [official](https://github.com/LeapLabTHU/FLatten-Transformer)                     |        [code](https://github.com/hp-l33/flash-bidirectional-linear-attention/blob/main/fbi_la/layers/focused_la/attention.py)         |

More models will be implemented gradually.

P.S.: The current implementation of MLLA is relatively basic and will be updated soon.

# Benchmark
Tested on an A800 80G GPU.
``` shell
B8-H16-D64:
         T  torch_fwd  triton_fwd  torch_bwd  triton_bwd
0    128.0   0.063488    0.049152   0.520192    0.651264
1    256.0   0.080896    0.056320   0.795648    0.599040
2    512.0   0.111616    0.070656   1.074176    1.065984
3   1024.0   0.169984    0.101376   1.014784    0.746496
4   2048.0   0.300032    0.165888   1.464320    1.364992
5   4096.0   0.532480    0.287744   2.741248    2.564096
6   8192.0   1.005568    0.521216   5.232128    4.940800
7  16384.0   1.924608    0.980992  10.235904    9.695744
```

# TODO
- improve memory efficiency during backpropagation
- replace ``torch.sum()`` operation
- implement more models
  - VSSD
  - RALA

# Acknowledgments
Thanks to the following repositories for their inspiration:
- [flash-attention](https://github.com/Dao-AILab/flash-attention)
- [flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention)

