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
0    128.0   0.064512    0.049152   0.837632    1.557504
1    256.0   0.162816    0.056320   1.797120    1.577472
2    512.0   0.153088    0.070656   1.767936    1.575936
3   1024.0   0.172032    0.101376   1.717248    1.610752
4   2048.0   0.301056    0.165888   1.814528    1.630208
5   4096.0   0.533504    0.287744   2.738688    2.561536
6   8192.0   1.006592    0.521216   5.226496    4.934656
7  16384.0   1.928192    0.980992  10.237952    9.684992
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

