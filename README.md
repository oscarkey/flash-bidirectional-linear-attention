<div align="center">

# Flash Bi-directional Linear Attention

</div>

The aim of this repository is to implement **bi-directional linear attention** for **non-causal** modeling using Triton.

<div align="center">
  <img width="600" alt="image" src="https://private-user-images.githubusercontent.com/74758580/387246938-cd89a618-5d54-41b7-9055-36ba28b29fbd.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzQ3OTEwODQsIm5iZiI6MTczNDc5MDc4NCwicGF0aCI6Ii83NDc1ODU4MC8zODcyNDY5MzgtY2Q4OWE2MTgtNWQ1NC00MWI3LTkwNTUtMzZiYTI4YjI5ZmJkLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDEyMjElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMjIxVDE0MTk0NFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTRkYjk5MDk5YTExNDZiZDZiNmMyNzlhYzk2ZmRiNjZiZjk4ZTdhNzhlMzRiNTA0MDU0NTRiYWI5NzYyYWU5ODQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.gCyr6rJJgGEbq9kJUZ70_SLI7-KNdmLS8A3tSfQatP4">
</div>



This project is currently maintained by an individual and remains a work in progress. As the maintainer is still in the early stages of learning Triton, many implementations may not be optimal. **Contributions and suggestions are welcome!**


# Models
Roughly sorted according to the timeline supported in FBi-LA

| Date    | Model     | Title                                                                  | Paper                                     | Code                                                          | FBi-LA impl                                                                                                           |
| :------ | :-------- | :--------------------------------------------------------------------- | :---------------------------------------: | :-----------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------: |
| 2024-11 | Linfusion | LinFusion: 1 GPU, 1 Minute, 16K Image                                  | [arxiv](https://arxiv.org/abs/2409.02097) | [official](https://github.com/Huage001/LinFusion)             | [code](https://github.com/hp-l33/flash-bidirectional-linear-attention/blob/main/fbi_la/layers/linfusion.py)           |
| 2024-11 | MLLA      | Demystify Mamba in Vision: A Linear Attention Perspective              | [arxiv](https://arxiv.org/abs/2405.16605) | [official](https://github.com/LeapLabTHU/MLLA)                | [code](https://github.com/hp-l33/flash-bidirectional-linear-attention/blob/main/fbi_la/layers/mlla.py)                |
| 2024-11 | Focused-LA| FLatten Transformer: Vision Transformer using Focused Linear Attention | [arxiv](https://arxiv.org/abs/2308.00442) | [official](https://github.com/LeapLabTHU/FLatten-Transformer) | [code](https://github.com/hp-l33/flash-bidirectional-linear-attention/blob/main/fbi_la/layers/focused_la.py)          |

More models will be implemented gradually.

P.S.: The current implementation of MLLA is relatively basic and will be updated soon.

# Usage

## Installation
``` shell
git clone https://github.com/hp-l33/flash-bidirectional-linear-attention.git
pip install -e flash-bidirectional-linear-attention/.
```

## Integrated Models
This library has integrated some models, which can be called directly. Taking [LinFusion](https://github.com/Huage001/LinFusion) as an example:
``` python
import torch
from diffusers import AutoPipelineForText2Image
from fbi_la.models import LinFusion

sd_repo = "Lykon/dreamshaper-8"

pipeline = AutoPipelineForText2Image.from_pretrained(
    sd_repo, torch_dtype=torch.float16, variant="fp16"
).to(torch.device("cuda"))

linfusion = LinFusion.construct_for(pipeline)

image = pipeline(
    "An astronaut floating in space. Beautiful view of the stars and the universe in the background.",
    generator=torch.manual_seed(123)
).images[0]
```

# Benchmarks
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
- replace ``torch.sum()`` and ``torch.mean()`` operations
- implement more models
  - VSSD
  - RALA

# Acknowledgments
Thanks to the following repositories for their inspiration:
- [flash-attention](https://github.com/Dao-AILab/flash-attention)
- [flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention)

