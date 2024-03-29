# Explorando o *Depth Anything*
***

O Depth Anything é treinado de forma conjunta em 1,5 milhões de imagens rotuladas e mais de 62 milhões de imagens não rotuladas, fornecendo os modelos fundamentais de Estimação de Profundidade Monocular (EPM) mais capazes, com as seguintes características:

- Estimação de profundidade relativa sem necessidade de ajustes, superior ao MiDaS v3.1 (BEiTL-512).
- Estimação de profundidade métrica sem necessidade de ajustes, superior ao ZoeDepth.
- Ajuste fino e avaliação ótimos no domínio em NYUv2 e KITTI.

Esse modelo também apresenta um aprimoramento no ControlNet condicionado à profundidade melhorado baseado em nosso Depth Anything.

![Imagem de Exemplo](assets/examples.png)

### Exemplos de Inferência
***

![Exemplo de Inferência 1](assets/output-inference1.gif)
***
![Exemplo de Inferência 2](assets/output-inference2.gif)
***
![Exemplo de Inferência 3](assets/output-inference3.gif)
***
![Exemplo de Inferência 4](assets/output-inference4.gif)
***
![Exemplo de Inferência 5](assets/output-inference5.gif)
***

### Citação do Trabalho Original
***

- Link do Artigo: [Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data](paper-depth-anything.pdf).

```
@inproceedings{depthanything,
  title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  booktitle={CVPR},
  year={2024}
}
```