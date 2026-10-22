# 许可证与第三方声明

## 自有程序与文档

`code/`中的自有程序、README及`docs/`项目说明采用[MIT许可证](LICENSE)。

## LaTeX模板

`paper/commons/enmcm26thesis.cls`改编自[jayxin/cumcm](https://github.com/jayxin/cumcm)的`commons/cumcmthesis.cls`。该项目基于[latexstudio/CUMCMThesis](https://github.com/latexstudio/CUMCMThesis)扩展，并在[LICENSE.txt](https://github.com/jayxin/cumcm/blob/main/LICENSE.txt)提供LPPL 1.3c条款。

本地派生模板及`cumcm2026.sty`、`preamble.tex`按LPPL-1.3c分发，许可证全文见[licenses/LPPL-1.3c.txt](licenses/LPPL-1.3c.txt)。完整上游工程可从[jayxin/cumcm](https://github.com/jayxin/cumcm)获取。

派生版使用独立文件名和Modified文档类标识；相对于上游，调整了图像目录、采用Fandol与TeX Gyre字体、裁剪程序附录专用listings配置，并适配正文样式和PDF元数据。维护状态为`unmaintained`。本项目的修改由本项目负责，上游作者不承担支持义务。

## 字体与运行依赖

- [TeX Gyre Termes/Heros](https://ctan.org/pkg/tex-gyre)采用GUST Font License。
- [Fandol](https://ctan.org/pkg/fandol)采用其随附GPL字体许可及相应嵌入例外，具体条款以发行包为准。

字体由TeX Live或MiKTeX提供。Python、SciPy、NumPy、pandas、PuLP/CBC、Matplotlib、openpyxl、Pillow，以及XeLaTeX、ctex等运行依赖，按各自许可证使用。

## 研究内容与赛题材料

论文、研究图表及结果的权利归原作者；这些研究内容独立于MIT软件授权。LaTeX模板的LPPL条款也不对论文内容主张权利。

赛题PDF、输入附件及官方工作簿模板的权利和使用条件由相应发布方规定，本项目的软件许可证不适用于这些第三方原件。
