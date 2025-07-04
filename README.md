# Super-VIO

**Super-VIO** (Visual-Inertial Odometry com SuperGlue + IMU) é uma pipeline que estima a trajetória de um drone (ou câmera nadir) usando correspondência de features visuais (SuperPoint + SuperGlue) e dados de movimento da IMU para recuperar a escala real. Os resultados são comparados com o Ground Truth (posições reais) para avaliação.

---

## 📌 Funcionalidades

- Detecção e correspondência robusta de features com SuperPoint e SuperGlue.
- Estimativa de pose relativa com Homografia + PnP.
- Cálculo da escala real usando dados da IMU (fusão visual-inercial).
- Atualização incremental da pose acumulada em forma de matriz homogênea.
- Visualização em tempo real da trajetória estimada vs ground truth.
- Exibição das correspondências de features entre imagens consecutivas.

---

## 🛠️ Requisitos

- Python 3.8+
- OpenCV 4.5+
- PyTorch
- matplotlib
- pandas
- NumPy

Instale as dependências com:

```bash
pip install -r requirements.txt

