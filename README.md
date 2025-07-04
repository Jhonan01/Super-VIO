# Super-VIO (Visual-Inertial Odometry com SuperPoint, SuperGlue e IMU)

**Super-VIO** é um sistema de Visual-Inertial Odometry que utiliza pares de imagens nadir, correspondência de pontos via SuperPoint + SuperGlue, e dados de IMU para estimar a pose absoluta do sensor com escalonamento real. Ideal para ambientes onde dados visuais e inerciais são sincronizados e disponíveis.

## 📌 Requisitos

- Python 3.8+
- CUDA (opcional, mas recomendado)
- PyTorch
- OpenCV (>=4.5)
- matplotlib
- pandas
- numpy

Instale as dependências com:

```bash
pip install -r requirements.txt
