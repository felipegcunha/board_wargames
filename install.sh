#!/bin/bash
# Baixar o instalador do Tesseract versão 5.4.0.20240606
wget https://github.com/tesseract-ocr/tesseract/archive/refs/tags/5.4.0.tar.gz
# Extrair o arquivo baixado
tar -zxvf 5.4.0.tar.gz
cd tesseract-5.4.0
# Instalar dependências necessárias
sudo apt-get update
sudo apt-get install -y automake ca-certificates g++ git libtool libleptonica-dev make pkg-config
# Configurar e compilar o Tesseract
./autogen.sh
./configure
make
sudo make install
sudo ldconfig
