{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMQU5DlW2FhxoMdSocjnf/c",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/rizkyakbarmaulidin16/cnn_hoax_classifier.py/blob/main/cnn_hoax_calssifier.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 760
        },
        "id": "Ykl-sMkdIm9J",
        "outputId": "becc5a42-65ab-49f3-8eb2-c3d4b638e8b6"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.11/dist-packages/keras/src/layers/core/embedding.py:90: UserWarning: Argument `input_length` is deprecated. Just remove it.\n",
            "  warnings.warn(\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "\u001b[1mModel: \"sequential\"\u001b[0m\n"
            ],
            "text/html": [
              "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\">Model: \"sequential\"</span>\n",
              "</pre>\n"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓\n",
              "┃\u001b[1m \u001b[0m\u001b[1mLayer (type)                   \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1mOutput Shape          \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1m      Param #\u001b[0m\u001b[1m \u001b[0m┃\n",
              "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩\n",
              "│ embedding (\u001b[38;5;33mEmbedding\u001b[0m)           │ ?                      │   \u001b[38;5;34m0\u001b[0m (unbuilt) │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ conv1d (\u001b[38;5;33mConv1D\u001b[0m)                 │ ?                      │   \u001b[38;5;34m0\u001b[0m (unbuilt) │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ global_max_pooling1d            │ ?                      │             \u001b[38;5;34m0\u001b[0m │\n",
              "│ (\u001b[38;5;33mGlobalMaxPooling1D\u001b[0m)            │                        │               │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ dense (\u001b[38;5;33mDense\u001b[0m)                   │ ?                      │   \u001b[38;5;34m0\u001b[0m (unbuilt) │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ dense_1 (\u001b[38;5;33mDense\u001b[0m)                 │ ?                      │   \u001b[38;5;34m0\u001b[0m (unbuilt) │\n",
              "└─────────────────────────────────┴────────────────────────┴───────────────┘\n"
            ],
            "text/html": [
              "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓\n",
              "┃<span style=\"font-weight: bold\"> Layer (type)                    </span>┃<span style=\"font-weight: bold\"> Output Shape           </span>┃<span style=\"font-weight: bold\">       Param # </span>┃\n",
              "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩\n",
              "│ embedding (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Embedding</span>)           │ ?                      │   <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> (unbuilt) │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ conv1d (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv1D</span>)                 │ ?                      │   <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> (unbuilt) │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ global_max_pooling1d            │ ?                      │             <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
              "│ (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">GlobalMaxPooling1D</span>)            │                        │               │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ dense (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                   │ ?                      │   <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> (unbuilt) │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ dense_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                 │ ?                      │   <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> (unbuilt) │\n",
              "└─────────────────────────────────┴────────────────────────┴───────────────┘\n",
              "</pre>\n"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "\u001b[1m Total params: \u001b[0m\u001b[38;5;34m0\u001b[0m (0.00 B)\n"
            ],
            "text/html": [
              "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Total params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> (0.00 B)\n",
              "</pre>\n"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "\u001b[1m Trainable params: \u001b[0m\u001b[38;5;34m0\u001b[0m (0.00 B)\n"
            ],
            "text/html": [
              "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> (0.00 B)\n",
              "</pre>\n"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "\u001b[1m Non-trainable params: \u001b[0m\u001b[38;5;34m0\u001b[0m (0.00 B)\n"
            ],
            "text/html": [
              "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Non-trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> (0.00 B)\n",
              "</pre>\n"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/10\n",
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m3s\u001b[0m 3s/step - accuracy: 0.5000 - loss: 0.6919 - val_accuracy: 0.5000 - val_loss: 0.6888\n",
            "Epoch 2/10\n",
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 88ms/step - accuracy: 0.6875 - loss: 0.6873 - val_accuracy: 0.5000 - val_loss: 0.6870\n",
            "Epoch 3/10\n",
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 143ms/step - accuracy: 1.0000 - loss: 0.6833 - val_accuracy: 0.5000 - val_loss: 0.6854\n",
            "Epoch 4/10\n",
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 108ms/step - accuracy: 1.0000 - loss: 0.6796 - val_accuracy: 0.5000 - val_loss: 0.6842\n",
            "Epoch 5/10\n",
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 138ms/step - accuracy: 1.0000 - loss: 0.6756 - val_accuracy: 0.5000 - val_loss: 0.6831\n",
            "Epoch 6/10\n",
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 155ms/step - accuracy: 1.0000 - loss: 0.6717 - val_accuracy: 0.7500 - val_loss: 0.6817\n",
            "Epoch 7/10\n",
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 98ms/step - accuracy: 1.0000 - loss: 0.6678 - val_accuracy: 0.7500 - val_loss: 0.6801\n",
            "Epoch 8/10\n",
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 105ms/step - accuracy: 1.0000 - loss: 0.6638 - val_accuracy: 0.7500 - val_loss: 0.6783\n",
            "Epoch 9/10\n",
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 138ms/step - accuracy: 1.0000 - loss: 0.6597 - val_accuracy: 0.7500 - val_loss: 0.6760\n",
            "Epoch 10/10\n",
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 176ms/step - accuracy: 1.0000 - loss: 0.6553 - val_accuracy: 1.0000 - val_loss: 0.6739\n",
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 103ms/step\n",
            "Prediksi probabilitas asli: [[0.49195287]]\n"
          ]
        }
      ],
      "source": [
        "# Install library jika belum\n",
        "# !pip install tensorflow\n",
        "\n",
        "import tensorflow as tf\n",
        "from tensorflow.keras.preprocessing.text import Tokenizer\n",
        "from tensorflow.keras.preprocessing.sequence import pad_sequences\n",
        "from tensorflow.keras.models import Sequential\n",
        "from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense\n",
        "import numpy as np\n",
        "\n",
        "# Dataset minimal 20 (contoh dummy data)\n",
        "texts = [\n",
        "    \"Berita ini benar dan sudah diverifikasi\",\n",
        "    \"Ini adalah hoaks yang tersebar di media sosial\",\n",
        "    \"Informasi ini valid dan resmi\",\n",
        "    \"Berita palsu yang menyesatkan\",\n",
        "    \"Hoaks berbahaya tentang kesehatan\",\n",
        "    \"Informasi asli dari pemerintah\",\n",
        "    \"Ini kabar bohong yang tidak jelas sumbernya\",\n",
        "    \"Fakta resmi sudah dikonfirmasi\",\n",
        "    \"Hoaks lama yang muncul kembali\",\n",
        "    \"Data valid sesuai dokumen resmi\",\n",
        "    \"Berita hoaks yang cepat viral\",\n",
        "    \"Pengumuman asli dari lembaga terkait\",\n",
        "    \"Informasi hoaks yang membingungkan\",\n",
        "    \"Siaran pers resmi pemerintah\",\n",
        "    \"Hoaks terbaru yang tidak terbukti\",\n",
        "    \"Data resmi sesuai laporan\",\n",
        "    \"Berita bohong yang banyak dipercaya\",\n",
        "    \"Fakta akurat sesuai penelitian\",\n",
        "    \"Hoaks lama dengan narasi baru\",\n",
        "    \"Informasi benar dan diverifikasi\"\n",
        "]\n",
        "\n",
        "# Label: 1 = asli, 0 = hoaks\n",
        "labels = np.array([\n",
        "    1,0,1,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1\n",
        "])\n",
        "\n",
        "# Tokenisasi\n",
        "tokenizer = Tokenizer(num_words=5000)\n",
        "tokenizer.fit_on_texts(texts)\n",
        "sequences = tokenizer.texts_to_sequences(texts)\n",
        "\n",
        "# Padding\n",
        "maxlen = 20\n",
        "X = pad_sequences(sequences, maxlen=maxlen)\n",
        "\n",
        "# Model CNN\n",
        "model = Sequential([\n",
        "    Embedding(input_dim=5000, output_dim=32, input_length=maxlen),\n",
        "    Conv1D(32, 3, activation='relu'),\n",
        "    GlobalMaxPooling1D(),\n",
        "    Dense(10, activation='relu'),\n",
        "    Dense(1, activation='sigmoid')\n",
        "])\n",
        "\n",
        "model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])\n",
        "model.summary()\n",
        "\n",
        "# Training\n",
        "history = model.fit(X, labels, epochs=10, validation_split=0.2)\n",
        "\n",
        "# Prediksi contoh baru\n",
        "new_texts = [\"Ini informasi bohong yang tidak diverifikasi\"]\n",
        "new_seq = tokenizer.texts_to_sequences(new_texts)\n",
        "new_pad = pad_sequences(new_seq, maxlen=maxlen)\n",
        "pred = model.predict(new_pad)\n",
        "print(\"Prediksi probabilitas asli:\", pred)\n"
      ]
    }
  ]
}