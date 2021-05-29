此处展示的是Bi-Real Net18经过200个epochs的训练结果，最后在测试集上的准确度为62.85%，模型保存为bireal_CIFAR_best.h5，转换为tflite格式后保存为bireal.tflite。


782/782 [==============================] - 105s 129ms/step - loss: 4.8101 - accuracy: 0.0567 - val_loss: 3.8626 - val_accuracy: 0.1115
Epoch 2/200
782/782 [==============================] - 106s 136ms/step - loss: 3.8455 - accuracy: 0.1132 - val_loss: 3.7216 - val_accuracy: 0.1432
Epoch 3/200
782/782 [==============================] - 107s 137ms/step - loss: 3.6151 - accuracy: 0.1485 - val_loss: 3.4748 - val_accuracy: 0.1789
Epoch 4/200
782/782 [==============================] - 108s 138ms/step - loss: 3.3692 - accuracy: 0.1857 - val_loss: 3.1900 - val_accuracy: 0.2232
Epoch 5/200
782/782 [==============================] - 108s 138ms/step - loss: 3.0937 - accuracy: 0.2362 - val_loss: 3.2403 - val_accuracy: 0.2312
Epoch 6/200
782/782 [==============================] - 109s 139ms/step - loss: 2.9044 - accuracy: 0.2677 - val_loss: 2.8581 - val_accuracy: 0.2861
Epoch 7/200
782/782 [==============================] - 106s 136ms/step - loss: 2.7012 - accuracy: 0.3088 - val_loss: 2.7653 - val_accuracy: 0.3089
Epoch 8/200
782/782 [==============================] - 107s 137ms/step - loss: 2.5612 - accuracy: 0.3402 - val_loss: 2.6364 - val_accuracy: 0.3403
Epoch 9/200
782/782 [==============================] - 108s 138ms/step - loss: 2.4577 - accuracy: 0.3582 - val_loss: 2.4187 - val_accuracy: 0.3658
Epoch 10/200
782/782 [==============================] - 108s 138ms/step - loss: 2.3782 - accuracy: 0.3757 - val_loss: 2.3395 - val_accuracy: 0.3994
Epoch 11/200
782/782 [==============================] - 108s 138ms/step - loss: 2.2760 - accuracy: 0.3971 - val_loss: 2.4292 - val_accuracy: 0.3743
Epoch 12/200
782/782 [==============================] - 109s 139ms/step - loss: 2.2016 - accuracy: 0.4141 - val_loss: 2.3936 - val_accuracy: 0.3857
Epoch 13/200
782/782 [==============================] - 109s 139ms/step - loss: 2.1736 - accuracy: 0.4205 - val_loss: 2.3503 - val_accuracy: 0.3902
Epoch 14/200
782/782 [==============================] - 110s 140ms/step - loss: 2.1236 - accuracy: 0.4304 - val_loss: 2.3023 - val_accuracy: 0.4092
Epoch 15/200
782/782 [==============================] - 109s 139ms/step - loss: 2.0703 - accuracy: 0.4446 - val_loss: 2.1326 - val_accuracy: 0.4417
Epoch 16/200
782/782 [==============================] - 108s 139ms/step - loss: 2.0716 - accuracy: 0.4430 - val_loss: 2.1200 - val_accuracy: 0.4548
Epoch 17/200
782/782 [==============================] - 109s 139ms/step - loss: 2.0148 - accuracy: 0.4553 - val_loss: 2.2465 - val_accuracy: 0.4322
Epoch 18/200
782/782 [==============================] - 109s 139ms/step - loss: 1.9555 - accuracy: 0.4672 - val_loss: 2.0887 - val_accuracy: 0.4501
Epoch 19/200
782/782 [==============================] - 109s 140ms/step - loss: 1.9498 - accuracy: 0.4679 - val_loss: 2.2600 - val_accuracy: 0.4318
Epoch 20/200
782/782 [==============================] - 110s 140ms/step - loss: 1.9187 - accuracy: 0.4824 - val_loss: 2.1180 - val_accuracy: 0.4618
Epoch 21/200
782/782 [==============================] - 108s 138ms/step - loss: 1.7080 - accuracy: 0.5244 - val_loss: 1.8524 - val_accuracy: 0.5107
Epoch 22/200
782/782 [==============================] - 109s 139ms/step - loss: 1.5988 - accuracy: 0.5530 - val_loss: 1.8555 - val_accuracy: 0.5099
Epoch 23/200
782/782 [==============================] - 135s 173ms/step - loss: 1.5824 - accuracy: 0.5583 - val_loss: 1.8936 - val_accuracy: 0.5090
Epoch 24/200
782/782 [==============================] - 128s 164ms/step - loss: 1.5510 - accuracy: 0.5663 - val_loss: 1.9041 - val_accuracy: 0.5043
Epoch 25/200
782/782 [==============================] - 136s 174ms/step - loss: 1.4993 - accuracy: 0.5754 - val_loss: 1.8079 - val_accuracy: 0.5243
Epoch 26/200
782/782 [==============================] - 136s 174ms/step - loss: 1.4936 - accuracy: 0.5774 - val_loss: 1.7900 - val_accuracy: 0.5376
Epoch 27/200
782/782 [==============================] - 139s 177ms/step - loss: 1.4830 - accuracy: 0.5817 - val_loss: 1.7188 - val_accuracy: 0.5386
Epoch 28/200
782/782 [==============================] - 129s 165ms/step - loss: 1.4661 - accuracy: 0.5866 - val_loss: 1.7740 - val_accuracy: 0.5281
Epoch 29/200
782/782 [==============================] - 130s 166ms/step - loss: 1.4499 - accuracy: 0.5937 - val_loss: 1.8053 - val_accuracy: 0.5318
Epoch 30/200
782/782 [==============================] - 125s 160ms/step - loss: 1.4420 - accuracy: 0.5930 - val_loss: 1.7779 - val_accuracy: 0.5371
Epoch 31/200
782/782 [==============================] - 122s 156ms/step - loss: 1.3988 - accuracy: 0.6021 - val_loss: 1.7486 - val_accuracy: 0.5366
Epoch 32/200
782/782 [==============================] - 117s 149ms/step - loss: 1.3992 - accuracy: 0.6001 - val_loss: 1.7923 - val_accuracy: 0.5431
Epoch 33/200
782/782 [==============================] - 109s 140ms/step - loss: 1.3915 - accuracy: 0.6041 - val_loss: 1.7497 - val_accuracy: 0.5499
Epoch 34/200
782/782 [==============================] - 110s 140ms/step - loss: 1.3561 - accuracy: 0.6099 - val_loss: 1.7714 - val_accuracy: 0.5418
Epoch 35/200
782/782 [==============================] - 112s 143ms/step - loss: 1.3648 - accuracy: 0.6098 - val_loss: 1.7244 - val_accuracy: 0.5449
Epoch 36/200
782/782 [==============================] - 109s 139ms/step - loss: 1.3500 - accuracy: 0.6115 - val_loss: 1.8428 - val_accuracy: 0.5288
Epoch 37/200
782/782 [==============================] - 108s 138ms/step - loss: 1.3199 - accuracy: 0.6228 - val_loss: 1.7326 - val_accuracy: 0.5460
Epoch 38/200
782/782 [==============================] - 107s 137ms/step - loss: 1.3107 - accuracy: 0.6239 - val_loss: 1.7095 - val_accuracy: 0.5603
Epoch 39/200
782/782 [==============================] - 106s 136ms/step - loss: 1.2896 - accuracy: 0.6288 - val_loss: 1.7533 - val_accuracy: 0.5476
Epoch 40/200
782/782 [==============================] - 109s 139ms/step - loss: 1.2914 - accuracy: 0.6340 - val_loss: 1.7451 - val_accuracy: 0.5533
Epoch 41/200
782/782 [==============================] - 108s 138ms/step - loss: 1.1755 - accuracy: 0.6579 - val_loss: 1.6315 - val_accuracy: 0.5804
Epoch 42/200
782/782 [==============================] - 109s 139ms/step - loss: 1.1049 - accuracy: 0.6767 - val_loss: 1.7208 - val_accuracy: 0.5693
Epoch 43/200
782/782 [==============================] - 111s 142ms/step - loss: 1.0855 - accuracy: 0.6816 - val_loss: 1.6399 - val_accuracy: 0.5812
Epoch 44/200
782/782 [==============================] - 110s 140ms/step - loss: 1.0665 - accuracy: 0.6857 - val_loss: 1.7001 - val_accuracy: 0.5729
Epoch 45/200
782/782 [==============================] - 121s 155ms/step - loss: 1.0510 - accuracy: 0.6923 - val_loss: 1.6189 - val_accuracy: 0.5835
Epoch 46/200
782/782 [==============================] - 125s 160ms/step - loss: 1.0394 - accuracy: 0.6932 - val_loss: 1.6098 - val_accuracy: 0.5836
Epoch 47/200
782/782 [==============================] - 115s 147ms/step - loss: 1.0295 - accuracy: 0.6996 - val_loss: 1.5915 - val_accuracy: 0.5931
Epoch 48/200
782/782 [==============================] - 111s 141ms/step - loss: 1.0352 - accuracy: 0.6953 - val_loss: 1.6156 - val_accuracy: 0.5872
Epoch 49/200
782/782 [==============================] - 113s 145ms/step - loss: 1.0114 - accuracy: 0.6998 - val_loss: 1.6070 - val_accuracy: 0.5907
Epoch 50/200
782/782 [==============================] - 109s 140ms/step - loss: 1.0096 - accuracy: 0.7024 - val_loss: 1.6276 - val_accuracy: 0.5805
Epoch 51/200
782/782 [==============================] - 109s 140ms/step - loss: 1.0089 - accuracy: 0.7009 - val_loss: 1.6792 - val_accuracy: 0.5790
Epoch 52/200
782/782 [==============================] - 109s 140ms/step - loss: 0.9939 - accuracy: 0.6987 - val_loss: 1.6884 - val_accuracy: 0.5815
Epoch 53/200
782/782 [==============================] - 109s 139ms/step - loss: 0.9712 - accuracy: 0.7120 - val_loss: 1.6403 - val_accuracy: 0.5858
Epoch 54/200
782/782 [==============================] - 110s 140ms/step - loss: 0.9723 - accuracy: 0.7113 - val_loss: 1.6552 - val_accuracy: 0.5855
Epoch 55/200
782/782 [==============================] - 127s 162ms/step - loss: 0.9601 - accuracy: 0.7116 - val_loss: 1.6551 - val_accuracy: 0.5838
Epoch 56/200
782/782 [==============================] - 130s 166ms/step - loss: 0.9608 - accuracy: 0.7136 - val_loss: 1.7058 - val_accuracy: 0.5815
Epoch 57/200
782/782 [==============================] - 127s 162ms/step - loss: 0.9534 - accuracy: 0.7132 - val_loss: 1.6052 - val_accuracy: 0.5942
Epoch 58/200
782/782 [==============================] - 125s 159ms/step - loss: 0.9414 - accuracy: 0.7174 - val_loss: 1.7044 - val_accuracy: 0.5816
Epoch 59/200
782/782 [==============================] - 129s 165ms/step - loss: 0.9332 - accuracy: 0.7207 - val_loss: 1.6654 - val_accuracy: 0.5873
Epoch 60/200
782/782 [==============================] - 132s 168ms/step - loss: 0.9221 - accuracy: 0.7208 - val_loss: 1.6564 - val_accuracy: 0.5858
Epoch 61/200
782/782 [==============================] - 110s 141ms/step - loss: 0.8537 - accuracy: 0.7439 - val_loss: 1.5910 - val_accuracy: 0.6082
Epoch 62/200
782/782 [==============================] - 109s 140ms/step - loss: 0.8146 - accuracy: 0.7524 - val_loss: 1.6179 - val_accuracy: 0.6045
Epoch 63/200
782/782 [==============================] - 107s 136ms/step - loss: 0.7951 - accuracy: 0.7579 - val_loss: 1.5964 - val_accuracy: 0.6079
Epoch 64/200
782/782 [==============================] - 108s 138ms/step - loss: 0.8053 - accuracy: 0.7552 - val_loss: 1.6406 - val_accuracy: 0.6036
Epoch 65/200
782/782 [==============================] - 110s 140ms/step - loss: 0.8088 - accuracy: 0.7532 - val_loss: 1.5987 - val_accuracy: 0.6080
Epoch 66/200
782/782 [==============================] - 113s 144ms/step - loss: 0.7840 - accuracy: 0.7624 - val_loss: 1.6017 - val_accuracy: 0.6093
Epoch 67/200
782/782 [==============================] - 111s 142ms/step - loss: 0.7877 - accuracy: 0.7603 - val_loss: 1.5899 - val_accuracy: 0.6060
Epoch 68/200
782/782 [==============================] - 114s 146ms/step - loss: 0.7647 - accuracy: 0.7665 - val_loss: 1.6244 - val_accuracy: 0.6024
Epoch 69/200
782/782 [==============================] - 111s 142ms/step - loss: 0.7689 - accuracy: 0.7651 - val_loss: 1.6066 - val_accuracy: 0.6045
Epoch 70/200
782/782 [==============================] - 112s 143ms/step - loss: 0.7649 - accuracy: 0.7648 - val_loss: 1.6226 - val_accuracy: 0.6083
Epoch 71/200
782/782 [==============================] - 112s 143ms/step - loss: 0.7558 - accuracy: 0.7665 - val_loss: 1.6154 - val_accuracy: 0.6103
Epoch 72/200
782/782 [==============================] - 113s 144ms/step - loss: 0.7689 - accuracy: 0.7624 - val_loss: 1.6165 - val_accuracy: 0.6094
Epoch 73/200
782/782 [==============================] - 115s 147ms/step - loss: 0.7514 - accuracy: 0.7713 - val_loss: 1.5770 - val_accuracy: 0.6140
Epoch 74/200
782/782 [==============================] - 115s 147ms/step - loss: 0.7505 - accuracy: 0.7717 - val_loss: 1.5793 - val_accuracy: 0.6159
Epoch 75/200
782/782 [==============================] - 113s 144ms/step - loss: 0.7329 - accuracy: 0.7736 - val_loss: 1.5940 - val_accuracy: 0.6110
Epoch 76/200
782/782 [==============================] - 115s 147ms/step - loss: 0.7296 - accuracy: 0.7739 - val_loss: 1.7078 - val_accuracy: 0.5985
Epoch 77/200
782/782 [==============================] - 111s 141ms/step - loss: 0.7292 - accuracy: 0.7733 - val_loss: 1.6753 - val_accuracy: 0.6059
Epoch 78/200
782/782 [==============================] - 112s 143ms/step - loss: 0.7341 - accuracy: 0.7731 - val_loss: 1.6785 - val_accuracy: 0.5986
Epoch 79/200
782/782 [==============================] - 111s 142ms/step - loss: 0.7245 - accuracy: 0.7754 - val_loss: 1.6270 - val_accuracy: 0.6134
Epoch 80/200
782/782 [==============================] - 107s 136ms/step - loss: 0.7242 - accuracy: 0.7775 - val_loss: 1.6544 - val_accuracy: 0.6064
Epoch 81/200
782/782 [==============================] - 112s 143ms/step - loss: 0.6754 - accuracy: 0.7926 - val_loss: 1.6204 - val_accuracy: 0.6168
Epoch 82/200
782/782 [==============================] - 109s 139ms/step - loss: 0.6618 - accuracy: 0.7952 - val_loss: 1.5751 - val_accuracy: 0.6238
Epoch 83/200
782/782 [==============================] - 107s 136ms/step - loss: 0.6427 - accuracy: 0.7977 - val_loss: 1.6584 - val_accuracy: 0.6122
Epoch 84/200
782/782 [==============================] - 107s 137ms/step - loss: 0.6501 - accuracy: 0.7973 - val_loss: 1.6386 - val_accuracy: 0.6107
Epoch 85/200
782/782 [==============================] - 108s 138ms/step - loss: 0.6407 - accuracy: 0.8001 - val_loss: 1.6520 - val_accuracy: 0.6136
Epoch 86/200
782/782 [==============================] - 108s 137ms/step - loss: 0.6328 - accuracy: 0.8033 - val_loss: 1.6329 - val_accuracy: 0.6170
Epoch 87/200
782/782 [==============================] - 117s 150ms/step - loss: 0.6272 - accuracy: 0.8031 - val_loss: 1.6961 - val_accuracy: 0.6075
Epoch 88/200
782/782 [==============================] - 152s 195ms/step - loss: 0.6322 - accuracy: 0.8020 - val_loss: 1.6361 - val_accuracy: 0.6170
Epoch 89/200
782/782 [==============================] - 154s 197ms/step - loss: 0.6232 - accuracy: 0.8072 - val_loss: 1.6616 - val_accuracy: 0.6154
Epoch 90/200
782/782 [==============================] - 129s 165ms/step - loss: 0.6268 - accuracy: 0.8057 - val_loss: 1.6624 - val_accuracy: 0.6127
Epoch 91/200
782/782 [==============================] - 137s 175ms/step - loss: 0.6336 - accuracy: 0.8028 - val_loss: 1.6923 - val_accuracy: 0.6136
Epoch 92/200
782/782 [==============================] - 150s 192ms/step - loss: 0.6187 - accuracy: 0.8066 - val_loss: 1.6735 - val_accuracy: 0.6117
Epoch 93/200
782/782 [==============================] - 125s 160ms/step - loss: 0.6206 - accuracy: 0.8033 - val_loss: 1.7081 - val_accuracy: 0.6129
Epoch 94/200
782/782 [==============================] - 149s 191ms/step - loss: 0.6007 - accuracy: 0.8114 - val_loss: 1.6392 - val_accuracy: 0.6216
Epoch 95/200
782/782 [==============================] - 159s 204ms/step - loss: 0.6002 - accuracy: 0.8126 - val_loss: 1.7073 - val_accuracy: 0.6147
Epoch 96/200
782/782 [==============================] - 158s 202ms/step - loss: 0.6061 - accuracy: 0.8100 - val_loss: 1.6944 - val_accuracy: 0.6185
Epoch 97/200
782/782 [==============================] - 157s 201ms/step - loss: 0.6004 - accuracy: 0.8102 - val_loss: 1.6587 - val_accuracy: 0.6190
Epoch 98/200
782/782 [==============================] - 158s 201ms/step - loss: 0.6081 - accuracy: 0.8111 - val_loss: 1.7420 - val_accuracy: 0.6067
Epoch 99/200
782/782 [==============================] - 157s 201ms/step - loss: 0.6010 - accuracy: 0.8121 - val_loss: 1.6502 - val_accuracy: 0.6220
Epoch 100/200
782/782 [==============================] - 157s 200ms/step - loss: 0.5892 - accuracy: 0.8140 - val_loss: 1.6494 - val_accuracy: 0.6199
Epoch 101/200
782/782 [==============================] - 126s 161ms/step - loss: 0.5825 - accuracy: 0.8195 - val_loss: 1.6769 - val_accuracy: 0.6163
Epoch 102/200
782/782 [==============================] - 127s 162ms/step - loss: 0.5596 - accuracy: 0.8242 - val_loss: 1.7064 - val_accuracy: 0.6213
Epoch 103/200
782/782 [==============================] - 116s 148ms/step - loss: 0.5538 - accuracy: 0.8260 - val_loss: 1.6442 - val_accuracy: 0.6285
Epoch 104/200
782/782 [==============================] - 114s 145ms/step - loss: 0.5557 - accuracy: 0.8267 - val_loss: 1.7292 - val_accuracy: 0.6154
Epoch 105/200
782/782 [==============================] - 109s 139ms/step - loss: 0.5577 - accuracy: 0.8226 - val_loss: 1.7062 - val_accuracy: 0.6178
Epoch 106/200
782/782 [==============================] - 111s 142ms/step - loss: 0.5506 - accuracy: 0.8285 - val_loss: 1.7006 - val_accuracy: 0.6182
Epoch 107/200
782/782 [==============================] - 109s 140ms/step - loss: 0.5506 - accuracy: 0.8269 - val_loss: 1.6911 - val_accuracy: 0.6182
Epoch 108/200
782/782 [==============================] - 109s 139ms/step - loss: 0.5531 - accuracy: 0.8254 - val_loss: 1.7292 - val_accuracy: 0.6174
Epoch 109/200
782/782 [==============================] - 115s 147ms/step - loss: 0.5585 - accuracy: 0.8264 - val_loss: 1.7555 - val_accuracy: 0.6121
Epoch 110/200
782/782 [==============================] - 142s 181ms/step - loss: 0.5423 - accuracy: 0.8273 - val_loss: 1.7330 - val_accuracy: 0.6199
Epoch 111/200
782/782 [==============================] - 151s 194ms/step - loss: 0.5381 - accuracy: 0.8286 - val_loss: 1.7415 - val_accuracy: 0.6199
Epoch 112/200
782/782 [==============================] - 141s 180ms/step - loss: 0.5268 - accuracy: 0.8330 - val_loss: 1.7048 - val_accuracy: 0.6179
Epoch 113/200
782/782 [==============================] - 166s 212ms/step - loss: 0.5446 - accuracy: 0.8300 - val_loss: 1.7863 - val_accuracy: 0.6084
Epoch 114/200
782/782 [==============================] - 121s 154ms/step - loss: 0.5329 - accuracy: 0.8297 - val_loss: 1.7522 - val_accuracy: 0.6135
Epoch 115/200
782/782 [==============================] - 112s 143ms/step - loss: 0.5432 - accuracy: 0.8287 - val_loss: 1.7075 - val_accuracy: 0.6202
Epoch 116/200
782/782 [==============================] - 112s 143ms/step - loss: 0.5256 - accuracy: 0.8342 - val_loss: 1.7098 - val_accuracy: 0.6165
Epoch 117/200
782/782 [==============================] - 110s 141ms/step - loss: 0.5306 - accuracy: 0.8309 - val_loss: 1.6958 - val_accuracy: 0.6211
Epoch 118/200
782/782 [==============================] - 109s 139ms/step - loss: 0.5343 - accuracy: 0.8284 - val_loss: 1.7396 - val_accuracy: 0.6147
Epoch 119/200
782/782 [==============================] - 108s 138ms/step - loss: 0.5328 - accuracy: 0.8315 - val_loss: 1.7049 - val_accuracy: 0.6192
Epoch 120/200
782/782 [==============================] - 108s 138ms/step - loss: 0.5253 - accuracy: 0.8333 - val_loss: 1.7461 - val_accuracy: 0.6149
Epoch 121/200
782/782 [==============================] - 113s 144ms/step - loss: 0.5066 - accuracy: 0.8392 - val_loss: 1.7263 - val_accuracy: 0.6170
Epoch 122/200
782/782 [==============================] - 116s 149ms/step - loss: 0.5089 - accuracy: 0.8372 - val_loss: 1.6909 - val_accuracy: 0.6232
Epoch 123/200
782/782 [==============================] - 114s 146ms/step - loss: 0.5140 - accuracy: 0.8373 - val_loss: 1.7135 - val_accuracy: 0.6212
Epoch 124/200
782/782 [==============================] - 110s 141ms/step - loss: 0.5210 - accuracy: 0.8355 - val_loss: 1.7132 - val_accuracy: 0.6193
Epoch 125/200
782/782 [==============================] - 110s 141ms/step - loss: 0.5125 - accuracy: 0.8383 - val_loss: 1.7412 - val_accuracy: 0.6171
Epoch 126/200
782/782 [==============================] - 107s 137ms/step - loss: 0.5053 - accuracy: 0.8395 - val_loss: 1.7681 - val_accuracy: 0.6187
Epoch 127/200
782/782 [==============================] - 107s 137ms/step - loss: 0.5057 - accuracy: 0.8372 - val_loss: 1.7315 - val_accuracy: 0.6197
Epoch 128/200
782/782 [==============================] - 108s 138ms/step - loss: 0.5100 - accuracy: 0.8387 - val_loss: 1.7046 - val_accuracy: 0.6211
Epoch 129/200
782/782 [==============================] - 111s 141ms/step - loss: 0.4969 - accuracy: 0.8407 - val_loss: 1.7462 - val_accuracy: 0.6192
Epoch 130/200
782/782 [==============================] - 107s 136ms/step - loss: 0.5029 - accuracy: 0.8406 - val_loss: 1.7488 - val_accuracy: 0.6145
Epoch 131/200
782/782 [==============================] - 108s 138ms/step - loss: 0.5034 - accuracy: 0.8378 - val_loss: 1.7771 - val_accuracy: 0.6135
Epoch 132/200
782/782 [==============================] - 108s 138ms/step - loss: 0.4966 - accuracy: 0.8408 - val_loss: 1.7332 - val_accuracy: 0.6181
Epoch 133/200
782/782 [==============================] - 105s 134ms/step - loss: 0.4917 - accuracy: 0.8450 - val_loss: 1.7371 - val_accuracy: 0.6210
Epoch 134/200
782/782 [==============================] - 109s 139ms/step - loss: 0.4964 - accuracy: 0.8405 - val_loss: 1.7565 - val_accuracy: 0.6188
Epoch 135/200
782/782 [==============================] - 106s 135ms/step - loss: 0.5075 - accuracy: 0.8375 - val_loss: 1.7351 - val_accuracy: 0.6185
Epoch 136/200
782/782 [==============================] - 112s 143ms/step - loss: 0.4998 - accuracy: 0.8394 - val_loss: 1.8085 - val_accuracy: 0.6091
Epoch 137/200
782/782 [==============================] - 114s 146ms/step - loss: 0.4938 - accuracy: 0.8435 - val_loss: 1.7137 - val_accuracy: 0.6265
Epoch 138/200
782/782 [==============================] - 106s 136ms/step - loss: 0.4997 - accuracy: 0.8377 - val_loss: 1.7249 - val_accuracy: 0.6232
Epoch 139/200
782/782 [==============================] - 109s 140ms/step - loss: 0.4934 - accuracy: 0.8428 - val_loss: 1.7189 - val_accuracy: 0.6211
Epoch 140/200
782/782 [==============================] - 108s 138ms/step - loss: 0.4940 - accuracy: 0.8421 - val_loss: 1.7254 - val_accuracy: 0.6244
Epoch 141/200
782/782 [==============================] - 105s 135ms/step - loss: 0.4816 - accuracy: 0.8464 - val_loss: 1.7439 - val_accuracy: 0.6219
Epoch 142/200
782/782 [==============================] - 107s 136ms/step - loss: 0.4737 - accuracy: 0.8505 - val_loss: 1.7260 - val_accuracy: 0.6224
Epoch 143/200
782/782 [==============================] - 106s 135ms/step - loss: 0.4753 - accuracy: 0.8491 - val_loss: 1.7590 - val_accuracy: 0.6151
Epoch 144/200
782/782 [==============================] - 106s 135ms/step - loss: 0.4749 - accuracy: 0.8481 - val_loss: 1.7520 - val_accuracy: 0.6203
Epoch 145/200
782/782 [==============================] - 107s 137ms/step - loss: 0.4799 - accuracy: 0.8487 - val_loss: 1.7517 - val_accuracy: 0.6217
Epoch 146/200
782/782 [==============================] - 107s 137ms/step - loss: 0.4859 - accuracy: 0.8460 - val_loss: 1.7668 - val_accuracy: 0.6171
Epoch 147/200
782/782 [==============================] - 108s 137ms/step - loss: 0.4859 - accuracy: 0.8442 - val_loss: 1.7530 - val_accuracy: 0.6225
Epoch 148/200
782/782 [==============================] - 106s 136ms/step - loss: 0.4769 - accuracy: 0.8462 - val_loss: 1.7698 - val_accuracy: 0.6232
Epoch 149/200
782/782 [==============================] - 105s 134ms/step - loss: 0.4804 - accuracy: 0.8467 - val_loss: 1.7701 - val_accuracy: 0.6191
Epoch 150/200
782/782 [==============================] - 106s 135ms/step - loss: 0.4682 - accuracy: 0.8501 - val_loss: 1.7118 - val_accuracy: 0.6260
Epoch 151/200
782/782 [==============================] - 105s 135ms/step - loss: 0.4734 - accuracy: 0.8490 - val_loss: 1.7777 - val_accuracy: 0.6203
Epoch 152/200
782/782 [==============================] - 105s 134ms/step - loss: 0.4767 - accuracy: 0.8463 - val_loss: 1.7728 - val_accuracy: 0.6176
Epoch 153/200
782/782 [==============================] - 105s 134ms/step - loss: 0.4726 - accuracy: 0.8474 - val_loss: 1.7573 - val_accuracy: 0.6183
Epoch 154/200
782/782 [==============================] - 105s 134ms/step - loss: 0.4827 - accuracy: 0.8448 - val_loss: 1.7880 - val_accuracy: 0.6158
Epoch 155/200
782/782 [==============================] - 105s 134ms/step - loss: 0.4756 - accuracy: 0.8476 - val_loss: 1.7919 - val_accuracy: 0.6116
Epoch 156/200
782/782 [==============================] - 108s 139ms/step - loss: 0.4784 - accuracy: 0.8478 - val_loss: 1.7921 - val_accuracy: 0.6177
Epoch 157/200
782/782 [==============================] - 117s 150ms/step - loss: 0.4792 - accuracy: 0.8449 - val_loss: 1.7442 - val_accuracy: 0.6217
Epoch 158/200
782/782 [==============================] - 114s 146ms/step - loss: 0.4710 - accuracy: 0.8496 - val_loss: 1.7468 - val_accuracy: 0.6231
Epoch 159/200
782/782 [==============================] - 107s 136ms/step - loss: 0.4681 - accuracy: 0.8513 - val_loss: 1.7315 - val_accuracy: 0.6271
Epoch 160/200
782/782 [==============================] - 108s 138ms/step - loss: 0.4630 - accuracy: 0.8508 - val_loss: 1.7599 - val_accuracy: 0.6189
Epoch 161/200
782/782 [==============================] - 108s 138ms/step - loss: 0.4668 - accuracy: 0.8507 - val_loss: 1.7681 - val_accuracy: 0.6205
Epoch 162/200
782/782 [==============================] - 110s 140ms/step - loss: 0.4757 - accuracy: 0.8483 - val_loss: 1.7249 - val_accuracy: 0.6239
Epoch 163/200
782/782 [==============================] - 120s 153ms/step - loss: 0.4656 - accuracy: 0.8520 - val_loss: 1.7737 - val_accuracy: 0.6149
Epoch 164/200
782/782 [==============================] - 149s 191ms/step - loss: 0.4588 - accuracy: 0.8515 - val_loss: 1.7947 - val_accuracy: 0.6139
Epoch 165/200
782/782 [==============================] - 142s 181ms/step - loss: 0.4603 - accuracy: 0.8522 - val_loss: 1.7672 - val_accuracy: 0.6210
Epoch 166/200
782/782 [==============================] - 149s 191ms/step - loss: 0.4607 - accuracy: 0.8512 - val_loss: 1.7498 - val_accuracy: 0.6204
Epoch 167/200
782/782 [==============================] - 150s 192ms/step - loss: 0.4592 - accuracy: 0.8535 - val_loss: 1.7621 - val_accuracy: 0.6229
Epoch 168/200
782/782 [==============================] - 147s 187ms/step - loss: 0.4700 - accuracy: 0.8491 - val_loss: 1.7562 - val_accuracy: 0.6215
Epoch 169/200
782/782 [==============================] - 150s 192ms/step - loss: 0.4676 - accuracy: 0.8521 - val_loss: 1.7584 - val_accuracy: 0.6218
Epoch 170/200
782/782 [==============================] - 147s 188ms/step - loss: 0.4685 - accuracy: 0.8500 - val_loss: 1.7616 - val_accuracy: 0.6168
Epoch 171/200
782/782 [==============================] - 117s 150ms/step - loss: 0.4546 - accuracy: 0.8556 - val_loss: 1.7409 - val_accuracy: 0.6232
Epoch 172/200
782/782 [==============================] - 105s 134ms/step - loss: 0.4658 - accuracy: 0.8512 - val_loss: 1.7459 - val_accuracy: 0.6255
Epoch 173/200
782/782 [==============================] - 105s 134ms/step - loss: 0.4664 - accuracy: 0.8498 - val_loss: 1.7775 - val_accuracy: 0.6212
Epoch 174/200
782/782 [==============================] - 103s 132ms/step - loss: 0.4644 - accuracy: 0.8508 - val_loss: 1.7575 - val_accuracy: 0.6217
Epoch 175/200
782/782 [==============================] - 103s 132ms/step - loss: 0.4736 - accuracy: 0.8521 - val_loss: 1.7708 - val_accuracy: 0.6204
Epoch 176/200
782/782 [==============================] - 104s 133ms/step - loss: 0.4634 - accuracy: 0.8509 - val_loss: 1.7416 - val_accuracy: 0.6250
Epoch 177/200
782/782 [==============================] - 103s 132ms/step - loss: 0.4545 - accuracy: 0.8542 - val_loss: 1.7291 - val_accuracy: 0.6245
Epoch 178/200
782/782 [==============================] - 103s 132ms/step - loss: 0.4576 - accuracy: 0.8556 - val_loss: 1.7606 - val_accuracy: 0.6235
Epoch 179/200
782/782 [==============================] - 103s 132ms/step - loss: 0.4581 - accuracy: 0.8543 - val_loss: 1.7748 - val_accuracy: 0.6218
Epoch 180/200
782/782 [==============================] - 103s 132ms/step - loss: 0.4572 - accuracy: 0.8551 - val_loss: 1.7697 - val_accuracy: 0.6193
Epoch 181/200
782/782 [==============================] - 103s 132ms/step - loss: 0.4618 - accuracy: 0.8522 - val_loss: 1.7641 - val_accuracy: 0.6227
Epoch 182/200
782/782 [==============================] - 103s 132ms/step - loss: 0.4523 - accuracy: 0.8555 - val_loss: 1.7575 - val_accuracy: 0.6208
Epoch 183/200
782/782 [==============================] - 103s 132ms/step - loss: 0.4600 - accuracy: 0.8531 - val_loss: 1.8130 - val_accuracy: 0.6184
Epoch 184/200
782/782 [==============================] - 103s 132ms/step - loss: 0.4632 - accuracy: 0.8522 - val_loss: 1.7883 - val_accuracy: 0.6200
Epoch 185/200
782/782 [==============================] - 103s 131ms/step - loss: 0.4545 - accuracy: 0.8574 - val_loss: 1.7649 - val_accuracy: 0.6184
Epoch 186/200
782/782 [==============================] - 103s 131ms/step - loss: 0.4553 - accuracy: 0.8563 - val_loss: 1.7560 - val_accuracy: 0.6192
Epoch 187/200
782/782 [==============================] - 103s 132ms/step - loss: 0.4544 - accuracy: 0.8545 - val_loss: 1.7823 - val_accuracy: 0.6236
Epoch 188/200
782/782 [==============================] - 104s 133ms/step - loss: 0.4588 - accuracy: 0.8555 - val_loss: 1.8080 - val_accuracy: 0.6155
Epoch 189/200
782/782 [==============================] - 103s 132ms/step - loss: 0.4594 - accuracy: 0.8533 - val_loss: 1.7499 - val_accuracy: 0.6247
Epoch 190/200
782/782 [==============================] - 103s 132ms/step - loss: 0.4630 - accuracy: 0.8518 - val_loss: 1.7599 - val_accuracy: 0.6219
Epoch 191/200
782/782 [==============================] - 103s 132ms/step - loss: 0.4536 - accuracy: 0.8533 - val_loss: 1.7620 - val_accuracy: 0.6222
Epoch 192/200
782/782 [==============================] - 103s 132ms/step - loss: 0.4631 - accuracy: 0.8493 - val_loss: 1.7791 - val_accuracy: 0.6217
Epoch 193/200
782/782 [==============================] - 104s 132ms/step - loss: 0.4577 - accuracy: 0.8512 - val_loss: 1.7623 - val_accuracy: 0.6186
Epoch 194/200
782/782 [==============================] - 104s 132ms/step - loss: 0.4579 - accuracy: 0.8545 - val_loss: 1.7892 - val_accuracy: 0.6214
Epoch 195/200
782/782 [==============================] - 104s 132ms/step - loss: 0.4534 - accuracy: 0.8552 - val_loss: 1.7889 - val_accuracy: 0.6172
Epoch 196/200
782/782 [==============================] - 103s 132ms/step - loss: 0.4646 - accuracy: 0.8551 - val_loss: 1.7746 - val_accuracy: 0.6264
Epoch 197/200
782/782 [==============================] - 103s 132ms/step - loss: 0.4578 - accuracy: 0.8569 - val_loss: 1.7881 - val_accuracy: 0.6204
Epoch 198/200
782/782 [==============================] - 106s 136ms/step - loss: 0.4561 - accuracy: 0.8536 - val_loss: 1.7681 - val_accuracy: 0.6258
Epoch 199/200
782/782 [==============================] - 108s 138ms/step - loss: 0.4500 - accuracy: 0.8576 - val_loss: 1.7345 - val_accuracy: 0.6247
Epoch 200/200
782/782 [==============================] - 109s 140ms/step - loss: 0.4551 - accuracy: 0.8554 - val_loss: 1.8023 - val_accuracy: 0.6157
train_acc : 0.8558200001716614
valid_acc : 0.6284999847412109
train_loss: 0.4507897198200226
valid_loss: 1.5751466751098633
best acc is...: 0.6284999847412109
