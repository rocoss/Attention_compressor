# Учебный проект, посвященный сжатию текста при помощью LSTM/GRU с механизмом внимания

## Описание модификации  
Чтобы улучшить результат исходного проекта код, написанный на *Tensorflow*, был переписан на Pytorch. В тестировании участвовали архитектура **GRU** и **LSTM**, в которые был добавлен механизм внимания. Тестирование проводилось на операционной системе Astra Linux на видеокарте NVIDIA GeForce RTX 3060 c 12288 MiB памяти.  Это позволило более чем в 20 раз сократить время, затрачиваемое на кодирование и декодирование, с незначительным улучшением размера и коэффициент сжатия. В ходе тестирование было проверено огромное количество гиперпараметров, но выбор пал на наиболее оптимальные
 
Размер батча для обучения

batch_size = 64  

Длина последовательности для обучения  
seq_length = 15  
Количество единиц в каждом слое GRU  
hidden_size = 1024  
Количество слоев GRU  
num_layers = 2  
 Размер слоя эмбеддинга  
embed_size = 512  
 Начальная скорость обучения для оптимизатора  
learning_rate = 0.0005

Гиперпараметры в тестируемых моделях с attention полностью идентичны.  


  Также было проведено сжатие стандартными архиваторами системы... Результаты в таблице. Хочется обратить внимание на колоссальную разницу во времени работы.

| Версия           | Исходный размер, байты | Размер после сжатия, байты | Коэффициент сжатия | Затраченное время, с |     |
| ---------------- | ---------------------- | -------------------------- | ------------------ | -------------------- | --- |
| Baseline         | 100000                 | 47150                      | 2.12               | 716                  |     |
| GRU_attCompress  | 100000                 | 34381                      | 2.91               | 44                   |     |
| LSTM_attCompress | 100000                 | 34809                      | 2.87               | 59                   |     |
| gzip             | 100000                 | 36145                      | 2.77               | 0.003563404083251953 |     |
| tar.gz           | 100000                 | 36242                      | 2.74               | 0.004873514175415039 |     |
  

  
### Оригинал  
https://ctlab.itmo.ru/gitlab/eabelyaev/lstmcompressor
