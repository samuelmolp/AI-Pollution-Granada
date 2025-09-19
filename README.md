# IA-Pollution-Granada

## AIM
Granada has been pointed out as the most polluted city in Spain. Thus, I decided to create a neural network model to predict NO2, SO2, PM10 and PM25 as my investigation in the IB Diploma. 

## PROCEDURE 
The historic data training set was obtained by ([Junta De Andalucía Open Data ](https://www.juntadeandalucia.es/datosabiertos/portal/dataset/datos-cuantitativos-diarios-del-indice-de-calidad-del-aire-en-andalucia/resource/a58a1e21-2800-4652-b99e-1921dd5f57d1)) to get pollution levels and AEMET to get weather parametres. 
The implementation was made in Python using Tensorflow and included three approaches: a linear implementation, one with hidden layers and a Recurrent Neural Network. During my research, I experimented with several architectures to optimize the best number of neurons, layers, epochs...

## RESULTS 
Results were highly satisfactory. PM10 and PM25 were predicted with less than 1.5% error in small time periods and less than 10% error in 24 hours predictions. NO2 and SO2 were forecasted with a higher error (up to 15% in small periods and 30% in 24 hours) but the neural network was proved to be effective under certain stable conditions (such as no heavy rain or high sudden changes). 
For example, for a prediction of NO2:

![Example](/Example.png)

Even if all models gave similar results under stable conditions, the RNN was proved to be the best approach when there are sudden changes or adverse weather conditions. The RNN was also the best approach as the predictions are of a longer period of time (more than 24 hours). 

