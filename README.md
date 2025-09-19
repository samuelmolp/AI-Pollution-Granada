# IA-Pollution-Granada

## AIM
Granada has been identified as the most polluted city in Spain. Thus, I decided to create a neural network model to predict NO2, SO2, PM10 and PM25 as my investigation in the IB Diploma. 

## PROCEDURE 
The historic data training set was obtained by ([Junta De Andalucía Open Data ](https://www.juntadeandalucia.es/datosabiertos/portal/dataset/datos-cuantitativos-diarios-del-indice-de-calidad-del-aire-en-andalucia/resource/a58a1e21-2800-4652-b99e-1921dd5f57d1)) to get pollution levels and AEMET (Spanish Meteorology Agency) to get weather parametres. 
The implementation was made in Python using Tensorflow and included three approaches: a linear model, a dense model and a Recurrent Neural Network. 
During my research, I conducted several experiments and tried multiple architectures to determine the optimal number of neurons, layers, and training parameters.

## RESULTS 
Results were highly satisfactory. PM10 and PM25 were predicted with less than 1.5% error for short-term predictions and less than 10% error for 24-hour predictions. NO2 and SO2 were forecasted with higher errors (up to 15% for short-term predictions and 30% for 24-hour predictions), but the neural network proved to be effective under certain stable conditions, such as no heavy rain or sudden large changes in concentrations.

For example, for a prediction of NO2:

![Example](/Example.png)

Even if all models gave similar results under stable conditions, the RNN was proved to be the best approach when there are sudden changes or adverse weather conditions. The RNN was also the best approach for longer-term predictions, exceeding 24 hours.

